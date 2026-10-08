from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, assert_type

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf
from typer.testing import CliRunner

from ember import EmberRunner
from ember.cli import app

if TYPE_CHECKING:
    from ember import EmberModel
    from ember.models import EmberModel as DirectEmberModel

    def check_model_result(runner: EmberRunner[EmberModel]) -> None:
        assert_type(runner.run(), DirectEmberModel)


class NoResultRunner(EmberRunner):
    def run(self) -> None:
        pass


class ConfigRunner(EmberRunner[DictConfig | ListConfig | None]):
    def run(self) -> DictConfig | ListConfig | None:
        return self.cfg


@dataclass(frozen=True)
class RunnerFiles:
    config: Path
    run: Path
    run_instance: Path
    run_with_instantiate: Path
    invalid: dict[str, Path]


@pytest.fixture(scope="module")
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def runner_files(tmp_path: Path) -> RunnerFiles:
    run = tmp_path / "run.py"
    run.write_text(
        """
from ember import EmberRunner

class MyRunner(EmberRunner[int]):
    def run(self) -> int:
        print(self.cfg.var)
        return 7
""".strip()
    )
    run_instance = tmp_path / "run_instance.py"
    run_instance.write_text(
        """
from ember import EmberRunner

class MyRunner(EmberRunner):
    def run(self):
        print(self.cfg.var)

runner = MyRunner()
""".strip()
    )
    run_with_instantiate = tmp_path / "run_with_instantiate.py"
    run_with_instantiate.write_text(
        """
import torch.nn as nn

from ember import EmberRunner
from ember.utils import instantiate

class MyRunner(EmberRunner):
    def run(self):
        model: nn.Module = instantiate(
            self.cfg.model_cls,
            self.cfg.model_params,
            local_path=self.script_dir,
            expected_type=nn.Module,
        )
        model.print_params()
""".strip()
    )
    (tmp_path / "runner_model.py").write_text(
        """
import torch.nn as nn

class TestModule(nn.Module):
    def __init__(self, param1: int, param2: float, param3: bool) -> None:
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    def print_params(self) -> None:
        print(self.param1, self.param2, self.param3)
""".strip()
    )

    invalid_sources = {
        "no_runner": "VALUE = 1",
        "multiple": """
from ember import EmberRunner

class RunnerOne(EmberRunner):
    def run(self):
        pass

class RunnerTwo(EmberRunner):
    def run(self):
        pass
""".strip(),
        "invalid_instance": "runner = object()",
        "bad_syntax": "def broken(:\n    pass",
    }
    invalid: dict[str, Path] = {}
    for name, source in invalid_sources.items():
        path = tmp_path / f"{name}.py"
        path.write_text(source)
        invalid[name] = path

    config = tmp_path / "config.yaml"
    config.write_text(
        """
var: test
model_cls: runner_model.TestModule
model_params:
  param1: 10
  param2: 0.3
  param3: true
""".strip()
    )
    return RunnerFiles(
        config=config,
        run=run,
        run_instance=run_instance,
        run_with_instantiate=run_with_instantiate,
        invalid=invalid,
    )


@pytest.mark.parametrize(
    "cfg",
    [
        OmegaConf.create({}),
        OmegaConf.create({"value": 1}),
        OmegaConf.create([]),
        OmegaConf.create([1]),
    ],
    ids=["empty_dict", "dict", "empty_list", "list"],
)
def test_in_memory_config_is_shared(cfg: DictConfig | ListConfig) -> None:
    runner = ConfigRunner(cfg=cfg)

    assert runner.cfg is cfg
    assert runner.cfg_path is None
    assert runner.run() is cfg

    if isinstance(cfg, DictConfig):
        assert isinstance(runner.cfg, DictConfig)
        cfg["value"] = 2
        assert runner.cfg["value"] == 2
        runner.cfg["value"] = 3
        assert cfg["value"] == 3
    else:
        cfg.append(2)
        assert runner.cfg[-1] == 2
        runner.cfg[-1] = 3
        assert cfg[-1] == 3


def test_runner_loads_config_with_positional_metadata(
    runner_files: RunnerFiles,
) -> None:
    runner = ConfigRunner(runner_files.config, runner_files.run.parent, 2)

    assert runner.cfg_path == runner_files.config
    assert runner.script_dir == runner_files.run.parent
    assert runner.verbosity == 2
    assert isinstance(runner.cfg, DictConfig)
    assert runner.cfg.var == "test"


def test_runner_without_config_returns_none() -> None:
    runner = ConfigRunner()

    assert runner.cfg is None
    assert runner.cfg_path is None
    assert runner.run() is None


@pytest.mark.parametrize("cfg", [OmegaConf.create({}), OmegaConf.create([])])
def test_conflicting_config_sources_fail_before_loading(
    tmp_path: Path, cfg: DictConfig | ListConfig
) -> None:
    with pytest.raises(ValueError, match="Supply either cfg or cfg_path"):
        ConfigRunner(cfg_path=tmp_path / "missing.yaml", cfg=cfg)


def test_runner_result_types() -> None:
    assert_type(NoResultRunner().run(), None)
    assert_type(ConfigRunner().run(), DictConfig | ListConfig | None)

    runners: list[EmberRunner[object]] = [NoResultRunner(), ConfigRunner()]
    for runner in runners:
        assert_type(runner.run(), object)


@pytest.mark.parametrize("runner_attribute", ["run", "run_instance"])
def test_cli_runs_discovered_runner_variants(
    cli_runner: CliRunner,
    runner_files: RunnerFiles,
    runner_attribute: str,
) -> None:
    result = cli_runner.invoke(
        app,
        [
            str(getattr(runner_files, runner_attribute)),
            "-c",
            str(runner_files.config),
        ],
    )

    assert result.exit_code == 0
    assert "test" in result.stdout


@pytest.mark.parametrize("config_source", ["{}", "{'var': 'in-memory'}", "[]", "[1]"])
def test_cli_preserves_exported_in_memory_config(
    cli_runner: CliRunner,
    runner_files: RunnerFiles,
    config_source: str,
) -> None:
    script = runner_files.run.parent / "configured_runner.py"
    script.write_text(
        f"""
from pathlib import Path
from omegaconf import OmegaConf
from ember import EmberRunner

cfg = OmegaConf.create({config_source})

class ConfiguredRunner(EmberRunner[int]):
    def run(self) -> int:
        assert self.cfg is cfg
        assert self.cfg_path is None
        assert self.script_dir == Path(__file__).parent
        assert self.verbosity == 2
        return 7

runner = ConfiguredRunner(cfg=cfg)
""".strip()
    )

    result = cli_runner.invoke(
        app, [str(script), "-c", str(runner_files.config), "-vv"]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == ""


def test_cli_runner_instantiates_local_objects(
    cli_runner: CliRunner,
    runner_files: RunnerFiles,
) -> None:
    result = cli_runner.invoke(
        app,
        [
            str(runner_files.run_with_instantiate),
            "-c",
            str(runner_files.config),
        ],
    )

    assert result.exit_code == 0
    assert "10 0.3 True" in result.stdout


def test_cli_reports_runner_discovery_errors(
    cli_runner: CliRunner,
    runner_files: RunnerFiles,
) -> None:
    cases = {
        "no_runner": "No Runner subclass found",
        "multiple": "Multiple Runner subclasses found",
        "invalid_instance": "Variable `runner` must be an EmberRunner instance",
        "bad_syntax": "failed to load script",
    }

    for name, message in cases.items():
        result = cli_runner.invoke(app, [str(runner_files.invalid[name])])
        assert result.exit_code == 1
        assert message in result.output
