from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ember.cli import app


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

class MyRunner(EmberRunner):
    def run(self):
        print(self.cfg.var)
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
