import sys
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, assert_type

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam, Optimizer

from ember.utils import instantiate


def build_activation(inplace: bool = False) -> nn.ReLU:
    return nn.ReLU(inplace=inplace)


class ActivationFactory:
    def __call__(self, inplace: bool = False) -> nn.ReLU:
        return build_activation(inplace)


if TYPE_CHECKING:

    def check_result_types() -> None:
        assert_type(instantiate(nn.ReLU), nn.ReLU)
        assert_type(instantiate(build_activation), nn.ReLU)
        assert_type(instantiate("torch.nn.ReLU"), Any)
        assert_type(instantiate("torch.optim.Adam", expected_type=Optimizer), Optimizer)
        assert_type(instantiate(nn.ReLU, expected_type=nn.Module), nn.Module)
        assert_type(instantiate(build_activation, expected_type=nn.Module), nn.Module)


@pytest.fixture
def local_modules() -> Iterator[None]:
    yield
    for name in tuple(sys.modules):
        if name.startswith("_ember_test_"):
            del sys.modules[name]


def test_optimizer() -> None:
    model = nn.Linear(4, 2)
    config = OmegaConf.create({"lr": 0.003})

    optimizer = instantiate(
        "torch.optim.Adam",
        model.parameters(),
        params=config,
        expected_type=Optimizer,
    )

    assert isinstance(optimizer, Adam)
    assert optimizer.param_groups[0]["lr"] == config.lr
    assert all(
        actual is expected
        for actual, expected in zip(
            optimizer.param_groups[0]["params"], model.parameters(), strict=True
        )
    )


@pytest.mark.parametrize(
    "target",
    [
        nn.ReLU,
        build_activation,
        f"{__name__}.build_activation",
        partial(nn.ReLU),
        ActivationFactory(),
    ],
)
def test_callable_targets(target: str | Callable[..., nn.ReLU]) -> None:
    activation = instantiate(
        target=target, params={"inplace": True}, expected_type=nn.Module
    )

    assert isinstance(activation, nn.ReLU)
    assert activation.inplace


def test_argument_forwarding(tmp_path: Path) -> None:
    dependency = torch.tensor([1.0])
    positional_mapping = {"dependency": dependency}
    nested = {"items": ["torch.nn.ReLU", dependency]}
    config = OmegaConf.create({"nested": {"value": 1}})
    params = MappingProxyType(
        {
            "nested": nested,
            "config": config.nested,
            "params": dependency,
            "local_path": tmp_path,
            "expected_type": object,
        }
    )
    calls = []

    def factory(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return dependency

    result = instantiate(
        factory, positional_mapping, dependency, params=params, local_path=tmp_path
    )

    assert result is dependency
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is positional_mapping
    assert args[1] is dependency
    assert kwargs.keys() == params.keys()
    assert all(kwargs[key] is value for key, value in params.items())


@pytest.mark.usefixtures("local_modules")
def test_local_module_precedence(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    module_template = """
import torch.nn as nn

class LocalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = "{source}"
"""
    (first_directory / "_ember_test_shared.py").write_text(
        module_template.format(source="first").strip()
    )
    (second_directory / "_ember_test_shared.py").write_text(
        "from _ember_test_neighbor import LocalModule\n"
    )
    (second_directory / "_ember_test_neighbor.py").write_text(
        module_template.format(source="second").strip()
    )

    original_path = sys.path.copy()
    first = instantiate(
        "_ember_test_shared.LocalModule",
        local_path=first_directory,
        expected_type=nn.Module,
    )
    second = instantiate(
        "_ember_test_shared.LocalModule",
        local_path=second_directory,
        expected_type=nn.Module,
    )

    assert first.source == "first"
    assert second.source == "second"
    assert sys.path == original_path
    cached_module = sys.modules["_ember_test_shared"]
    instantiate("_ember_test_shared.LocalModule", local_path=second_directory)
    assert sys.modules["_ember_test_shared"] is cached_module


@pytest.mark.usefixtures("local_modules")
def test_local_package(tmp_path: Path) -> None:
    package = tmp_path / "_ember_test_package"
    package.mkdir()
    (package / "__init__.py").write_text("from .model import Model\n")
    (package / "model.py").write_text("class Model: pass\n")

    model = instantiate("_ember_test_package.Model", local_path=tmp_path)
    repeated = instantiate("_ember_test_package.Model", local_path=tmp_path)

    assert type(model) is type(repeated)
    assert type(model).__module__ == "_ember_test_package.model"
    assert isinstance(instantiate("torch.nn.ReLU", local_path=tmp_path), nn.ReLU)


@pytest.mark.usefixtures("local_modules")
@pytest.mark.parametrize("cached", [False, True])
def test_missing_dependency(tmp_path: Path, cached: bool) -> None:
    original_module = None
    if cached:
        previous = tmp_path / "previous"
        previous.mkdir()
        (previous / "_ember_test_broken.py").write_text("class Broken: pass\n")
        instantiate("_ember_test_broken.Broken", local_path=previous)
        original_module = sys.modules["_ember_test_broken"]
    (tmp_path / "_ember_test_broken.py").write_text(
        """
import _ember_test_missing_dependency

class Broken:
    pass
""".strip()
    )

    original_path = sys.path.copy()
    with pytest.raises(ModuleNotFoundError) as error:
        instantiate("_ember_test_broken.Broken", local_path=tmp_path)

    assert error.value.name == "_ember_test_missing_dependency"
    assert sys.path == original_path
    assert sys.modules.get("_ember_test_broken") is original_module


def test_import_aliases() -> None:
    with pytest.raises(ModuleNotFoundError) as error:
        instantiate("nn.ReLU")
    assert error.value.name == "nn"
    assert isinstance(instantiate(nn.ReLU), nn.ReLU)


@pytest.mark.parametrize(
    "target",
    [
        "",
        "ReLU",
        ".torch.nn.ReLU",
        "torch..ReLU",
        "torch.nn.",
        " torch.nn.ReLU",
        "torch/nn.ReLU",
        "torch.nn.class",
        "torch.nn.1ReLU",
    ],
)
def test_invalid_target(target: str) -> None:
    with pytest.raises(ValueError, match="Invalid target"):
        instantiate(target)


def test_missing_and_non_callable_targets() -> None:
    with pytest.raises(ImportError, match="Target attribute not found"):
        instantiate("sys.ember_missing_target")
    targets: list[Any] = ["sys.version", None]
    for target in targets:
        with pytest.raises(TypeError, match="not callable"):
            instantiate(target)


def test_expected_type() -> None:
    with pytest.raises(TypeError, match=r"pathlib\.Path must inherit from Module"):
        instantiate("pathlib.Path", expected_type=nn.Module)

    calls = []

    class Incompatible:
        def __init__(self) -> None:
            calls.append("class")

    with pytest.raises(TypeError, match="Incompatible must inherit from Module"):
        instantiate(Incompatible, expected_type=nn.Module)
    assert calls == []

    def factory() -> object:
        calls.append("factory")
        return object()

    with pytest.raises(TypeError, match="factory must return an instance of Module"):
        instantiate(factory, expected_type=nn.Module)
    assert calls == ["factory"]

    class UnexpectedResult:
        def __new__(cls) -> Any:
            return object()

    with pytest.raises(TypeError, match="must return an instance of UnexpectedResult"):
        instantiate(UnexpectedResult, expected_type=UnexpectedResult)


def test_constructor_errors() -> None:
    with pytest.raises(
        TypeError, match=r"Failed to instantiate torch\.nn\.ReLU"
    ) as error:
        instantiate("torch.nn.ReLU", params={"bogus": True})
    assert isinstance(error.value.__cause__, TypeError)

    with pytest.raises(TypeError, match="Failed to instantiate") as error:
        instantiate(nn.ReLU, True, params={"inplace": False})
    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "exception", [TypeError("factory failed"), ValueError("factory failed")]
)
def test_factory_errors(exception: Exception) -> None:
    def factory() -> object:
        raise exception

    with pytest.raises(type(exception)) as error:
        instantiate(factory)

    if isinstance(exception, TypeError):
        assert error.value.__cause__ is exception
    else:
        assert error.value is exception
