import pytest
import torch.nn as nn

from ember.utils import instantiate


def test_local_module_precedence(tmp_path) -> None:
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
    (first_directory / "shared_module.py").write_text(
        module_template.format(source="first").strip()
    )
    (second_directory / "shared_module.py").write_text(
        module_template.format(source="second").strip()
    )

    first = instantiate(
        "shared_module.LocalModule",
        local_path=first_directory,
        expected_type=nn.Module,
    )
    second = instantiate(
        "shared_module.LocalModule",
        local_path=second_directory,
        expected_type=nn.Module,
    )

    assert first.source == "first"
    assert second.source == "second"


def test_missing_dependency(tmp_path) -> None:
    (tmp_path / "broken_module.py").write_text(
        """
import missing_dependency

class Broken:
    pass
""".strip()
    )

    with pytest.raises(ModuleNotFoundError) as error:
        instantiate("broken_module.Broken", local_path=tmp_path)

    assert error.value.name == "missing_dependency"


def test_import_aliases() -> None:
    activation = instantiate("nn.ReLU")

    assert isinstance(activation, nn.ReLU)


def test_invalid_class_spec() -> None:
    with pytest.raises(ValueError, match="Invalid class_spec: ReLU"):
        instantiate("ReLU")


def test_expected_type() -> None:
    with pytest.raises(TypeError, match=r"pathlib\.Path must inherit from Module"):
        instantiate("pathlib.Path", expected_type=nn.Module)


def test_constructor_errors() -> None:
    with pytest.raises(TypeError, match=r"Failed to instantiate torch\.nn\.ReLU"):
        instantiate("torch.nn.ReLU", params={"bogus": True})
