import pytest
import torch.nn as nn

from ember.utils import instantiate


def test_instantiate_prefers_matching_local_module_over_cached_module(tmp_path) -> None:
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


def test_instantiate_preserves_missing_dependency_errors(tmp_path) -> None:
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


def test_instantiate_supports_import_aliases() -> None:
    activation = instantiate("nn.ReLU")

    assert isinstance(activation, nn.ReLU)


def test_instantiate_rejects_invalid_class_spec() -> None:
    with pytest.raises(ValueError, match="Invalid class_spec: ReLU"):
        instantiate("ReLU")


def test_instantiate_rejects_wrong_expected_type() -> None:
    with pytest.raises(TypeError, match=r"pathlib\.Path must inherit from Module"):
        instantiate("pathlib.Path", expected_type=nn.Module)


def test_instantiate_wraps_constructor_type_errors() -> None:
    with pytest.raises(TypeError, match=r"Failed to instantiate torch\.nn\.ReLU"):
        instantiate("torch.nn.ReLU", params={"bogus": True})
