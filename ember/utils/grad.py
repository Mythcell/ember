import torch.nn as nn


def set_requires_grad(
    model: nn.Module,
    value: bool,
    layer_names: list[str] | None = None,
    layer_indices: list[int] | None = None,
    recurse: bool = False,
) -> None:
    """Set `requires_grad` globally or for selected named modules.

    Args:
        model: Module whose parameters are updated.
        value: Required gradient state.
        layer_names: Optional names from `model.named_modules()` to select.
        layer_indices: Optional indices from `model.named_modules()` to select.
        recurse: Whether selected modules include parameters from their descendants.
            This has no effect when no selectors are provided.
    """
    if layer_names or layer_indices:
        layer_name_set = set(layer_names) if layer_names else set()
        layer_index_set = set(layer_indices) if layer_indices else set()
        for idx, (name, module) in enumerate(model.named_modules()):
            if name in layer_name_set or idx in layer_index_set:
                for param in module.parameters(recurse=recurse):
                    param.requires_grad = value
    else:
        for param in model.parameters():
            param.requires_grad = value
