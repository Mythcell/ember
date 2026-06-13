import torch.nn as nn


def set_requires_grad(
    model: nn.Module,
    value: bool,
    layer_names: list[str] | None = None,
    layer_indices: list[int] | None = None,
    recurse: bool = False,
) -> None:
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
