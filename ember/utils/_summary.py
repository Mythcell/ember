from typing import Any

import torch
import torch.nn as nn

InputSize = tuple[int, ...] | torch.Size | list[tuple[int, ...] | torch.Size] | None


def model_summary(
    model: nn.Module,
    input_size: InputSize,
    input_data: torch.Tensor | list[torch.Tensor] | None,
    depth: int,
    **kwargs,
) -> Any:
    """Display a model summary with torchinfo."""
    import torchinfo

    return torchinfo.summary(
        model=model,
        input_size=input_size,
        input_data=input_data,
        depth=depth,
        **kwargs,
    )
