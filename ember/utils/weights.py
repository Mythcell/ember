from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

INIT_FN_MAP: dict[str, Callable[..., Any]] = {
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "orthogonal": nn.init.orthogonal_,
    "ones": nn.init.ones_,
    "zeros": nn.init.zeros_,
    "constant": nn.init.constant_,
}


def init_weights(
    model: nn.Module,
    config: dict[type[nn.Module], dict[str, Any]],
    verbose: bool = False,
) -> None:
    """Initialize model weights according to exact module types.

    Args:
        model: Module tree to initialize.
        config: Mapping from exact module types to initialization settings. Modules
            with weights require a `mode` from `INIT_FN_MAP`; `bias_val` is applied
            separately, and remaining values are passed to the initializer.
        verbose: Whether to print successful initialization operations.

    Raises:
        ValueError: If a matching configuration omits `mode` or names an unknown mode.
    """
    for module_name, m in model.named_modules():
        module_type = type(m)
        if module_type in config:
            params = config[module_type].copy()
            if hasattr(m, "weight") and m.weight is not None:
                try:
                    mode = params.pop("mode")
                except KeyError as err:
                    raise ValueError(
                        f"Configuration for {module_type.__name__} "
                        f"is missing the required 'mode' key."
                    ) from err

                if mode not in INIT_FN_MAP:
                    raise ValueError(f"Invalid initialization mode: {mode}")

                # Pop bias_val here as it's not a weight init parameter
                params.pop("bias_val", None)
                init_fn = INIT_FN_MAP[mode]

                try:
                    init_fn(m.weight, **params)
                    if verbose:
                        print(
                            f"Initialized '{module_name}' ({module_type.__name__}) "
                            f"weights with {mode}({params})"
                        )
                except TypeError as e:
                    print(
                        f"Warning: Could not initialize '{module_name}'. "
                        f"Check if kwargs {params} are valid for {mode}. Error: {e}"
                    )
            bias = getattr(m, "bias", None)
            if isinstance(bias, torch.Tensor) and "bias_val" in config[module_type]:
                bias_val = config[module_type]["bias_val"]
                nn.init.constant_(bias, bias_val)
                if verbose:
                    print(
                        f"Initialized '{module_name}' ({module_type.__name__}) "
                        f"bias to {bias_val}"
                    )
