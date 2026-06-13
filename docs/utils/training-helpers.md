# Training Helpers

These helpers are small conveniences around common PyTorch training tasks. They
are useful, but they are not a replacement for PyTorch APIs.

## Freezing Parameters

`EmberModel.freeze()` and `EmberModel.unfreeze()` call
`ember.utils.grad.set_requires_grad()` internally.

```python
em.freeze()
em.unfreeze(layer_names=["_forward_module.encoder"])
```

| Argument | Notes |
| --- | --- |
| `layer_names` | Match names from `model.named_modules()`. Fabric-wrapped models may prefix names. |
| `layer_indices` | Match modules by `named_modules()` enumeration index. |
| `recurse` | Passed to `module.parameters(recurse=...)` for matched modules. |

With no names or indices, every model parameter is toggled.

## Weight Initialization

`init_weights(model, config, verbose=False)` initializes modules by exact layer
type.

```python
import torch.nn as nn

from ember.utils.weights import init_weights

init_weights(
    model,
    {
        nn.Conv2d: {"mode": "kaiming_normal", "nonlinearity": "relu"},
        nn.Linear: {"mode": "xavier_uniform", "bias_val": 0.0},
    },
)
```

Supported modes are `kaiming_uniform`, `kaiming_normal`, `xavier_uniform`,
`xavier_normal`, `orthogonal`, `ones`, `zeros`, and `constant`. Extra keys are
passed to the matching `torch.nn.init` function, except `bias_val`, which sets a
constant bias when the module has one.

## Parameter Schedulers

Ember includes epoch-based parameter schedulers for values such as VAE beta
weights:

```python
from ember.schedulers import LinearParamScheduler

beta = LinearParamScheduler(
    start_value=0.0,
    end_value=1.0,
    startup_epochs=5,
    interp_epochs=20,
)
value = beta(epoch)
```

| Scheduler | Behavior |
| --- | --- |
| `LinearParamScheduler` | Hold the start value, then linearly interpolate. |
| `ExponentialParamScheduler` | Hold the start value, then interpolate with growth or decay profile. |
| `PiecewiseLinearParamScheduler` | Interpolate across `(value, epoch)` milestones. |
| `PiecewiseLinearLRScheduler` | PyTorch LR scheduler using piecewise linear milestones. |

Milestone epochs must be positive and unique. Values are clamped to the final
milestone after the last segment.
