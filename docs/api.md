# API Reference

This page is a hand-written import map for Ember's documented surface. It is not
generated from docstrings.

## Preferred Top-Level Imports

```python
from ember import EmberData, EmberModel, EmberRunner, EmberTracker
```

| Symbol | Purpose |
| --- | --- |
| `EmberModel` | Fabric-backed `fit()` / `evaluate()` / `predict()` interface for a PyTorch model. |
| `EmberData` | Optional data setup and dataloader container. |
| `EmberTracker` | Metric history and Fabric logging helper. |
| `EmberRunner` | Base class for CLI runner scripts. |

## Submodule Imports

```python
from ember.callbacks import EarlyStopping, EmberCallback, ModelCheckpoint
from ember.callbacks import OptunaTrialPruner
from ember.mnist import EmberMNIST
from ember.schedulers import LinearParamScheduler
from ember.utils import instantiate
```

| Symbol | Purpose |
| --- | --- |
| `EmberCallback` | Base lifecycle hook class. |
| `ModelCheckpoint` | Fabric checkpoint saving callback. |
| `EarlyStopping` | Metric-based early stopping callback. |
| `OptunaTrialPruner` | Optuna pruning callback. |
| `EmberMNIST` | Small MNIST `EmberData` implementation used by examples. |
| `instantiate` | Construct trusted classes from `module.Class` strings and parameter dictionaries. |
| `ember.schedulers` | Small parameter and LR schedulers for epoch-based values. |

## Utility Modules

- `ember.utils.weights.init_weights`: exact-layer-type weight initialization.
- `ember.utils.grad`: gradient and `requires_grad` helpers used by
  `EmberModel.freeze()` and `EmberModel.unfreeze()`.
- See [Training Helpers](utils/training-helpers.md) for the documented freezing,
  initialization, and scheduler helpers.

## EmberModel Reference

See [EmberModel](core/embermodel.md) for usage patterns, default-loop behavior,
and subclassing guidance.

### Method Reference

!!! warning

    `fit()`, `evaluate()`, and `predict()` are public orchestration methods, not
    subclass extension points. Do not override them in custom `EmberModel`
    subclasses. Override the hook methods below instead: `train_step()`,
    `val_step()`, `eval_step()`, `predict_step()`, `setup_tracker()`,
    `calculate_metric()`, or `scheduler_step()`.

`fit(...)`:
Train for additional epochs or steps. Requires either dataloaders or an
`EmberData` instance, plus either `epochs` or `steps`. Empty training
dataloaders raise `ValueError`. `batch_limit=0` means all batches; a positive
integer limits batches directly; a nonzero float uses that fraction of the
dataloader and processes at least one batch. Use `skip_validation=True` to skip
validation for one call, or set `no_validation=True` on the model to make that
the default.

`evaluate(...)`:
Run evaluation once and return a fresh metric dictionary. Explicit `metrics`
override constructor metrics for that call. Empty evaluation dataloaders raise
`ValueError`.

`predict(...)`:
Run inference and return a list of model outputs. Batch outputs are
concatenated by output position by default. Empty prediction dataloaders raise
`ValueError`. Pass `concatenate_preds=False` when you want per-batch prediction
outputs instead of concatenated tensors.

### Hook Reference

`train_step(batch, batch_idx, epoch)`:
Default supervised forward pass, loss computation, backward pass, optional
accumulation/clipping, and optimizer step. Override when training is not simple
supervised learning.

`val_step(batch, batch_idx, epoch)`:
Default validation forward and loss in inference mode. Override when validation
needs custom outputs or metrics.

`eval_step(batch, batch_idx, eval_tracker)`:
Default one-off evaluation forward and loss. Override when evaluation differs
from validation.

`predict_step(batch, batch_idx)`:
Returns `self.model(batch[0])`. Override for multiple outputs or custom
post-processing.

`setup_tracker()`:
Registers default train/validation loss and metric keys. Override to add custom
history keys.

`calculate_metric(metric, batch, y_pred)`:
Calls `metric(y_pred, batch[-1])`. Override when metrics need different inputs.

`scheduler_step()`:
Steps schedulers after each epoch. Override when scheduler behavior depends on
custom state.

## Extension Patterns

- VAE-style subclasses use custom loss signatures while still returning
  `(loss, y_pred)` when default loss tracking is useful.
- GAN-style subclasses register manual tracker keys and return `None` from step
  methods when automatic supervised metrics do not apply.
