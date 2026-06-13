# EmberModel

`EmberModel` is the main interface to Ember's lightweight training library. It
wraps a standard `torch.nn.Module`, a Lightning `Fabric` instance, optimizer
state, and optional loss, metrics, augmentation, and schedulers.

The built-in loop is deliberately narrow. It works well for conventional
supervised PyTorch workloads with tuple-like `(input, target)` batches, while
remaining easy to override when the experiment needs different step logic.

In general, you should subclass `EmberModel` for custom objectives,
nonstandard batch structures, multi-part models, or any workflow that does not
match the default assumptions.

## Construction

```python
from lightning import Fabric
from torchmetrics import Accuracy

from ember import EmberModel

em = EmberModel(
    fabric=Fabric(accelerator="auto"),
    model=model,
    optimizers=[optimizer],
    loss_fn=loss_fn,
    metrics=[Accuracy(task="multiclass", num_classes=10)],
    schedulers=[scheduler],
)
```

By default, `EmberModel` calls `fabric.setup()` on the model and optimizers.
Pass `setup_fabric=False` or override `fabric_setup()` when Fabric setup needs
to be customized.

| Argument | Purpose |
| --- | --- |
| `fabric` | Lightning Fabric accelerator and logger owner. |
| `model` | The wrapped PyTorch module. |
| `optimizers` | Optimizers used by the default loop or custom steps. |
| `loss_fn` | Loss used by the default `train_step()`, `val_step()`, and `eval_step()`. Passing `None` is only safe when custom steps do not rely on the default loss path. |
| `metrics` | TorchMetrics objects used for automatic metric calculation. |
| `aug_fn` | Batch augmentation function used by `augment_batch()`. |
| `schedulers` | Schedulers stepped after each epoch. |
| `log_lr` | `"auto"` logs LR only when schedulers exist; `True` always registers LR keys. |
| `random_seed` | Optional Fabric seed for reproducible training setup. |
| `deterministic` | Enables deterministic PyTorch algorithms and disables cuDNN benchmarking. |
| `no_validation` | Skips validation even if validation data is available. |
| `reduce_lr_on_plateau_monitor` | History key used for `ReduceLROnPlateau`; defaults to `val_loss`. |

## Default Batch Assumptions

The default `train_step()`, `val_step()`, and `eval_step()` assume that a batch
is tuple-like:

- `batch[0]` is the model input;
- `batch[-1]` is the target;
- `self.model(batch[0])` returns predictions;
- `self.loss_fn(y_pred, batch[-1])` returns a scalar loss;
- automated metrics can be called as `metric(y_pred, batch[-1])`.

This covers many supervised datasets shaped like `(x, y)`. A single-item batch
also works because `batch[0]` and `batch[-1]` refer to the same item, which can
be useful for simple reconstruction tasks. For more specialized autoencoders,
custom target handling, nonstandard metric inputs, contrastive learning, GANs,
or multi-loss objectives, override the relevant step and metric hooks directly.

## The Default Loop

`fit()`, `evaluate()`, and `predict()` are public orchestration methods, not
subclass extension points. Custom `EmberModel` subclasses should not override
them. Override the smaller hook methods instead: `train_step()`, `val_step()`,
`eval_step()`, `predict_step()`, `setup_tracker()`, `calculate_metric()`, or
`scheduler_step()`.

The default loop handles:

- Fabric dataloader setup;
- train and validation phase switching;
- progress bars and metric printing;
- [callback hooks](callbacks.md);
- `epochs` or `steps` stopping;
- implicit continuation when `fit()` is called again;
- gradient accumulation with `accumulate_grad_batches`;
- gradient clipping with `gradients_clip_val` or `gradients_max_norm`;
- `torchmetrics` updates when a step returns `(loss, y_pred)`;
- epoch-level metric synchronization through [`EmberTracker`](tracking.md);
- scheduler stepping after each epoch;
- optional learning-rate tracking when schedulers are present, or when
  `log_lr=True`.

For `ReduceLROnPlateau`, `scheduler_step()` looks up
`reduce_lr_on_plateau_monitor`, which defaults to `val_loss`. If that key has no
history, the plateau scheduler is skipped for that epoch.

## Step Methods

`EmberModel` includes all the key lifecycle steps: `train_step`, `val_step`,
`eval_step`, and `predict_step`. The step-method return value is **optional**
and is only used for automatic metric calculation.

Return `(loss, y_pred)` when the default loop should update `train_loss`,
`val_loss`, and any constructor metrics:

```python
def train_step(self, batch, batch_idx: int, epoch: int):
    x, y = batch[0], batch[-1]
    (optimizer, *_) = self.optimizers

    y_pred = self.model(x)
    loss = self.loss_fn(y_pred, y)
    self.fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    return loss, y_pred
```

If you are manually calculating metrics or updating the tracker yourself, you
can simply omit the return value (i.e. return `None`).

```python
class CustomModel(EmberModel):
    def setup_tracker(self) -> None:
        self.tracker.register(["loss_a", "loss_b"], key_type="train")

    def train_step(self, batch, batch_idx: int, epoch: int) -> None:
        loss_a, loss_b = compute_losses(batch)
        loss = loss_a + loss_b
        self.fabric.backward(loss)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        batch_size = batch[0].shape[0]
        self.tracker.update("loss_a", loss_a.detach(), batch_size=batch_size)
        self.tracker.update("loss_b", loss_b.detach(), batch_size=batch_size)
```

GAN-style loops use the same pattern: register custom keys such as `g_loss` and
`d_loss`, update those keys inside `train_step()`, and return `None` because
automated supervised metrics are not meaningful there.

## Metric Semantics

Automatic constructor metrics are treated as batch callables. When a step
returns `(loss, y_pred)`, the default loop calls
`calculate_metric(metric, batch, y_pred)`, whose default implementation is
`metric(y_pred, batch[-1])`. The returned batch value is sent to
[`EmberTracker`](tracking.md), which stores a batch-size-weighted mean at epoch
sync time.

This path does not call TorchMetrics `.compute()` or `.reset()`. It is useful
for simple metrics where averaging per-batch values is acceptable. For macro F1,
AUROC, ranking metrics, calibration metrics, and other stateful or non-additive
metrics, update the metric manually and log a computed epoch value through a
custom tracker key. See [Tracking](tracking.md) for a true epoch-metric pattern.

Override `calculate_metric()` when the default input signature is the only
mismatch. Override step methods and return `None` when metric ownership should
move fully into the subclass.

## Validation And Evaluation

`val_step()` runs inside `torch.inference_mode()` during `fit()`. It may return
`(loss, y_pred)` for default validation metric updates, or `None` if the
subclass updates validation keys manually.

`evaluate()` creates a separate `EmberTracker` with Fabric logging disabled and
calls `eval_step(batch, batch_idx, eval_tracker)`. This allows custom
subclasses to write evaluation-specific values without mutating training
history.

```python
metrics = em.evaluate(eval_data=test_loader)
```

If an `EmberData` instance is supplied, `evaluate()` uses
`emberdata.eval_dataloader()`.

## Prediction

`predict()` calls `predict_step()` under `torch.inference_mode()`. The default
step returns `self.model(batch[0])`.

If `predict_step()` returns a single tensor, Ember wraps it internally and still
returns a one-item list. If it returns a tuple of tensors, each output is
handled separately. With `concatenate_preds=True`, batch outputs are
concatenated by output position:

```python
preds = em.predict(data=predict_loader)
```

Set `concatenate_preds=False` to keep a list of per-batch outputs.

## Dataloaders And EmberData

You can pass dataloaders directly:

```python
em.fit(train_data=train_loader, val_data=val_loader, epochs=10)
```

Or provide an `EmberData` instance:

```python
em.fit(emberdata=data, epochs=10)
```

When `emberdata` is supplied, it takes priority over direct dataloader
arguments. `fit()` calls the `EmberData` setup path for stage `"fit"`, then
uses `train_dataloader()` and `val_dataloader()`. `evaluate()` and `predict()`
use their matching stages and dataloader methods.

If stage execution fails after an `EmberData` instance is supplied, Ember still
calls `cleanup(stage=...)` before re-raising the original exception. See
[Data](data.md) for the full `EmberData` lifecycle.

## Checkpoints And Weights

`save_checkpoint()` and `load_checkpoint()` use Fabric checkpointing and include:

- `self.model`;
- each optimizer as `optimizer_0`, `optimizer_1`, and so on;
- each scheduler as `scheduler_0`, `scheduler_1`, and so on;
- `self.tracker`;
- `self.step_count`.

`save_weights()` and `load_weights()` are thinner helpers for model weights
only.

Repeated calls to `fit()` continue from the current `epoch_count` and
`step_count`. Loading a checkpoint restores model weights, optimizer state,
scheduler state, tracker history, and step count, then marks the model as
resuming.

`save_checkpoint(path)` creates parent directories and writes Fabric checkpoint
state. `load_checkpoint(path)` restores the model, optimizers, schedulers,
tracker, and step count.

`load_best_checkpoint()` and `load_last_checkpoint()` load paths set by
[`ModelCheckpoint`](callbacks.md). `save_weights(path)` and
`load_weights(path)` handle model weights only.

`summary(...)` displays a TorchInfo model summary for the wrapped model. The
TorchInfo import happens when `summary()` is called.

`freeze(...)` and `unfreeze(...)` toggle `requires_grad` globally or by module
name/index.

## When to Subclass

### Default `EmberModel` Limitations

The default `EmberModel` is deliberately opinionated towards a Keras-style
experience. However, it is not a drop-in replacement for Keras or PyTorch
Lightning. In particular, key limitations of the default `EmberModel` are:

- it assumes a single model and a single optimizer;
- the default supervised steps assume `batch[0]` input and `batch[-1]` target;
- automated metric calculation assumes metrics accept `(y_pred, target)`;
- automated metric history is a weighted average of per-batch metric outputs,
  not true TorchMetrics epoch `.compute()` semantics;

If your project uses more complex dataloaders, multiple optimizers, or
fine-grained loss calculation, then it is worth subclassing `EmberModel` and
overriding the relevant step and lifecycle hooks. Do not override `fit()`,
`evaluate()`, or `predict()`; keep those methods as the stable outer entry
points. See the [API Reference](../api.md).

### Subclass Responsibilities

When you override a step method, it is up to you to control the behavior it
replaces. For example, if a custom `train_step()` performs optimization
manually, then it is responsible for:

- zeroing gradients at the right time;
- calling `self.fabric.backward()`;
- stepping the relevant optimizer(s);
- handling multiple optimizers;
- applying gradient accumulation if desired;
- applying gradient clipping if desired;
- applying augmentation in the right place;
- registering and updating custom tracker keys if default metrics are not used.

The outer loop still handles callback hooks, epoch boundaries, tracker
synchronization, progress bars, scheduler stepping, and callback execution
around the custom step.

## Other Limitations

- Default gradient accumulation and clipping only apply to the default
  optimization pattern unless a subclass implements equivalent behavior;
- Fabric-compatible distributed primitives are present, but cluster and
  multi-node workflows have limited explicit testing.
