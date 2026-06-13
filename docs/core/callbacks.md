# Callbacks

Callbacks inherit from `EmberCallback` and integrate with `EmberModel.fit()`.

```python
em.fit(
    train_data=train_loader,
    val_data=val_loader,
    epochs=100,
    callbacks=[checkpoint, early_stop],
)
```

## Lifecycle Hooks

Override only the hooks you need.

| Hook | Called |
| --- | --- |
| `on_train_start` / `on_train_end` | Around the full `fit()` call. |
| `on_epoch_start` / `on_epoch_end` | Around each epoch, including validation and scheduler stepping. |
| `on_train_epoch_start` / `on_train_epoch_end` | Around the train loop within an epoch. |
| `on_train_batch_start` / `on_train_batch_end` | Around each `train_step()`. |
| `on_validation_epoch_start` / `on_validation_epoch_end` | Around validation when validation runs. |
| `on_validation_batch_start` / `on_validation_batch_end` | Around each `val_step()`. |

Batch hooks receive `batch_idx`. Other hooks receive the active model, epoch,
and step.

`on_train_end` runs after a successful `fit()` loop. For cleanup that must run
even when a step method or callback raises, use [`EmberData.cleanup()`](data.md)
or your own `try` / `finally` around external resources.

```python
from ember.callbacks import EmberCallback


class PrintEpoch(EmberCallback):
    def on_epoch_end(self, embermodel, epoch: int = 0, step: int = 0) -> None:
        if embermodel.fabric.is_global_zero:
            print(epoch, embermodel.history)
```

## ModelCheckpoint

`ModelCheckpoint` saves Fabric checkpoints through
`embermodel.save_checkpoint()`.

```python
from ember.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_dir="ckpts",
    save_prefix="model",
    save_top_k=1,
    save_last=True,
)
```

| Option | Notes |
| --- | --- |
| `monitor` | Must be a registered history key, such as `val_loss`. |
| `mode` | `"min"` treats lower values as better; `"max"` treats higher values as better. |
| `save_best` | Convenience switch for one best checkpoint when `save_top_k` is left as `-1`. |
| `save_top_k` | `0` disables best checkpoints; positive values retain the best K. |
| `save_last` | Saves `{prefix}-epoch_{epoch:05d}_last.ckpt` at train end. |
| `save_every_n_epochs` | Saves periodic checkpoints independent of monitored metric availability. |
| `save_every_n_steps` | Saves completed-step checkpoints such as `model-step_0000002.ckpt`. |
| `warmup_epochs` | Skips epoch-end checkpoint handling before the warmup epoch count. |

Best checkpoints are named `{prefix}-epoch_{epoch:05d}.ckpt`. Periodic
checkpoints are named `{prefix}-epoch_{epoch:05d}_periodic.ckpt`.

Constructor validation catches obvious invalid values: `mode` must be `"min"` or
`"max"`, `save_top_k` must be at least `-1`, and periodic save intervals and
warmup epochs must be non-negative. An explicit `save_top_k=0` disables best
checkpoints even when `save_best=True`.

If validation is skipped and the monitored key has no values, best-checkpoint
logic is skipped for that epoch. Periodic and last checkpoints can still be
written.

## EarlyStopping

`EarlyStopping` sets `embermodel.should_train = False` once the monitored metric
has stalled.

```python
from ember.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    min_delta=1e-4,
)
```

| Option | Notes |
| --- | --- |
| `monitor` | Must be a registered history key. |
| `patience` | Number of epochs since the best value before stopping. |
| `mode` | `"min"` or `"max"`. |
| `min_delta` | Minimum improvement required to reset patience. |
| `threshold` | Optional floor or ceiling for when an update is allowed to count as an improvement. |

Constructor validation requires `mode` to be `"min"` or `"max"` and `patience`
to be non-negative.

If the monitored history is empty, for example because validation was skipped,
the callback does nothing. A NaN monitored value stops future training.

Pass `force_train=True` to `fit()` to continue after early stopping.

## Optuna Trial Pruning

`OptunaTrialPruner` reports a monitored value to an Optuna trial at epoch end
and raises `optuna.TrialPruned` when the trial should stop.

```python
from ember.callbacks import OptunaTrialPruner

callbacks=[OptunaTrialPruner(trial, monitor="val_loss")]
```

Use it inside an Optuna objective after registering the monitored key through
the default tracker or a custom `setup_tracker()`.

Optuna is a core dependency, but `ember.callbacks` does not import Optuna unless
this pruning path is actually used.
