# Tracking

`EmberTracker` is a small metric history and Fabric logging helper. It is used by
`EmberModel`, but it can also be used directly in custom loops.

## Core Model

A tracker stores four pieces of state:

- `history`: finalized epoch values for each registered key;
- `_key_types`: a type label for each key, usually `"train"` or `"val"`;
- `_increments`: accumulated weighted sums for the current epoch;
- `_counts`: accumulated weights for the current epoch.

Register keys before updating them:

```python
self.tracker.register(["g_loss", "d_loss"], key_type="train")
```

Update keys with tensors:

```python
self.tracker.update("g_loss", g_loss.detach(), batch_size=batch_size)
```

`update()` multiplies the value by `batch_size` and increments the matching
count. At synchronization time, the tracker stores the weighted mean. Updates
for keys that have not been registered are ignored, so custom metrics should be
registered before the first step that updates them.

If you pass `step`, the raw step value is also logged through Fabric on the
global-zero process:

```python
self.tracker.update("train_loss", loss.detach(), batch_size=n, step=step)
```

That logs `train_loss_step`. Epoch-level values are logged when `sync_epoch()` is
called with a `step` or `epoch` argument.

## Automatic Use In EmberModel

The default `EmberModel.setup_tracker()` registers:

- `train_loss`;
- `val_loss`;
- `train_{MetricName}` for each constructor metric;
- `val_{MetricName}` for each constructor metric.

If scheduler learning-rate logging is enabled, it also registers `lr_0`,
`lr_1`, and so on with key type `"train"`.

At the end of each training epoch, `fit()` calls:

```python
self.tracker.sync_epoch(key_type="train", step=self.step_count)
```

At the end of each validation epoch, it calls:

```python
self.tracker.sync_epoch(key_type="val", step=self.step_count)
```

This means custom subclasses usually only need to register keys and call
`update()` inside step methods. They do not need to call `sync_epoch()` for
normal train and validation keys.

## Batch Averages And True Epoch Metrics

`EmberTracker` stores weighted means of values passed to `update()`. In the
default `EmberModel` loop, automatic TorchMetrics objects are called once per
batch, and those returned batch values are averaged by the tracker. Ember does
not call TorchMetrics `.compute()` or `.reset()` for automatic metrics.

For metrics that need true epoch state, keep the TorchMetric object under your
control and log its computed value manually:

```python
from ember import EmberModel
from ember.callbacks import EmberCallback
from torchmetrics.classification import MulticlassAUROC


class EpochMetricModel(EmberModel):
    def setup_tracker(self) -> None:
        super().setup_tracker()
        self.tracker.register("val_auc_epoch", key_type="manual_val")
        self.val_auc = self.fabric.to_device(MulticlassAUROC(num_classes=10))

    def val_step(self, batch, batch_idx: int, epoch: int):
        x, y = batch[0], batch[-1]
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.val_auc.update(y_pred.softmax(dim=-1), y)
        return loss, y_pred


class SyncValAUC(EmberCallback):
    def on_validation_epoch_end(
        self, embermodel: EpochMetricModel, epoch: int = 0, step: int = 0
    ) -> None:
        auc = embermodel.val_auc.compute()
        embermodel.tracker.update("val_auc_epoch", auc.detach(), batch_size=1)
        embermodel.tracker.sync_epoch(keys="val_auc_epoch", step=step)
        embermodel.val_auc.reset()
        embermodel.history = embermodel.tracker.get_history_items()
```

Use this pattern for macro F1, AUROC, ranking metrics, calibration metrics, or
other stateful/non-additive metrics where averaging per-batch values would be
misleading.

## Custom Metrics

For manual metrics, override `setup_tracker()` and register the keys you will
update:

```python
class GANModel(EmberModel):
    def setup_tracker(self) -> None:
        self.tracker.register(["g_loss", "d_loss"], key_type="train")

    def train_step(self, batch, batch_idx: int, epoch: int) -> None:
        g_loss, d_loss = train_gan_batch(batch)
        batch_size = batch[0].shape[0]

        self.tracker.update("g_loss", g_loss.detach(), batch_size=batch_size)
        self.tracker.update("d_loss", d_loss.detach(), batch_size=batch_size)
        return None
```

Returning `None` is correct here. The default `(loss, y_pred)` return is only
needed when you want `EmberModel` to calculate its built-in loss and
TorchMetrics updates.

## Custom Key Types

Keys with type `"train"` and `"val"` are synchronized automatically by
`EmberModel.fit()`. Other key types are allowed, but you must synchronize them
manually:

```python
self.tracker.register("sample_quality", key_type="diagnostic")

# Later, for example from a callback:
self.tracker.sync_epoch(key_type="diagnostic", step=embermodel.step_count)
```

`sync_epoch()` performs Fabric `all_reduce` calls and a barrier, so it is a
blocking operation. Call it at coarse boundaries, not for every batch.

## Evaluation Trackers

`evaluate()` creates a separate tracker:

```python
eval_tracker = EmberTracker(fabric=self.fabric, fabric_logging=False)
```

It registers validation-style keys, runs `eval_step()`, synchronizes once, and
returns a dictionary of lists. Because Fabric logging is disabled for this
tracker, evaluation can compute metrics without polluting the training logger.

## State And History

`EmberTracker` implements `state_dict()` and `load_state_dict()`, so it is saved
inside `EmberModel` checkpoints.

For user-facing history, use:

```python
history = self.tracker.get_history_items()
```

This converts tensors to plain Python floats. `EmberModel.history` is refreshed
from this method after synchronization.

`get_history_length()` returns the number of tracked epochs, either for one key
or the maximum length across all keys. `EmberModel` uses it to maintain
`epoch_count` when resuming or continuing training.
