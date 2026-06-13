# Custom Loop Patterns

The default `EmberModel` loop covers ordinary supervised training. For unusual
objectives, keep `fit()` and override the step methods that differ.

## What Still Comes From EmberModel

Even in a custom subclass, Ember's lightweight training library still owns the
outer loop through `EmberModel`:

- Fabric-aware dataloader setup inside `fit()`, `evaluate()`, and `predict()`;
- callback execution, progress bars, and train/validation phase switching;
- epoch-level tracker synchronization for keys registered with type `"train"`
  and `"val"`;
- scheduler stepping through `scheduler_step()` at the end of each epoch;
- optional learning-rate tracking when scheduler logging is enabled;
- checkpoint save/load for the model, optimizers, schedulers, tracker, and step
  count.

That means most custom work should stay inside `train_step()`, `val_step()`,
`eval_step()`, `predict_step()`, `setup_tracker()`, `calculate_metric()`, and
`scheduler_step()` rather than replacing `fit()` itself.

## Construction Points To Know

Custom loops still start from the same constructor surface:

- `optimizers` are always supplied explicitly as a list and are available as
  `self.optimizers` inside custom steps;
- `schedulers` are optional, but if you pass them they are stepped by the
  default `scheduler_step()` unless you override that method;
- `metrics` are optional TorchMetrics instances. If a step returns
  `(loss, y_pred)`, the outer loop updates them automatically through
  `calculate_metric()`;
- `setup_tracker()` is called during construction, so tracker registration
  belongs there rather than inside the step methods. If you override
  `setup_tracker()` and still want the default `train_loss`, `val_loss`, or
  metric keys, call `super().setup_tracker()` first or register those keys
  yourself;
- `EmberTracker.update()` ignores unregistered keys, so a misspelled or missing
  custom key will not appear in history;
- `fabric_setup()` runs during construction by default and calls
  `fabric.setup(self.model, *self.optimizers)`.

If your model wrapper or optimizer setup needs different Fabric initialization,
override `fabric_setup()` or pass `setup_fabric=False` and handle setup
yourself.

## VAE-Style Step

A VAE usually needs a custom loss signature, but it can still return
`(loss, y_pred)` when the default loop should track loss or TorchMetrics.

```python
from ember import EmberModel


class EmberVAE(EmberModel):
    def train_step(self, batch, batch_idx: int, epoch: int):
        x = batch[0]
        x = self.aug_fn(x)

        is_accumulating = (batch_idx + 1) % self.accumulate_grad_batches != 0
        with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
            x_out, z_mean, z_log_var = self.model(x)
            loss = self.loss_fn(x_out, x, z_mean, z_log_var, epoch)
            self.fabric.backward(loss / self.accumulate_grad_batches)

        if not is_accumulating:
            self.optimizers[0].step()
            self.optimizers[0].zero_grad()

        return loss, x_out
```

Returning `(loss, x_out)` lets the outer loop update registered loss keys and
any compatible metrics. If a metric does not make sense for reconstructed
outputs, omit constructor metrics or update your own keys manually.

This pattern is useful when the loop structure is still standard, but the model
forward pass or loss signature differs from simple supervised classification.
The subclass owns the custom forward/loss path, while `EmberModel` still owns
epoch bookkeeping, tracker synchronization, callbacks, and scheduler stepping.

## GAN-Style Step

GANs are a better example of manual tracking. There is no single supervised
`(loss, y_pred)` pair, so the subclass registers custom keys and returns `None`.

```python
from ember import EmberModel


class EmberGAN(EmberModel):
    def setup_tracker(self) -> None:
        self.tracker.register(["g_loss", "d_loss"], key_type="train")

    def train_step(self, batch, batch_idx: int, epoch: int) -> None:
        g_loss, d_loss = train_generator_and_discriminator(batch, self.optimizers)
        batch_size = batch[0].shape[0]

        self.tracker.update("d_loss", d_loss.detach(), batch_size=batch_size)
        self.tracker.update("g_loss", g_loss.detach(), batch_size=batch_size)
        return None
```

`fit()` still handles callbacks, epoch boundaries, tracker synchronization,
progress bars, and scheduler stepping. The subclass owns optimizer stepping,
gradient accumulation, clipping, and metric updates for the custom step.

If you also need validation-side custom keys, register them with key type
`"val"` in `setup_tracker()` and update them inside `val_step()`. They will be
aggregated automatically at the end of each validation epoch.

## Metrics, Schedulers, And Tracker Hooks

For custom loops, the key question is whether you still want Ember's automatic
metric path.

- Return `(loss, y_pred)` when your outputs are compatible with constructor
  metrics and `calculate_metric(metric, batch, y_pred)`.
- Return `None` when the subclass should own all logging and aggregation.
- Override `calculate_metric()` when metrics need something other than
  `(y_pred, batch[-1])`.
- Override `scheduler_step()` when scheduler behavior depends on custom state or
  on a metric other than the default tracked history.

The default scheduler path already handles plain schedulers and
`ReduceLROnPlateau`. For plateau schedulers, Ember reads the history key given
by `reduce_lr_on_plateau_monitor`, which defaults to `val_loss`.

For manual metric aggregation, true epoch metrics, and standalone tracker use,
see [Tracking](../core/tracking.md).

## Standalone EmberTracker

`EmberTracker` can also be used outside `EmberModel` in fully custom loops. The
core API is the same as the custom-step examples above: register keys, update
them during the loop, then synchronize at an epoch boundary.

```python
from ember import EmberTracker

tracker = EmberTracker(fabric=fabric, fabric_logging=False)
tracker.register(["loss", "aux_metric"], key_type="train")

for step, batch in enumerate(train_loader, start=1):
    loss, aux_metric = run_step(batch)
    batch_size = batch[0].shape[0]

    tracker.update("loss", loss.detach(), batch_size=batch_size, step=step)
    tracker.update("aux_metric", aux_metric.detach(), batch_size=batch_size)

tracker.sync_epoch(key_type="train", step=step)
history = tracker.get_history_items()
```

See [Tracking](../core/tracking.md) for the full tracker API, synchronization
behavior, and checkpointing details.

## Single Model Wrapper

`EmberModel` expects one main `nn.Module` for Fabric setup and checkpointing. For
multi-module systems, wrap the parts in one module:

```python
import torch.nn as nn


class GANWrapper(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
```

This keeps checkpointing simple while still allowing custom code to address each
submodule directly.
