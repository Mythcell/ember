# Ember

Ember is a lightweight Fabric-based training library for PyTorch. It keeps the
parts of a trainer that are useful during iterative model work while leaving
the training script and model structure under your control.

!!! note

    `ember` is not meant to be a full-fledged ML framework. It is a compact
    training library for quickly prototyping and training PyTorch models with
    Keras-style `fit()` / `evaluate()` / `predict()` ergonomics.

At the heart is `EmberModel`: a Keras-inspired model wrapper / trainer combo
based around a standard `torch.nn.Module`. For typical supervised training, the
default loop handles device setup through Lightning Fabric, metric tracking,
gradient accumulation, gradient clipping, checkpoint callbacks, evaluation, and
prediction.

For more complex setups that require custom loops, subclass `EmberModel` and
override the relevant step methods.

## Start Here

- [Installation](getting-started/installation.md): set up Ember from a local
  checkout or editable install.
- [Quickstart](getting-started/quickstart.md): train a small MNIST classifier
  with the default `EmberModel` loop.
- [EmberModel](core/embermodel.md): understand the main training interface,
  hook methods, and default-loop assumptions.
- [Data](core/data.md): use `EmberData` or pass PyTorch dataloaders directly.
- [Tracking](core/tracking.md): understand metric history, logging, and manual
  metric patterns.
- [Callbacks](core/callbacks.md): add checkpointing, early stopping, or custom
  lifecycle hooks.
- [Runners](core/runners.md) and [CLI](cli.md): run experiment scripts with
  optional YAML configs.
- [Custom Loops](examples/custom-loops.md): adapt Ember to objectives that do
  not fit the default supervised loop.
- [API Reference](api.md): see the documented import surface at a glance.

!!! info

    Portions of this documentation were drafted and formatted with AI assistance,
    then reviewed and edited by the maintainer.

## What `ember` Provides

- `EmberModel` with `fit()`, `evaluate()`, `predict()`, checkpoint helpers, and
  subclassable `train_step()`, `val_step()`, `eval_step()`, and `predict_step()`.
- `EmberCallback`, `ModelCheckpoint`, and `EarlyStopping` for focused training
  lifecycle hooks.
- `EmberTracker` for epoch-level metric history and Fabric logger integration.
- `EmberData`, a lightweight optional data container inspired by
  `LightningDataModule`.
- `EmberRunner`, a script runner abstraction for experiments with optional YAML
  configs.
- `instantiate()`, a small utility for constructing trusted classes from
  `module.Class` specs.
- General utilities for freezing, initialization, schedulers, and Hydra-style
  config behavior without adopting a larger framework.

## Scope

Ember is deliberately lightweight. It is not intended to become a full-fledged
framework. It is a compact Lightning Fabric-based training library for PyTorch
code that benefits from a reusable `fit()` / `evaluate()` / `predict()`
model-trainer with Fabric-compatible primitives. Distributed, cluster, and
multi-node workflows are possible through Fabric, but have had limited explicit
testing.

## What To Expect

### Differences from Keras

- Ember keeps the `fit()` / `evaluate()` / `predict()` ergonomics, but it does
  not try to hide PyTorch. You still construct modules, optimizers, and losses
  explicitly.
- Subclassing is the normal extension path. When the default loop stops fitting
  the problem, override step methods rather than learning a separate graph or
  layer API.
- There is less built-in abstraction around data, model compilation, and
  callbacks. In return, the training code stays close to ordinary PyTorch.
- Expect a smaller surface area and fewer batteries included. Ember is intended
  for research workflows that value clarity and control over framework breadth.

### Differences from PyTorch Lightning

- Ember uses Fabric primitives, but not the full Lightning Trainer model. The
  loop is smaller, more explicit, and easier to read end to end.
- `EmberModel` is closer to a lightweight training library than a full training
  framework. You pass concrete optimizers, losses, metrics, and schedulers
  directly rather than configuring a large trainer object.
- Customization happens by overriding focused methods on `EmberModel`, not by
  adopting Lightning's broader module and trainer contract.
- Expect fewer built-in integrations and less framework automation, but less
  indirection when debugging or shaping an experiment around unusual logic.
