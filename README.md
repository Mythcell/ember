# ember

A lightweight model training library. *Minimal boilerplate for efficient research*.

---

## Overview

`ember` is a lightweight Lightning Fabric-based training library, designed
for rapid prototyping. It provides a Keras-style model wrapper / trainer combo for
PyTorch projects that benefit from a reusable `fit()` / `evaluate()` /
`predict()` loop without adopting the full Lightning Trainer abstraction.

### Install

From a local checkout, run

```bash
uv sync
```

or, to install into an existing environment,

```bash
pip install -e .
```

### Features

- Built on [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) primitives.
- Flexible `EmberModel` model wrapper / trainer combo.
- Easily implement custom PyTorch logic by overriding `train_step`, `val_step`, etc.
- Minimalist CLI for running `EmberRunner` scripts with optional YAML configs and Hydra-style object instantiation.
- Custom `EmberCallback` callback system with full lifecycle hooks.
- Built-in `EmberTracker` metric/history tracking inspired by Keras' `History` object.
- Lightweight, optional `EmberData` container inspired by Lightning's `LightningDataModule`.
- Default train loop integration with `torchmetrics` metrics.
- Supports hyperparameter tuning with `optuna` (including trial pruning).
- Implicit resume: simply run `model.fit()` again to continue training.

### Why not use Keras or PyTorch Lightning?

I wanted something in between. I initially conceived `ember` to help port my
personal TensorFlow/Keras projects to PyTorch. I wanted the convenience and familiarity of
a Keras-style model-trainer combo while staying close to underlying PyTorch modules, optimizers, losses, and dataloaders.

PyTorch Lightning is an excellent and mature ecosystem, but its full Trainer abstractions felt a little too cumbersome for the quick prototyping I wanted with my personal projects. This is why `ember` uses Lightning Fabric.

> [!NOTE]
> `ember` is not meant to be a full-fledged ML framework, but rather a compact
> lightweight training library to quickly prototype and train models.
>
> It is aimed at those who like the Keras-style ergonomics of `fit()` / `evaluate()` /
> `predict()`, and who want their training code to feel close to ordinary PyTorch.

## Examples

### Hello World (MNIST)

Below is an example with MNIST:

```python
import torch
import torch.nn as nn
from lightning import Fabric
from torchmetrics import Accuracy

from ember import EmberModel
from ember.callbacks import ModelCheckpoint
from ember.mnist import EmberMNIST


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


fabric = Fabric(accelerator="auto")
model = SmallCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = EmberMNIST(batch_size=128, val_fraction=0.2, root="data")

em = EmberModel(
    fabric=fabric,
    model=model,
    optimizers=[optimizer],
    loss_fn=nn.CrossEntropyLoss(),
    metrics=[Accuracy(task="multiclass", num_classes=10)],
)

em.fit(
    emberdata=data,
    epochs=5,
    callbacks=[ModelCheckpoint(monitor="val_loss", save_dir="ckpts")],
)

metrics = em.evaluate(emberdata=data)
preds = em.predict(emberdata=data)
```

`evaluate()` returns a dictionary of metric history lists for that evaluation
run, using validation-style keys such as `val_loss` and
`val_MulticlassAccuracy`.

`predict()` returns a list of output tensors. For this single-output classifier,
`preds[0]` contains the concatenated logits.

## EmberModel

### Default `EmberModel`

`ember` was designed to be flexible enough for a wide range of PyTorch
experiments, but the default `EmberModel` loop is deliberately narrow. Out of
the box, it assumes a single-model, single-optimizer, supervised workflow.

The default `train_step()`, `val_step()`, and `eval_step()` assume that a batch
is tuple-like:

- `batch[0]` is the model input;
- `batch[-1]` is the target;
- `self.model(batch[0])` returns predictions;
- `self.loss_fn(y_pred, batch[-1])` returns a scalar loss;
- automated metrics can be called as `metric(y_pred, batch[-1])`.

This is enough for many supervised learning workflows with `(x, y)`-style
datasets. Single-item batches are treated as inputs and targets, which is useful
for autoencoders or other reconstruction workflows. The default supervised
steps require `loss_fn`; passing `loss_fn=None` is only intended for custom
`EmberModel`s or for when you are doing manual logging.

### Custom `EmberModel` (recommended)

If your requirements do not neatly fit into the default batch assumptions, or
require any of the following:

- multiple optimizers;
- handling multiple model outputs or managing submodules;
- freezing or otherwise manipulating submodules in the train loop;

then you'll likely want to override `train_step`, `val_step`, etc. directly.

Subclassing `EmberModel` is the **intended** way to support custom objectives
(e.g. contrastive learning), multi-module / multi-optimizer architectures
(e.g. GAN), custom batch structures, and more specialized PyTorch patterns.

If you want to be in full control (chances are you probably do), then this is
the **recommended** approach.

## Dataloaders

`EmberData` is a convenience wrapper directly inspired by Lightning's
`LightningDataModule`:
```python
em.fit(emberdata=data, epochs=10)
```
It is most useful for self-contained data preprocessing workflows.

Of course, passing `DataLoader`s directly to `.fit()`, `.evaluate()` and
`.predict()` is fine as well:
```python
em.fit(train_data=train_loader, val_data=val_loader, epochs=10)
```

When `emberdata` is supplied, it takes priority over direct dataloader arguments.
`fit()` calls `setup(stage="fit")`, then uses `train_dataloader()` and
`val_dataloader()`. `evaluate()` and `predict()` use their matching stages and
dataloader methods.

When using a `DataLoader` with a single-element tuple, the input is
automatically treated as the target. For example:

```python
autoenc = MyAutoencoder()
train_data = DataLoader(TensorDataset(images))
# ...
em.fit(epochs=100, train_data=train_data)
```

`ember.mnist.EmberMNIST` is a small built-in `EmberData` implementation used in
examples. It uses the current working directory by default. Passing `root` keeps
downloaded MNIST files in a predictable location for scripts and examples.

## CLI, Configs and Runners

### CLI

`ember` contains a minimal CLI that runs scripts containing an `EmberRunner`
class. It also includes an `instantiate` utility to construct objects directly
from type/param YAML specs, giving you a small Hydra-style workflow without the
overhead of rigid schemas and registries.

This is useful for iterative, reproducible experiments driven by an
**optional** YAML config, e.g.:

```bash
ember run.py -c config.yaml
```

> [!WARNING]
> `EmberRunner` scripts and config-driven object instantiation can execute
> arbitrary Python code. The CLI imports the runner module, and specs passed to
> `instantiate()` import modules and call constructors. Only run scripts and
> configs that you trust.

### Minimal Runner

An `EmberRunner` is just a class with a `run()` method. The CLI loads the config
with OmegaConf and injects `self.cfg`, `self.cfg_path`, `self.script_dir` and
`self.verbosity`.

```python
# run.py
from ember import EmberRunner

class MyRunner(EmberRunner):
    def run(self) -> None:
        print(self.cfg.batch_size)
```

Runner scripts can be discovered in two ways: define a single `EmberRunner`
subclass, or export a module-level `runner = MyRunner()` when you want explicit
construction.

### Configs

Configs are plain YAML and are entirely **optional**. There is no set schema or
format: the CLI injects `self.cfg` directly to your runner class.

```yaml
model:
  type: models.MyCNN
  params:
    channels: 32
    num_classes: 10
```

### Hydra-style Object Instantiation

`ember.utils.instantiate()` supports simple Hydra-style object instantiation
from a string module spec and parameter dictionary, and can also resolve local
imports. Let's see how we can initialize a model using the above YAML config:

```python
# run.py
import torch.nn as nn

from ember import EmberRunner
from ember.utils import instantiate


class MyRunner(EmberRunner):
    def run(self) -> None:
        model = instantiate(
            self.cfg.model.type,  # models.MyCNN
            params=self.cfg.model.params,  # {channels: 32, num_classes: 10}
            local_path=self.script_dir,  # where to find models.py
            expected_type=nn.Module,  # optional type validation, raises TypeError
        )
```
this will work assuming `run.py`, `models.py` and `config.yaml` are in the same file.

### More on import resolution

`ember`'s `instantiate()` is essentially a wrapper for `importlib` with a
slightly more strict resolution order. In particular, when `local_path` is
provided, `instantiate()` first checks for local files relative to that path:

- `models.MyCNN` can resolve to `models.py`;
- `package.Model` can resolve to `package/__init__.py`;
- nested local packages are supported when matching files exist.

Runner scripts should therefore pass `self.script_dir` as you'll probably put
your script next to your model files. If `instantiate()` doesn't find anything,
it temporarily adds `local_path` to `sys.path` and falls back to normal
`importlib.import_module()` behavior. Alternatively, you can import the relevant
package or module into the namespace, i.e. `import models`, then use
`models.MyCNN`.

`instantiate()` requires a dotted `"module.Class"` string specification. The
importer also checks `sys.modules` and the active call stack for module aliases:

```python
import torch.nn as nn

from ember.utils import instantiate

activation = instantiate("nn.ReLU")  # equivalent to nn.ReLU()
```

## TorchMetrics and Logging

You can define TorchMetrics metrics when creating an `EmberModel`. In the MNIST
example from before, supplying this metric:
```python
em = EmberModel(
    # ...
    metrics=[Accuracy(task="multiclass", num_classes=10)],
)
```
automatically registers `"train_MulticlassAccuracy"` and
`"val_MulticlassAccuracy"` keys in the `EmberTracker` logger (more generally,
`metric._get_name()`).

### Automatic metric calculation

`train_step` and `val_step` support automatic TorchMetrics metric calculation
and logging whenever you return a tuple `(loss, y_pred)`.

```python
def train_step(
    self, batch, batch_idx: int, epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch[0], batch[-1]
    (opt, *_) = self.optimizers
    y_pred = self.model(x)
    loss = self.loss_fn(y_pred, y)
    self.fabric.backward(loss)
    opt.step()
    opt.zero_grad()
    return loss, y_pred
```

The automatic path is deliberately simple: the default loop calls
`metric(y_pred, batch[-1])` once per batch and sends the returned batch value to
`EmberTracker`. The tracker stores and logs a batch-size-weighted mean of those
batch values. If you are using a TorchMetrics metric with a different call
signature, override `calculate_metric` in a subclass.

Note the automatic metric path does not call TorchMetrics `.compute()` or
`.reset()`. For true epoch-level metrics, you can calculate metrics manually:

### Custom metric logging

For custom `EmberModel` classes, it is often best to register and update
metrics manually with the built-in `EmberTracker`. Return `None` from your step
method when you own the metric updates yourself:

```python
class CustomModel(EmberModel):
    def setup_tracker(self) -> None:
        self.tracker.register(["loss_a", "loss_b", "my_metric"], key_type="train")

    def train_step(self, batch, batch_idx: int, epoch: int) -> None:
        # ...
        loss_a = some_loss_fn(y_pred, y)
        loss_b = some_other_loss_fn(y_pred, y)
        my_metric = calculate_my_metric(y_pred, y)
        self.tracker.update("loss_a", loss_a.detach(), batch_size=batch_size)
        self.tracker.update("loss_b", loss_b.detach(), batch_size=batch_size)
        self.tracker.update("my_metric", my_metric.detach(), batch_size=batch_size)

        # no need to return (loss, y_pred)
```

## Caveats and Limitations

- `EmberModel` is designed to operate on a single `nn.Module` model object. As
  such you'll need to wrap multi-module models, like GANs, into a single
  `nn.Module`.
- Automatic metric logging stores weighted averages of per-batch metric outputs.
  Use manual metric handling for true epoch-level TorchMetrics semantics or
  custom loss metrics.
- `EmberTracker`'s logging has only been tested with the default `TensorBoardLogger`.
  If you want to integrate with `wandb`, `mlflow`, `aim` or other formats, you
  may want to use the `EmberRunner` approach.
- Distributed training has had limited testing and only on local machines. That
  said, the code is Fabric-compatible and rank-aware, for example
  rank-zero-only checkpoint saving and metric reduction. Use with caution.

## Why `ember`?

I wanted the name to imply some sort of "lightweight PyTorch", which led to
`torchlite` and eventually to `ember`, named after the Embermage character class
from *Torchlight II*.

I then gave it the *unofficial* recursive bacronym "Ember Minimal Boilerplate
for Efficient Research".
