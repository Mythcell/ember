# ember

A lightweight model training library. *Minimal boilerplate for efficient research*.

---

## Overview

`ember` is a lightweight Lightning Fabric-based training library, designed
for rapid prototyping. It provides a Keras-style model wrapper / trainer combo for
PyTorch projects that benefit from a reusable `fit()` / `evaluate()` /
`predict()` loop without adopting the full Lightning Trainer abstraction.

### Install

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Mythcell/ember.git
cd ember
```

#### uv (recommended)

Create and sync a local uv-managed environment from the project metadata:

```bash
uv sync
```

#### pip

Install Ember into an existing environment with an editable install:

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

`evaluate()` returns metric history lists for that evaluation run. `predict()`
returns a list of output tensors; for this single-output classifier, `preds[0]`
contains the concatenated logits.

## EmberModel

`EmberModel` is the main model wrapper / trainer combo. The default loop is
deliberately narrow: it works well for conventional supervised PyTorch workloads
with tuple-like `(input, target)` batches, a single model, and a single optimizer.

When the default assumptions stop matching the experiment, subclassing
`EmberModel` is the intended escape hatch. Override `train_step()`,
`val_step()`, `eval_step()`, `predict_step()`, or metric hooks to support custom
objectives, unusual batch structures, GAN-style loops, multiple losses, and
other research-shaped training code.

See the
[EmberModel docs](https://mythcell.github.io/ember/core/embermodel/) for the
full default-loop behavior, hook reference, checkpoint helpers, and subclassing
guidance. For worked custom-loop patterns, see
[Custom Loop Patterns](https://mythcell.github.io/ember/examples/custom-loops/).

## Dataloaders

`EmberData` is an optional convenience wrapper inspired by Lightning's
`LightningDataModule`. It is most useful when data setup, downloads, teardown,
or repeated `fit()` / `evaluate()` / `predict()` calls benefit from living in
one object:

```python
em.fit(emberdata=data, epochs=10)
```

Passing `DataLoader`s directly is also fine, and is often enough for small
scripts:

```python
em.fit(train_data=train_loader, val_data=val_loader, epochs=10)
```

See the [Data docs](https://mythcell.github.io/ember/core/data/) for the full
`EmberData` lifecycle and direct dataloader behavior.

## CLI, Configs and Runners

`ember` contains a minimal CLI that runs scripts containing an `EmberRunner`
class. YAML configs are completely optional; a runner can be a plain trusted
Python entrypoint:

```bash
ember path/to/run.py
ember path/to/run.py -c config.yaml
```

When you do use a config, it is plain YAML and intentionally schema-light. If
you want a compact Hydra-style workflow, `ember.utils.instantiate()` can
construct trusted classes from `module.Class` specs and parameter dictionaries:

```yaml
model:
  type: models.MyCNN
  params:
    channels: 32
    num_classes: 10
```

```python
# run.py
import torch.nn as nn

from ember import EmberRunner
from ember.utils import instantiate


class MyRunner(EmberRunner):
    def run(self) -> None:
        model = instantiate(
            self.cfg.model.type,
            params=self.cfg.model.params,
            local_path=self.script_dir,
            expected_type=nn.Module,
        )
```

> [!WARNING]
> `EmberRunner` scripts and config-driven object instantiation can execute
> arbitrary Python code. The CLI imports the runner module, and specs passed to
> `instantiate()` import modules and call constructors. Only run scripts and
> configs that you trust.

See the [CLI docs](https://mythcell.github.io/ember/cli/) for command syntax,
[Runners](https://mythcell.github.io/ember/core/runners/) for runner patterns,
and [Instantiation](https://mythcell.github.io/ember/utils/instantiation/) for
local import roots, relative specs, aliases, and type guards.

## TorchMetrics and Logging

You can pass TorchMetrics metrics when creating an `EmberModel`:

```python
em = EmberModel(
    # ...
    metrics=[Accuracy(task="multiclass", num_classes=10)],
)
```

The automatic path is deliberately simple: metrics are called once per batch and
logged through `EmberTracker` as batch-size-weighted means. For true epoch-level
TorchMetrics semantics, or for metrics with nonstandard call signatures, own the
metric updates in a custom `EmberModel` subclass.

See [Tracking](https://mythcell.github.io/ember/core/tracking/) for metric
history, Fabric logging, and manual metric patterns.

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
