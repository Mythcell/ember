# Runners

`EmberRunner` is a small base class for trusted local experiment scripts. A
runner remains ordinary Python code; optional YAML configs are a convenience for
experiment parameters and simple object specifications.

See [CLI](../cli.md) for command syntax, runner discovery order, and the
runtime metadata injected before `run()` is called.

!!! danger

    Ember runner scripts and config-driven instantiation can execute arbitrary
    Python code. The CLI imports the runner module, and any specs passed to
    `instantiate()` import modules and call constructors. Only run scripts and
    configs that you trust.

Treat runners as a convenience for trusted local experiment scripts, not as a
sandbox for third-party code or configs.

## Minimal Runner

Define `run()` with the experiment logic you want the CLI to execute:

```python
from ember import EmberRunner


class TrainRunner(EmberRunner):
    def run(self) -> None:
        if self.cfg is not None:
            print(self.cfg.batch_size)
```

Use `self.script_dir` when resolving files that live next to the runner, such
as local model modules, configs, or data directories. That keeps runners
independent of the current working directory used to launch the CLI.

Use `self.project_root` when the runner lives inside a package and needs to
import other local package modules without requiring an editable install.

## Config-Driven Runner

The repository includes a runnable example in `examples/runner`. Its config
uses simple type specifications:

```yaml
model_type: models.SmallCNN
model_params: {}
loss_fn: nn.CrossEntropyLoss
metric_type: torchmetrics.Accuracy
metric_params:
  task: multiclass
  num_classes: 10
batch_size: 64
lr: 0.001
accelerator: auto
epochs: 5
model_checkpoint_params:
  monitor: val_MulticlassAccuracy
  mode: max
```

The runner resolves local model code relative to the script directory:

```python
import torch.nn as nn
import torchmetrics

from ember import EmberRunner
from ember.utils import instantiate


class MyRunner(EmberRunner):
    def run(self) -> None:
        model = instantiate(
            self.cfg.model_type,
            params=self.cfg.model_params,
            local_path=self.script_dir,
            expected_type=nn.Module,
        )
        loss_fn = instantiate(self.cfg.loss_fn, expected_type=nn.Module)
        metric = instantiate(
            self.cfg.metric_type,
            params=self.cfg.metric_params,
            expected_type=torchmetrics.Metric,
        )
```

This provides Hydra-style object construction without making the entire project
config-first. See [Instantiation](../utils/instantiation.md) for the detailed
rules and safety caveats.

## Nested Package Runner

For a project laid out as a package, use fully qualified specs with
`local_path=self.project_root`:

```python
from ember import EmberRunner
from ember.utils import instantiate


class TrainRunner(EmberRunner):
    def run(self) -> None:
        data = instantiate(
            "my_project.data.TrainingData",
            local_path=self.project_root,
        )
```

Relative specs are supported too, but they need an explicit package anchor:

```python
data = instantiate(
    "..data.TrainingData",
    local_path=self.project_root,
    package="my_project.runners",
)
```
