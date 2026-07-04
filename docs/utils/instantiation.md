# Instantiation

`ember.utils.instantiate()` is a small Hydra-style object factory. It creates a
class instance from a string spec and a parameter dictionary:

```python
from ember.utils import instantiate

model = instantiate(
    "models.SmallCNN",
    params={"channels": 32, "num_classes": 10},
    local_path=self.script_dir,
)
```

The spec must be a class path in `module.Class` form. Nested modules are fine,
for example `torch.nn.ReLU` or `my_package.models.SmallCNN`.

!!! danger

    `instantiate()` can execute arbitrary Python code. It imports the module
    named by the config spec and then calls the resolved constructor, so a
    malicious or untrusted spec can run code during import or object creation.
    Do not use it with configs, paths, or modules from untrusted sources.

Instantiation imports Python modules and calls constructors. Only use it with
trusted local code and trusted configs. It is a convenience for experiment
scripts, not a sandbox.

## Config Pattern

A typical runner config might look like this:

```yaml
model:
  type: models.SmallCNN
  params:
    channels: 32
    num_classes: 10

optimizer:
  type: torch.optim.Adam
  params:
    lr: 0.001
```

Then the runner can construct objects from those specs:

```python
import torch.nn as nn

from ember import EmberRunner
from ember.utils import instantiate


class TrainRunner(EmberRunner):
    def run(self) -> None:
        model = instantiate(
            self.cfg.model.type,
            params=self.cfg.model.params,
            local_path=self.script_dir,
            expected_type=nn.Module,
        )

        optimizer = instantiate(
            self.cfg.optimizer.type,
            params={
                **self.cfg.optimizer.params,
                "params": model.parameters(),
            },
        )
```

This gives you the useful part of Hydra-style construction without requiring a
full configuration framework.

## Building Nested Objects

`instantiate()` constructs one class at a time. It does not recursively walk a
config tree and instantiate every nested object automatically.

For nested setups, keep the runner explicit:

```yaml
model:
  type: models.SmallCNN
  params:
    channels: 32

optimizer:
  type: torch.optim.Adam
  params:
    lr: 0.001

scheduler:
  type: torch.optim.lr_scheduler.CosineAnnealingLR
  params:
    T_max: 20
```

After constructing `model` and `optimizer` as above, pass constructed
dependencies explicitly:

```python
scheduler = instantiate(
    self.cfg.scheduler.type,
    params={**self.cfg.scheduler.params, "optimizer": optimizer},
)
```

This explicit style is a little more verbose than Hydra, but it keeps object
lifetimes and dependency injection clear in experiment scripts.

## Local Module Resolution

When `local_path` is provided, `instantiate()` temporarily treats that directory
as an import root and can resolve local files relative to that path:

- `models.SmallCNN` can resolve to `models.py`;
- `package.Model` can resolve to `package/__init__.py`;
- nested local packages are supported when matching files exist.

Flat runner scripts usually pass `self.script_dir` when model files live next to
the runner. Nested package projects should pass `self.project_root` and use a
fully qualified local spec:

```python
data = instantiate(
    "my_project.data.TrainingData",
    local_path=self.project_root,
)
```

If a matching local module is already cached from another location, an explicit
`local_path` still wins so repeated experiments can resolve the requested local
file.

## Relative Specs

Relative specs need an explicit package anchor, just like Python relative
imports:

```python
data = instantiate(
    "..data.TrainingData",
    local_path=self.project_root,
    package="my_project.runners",
)
```

Without `package=...`, relative specs raise `ImportError` with a message
explaining that a package anchor is required.

## Type Guards

Use `expected_type` when configs should only instantiate a specific kind of
object:

```python
model = instantiate(
    self.cfg.model.type,
    params=self.cfg.model.params,
    local_path=self.script_dir,
    expected_type=nn.Module,
)
```

If the resolved class is not a subclass of `expected_type`, `instantiate()`
raises `TypeError` before construction.

## Aliases And Imported Modules

The importer checks `sys.modules` and the active call stack for module aliases.
This allows simple specs such as `nn.ReLU` when `torch.nn` has already been
imported as `nn` in the relevant scope:

```python
import torch.nn as nn

activation = instantiate("nn.ReLU")
```

For configs, fully qualified module paths are usually clearer and more
portable.
