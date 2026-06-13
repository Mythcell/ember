# Data

## Data Containers

`EmberData` is an optional data container. Use it when an experiment benefits
from keeping data setup and dataloader construction in one object.

```python
from torch.utils.data import DataLoader

from ember import EmberData


class MyData(EmberData):
    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            self.train = ...
            self.val = ...
        elif stage in {"eval", "predict"}:
            self.eval_data = ...

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64)

    def eval_dataloader(self):
        return DataLoader(self.eval_data, batch_size=64)

    def predict_dataloader(self):
        return DataLoader(self.eval_data, batch_size=64)
```

## Container Lifecycle

`pre_setup()` runs once on the global rank-zero process before `setup()`. Use
it for one-off work such as downloads. `setup(stage=...)` runs once per stage
on all processes. `cleanup(stage=...)` is called after `fit()`, `evaluate()`,
and `predict()`, including when the stage exits because a step method or
callback raises.

| Method | Purpose |
| --- | --- |
| `pre_setup()` | One-time rank-zero work before any stage setup. |
| `setup(stage="fit")` | Build datasets or state for `fit`, `eval`, or `predict`. |
| `train_dataloader()` | Return the training dataloader for `fit()`. |
| `val_dataloader()` | Return validation dataloader, or `None` to skip validation. |
| `eval_dataloader()` | Return evaluation dataloader for `evaluate()`. |
| `predict_dataloader()` | Return prediction dataloader for `predict()`. |
| `cleanup(stage="fit")` | Release resources after a stage completes or fails. |
| `reset()` | Allow `pre_setup()` and per-stage `setup()` to run again. |

Only assign shared state in `setup()`, not `pre_setup()`. `pre_setup()` runs on
global zero only, so instance attributes set there are not guaranteed to exist
on other ranks.

## Direct Dataloaders

Direct dataloaders remain fully supported:

```python
em.fit(train_data=train_loader, val_data=val_loader, epochs=10)
```

Use direct dataloaders for simple scripts. Use `EmberData` when setup,
teardown, downloads, or repeated `fit` / `evaluate` / `predict` entrypoints
benefit from one container object.

## Built-In MNIST Data

`EmberMNIST` is a small built-in `EmberData` implementation available from
`ember.mnist`. By default it downloads and reads MNIST from the current working
directory; pass `root="path/to/data"` to use a specific dataset directory.
