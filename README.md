# ember

*Minimal boilerplate for efficient research.*

Ember is a lightweight, [Lightning Fabric](https://lightning.ai/docs/fabric/stable/)-based training library for PyTorch.

Key features:

- Flexible `EmberModel` model wrapper / trainer combo inspired by Keras.
- Easily implement custom PyTorch logic by overriding `train_step`, `val_step`, etc.
- Minimalist CLI for running `EmberRunner` scripts with optional YAML configs and Hydra-style object instantiation.
- Custom `EmberCallback` callback system with full lifecycle hooks.
- Built-in `EmberTracker` metric/history tracking inspired by Keras' `History` object.
- Lightweight, optional `EmberData` container inspired by Lightning's `LightningDataModule`.
- Default train loop integration with TorchMetrics.
- Supports hyperparameter tuning with `optuna` (including trial pruning).
- Implicit resume: simply run `model.fit()` again to continue training.

Check out the [runner example](examples/runner/README.md) to see config-driven training in action,
or the [MNIST VAE example](examples/mnist_vae/run.py) for working with custom training steps.

## Installation

```bash
git clone https://github.com/Mythcell/ember.git
cd ember
uv sync
```

Note that Ember currently supports Python 3.12 and 3.13

## Quick Example

```python
import torch
import torch.nn as nn
from lightning import Fabric
from torchmetrics import Accuracy

from ember import EmberModel

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

em = EmberModel(
    fabric=Fabric(accelerator="auto"),
    model=model,
    optimizers=[optimizer],
    loss_fn=nn.CrossEntropyLoss(),
    metrics=[Accuracy(task="multiclass", num_classes=10)],
)

em.fit(train_data=train_loader, val_data=val_loader, epochs=5)
metrics = em.evaluate(eval_data=test_loader)
predictions = em.predict(data=predict_loader)
```

## Why not use Keras or PyTorch Lightning?

I wanted something in between. I initially conceived `ember` to help port my personal TensorFlow/Keras projects to PyTorch. I wanted the convenience and familiarity of a Keras-style model-trainer combo while leveraging the flexibily and power of PyTorch.

For more about the design philosophy, see my [blog post](https://mythcell.me/blog/2026/ember/).

## Why `ember`?

I wanted the name to imply some sort of "lightweight PyTorch", which led to
`torchlite` and eventually to `ember`, named after the Embermage character class
from *Torchlight II*.

I then gave it the *unofficial* recursive bacronym "Ember Minimal Boilerplate
for Efficient Research".
