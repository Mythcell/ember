# Quickstart

This example uses the default `EmberModel` loop with a standard PyTorch module
and the built-in `EmberMNIST` data helper.

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

`EmberMNIST` uses the current working directory by default. Pass `root` to keep
downloaded data in a specific directory.

You can also pass dataloaders directly:

```python
em.fit(train_data=train_loader, val_data=val_loader, epochs=10)
```

`EmberData` is a convenience, not a requirement.
