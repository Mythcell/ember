from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from lightning import Fabric
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy

from ember import EmberData, EmberModel

NUM_FEATURES = 4
NUM_CLASSES = 2


def make_dataset(num_samples: int, *, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, NUM_FEATURES, generator=generator)
    targets = torch.randint(NUM_CLASSES, (num_samples,), generator=generator)
    return TensorDataset(features, targets)


class SyntheticData(EmberData):
    def __init__(self, *, no_validation: bool = False) -> None:
        super().__init__()
        self.train_data = make_dataset(8, seed=100)
        self.val_data = make_dataset(4, seed=200)
        self.eval_data = make_dataset(6, seed=300)
        self.no_validation = no_validation

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=2, shuffle=False)

    def val_dataloader(self) -> DataLoader | None:
        if self.no_validation:
            return None
        return DataLoader(self.val_data, batch_size=2, shuffle=False)

    def eval_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_data, batch_size=2, shuffle=False)

    def pred_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_data, batch_size=2, shuffle=False)


class TinyClassifier(nn.Linear):
    def __init__(self) -> None:
        super().__init__(NUM_FEATURES, NUM_CLASSES)
        with torch.no_grad():
            self.weight.copy_(
                torch.tensor(
                    [
                        [0.10, -0.20, 0.30, -0.40],
                        [-0.10, 0.20, -0.30, 0.40],
                    ]
                )
            )
            self.bias.zero_()


ModelFactory = Callable[..., EmberModel]


@pytest.fixture
def model_factory() -> ModelFactory:
    def make_model(
        *,
        model: nn.Module | None = None,
        model_class: type[EmberModel] = EmberModel,
        aug_fn: nn.Module | None = None,
        metrics: bool = True,
        no_validation: bool = False,
        scheduler: bool = True,
    ) -> EmberModel:
        module = model if model is not None else TinyClassifier()
        fabric = Fabric(devices=1, accelerator="auto")
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] | None = (
            [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.5,
                    patience=0,
                )
            ]
            if scheduler
            else None
        )
        metric_list = (
            [Accuracy(task="multiclass", num_classes=NUM_CLASSES)] if metrics else None
        )
        return model_class(
            fabric=fabric,
            model=module,
            optimizers=[optimizer],
            schedulers=schedulers,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metric_list,
            aug_fn=aug_fn,
            no_validation=no_validation,
        )

    return make_model


@pytest.fixture
def synthetic_data() -> SyntheticData:
    return SyntheticData()


@pytest.fixture
def fitted_model(
    model_factory: ModelFactory,
    synthetic_data: SyntheticData,
) -> EmberModel:
    model = model_factory()
    model.fit(
        emberdata=synthetic_data,
        epochs=2,
        show_metrics=False,
        show_progress_bar=False,
    )
    return model
