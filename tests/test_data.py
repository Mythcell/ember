from pathlib import Path
from typing import ClassVar

import pytest
import torch
import torch.nn as nn
from conftest import ModelFactory, make_dataset
from lightning import Fabric
from torch.utils.data import DataLoader, Dataset

from ember import EmberData, EmberModel
from ember.mnist import EmberMNIST


class FakeMNIST(Dataset):
    calls: ClassVar[list[tuple[Path, bool, bool]]] = []

    def __init__(
        self,
        root: str | Path,
        train: bool,
        download: bool = False,
        transform=None,
    ) -> None:
        self.calls.append((Path(root), train, download))
        generator = torch.Generator().manual_seed(600 if train else 700)
        size = 12 if train else 5
        self.data = torch.rand(size, 1, 28, 28, generator=generator)
        self.targets = torch.randint(10, (size,), generator=generator)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


@pytest.fixture
def fake_mnist(monkeypatch: pytest.MonkeyPatch) -> type[FakeMNIST]:
    FakeMNIST.calls = []
    monkeypatch.setattr("ember.mnist.MNIST", FakeMNIST)
    return FakeMNIST


def test_embermnist_pre_setup_uses_configured_root_without_io(
    fake_mnist: type[FakeMNIST],
    tmp_path: Path,
) -> None:
    data = EmberMNIST(root=tmp_path, transform=nn.Identity())

    data.pre_setup()

    assert fake_mnist.calls == [
        (tmp_path, True, True),
        (tmp_path, False, True),
    ]


def test_embermnist_builds_all_stage_dataloaders_from_fake_data(
    fake_mnist: type[FakeMNIST],
    tmp_path: Path,
) -> None:
    data = EmberMNIST(
        root=tmp_path,
        transform=nn.Identity(),
        batch_size=2,
        val_fraction=0.25,
    )

    data.setup("fit")
    assert len(data.train_data) == 9
    assert len(data.val_data) == 3
    assert len(data.train_dataloader()) == 5
    val_dataloader = data.val_dataloader()
    assert val_dataloader is not None
    assert len(val_dataloader) == 2

    data.setup("eval")
    assert len(data.eval_data) == 5
    assert len(data.eval_dataloader()) == 3

    data.setup("pred")
    assert len(data.pred_dataloader()) == 3
    assert fake_mnist.calls == [
        (tmp_path, True, False),
        (tmp_path, False, False),
        (tmp_path, False, False),
    ]


def test_embermnist_can_disable_validation() -> None:
    assert EmberMNIST(no_val=True).val_dataloader() is None
    assert EmberMNIST(val_fraction=0).val_dataloader() is None


class CleanupData(EmberData):
    def __init__(self) -> None:
        super().__init__()
        self.data = make_dataset(2, seed=800)
        self.cleanup_stages: list[str] = []

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data, batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data, batch_size=1, shuffle=False)

    def eval_dataloader(self) -> DataLoader:
        return DataLoader(self.data, batch_size=1, shuffle=False)

    def pred_dataloader(self) -> DataLoader:
        return DataLoader(self.data, batch_size=1, shuffle=False)

    def cleanup(self, stage: str = "fit") -> None:
        self.cleanup_stages.append(stage)


class FailingCleanupData(CleanupData):
    def cleanup(self, stage: str = "fit") -> None:
        super().cleanup(stage)
        raise RuntimeError("cleanup failed")


class FailingTrainModel(EmberModel):
    def train_step(self, batch, batch_idx: int, epoch: int):
        raise RuntimeError("train failed")


class FailingValModel(EmberModel):
    def val_step(self, batch, batch_idx: int, epoch: int):
        raise RuntimeError("val failed")


class FailingEvalModel(EmberModel):
    def eval_step(self, batch, batch_idx: int, eval_tracker=None):
        raise RuntimeError("eval failed")


class FailingPredictModel(EmberModel):
    def pred_step(self, batch, batch_idx: int):
        raise RuntimeError("pred failed")


def test_emberdata_cleanup_runs_after_each_successful_stage(
    model_factory: ModelFactory,
) -> None:
    data = CleanupData()
    model = model_factory(metrics=False, scheduler=False)

    model.fit(
        emberdata=data,
        epochs=1,
        show_metrics=False,
        show_progress_bar=False,
    )
    model.evaluate(
        emberdata=data,
        show_metrics=False,
        show_progress_bar=False,
    )
    model.predict(emberdata=data, show_progress_bar=False)

    assert data.cleanup_stages == ["fit", "eval", "pred"]


def run_failing_stage(model: EmberModel, data: CleanupData, stage: str) -> None:
    if stage in {"train", "val"}:
        model.fit(
            emberdata=data,
            epochs=1,
            show_metrics=False,
            show_progress_bar=False,
        )
    elif stage == "eval":
        model.evaluate(
            emberdata=data,
            show_metrics=False,
            show_progress_bar=False,
        )
    else:
        model.predict(emberdata=data, show_progress_bar=False)


def test_emberdata_cleanup_runs_when_model_stages_fail(
    model_factory: ModelFactory,
) -> None:
    cases = [
        ("train", FailingTrainModel),
        ("val", FailingValModel),
        ("eval", FailingEvalModel),
        ("pred", FailingPredictModel),
    ]

    for stage, model_class in cases:
        data = CleanupData()
        model = model_factory(
            model_class=model_class,
            metrics=False,
            scheduler=False,
        )
        with pytest.raises(RuntimeError, match=f"{stage} failed"):
            run_failing_stage(model, data, stage)
        expected_stage = "fit" if stage in {"train", "val"} else stage
        assert data.cleanup_stages == [expected_stage]


def test_cleanup_exception_precedence(model_factory: ModelFactory) -> None:
    failing_stage_data = FailingCleanupData()
    failing_model = model_factory(
        model_class=FailingTrainModel,
        metrics=False,
        scheduler=False,
    )

    with pytest.raises(RuntimeError, match="train failed"):
        failing_model.fit(
            emberdata=failing_stage_data,
            epochs=1,
            show_metrics=False,
            show_progress_bar=False,
        )
    assert failing_stage_data.cleanup_stages == ["fit"]

    successful_stage_data = FailingCleanupData()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        model_factory(metrics=False, scheduler=False).fit(
            emberdata=successful_stage_data,
            epochs=1,
            show_metrics=False,
            show_progress_bar=False,
        )
    assert successful_stage_data.cleanup_stages == ["fit"]


class LifecycleData(EmberData):
    def __init__(self) -> None:
        super().__init__()
        self.pre_setup_calls = 0
        self.setup_calls: list[str] = []

    def pre_setup(self) -> None:
        self.pre_setup_calls += 1

    def setup(self, stage: str = "fit") -> None:
        self.setup_calls.append(stage)


def test_emberdata_setup_runs_once_per_stage_until_reset() -> None:
    fabric = Fabric(devices=1, accelerator="auto")
    data = LifecycleData()

    data._setup_emberdata(fabric, stage="fit")
    data._setup_emberdata(fabric, stage="fit")
    data._setup_emberdata(fabric, stage="eval")
    data._setup_emberdata(fabric, stage="pred")

    assert data.pre_setup_calls == 1
    assert data.setup_calls == ["fit", "eval", "pred"]

    data.reset()
    data._setup_emberdata(fabric, stage="fit")
    assert data.pre_setup_calls == 2
    assert data.setup_calls == ["fit", "eval", "pred", "fit"]
