from unittest.mock import MagicMock, patch

import pytest
import torch
from conftest import ModelFactory, make_dataset
from lightning import Fabric
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassConfusionMatrix

from ember import EmberModel, EmberTracker


class CustomTrackingModel(EmberModel):
    def setup_tracker(self) -> None:
        self.tracker.register(["custom_loss", "custom_aux"], key_type="train")

    def train_step(self, batch, batch_idx: int, epoch: int) -> None:
        inputs = batch[0]
        optimizer = self.optimizers[0]
        predictions = self.model(inputs)
        loss = predictions.square().mean()
        self.fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.tracker.update("custom_loss", loss.detach(), batch_size=inputs.shape[0])
        self.tracker.update(
            "custom_aux",
            torch.tensor(float(batch_idx + 1), device=self.fabric.device),
            batch_size=inputs.shape[0],
        )


def test_custom_history(
    model_factory: ModelFactory,
) -> None:
    model = model_factory(
        model_class=CustomTrackingModel,
        no_validation=True,
        scheduler=False,
    )
    data = DataLoader(make_dataset(2, seed=900), batch_size=1, shuffle=False)

    model.fit(
        train_data=data,
        epochs=1,
        show_metrics=False,
        show_progress_bar=False,
    )

    assert set(model.history) == {"custom_loss", "custom_aux"}
    assert model.history["custom_loss"][0] >= 0
    assert model.history["custom_aux"] == [1.5]


def test_zero_step_and_epoch() -> None:
    device = Fabric(devices=1, accelerator="auto").device
    fabric = MagicMock()
    fabric.device = device
    fabric.is_global_zero = True
    fabric.all_reduce = lambda value, **kwargs: value
    fabric.broadcast = lambda value: value
    tracker = EmberTracker(fabric, fabric_logging=True)
    tracker.register("loss", key_type="train")

    step_value = torch.tensor(0.5, device=device)
    tracker.update("loss", step_value, step=0)
    fabric.log.assert_called_once_with("loss_step", step_value, step=0)

    tracker.update("loss", torch.tensor(2.0, device=device), batch_size=1)
    tracker.sync_epoch(key_type="train", epoch=0)
    assert len(tracker.history["loss"]) == 1
    assert fabric.log.call_count == 2


def test_named_metrics() -> None:
    tracker = EmberTracker(Fabric(accelerator="cpu", devices=1))
    first, second = MeanMetric(), MeanMetric()
    tracker.register("loss", "train")
    tracker.register_torchmetric("first", first, "train")
    tracker.register_torchmetric("second", second, "train")
    unused = MeanMetric()
    tracker.register_torchmetric("unused", unused, "train")
    with pytest.raises(ValueError, match="already registered"):
        tracker.register_torchmetric("loss", MeanMetric(), "train")
    with pytest.raises(ValueError, match="already registered"):
        tracker.register_torchmetric("first", MeanMetric(), "train")
    with pytest.raises(ValueError, match="already registered"):
        tracker.register("first", "train")
    with pytest.raises(ValueError, match="separate instance"):
        tracker.register_torchmetric("shared", first, "val")
    with pytest.raises(KeyError):
        tracker.update_torchmetric("unknown", torch.tensor(1.0))
    with pytest.raises(TypeError, match="scalar key"):
        tracker.update_torchmetric("loss", torch.tensor(1.0))
    with pytest.raises(TypeError, match="TorchMetric key"):
        tracker.update("first", torch.tensor(1.0))

    tracker.update("unknown", torch.tensor(10.0))
    tracker.update("loss", torch.tensor(2.0), batch_size=3)
    tracker.update("loss", torch.tensor(4.0))
    tracker.update_torchmetric("first", torch.tensor([2.0, 4.0]), weight=1.0)
    tracker.update_torchmetric("second", torch.tensor(7.0))
    with patch.object(unused, "compute", side_effect=AssertionError("unused metric")):
        tracker.sync_epoch("train", epoch=0)
    assert tracker.get_history_items() == {
        "loss": [2.5],
        "first": [3.0],
        "second": [7.0],
        "unused": [0.0],
    }
    assert not first.update_called and not second.update_called
    tracker.sync_epoch("train", epoch=1)
    assert tracker.get_history_items()["loss"] == [2.5, 0.0]
    assert tracker.get_history_items()["first"] == [3.0, 0.0]
    tracker.register("loss", "train")
    assert tracker.get_history_items()["loss"] == []
    tracker.sync_epoch("empty")


def test_metric_failure() -> None:
    tracker = EmberTracker(Fabric(accelerator="cpu", devices=1))
    metric = MulticlassConfusionMatrix(num_classes=2)
    tracker.register("loss", "val")
    tracker.register_torchmetric("confusion", metric, "val")
    with pytest.raises(ValueError):
        tracker.update_torchmetric("confusion", torch.zeros(3, 4), torch.zeros(2))
    assert not metric.update_called

    tracker.update("loss", torch.tensor(10.0))
    tracker.update_torchmetric("confusion", torch.tensor([0, 1]), torch.tensor([0, 1]))
    with pytest.raises(
        ValueError, match="confusion must compute a single tensor value"
    ):
        tracker.sync_epoch("val")
    assert not metric.update_called
    assert tracker.get_history_items() == {"loss": [], "confusion": []}
    tracker.update("loss", torch.tensor(2.0))
    tracker.sync_epoch(keys="loss")
    assert tracker.get_history_items()["loss"] == [2.0]


def test_checkpoint_reset() -> None:
    fabric = Fabric(accelerator="cpu", devices=1)

    def make_tracker() -> EmberTracker:
        tracker = EmberTracker(fabric)
        tracker.register("loss", "train")
        tracker.register_torchmetric("score", MeanMetric(), "train")
        return tracker

    tracker = make_tracker()
    tracker.update("loss", torch.tensor(1.0))
    tracker.update_torchmetric("score", torch.tensor(2.0))
    tracker.sync_epoch("train", epoch=3)
    tracker.update("loss", torch.tensor(100.0))
    tracker.update_torchmetric("score", torch.tensor(100.0))
    state = tracker.state_dict()
    restored = EmberTracker(fabric)
    restored.register_torchmetric("score", MeanMetric(), "train")
    restored.register_torchmetric("new", MeanMetric(), "val")
    restored.update_torchmetric("new", torch.tensor(10.0))
    restored.sync_epoch("val", epoch=0)
    restored.update_torchmetric("score", torch.tensor(99.0))
    restored.load_state_dict(state)
    assert restored.get_history_items() == {**tracker.get_history_items(), "new": []}
    restored.update("loss", torch.tensor(4.0))
    restored.update_torchmetric("score", torch.tensor(5.0))
    restored.sync_epoch("train", epoch=4)
    assert restored.get_history_items() == {
        "loss": [1.0, 4.0],
        "score": [2.0, 5.0],
        "new": [],
    }

    bare = EmberTracker(fabric)
    bare.load_state_dict(state)
    assert bare.get_history_items() == {"loss": [1.0]}
    bare.update("loss", torch.tensor(3.0))
    bare.sync_epoch("train", epoch=4)
    assert bare.get_history_items() == {"loss": [1.0, 3.0]}

    mismatched = EmberTracker(fabric)
    mismatched.register(["loss", "score"], "train")
    with pytest.raises(ValueError, match="value kind conflicts for score"):
        mismatched.load_state_dict(state)
    with pytest.raises(ValueError, match="requires history and key groups"):
        restored.load_state_dict({"history": {}})
