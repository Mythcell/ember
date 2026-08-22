from unittest.mock import MagicMock

import torch
from conftest import ModelFactory, make_dataset
from lightning import Fabric
from torch.utils.data import DataLoader

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


def test_history_exposes_registered_tracker_values(fitted_model: EmberModel) -> None:
    required_keys = {
        "train_loss",
        "val_loss",
        "train_MulticlassAccuracy",
        "val_MulticlassAccuracy",
        "lr_0",
    }

    assert required_keys <= fitted_model.history.keys()
    assert all(len(fitted_model.history[key]) == 2 for key in required_keys)
    assert (
        fitted_model.history.keys() == fitted_model.tracker.get_history_items().keys()
    )


def test_custom_tracker_keys_replace_default_history(
    model_factory: ModelFactory,
) -> None:
    model = model_factory(
        model_class=CustomTrackingModel,
        metrics=False,
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


def test_zero_is_a_valid_logging_step_and_epoch() -> None:
    device = Fabric(devices=1, accelerator="auto").device
    fabric = MagicMock()
    fabric.device = device
    fabric.is_global_zero = True
    fabric.all_reduce = lambda value: value
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
