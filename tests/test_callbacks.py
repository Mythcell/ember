import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock

import optuna
import pytest
import torch
from conftest import ModelFactory, SyntheticData, make_dataset
from torch.utils.data import DataLoader

from ember import EmberModel
from ember.callbacks import (
    EarlyStopping,
    EmberCallback,
    ModelCheckpoint,
    OptunaTrialPruner,
)


def make_loader(num_samples: int = 2) -> DataLoader:
    return DataLoader(
        make_dataset(num_samples, seed=500 + num_samples),
        batch_size=1,
        shuffle=False,
    )


def make_checkpoint_callback(save_dir: Path, **kwargs) -> ModelCheckpoint:
    params = {
        "monitor": "val_loss",
        "save_dir": save_dir,
        "save_best": False,
        "save_last": False,
        "verbose": False,
    }
    params.update(kwargs)
    return ModelCheckpoint(**params)


def test_model_checkpoint_rejects_invalid_values(tmp_path: Path) -> None:
    invalid_mode = cast("Any", "middle")
    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        ModelCheckpoint(monitor="val_loss", save_dir=tmp_path, mode=invalid_mode)
    with pytest.raises(ValueError, match="save_top_k must be >= -1"):
        ModelCheckpoint(monitor="val_loss", save_dir=tmp_path, save_top_k=-2)
    with pytest.raises(ValueError, match="save_every_n_epochs must be >= 0"):
        ModelCheckpoint(monitor="val_loss", save_dir=tmp_path, save_every_n_epochs=-1)
    with pytest.raises(ValueError, match="save_every_n_steps must be >= 0"):
        ModelCheckpoint(monitor="val_loss", save_dir=tmp_path, save_every_n_steps=-1)
    with pytest.raises(ValueError, match="warmup_epochs must be >= 0"):
        ModelCheckpoint(monitor="val_loss", save_dir=tmp_path, warmup_epochs=-1)


def test_early_stopping_rejects_invalid_values() -> None:
    invalid_mode = cast("Any", "middle")
    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        EarlyStopping(monitor="val_loss", mode=invalid_mode)
    with pytest.raises(ValueError, match="patience must be >= 0"):
        EarlyStopping(monitor="val_loss", patience=-1)
    with pytest.raises(ValueError, match="warmup_epochs must be >= 0"):
        EarlyStopping(monitor="val_loss", warmup_epochs=-1)


def fake_embermodel(history: dict[str, list[float]]) -> EmberModel:
    value = SimpleNamespace(
        history=history,
        should_train=True,
        fabric=SimpleNamespace(is_global_zero=True),
    )
    return cast("EmberModel", value)


def test_early_stopping_follows_synthetic_metric_history() -> None:
    model = fake_embermodel({"val_loss": []})
    callback = EarlyStopping(monitor="val_loss", patience=1)
    callback.on_train_start(model)

    for epoch, loss in enumerate((1.0, 0.5, 0.5)):
        model.history["val_loss"].append(loss)
        callback.on_epoch_end(model, epoch=epoch)
        if epoch < 2:
            assert model.should_train

    assert callback.best_value == 0.5
    assert callback.best_value_epoch == 1
    assert not model.should_train


def test_early_stopping_begins_monitoring_after_warmup() -> None:
    model = fake_embermodel({"val_loss": []})
    callback = EarlyStopping(
        monitor="val_loss",
        patience=1,
        stopping_threshold=0.2,
        warmup_epochs=2,
    )
    callback.on_train_start(model)

    for epoch, loss in enumerate((float("nan"), 0.1, 1.0, 1.0)):
        model.history["val_loss"].append(loss)
        callback.on_epoch_end(model, epoch=epoch)
        if epoch < 3:
            assert model.should_train

    assert callback.best_value == 1.0
    assert callback.best_value_epoch == 2
    assert not model.should_train


def test_early_stopping_recovers_only_post_warmup_history() -> None:
    model = fake_embermodel({"val_loss": [0.1, 0.2, 1.0]})
    callback = EarlyStopping(monitor="val_loss", patience=1, warmup_epochs=2)

    callback.on_train_start(model, epoch=3)

    assert callback.best_value == 1.0
    assert callback.best_value_epoch == 2


@pytest.mark.parametrize(
    ("mode", "metric_history", "stopping_threshold"),
    [
        ("min", (1.0, 0.5), 0.5),
        ("max", (0.5, 1.0), 1.0),
    ],
)
def test_early_stopping_stops_when_threshold_is_reached(
    mode: str,
    metric_history: tuple[float, ...],
    stopping_threshold: float,
) -> None:
    model = fake_embermodel({"metric": []})
    callback = EarlyStopping(
        monitor="metric",
        mode=cast("Any", mode),
        stopping_threshold=stopping_threshold,
    )
    callback.on_train_start(model)

    for epoch, value in enumerate(metric_history):
        model.history["metric"].append(value)
        callback.on_epoch_end(model, epoch=epoch)

    assert not model.should_train


def test_callbacks_require_registered_metrics() -> None:
    model = fake_embermodel({})

    with pytest.raises(ValueError, match="val_loss is not a registered metric"):
        EarlyStopping("val_loss").on_train_start(model)
    with pytest.raises(ValueError, match="val_loss is not a registered metric"):
        OptunaTrialPruner(MagicMock(), "val_loss").on_train_start(model)


class RecordingCallback(EmberCallback):
    def __init__(self) -> None:
        self.events: list[tuple[str, int, int, int | None]] = []

    def record(
        self,
        name: str,
        epoch: int = 0,
        step: int = 0,
        batch_idx: int | None = None,
    ) -> None:
        self.events.append((name, epoch, step, batch_idx))

    def on_train_start(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("train_start", epoch, step)

    def on_train_end(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("train_end", epoch, step)

    def on_epoch_start(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("epoch_start", epoch, step)

    def on_epoch_end(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("epoch_end", epoch, step)

    def on_train_epoch_start(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("train_epoch_start", epoch, step)

    def on_train_epoch_end(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("train_epoch_end", epoch, step)

    def on_train_batch_start(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        self.record("train_batch_start", epoch, step, batch_idx)

    def on_train_batch_end(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        self.record("train_batch_end", epoch, step, batch_idx)

    def on_validation_epoch_start(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("validation_epoch_start", epoch, step)

    def on_validation_epoch_end(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.record("validation_epoch_end", epoch, step)

    def on_validation_batch_start(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        self.record("validation_batch_start", epoch, step, batch_idx)

    def on_validation_batch_end(
        self,
        embermodel: EmberModel,
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        self.record("validation_batch_end", epoch, step, batch_idx)


def test_callback_hooks_receive_ordered_epoch_and_step_state(
    model_factory: ModelFactory,
) -> None:
    model = model_factory(metrics=False, scheduler=False)
    callback = RecordingCallback()

    model.fit(
        train_data=make_loader(2),
        val_data=make_loader(1),
        epochs=1,
        callbacks=[callback],
        show_metrics=False,
        show_progress_bar=False,
    )

    assert callback.events == [
        ("train_start", 0, 0, None),
        ("epoch_start", 0, 0, None),
        ("train_epoch_start", 0, 0, None),
        ("train_batch_start", 0, 0, 0),
        ("train_batch_end", 0, 1, 0),
        ("train_batch_start", 0, 1, 1),
        ("train_batch_end", 0, 2, 1),
        ("train_epoch_end", 0, 2, None),
        ("validation_epoch_start", 0, 2, None),
        ("validation_batch_start", 0, 2, 0),
        ("validation_batch_end", 0, 2, 0),
        ("validation_epoch_end", 0, 2, None),
        ("epoch_end", 0, 2, None),
        ("train_end", 0, 2, None),
    ]


def test_validation_callbacks_are_safe_when_validation_is_disabled(
    model_factory: ModelFactory,
    tmp_path: Path,
) -> None:
    model = model_factory(no_validation=True, metrics=False, scheduler=False)
    recording_callback = RecordingCallback()
    callbacks: list[EmberCallback] = [
        recording_callback,
        EarlyStopping(monitor="val_loss", patience=1),
        ModelCheckpoint(
            monitor="val_loss",
            save_dir=tmp_path,
            save_best=True,
            save_last=False,
            verbose=False,
        ),
    ]

    model.fit(
        train_data=make_loader(1),
        val_data=make_loader(1),
        epochs=2,
        callbacks=callbacks,
        show_metrics=False,
        show_progress_bar=False,
    )

    assert not any(
        name.startswith("validation_") for name, *_ in recording_callback.events
    )
    assert model.epoch_count == 2
    assert model.history["val_loss"] == []
    assert list(tmp_path.glob("*.ckpt")) == []


def test_top_k_checkpoints_retain_the_best_values(
    model_factory: ModelFactory,
    tmp_path: Path,
) -> None:
    cases = {
        "min": ([0.3, 0.5, 0.4, 0.6, 0.2], [(0.3, 0), (0.2, 4)]),
        "max": ([0.7, 0.5, 0.6, 0.4, 0.8], [(0.7, 0), (0.8, 4)]),
    }

    for mode, (history, expected) in cases.items():
        save_dir = tmp_path / mode
        model = model_factory()
        callback = make_checkpoint_callback(save_dir, mode=mode, save_top_k=2)
        callback.on_train_start(model)

        for epoch, value in enumerate(history):
            model.history["val_loss"].append(value)
            callback.on_epoch_end(model, epoch=epoch)

        assert callback.best_checkpoints == expected
        assert sorted(path.name for path in save_dir.glob("*.ckpt")) == [
            "model-epoch_00000.ckpt",
            "model-epoch_00004.ckpt",
        ]
        model.load_best_checkpoint()


def test_checkpoint_cadence_and_last_checkpoint(
    model_factory: ModelFactory,
    tmp_path: Path,
) -> None:
    model = model_factory(no_validation=True, metrics=False, scheduler=False)
    callback = ModelCheckpoint(
        monitor="train_loss",
        save_dir=tmp_path,
        save_best=False,
        save_last=True,
        save_every_n_epochs=2,
        save_every_n_steps=2,
        verbose=False,
    )

    model.fit(
        train_data=make_loader(4),
        epochs=3,
        callbacks=[callback],
        show_metrics=False,
        show_progress_bar=False,
    )

    assert sorted(path.name for path in tmp_path.glob("*_periodic.ckpt")) == [
        "model-epoch_00000_periodic.ckpt",
        "model-epoch_00002_periodic.ckpt",
    ]
    assert sorted(path.name for path in tmp_path.glob("model-step_*.ckpt")) == [
        f"model-step_{step:07d}.ckpt" for step in range(2, 13, 2)
    ]
    assert callback.last_ckpt_path == tmp_path / "model-epoch_00002_last.ckpt"
    model.load_last_checkpoint()


def test_zero_top_k_disables_best_checkpoints(
    model_factory: ModelFactory,
    tmp_path: Path,
) -> None:
    model = model_factory(no_validation=True, metrics=False, scheduler=False)
    callback = ModelCheckpoint(
        monitor="train_loss",
        save_dir=tmp_path,
        save_best=True,
        save_top_k=0,
        save_last=False,
        verbose=False,
    )

    model.fit(
        train_data=make_loader(2),
        epochs=2,
        callbacks=[callback],
        show_metrics=False,
        show_progress_bar=False,
    )

    assert list(tmp_path.glob("*.ckpt")) == []


def test_checkpoint_restores_model_optimizer_scheduler_and_history(
    fitted_model: EmberModel,
    model_factory: ModelFactory,
    synthetic_data: SyntheticData,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    weights_before = {
        key: value.cpu().clone()
        for key, value in fitted_model.model.state_dict().items()
    }
    optimizer_before = copy.deepcopy(fitted_model.optimizers[0].state_dict())
    assert fitted_model.schedulers is not None
    scheduler_before = copy.deepcopy(fitted_model.schedulers[0].state_dict())
    history_before = copy.deepcopy(fitted_model.history)
    fitted_model.save_checkpoint(checkpoint_path)

    restored_model = model_factory()
    restored_model.load_checkpoint(checkpoint_path)

    for key, weight_before in weights_before.items():
        assert torch.equal(weight_before, restored_model.model.state_dict()[key].cpu())

    optimizer_after = restored_model.optimizers[0].state_dict()
    for parameter, state_before in optimizer_before["state"].items():
        state_after = optimizer_after["state"][parameter]
        for key, value_before in state_before.items():
            value_after = state_after[key]
            if isinstance(value_before, torch.Tensor):
                assert torch.equal(value_before.cpu(), value_after.cpu())
            else:
                assert value_before == value_after

    assert restored_model.schedulers is not None
    assert scheduler_before == restored_model.schedulers[0].state_dict()
    assert restored_model.history == history_before
    assert restored_model.resuming

    restored_model.fit(
        emberdata=synthetic_data,
        epochs=1,
        show_metrics=False,
        show_progress_bar=False,
    )
    assert restored_model.epoch_count == fitted_model.epoch_count + 1


def test_optuna_pruner_reports_synthetic_history() -> None:
    reporting_trial = MagicMock()
    reporting_trial.should_prune.return_value = False
    model = fake_embermodel({"val_loss": [0.8, 0.4]})
    callback = OptunaTrialPruner(reporting_trial, "val_loss")
    callback.on_train_start(model)
    callback.on_epoch_end(model, epoch=1)
    reporting_trial.report.assert_called_once_with(0.4, step=1)


def test_optuna_pruner_raises_when_trial_should_prune() -> None:
    model = fake_embermodel({"val_loss": [0.4]})
    pruning_trial = MagicMock()
    pruning_trial.should_prune.return_value = True
    callback = OptunaTrialPruner(pruning_trial, "val_loss")
    with pytest.raises(optuna.TrialPruned):
        callback.on_epoch_end(model, epoch=0)


class RecordingTqdm:
    instances: ClassVar[list["RecordingTqdm"]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.closed = False
        RecordingTqdm.instances.append(self)

    def update(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


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
        raise RuntimeError("predict failed")


def run_failing_stage(model: EmberModel, stage: str) -> None:
    if stage == "train":
        model.fit(train_data=make_loader(), epochs=1)
    elif stage == "val":
        model.fit(train_data=make_loader(), val_data=make_loader(), epochs=1)
    elif stage == "eval":
        model.evaluate(eval_data=make_loader())
    else:
        model.predict(data=make_loader())


def test_progress_bars_close_when_model_steps_fail(
    model_factory: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ember.models.tqdm", RecordingTqdm)
    cases: list[tuple[str, type[EmberModel], int]] = [
        ("train", FailingTrainModel, 1),
        ("val", FailingValModel, 2),
        ("eval", FailingEvalModel, 1),
        ("predict", FailingPredictModel, 1),
    ]

    for stage, model_class, expected_bars in cases:
        RecordingTqdm.instances = []
        model = model_factory(
            model_class=model_class,
            metrics=False,
            scheduler=False,
        )
        with pytest.raises(RuntimeError, match=f"{stage} failed"):
            run_failing_stage(model, stage)
        assert len(RecordingTqdm.instances) == expected_bars
        assert all(progress.closed for progress in RecordingTqdm.instances)
