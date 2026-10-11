from pathlib import Path

import pytest
import torch
import torch.nn as nn
from conftest import ModelFactory, SyntheticData, TinyClassifier, make_dataset
from lightning import Fabric
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Metric, Precision
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score

from ember import EmberModel


def make_loader(num_samples: int, *, batch_size: int = 1) -> DataLoader:
    return DataLoader(
        make_dataset(num_samples, seed=400 + num_samples),
        batch_size=batch_size,
        shuffle=False,
    )


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float = 0.1) -> None:
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def make_counting_model() -> tuple[EmberModel, CountingSGD]:
    module = TinyClassifier()
    optimizer = CountingSGD(module.parameters())
    model = EmberModel(
        fabric=Fabric(devices=1, accelerator="auto"),
        model=module,
        optimizers=[optimizer],
        loss_fn=nn.CrossEntropyLoss(),
        no_validation=True,
    )
    return model, optimizer


class FixedLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale


class TargetOverrideMetricModel(EmberModel):
    def update_metric(
        self,
        metric: Metric,
        batch: tuple[torch.Tensor, torch.Tensor] | tuple,
        y_pred: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        metric.update(y_pred, torch.zeros_like(batch[-1]))


def make_metric_model(
    metrics: list[Metric],
    *,
    model_class: type[EmberModel] = EmberModel,
) -> EmberModel:
    module = FixedLogitModel()
    return model_class(
        fabric=Fabric(devices=1, accelerator="auto"),
        model=module,
        optimizers=[torch.optim.SGD(module.parameters(), lr=0)],
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
    )


def make_metric_loader() -> DataLoader:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1])
    return DataLoader(TensorDataset(logits, targets), batch_size=3, shuffle=False)


@pytest.mark.parametrize("as_module", [False, True])
def test_paired_augmentation(
    model_factory: ModelFactory,
    as_module: bool,
) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def flip(
        inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((inputs, targets))
        return inputs.flip(-1), targets.flip(-1)

    class PairedFlip(nn.Module):
        def forward(
            self, inputs: torch.Tensor, targets: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return flip(inputs, targets)

    model = model_factory(aug_fn=PairedFlip() if as_module else flip)
    model.augment_targets = True
    grid = torch.arange(16).reshape(2, 1, 2, 4)
    inputs = grid.repeat(1, 3, 1, 1).float()
    targets = grid.repeat(1, 2, 1, 1)

    augmented_inputs, augmented_targets = model.augment_batch((inputs, targets))

    ((received_inputs, received_targets),) = calls
    assert received_inputs is inputs
    assert received_targets is targets
    torch.testing.assert_close(augmented_inputs, inputs.flip(-1))
    torch.testing.assert_close(augmented_targets, targets.flip(-1))


def test_input_only_augmentation(model_factory: ModelFactory) -> None:
    def shift(inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1

    model = model_factory(aug_fn=shift)
    inputs, targets = make_dataset(2, seed=600).tensors

    augmented_inputs, augmented_targets = model.augment_batch((inputs, targets))

    torch.testing.assert_close(augmented_inputs, inputs + 1)
    assert augmented_targets is targets


@pytest.mark.parametrize("augment_targets", [False, True])
def test_no_augmentation(
    model_factory: ModelFactory,
    augment_targets: bool,
) -> None:
    model = model_factory()
    model.augment_targets = augment_targets
    inputs, targets = make_dataset(2, seed=600).tensors

    augmented_inputs, augmented_targets = model.augment_batch((inputs, targets))

    assert augmented_inputs is inputs
    assert augmented_targets is targets


def test_fit_history(
    model_factory: ModelFactory,
    synthetic_data: SyntheticData,
) -> None:
    model = model_factory()

    model.fit(
        emberdata=synthetic_data,
        epochs=2,
        show_metrics=False,
        show_progress_bar=False,
    )

    assert model.epoch_count == 2
    assert model.step_count == 8
    assert len(model.history["train_loss"]) == 2
    assert len(model.history["val_loss"]) == 2
    assert all(loss > 0 for loss in model.history["train_loss"])
    assert model.history["lr_0"]


def test_metric_lifecycle() -> None:
    metric = MulticlassF1Score(num_classes=2, average="macro")
    model = make_metric_model([metric])
    data = make_metric_loader()

    model.fit(
        train_data=data,
        val_data=data,
        epochs=2,
        show_metrics=False,
        show_progress_bar=False,
    )

    expected = pytest.approx([1 / 3, 1 / 3])
    assert model.history["train_MulticlassF1Score"] == expected
    assert model.history["val_MulticlassF1Score"] == expected
    assert not metric.update_called


def test_metric_override() -> None:
    metric = Accuracy(task="multiclass", num_classes=2)
    model = make_metric_model([metric], model_class=TargetOverrideMetricModel)

    results = model.evaluate(
        eval_data=make_metric_loader(),
        show_metrics=False,
        show_progress_bar=False,
    )

    assert results["val_MulticlassAccuracy"] == [1.0]
    assert not metric.update_called


def test_metric_validation() -> None:
    duplicate_metrics = [
        Accuracy(task="multiclass", num_classes=2),
        Accuracy(task="multiclass", num_classes=2),
    ]
    with pytest.raises(ValueError, match="Metric names must be unique"):
        make_metric_model(duplicate_metrics)

    metric = MulticlassConfusionMatrix(num_classes=2)
    model = make_metric_model([])
    with pytest.raises(
        ValueError,
        match="MulticlassConfusionMatrix must compute a single tensor value",
    ):
        model.evaluate(
            eval_data=make_metric_loader(),
            metrics=[metric],
            show_metrics=False,
            show_progress_bar=False,
        )
    assert not metric.update_called


def test_step_resume(
    model_factory: ModelFactory,
    synthetic_data: SyntheticData,
) -> None:
    model = model_factory(no_validation=True)

    model.fit(
        emberdata=synthetic_data,
        steps=5,
        show_metrics=False,
        show_progress_bar=False,
    )
    assert model.step_count == 5
    assert model.epoch_count == 2
    assert not model.resuming

    model.fit(
        emberdata=synthetic_data,
        steps=2,
        show_metrics=False,
        show_progress_bar=False,
    )
    assert model.step_count == 7
    assert model.epoch_count == 3
    assert model.resuming


def test_epoch_limit(
    model_factory: ModelFactory,
    synthetic_data: SyntheticData,
) -> None:
    model = model_factory(no_validation=True)

    model.fit(
        emberdata=synthetic_data,
        epochs=1,
        steps=10_000,
        show_metrics=False,
        show_progress_bar=False,
    )

    assert model.epoch_count == 1
    assert model.step_count == 4


def test_training_limit_required(
    model_factory: ModelFactory,
    synthetic_data: SyntheticData,
) -> None:
    model = model_factory()

    with pytest.raises(ValueError):
        model.fit(emberdata=synthetic_data)

    assert model.step_count == 0


def test_gradient_accumulation() -> None:
    cases = [
        (1, None, 3),
        (None, 3, 5),
    ]

    for epochs, steps, num_samples in cases:
        model, optimizer = make_counting_model()
        model.fit(
            train_data=make_loader(num_samples),
            epochs=epochs,
            steps=steps,
            accumulate_grad_batches=2,
            show_metrics=False,
            show_progress_bar=False,
        )

        assert optimizer.step_calls == 2
        assert all(parameter.grad is None for parameter in model.model.parameters())


def test_batch_limits() -> None:
    for invalid_limit in (-1, -0.1, 1.1):
        model, _ = make_counting_model()
        with pytest.raises(ValueError, match="batch_limit"):
            model.fit(
                train_data=make_loader(2),
                epochs=1,
                batch_limit=invalid_limit,
                show_metrics=False,
                show_progress_bar=False,
            )

    model, _ = make_counting_model()
    model.fit(
        train_data=make_loader(10),
        epochs=1,
        batch_limit=0.01,
        show_metrics=False,
        show_progress_bar=False,
    )
    assert model.step_count == 1


def test_empty_dataloaders(model_factory: ModelFactory) -> None:
    empty_data = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))
    )

    with pytest.raises(ValueError, match="Training dataloader is empty"):
        model_factory().fit(
            train_data=empty_data,
            epochs=1,
            show_metrics=False,
            show_progress_bar=False,
        )

    with pytest.raises(ValueError, match="Validation dataloader is empty"):
        model_factory().fit(
            train_data=make_loader(1),
            val_data=empty_data,
            epochs=1,
            show_metrics=False,
            show_progress_bar=False,
        )

    skipped_model = model_factory(no_validation=True)
    skipped_model.fit(
        train_data=make_loader(1),
        val_data=empty_data,
        epochs=1,
        show_metrics=False,
        show_progress_bar=False,
    )
    assert skipped_model.step_count == 1

    with pytest.raises(ValueError, match="Evaluation dataloader is empty"):
        model_factory().evaluate(
            eval_data=empty_data,
            show_metrics=False,
            show_progress_bar=False,
        )

    with pytest.raises(ValueError, match="Prediction dataloader is empty"):
        model_factory().predict(data=empty_data, show_progress_bar=False)


def test_plateau_scheduler(
    model_factory: ModelFactory,
) -> None:
    model = model_factory()
    optimizer = model.optimizers[0]
    initial_lr = optimizer.param_groups[0]["lr"]

    model.history["val_loss"] = [1.0]
    model.scheduler_step()
    assert optimizer.param_groups[0]["lr"] == initial_lr

    model.history["val_loss"].append(2.0)
    model.scheduler_step()
    assert optimizer.param_groups[0]["lr"] == initial_lr * 0.5

    skipped_model = model_factory(no_validation=True)
    skipped_model.scheduler_step()
    assert skipped_model.optimizers[0].param_groups[0]["lr"] == initial_lr


class MultiOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scaled = inputs * self.scale
        return scaled, scaled + 1, scaled.sum(dim=1, keepdim=True)


@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_predict_outputs(
    model_factory: ModelFactory,
    multiple_outputs: bool,
) -> None:
    model = model_factory(
        model=MultiOutputModel() if multiple_outputs else FixedLogitModel(),
        metrics=False,
        no_validation=True,
        scheduler=False,
    )
    data = make_loader(4, batch_size=2)

    predictions = model.predict(data=data, show_progress_bar=False)

    dataset = data.dataset
    assert isinstance(dataset, TensorDataset)
    inputs = dataset.tensors[0]
    expected = (
        (inputs, inputs + 1, inputs.sum(dim=1, keepdim=True))
        if multiple_outputs
        else (inputs,)
    )
    for prediction, expected_output in zip(predictions, expected, strict=True):
        assert isinstance(prediction, torch.Tensor)
        torch.testing.assert_close(prediction.cpu(), expected_output)


class CustomEvalTrackerModel(EmberModel):
    def setup_tracker(self) -> None:
        super().setup_tracker()
        self.tracker.register("val_custom_score", key_type="val")

    def eval_step(self, batch, batch_idx: int, eval_tracker=None):
        output = super().eval_step(batch, batch_idx, eval_tracker)
        if output is not None and eval_tracker is not None:
            _, predictions = output
            eval_tracker.update(
                "val_custom_score",
                torch.tensor(0.25, device=self.fabric.device),
                batch_size=predictions.shape[0],
            )
        return output


def test_evaluation_isolation(
    model_factory: ModelFactory,
) -> None:
    model = model_factory(
        model_class=CustomEvalTrackerModel,
        metrics=False,
    )
    model.metrics = model.fabric.to_device(
        [Precision(task="multiclass", num_classes=2, average="macro")]
    )
    history_before = {key: list(values) for key, values in model.history.items()}
    evaluation_metric = Accuracy(task="multiclass", num_classes=2)

    metrics = model.evaluate(
        eval_data=make_loader(4, batch_size=2),
        metrics=[evaluation_metric],
        show_metrics=False,
        show_progress_bar=False,
    )

    assert set(metrics) == {
        "val_loss",
        "val_MulticlassAccuracy",
        "val_custom_score",
    }
    assert model.history == history_before
    assert not evaluation_metric.update_called


class LayeredClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 3)
        self.output = nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(inputs)))


def test_weights_and_freezing(
    fitted_model: EmberModel,
    model_factory: ModelFactory,
    tmp_path: Path,
) -> None:
    weights_path = tmp_path / "weights.pth"
    weights_before = {
        key: value.detach().cpu().clone()
        for key, value in fitted_model.model.state_dict().items()
    }
    fitted_model.save_weights(weights_path)

    with torch.no_grad():
        for value in fitted_model.model.state_dict().values():
            if value.is_floating_point():
                value.add_(1)
    fitted_model.load_weights(weights_path)

    for key, value_before in weights_before.items():
        assert torch.equal(
            value_before,
            fitted_model.model.state_dict()[key].detach().cpu(),
        )

    fitted_model.freeze()
    assert all(
        not parameter.requires_grad for parameter in fitted_model.model.parameters()
    )
    fitted_model.unfreeze()
    assert all(parameter.requires_grad for parameter in fitted_model.model.parameters())

    layered_module = LayeredClassifier()
    layered_model = model_factory(model=layered_module)
    layered_model.freeze(layer_names=["_forward_module.hidden"])
    hidden_parameters = list(layered_module.hidden.parameters())
    assert all(not parameter.requires_grad for parameter in hidden_parameters)
    assert all(
        parameter.requires_grad for parameter in layered_module.output.parameters()
    )
    layered_model.unfreeze(layer_names=["_forward_module.hidden"])
    assert all(parameter.requires_grad for parameter in hidden_parameters)
