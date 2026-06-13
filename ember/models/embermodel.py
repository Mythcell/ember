import contextlib
from collections.abc import Callable, Iterator, Sequence
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, final

import torch
import torch.nn as nn
import torchmetrics
from lightning.fabric import Fabric
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ember.utils.grad import set_requires_grad
from ember.utils._summary import model_summary

if TYPE_CHECKING:
    from lightning.fabric.wrappers import _FabricModule

    from ember.callbacks import EmberCallback

from ember.data import EmberData
from ember.tracking import EmberTracker


class EmberModel:
    """
    A class for Fabric-accelerated model training and inference.
    """

    def __init__(
        self,
        fabric: Fabric,
        model: nn.Module,
        optimizers: list[Optimizer],
        loss_fn: nn.Module | None = None,
        metrics: list[torchmetrics.Metric] | None = None,
        aug_fn: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
        schedulers: list[LRScheduler] | None = None,
        setup_fabric: bool = True,
        log_lr: Literal["auto"] | bool = "auto",
        random_seed: int | None = None,
        deterministic: bool = False,
        no_validation: bool = False,
        reduce_lr_on_plateau_monitor: str = "val_loss",
    ) -> None:
        """
        A class for Fabric-accelerated model training and inference.

        EmberModel supports flexible model training and inference. The default
        training, validation, evaluation and inference loops are designed for
        supervised, tuple-like `(input, target)` batches. For other PyTorch workflows,
        subclass EmberModel and override `train_step`, `val_step`, `eval_step`,
        `predict_step`, or the tracker/scheduler hooks as needed.

        Args:
            fabric: The Fabric accelerator. Override `fabric_setup()` for custom Fabric
                initialization.
            model: The PyTorch module.
            optimizers: A list of torch Optimizers for training.
            loss_fn: The default loss function to use. Custom losses should be defined
                as `nn.Module` classes.
            metrics: List of torchmetric Metrics for automatic metric calculation.
                Override `calculate_metric` for custom metric calculations.
            aug_fn: Function to apply to batches from the dataloader. Override
                `augment_batch` for custom data augmentation.
            schedulers: A list of schedulers to use. Override `scheduler_step` for
                custom scheduler steps.
            setup_fabric: Whether to run the `fabric_setup` function. Default is True.
            log_lr: Whether to log learning rates, either True, False or "auto".
                If "auto", will only log learning rates if a scheduler is also provided.
                Defaults to "auto".
            random_seed: Seed to initialize for training.
            deterministic: Flag for using deterministic algorithms. Default is False.
            no_validation: Whether to always skip validation stage. Default is False.
            reduce_lr_on_plateau_monitor: The default metric to monitor for the
                `ReduceLROnPlateau` scheduler, if present. Default is "val_loss".
        """
        super().__init__()
        self.fabric = fabric
        self.model = model
        self.optimizers = optimizers
        if self.fabric.world_size > 1:
            self.fabric.launch()
        if setup_fabric:
            self.fabric_setup()
        self.loss_fn = loss_fn
        self.metrics = self.fabric.to_device(metrics) if metrics else []
        self.aug_fn: Callable[[torch.Tensor], torch.Tensor] = aug_fn or nn.Identity()
        self.schedulers = schedulers
        self.reduce_lr_on_plateau_monitor = reduce_lr_on_plateau_monitor

        self.tracker = EmberTracker(fabric=self.fabric)
        self.setup_tracker()
        self._setup_lr_tracking(log_lr)
        self.history = self.tracker.get_history_items()

        if random_seed is not None:
            self.fabric.seed_everything(random_seed, workers=True)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False

        self.no_validation = no_validation
        self.should_train: bool = True
        self.resuming: bool = False

        # EmberCallback integration
        self.best_ckpt_path: Path | None = None
        self.last_ckpt_path: Path | None = None

        self.epoch_count: int = 0
        self.step_count: int = 0

        # fit() parameters with sensible defaults
        self.accumulate_grad_batches: int = 1
        self.augment_targets: bool = False
        self.gradients_clip_val: float | None = None
        self.gradients_max_norm: int | float | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def fabric_setup(self) -> None:
        """
        Runs `fabric.setup()` to initialize the model and optimizer(s).
        Custom `EmberModel` classes should override this function.
        """
        self.model, *self.optimizers = self.fabric.setup(self.model, *self.optimizers)

    def _setup_dataloader_with_fabric(self, data: DataLoader) -> DataLoader:
        return cast("DataLoader", self.fabric.setup_dataloaders(data))

    @contextlib.contextmanager
    def _progress_bar(
        self, total: int, desc: str, show_progress_bar: bool
    ) -> Iterator[Any]:
        progress_bar = None
        if self.fabric.is_global_zero:
            progress_bar = tqdm(
                total=total,
                desc=desc,
                disable=not show_progress_bar,
            )
        try:
            yield progress_bar
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def _require_non_empty_dataloader(self, data: DataLoader, name: str) -> int:
        dataloader_len = len(data)
        if dataloader_len == 0:
            raise ValueError(f"{name} is empty")
        return dataloader_len

    def _resolve_batch_limit(
        self, batch_limit: float | int, dataloader_len: int
    ) -> int:
        if isinstance(batch_limit, float):
            if batch_limit < 0 or batch_limit > 1:
                raise ValueError("float batch_limit must be between 0 and 1")
            if batch_limit == 0:
                return 0
            return max(1, int(batch_limit * dataloader_len))
        if batch_limit < 0:
            raise ValueError("batch_limit must be non-negative")
        return batch_limit

    def train_step(
        self, batch: tuple[torch.Tensor, ...] | tuple, batch_idx: int, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Conducts a single training step.

        Processes the input batch, backpropagates the loss and updates the optimizer.
        `EmberModel` subclasses should override this function instead of `fit()` when
        implementing custom training steps. You may return a tuple (loss, y_pred) for
        automatic metric calculation.

        Args:
            batch: A single batch from a `DataLoader`.
            batch_idx: Index of the current batch.
            epoch: The current epoch.

        Returns:
            Optional tuple of (loss, y_pred) for default logging and metric calculation.
        """
        batch = self.augment_batch(batch)
        x, y = batch[0], batch[-1]
        (opt, *_) = self.optimizers

        is_accumulating = (batch_idx + 1) % self.accumulate_grad_batches != 0
        with self.fabric.no_backward_sync(
            cast("_FabricModule", self.model), enabled=is_accumulating
        ):
            y_pred = self.model(x)
            loss_fn = cast("Callable[..., torch.Tensor]", self.loss_fn)
            loss = loss_fn(y_pred, y)
            self.fabric.backward(loss / self.accumulate_grad_batches)

        if not is_accumulating:
            if (
                self.gradients_clip_val is not None
                or self.gradients_max_norm is not None
            ):
                self.fabric.clip_gradients(
                    self.model,
                    opt,
                    clip_val=self.gradients_clip_val,
                    max_norm=self.gradients_max_norm,
                )
            opt.step()
            opt.zero_grad()

        return loss, y_pred

    def val_step(
        self, batch: tuple[torch.Tensor, ...] | tuple, batch_idx: int, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Conducts a single validation step.

        `EmberModel` subclasses should override this function instead of `fit()` when
        implementing custom validation steps. Note that this function is run within a
        `torch.inference_mode()` context. You may return a tuple (loss, y_pred) for
        automatic metric calculation.

        Args:
            batch: A single batch from a `DataLoader`.
            batch_idx: Index of the current batch.
            epoch: The current epoch.

        Returns:
            Optional tuple (`loss`, `y_pred`) for default logging and metric
            calculation.
        """
        x, y = batch[0], batch[-1]
        y_pred = self.model(x)
        loss_fn = cast("Callable[..., torch.Tensor]", self.loss_fn)
        loss = loss_fn(y_pred, y)
        return loss, y_pred

    def eval_step(
        self,
        batch: tuple[torch.Tensor, ...] | tuple,
        batch_idx: int,
        eval_tracker: EmberTracker | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Conducts a single evaluation step.

        `EmberModel` subclasses should override this function instead of `evaluate()`.
        Note that this function is run within a `torch.inference_mode()` context.
        You may return a tuple (loss, y_pred) for automatic metric calculation.

        Args:
            batch: A single batch from a `DataLoader`.
            batch_idx: Index of the current batch.
            eval_tracker: Optional tracker instance passed by `evaluate()`.

        Returns:
            Optional tuple (`loss`, `y_pred`) for default logging and metric
            calculation.
        """
        x, y = batch[0], batch[-1]
        y_pred = self.model(x)
        loss_fn = cast("Callable[..., torch.Tensor]", self.loss_fn)
        loss = loss_fn(y_pred, y)
        return loss, y_pred

    def predict_step(
        self, batch: tuple[torch.Tensor, ...] | tuple, batch_idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Processes an input batch and returns the model's predictions.

        `EmberModel` subclasses should override this function instead of `predict()`.

        Args:
            batch: A single batch from a `DataLoader`.
            batch_idx: Index of the current batch.

        Returns:
            A single `torch.Tensor` or tuple of tensors.
        """
        x = batch[0]
        return cast("torch.Tensor | tuple[torch.Tensor, ...]", self.model(x))

    def calculate_metric(
        self,
        metric: torchmetrics.Metric,
        batch: tuple[torch.Tensor, torch.Tensor] | tuple,
        y_pred: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Calculates the metric value for the given Torchmetric metric.

        Args:
            metric: The Torchmetric metric to compute.
            batch: A single batch from a `DataLoader`.
                Targets are treated as `batch[-1]`.
            y_pred: The Tensor of model predictions.

        Returns:
            A tensor with the value of the computed metric.
        """
        y = batch[-1]
        return cast("torch.Tensor", metric(y_pred, y))

    def augment_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | tuple,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple:
        """
        Applies data augmentation to the current batch.

        Args:
            batch: A single batch from a `DataLoader`.

        Returns:
            Tuple of inputs and targets.
        """
        x, y = batch[0], batch[-1]
        if self.augment_targets:
            aug = self.aug_fn(torch.cat((x, y), dim=0))
            x, y = aug[: x.shape[0]], aug[x.shape[0] :]
        else:
            x = self.aug_fn(x)
        return x, y

    def scheduler_step(self) -> None:
        """
        Calls the `step()` function for the model scheduler (if present).

        If the scheduler is `ReduceLROnPlateau`, this function uses the monitored
        metric (default `val_loss`) to step. If the monitored metric has no values
        (e.g. validation was skipped), the plateau scheduler step is skipped.
        """
        if not self.schedulers:
            return
        for scheduler in self.schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                monitor_values = self.history.get(self.reduce_lr_on_plateau_monitor, [])
                if monitor_values:
                    scheduler.step(metrics=monitor_values[-1])
            else:
                scheduler.step()

    def setup_tracker(self) -> None:
        """
        Defines and registers keys for metric tracking with the built-in `EmberTracker`.
        By default this includes the following keys: "train_loss", "val_loss",
        plus keys for each present `torchmetrics.Metric` prefixed with "train\\_"
        and "val\\_".

        All subclasses should override this function and register custom keys here.
        """
        train_keys = self._get_default_tracker_keys("train", self.metrics)
        val_keys = self._get_default_tracker_keys("val", self.metrics)

        self.tracker.register(train_keys, key_type="train")
        self.tracker.register(val_keys, key_type="val")

    def _get_default_tracker_keys(
        self,
        key_type: Literal["train", "val"],
        metrics: list[torchmetrics.Metric] | None = None,
    ) -> list[str]:
        metric_list = self.metrics if metrics is None else metrics
        keys = [f"{key_type}_loss"]
        keys.extend([f"{key_type}_{metric._get_name()}" for metric in metric_list])
        return keys

    def _get_custom_validation_tracker_keys(self) -> list[str]:
        default_val_keys = set(self._get_default_tracker_keys("val", self.metrics))
        return [
            key
            for key in self.tracker.get_keys_with_type("val")
            if key not in default_val_keys
        ]

    def _get_evaluation_tracker_keys(
        self, metrics: list[torchmetrics.Metric]
    ) -> list[str]:
        eval_keys = self._get_default_tracker_keys("val", metrics)
        eval_keys.extend(self._get_custom_validation_tracker_keys())
        return list(dict.fromkeys(eval_keys))

    def _optimizer_has_pending_gradients(self, optimizer: Optimizer) -> bool:
        if self.accumulate_grad_batches <= 1:
            return False
        return any(
            param.grad is not None
            for param_group in optimizer.param_groups
            for param in param_group["params"]
        )

    def _flush_pending_gradients(self) -> None:
        for optimizer in self.optimizers:
            if not self._optimizer_has_pending_gradients(optimizer):
                continue
            if (
                self.gradients_clip_val is not None
                or self.gradients_max_norm is not None
            ):
                self.fabric.clip_gradients(
                    self.model,
                    optimizer,
                    clip_val=self.gradients_clip_val,
                    max_norm=self.gradients_max_norm,
                )
            optimizer.step()
            optimizer.zero_grad()

    def _setup_lr_tracking(self, log_lr: Literal["auto"] | bool = "auto") -> None:
        """Registers lr keys for learning rate tracking."""
        if (isinstance(log_lr, str) and self.schedulers) or log_lr is True:
            self.tracker.register(
                [f"lr_{i}" for i, _ in enumerate(self.optimizers)], key_type="train"
            )
            self.log_lr = True
        else:
            self.log_lr = False

    @rank_zero_only
    def _print_start_message(self, epochs: int | None, steps: int | None) -> None:
        msg = ""
        if self.resuming:
            msg += f"Continuing training on device {self.fabric.device} for "
        else:
            msg += f"Commencing training on device {self.fabric.device} for "
        if steps:
            msg += f"{steps} steps." if not epochs else f"up to {steps} steps or "
        if epochs:
            msg += f"{epochs} epochs."
        print(msg)

    @rank_zero_only
    def _print_metric_values(self, key_type: Literal["train", "val"]) -> None:
        metric_keys = self.tracker.get_keys_with_type(key_type)
        if not metric_keys:
            return
        for key in metric_keys:
            print(f"{key}: {self.history[key][-1]}")

    @final
    def fit(
        self,
        train_data: DataLoader | None = None,
        val_data: DataLoader | None = None,
        epochs: int | None = None,
        steps: int | None = None,
        callbacks: list["EmberCallback"] | None = None,
        emberdata: EmberData | None = None,
        accumulate_grad_batches: int = 1,
        augment_targets: bool = False,
        gradients_clip_val: float | None = None,
        gradients_max_norm: int | float | None = None,
        show_progress_bar: bool = True,
        show_metrics: bool = True,
        batch_limit: float | int = 0,
        force_train: bool = False,
        skip_validation: bool = False,
    ) -> None:
        """
        Fits the model using the provided training and validation data for the specified
        number of epochs or steps. Subsequent calls to `fit()` for the same `EmberModel`
        instance will resume training for additional epochs.
        If you provide values for both epochs and steps, training will continue until
        the first target is reached.
        Models with custom training loops should override `train_step` and `val_step`.

        Args:
            train_data: DataLoader for training.
            val_data: DataLoader for validation.
            epochs: Number of epochs to train the model for.
            steps: Number of steps to train the model for.
            callbacks: Optional list of EmberCallback callbacks.
            emberdata: Optional EmberData object to supply the dataloaders. Will take
                priority over train_data and val_data.
            accumulate_grad_batches: Accumulates gradients for every [n] batches
                before stepping the optimizer.
            augment_targets: Whether to apply the same data augmentation to the targets
                (i.e. labels) as well as the inputs, such as for an image-to-image
                model.
            gradients_clip_val: Clips gradients to within value (+/-). See
                `fabric.clip_gradients()`.
            gradients_max_norm: Clips gradients such that their total norm does not
                exceed this value. See `fabric.clip_gradients()`.
            show_progress_bar: Whether to show progress bars for each epoch. Default is
                True.
            show_metrics: Whether to print metric values at the end of every epoch.
                Default is True.
            batch_limit: A limit on the fraction of the dataloader to process (float),
                or the maximum batches per epoch (int). Default is 0 (i.e. all).
            force_train: Forcibly train for the specified number of epochs or steps,
                ignoring any early stopping.
            skip_validation: Only perform the training loop and skip the
                validation stage entirely.

        Raises:
            ValueError: If no training data is provided.
        """
        stage_failed = False
        try:
            # acquire dataloaders
            if isinstance(emberdata, EmberData):
                emberdata._setup_emberdata(self.fabric, stage="fit")
                train_data = emberdata.train_dataloader()
                val_data = emberdata.val_dataloader()
            if train_data is None:
                raise ValueError("No training data provided")
            train_data_len = self._require_non_empty_dataloader(
                train_data, "Training dataloader"
            )
            batch_limit_train = self._resolve_batch_limit(batch_limit, train_data_len)
            batch_limit_val = 0
            if val_data is None:
                if self.fabric.is_global_zero and not (
                    self.no_validation or skip_validation
                ):
                    print(
                        "W: No validation data provided. Skipping validation stage.\n"
                    )
                skip_validation = True
                train_data = self._setup_dataloader_with_fabric(train_data)
            else:
                run_validation = not (self.no_validation or skip_validation)
                if run_validation:
                    val_data_len = self._require_non_empty_dataloader(
                        val_data, "Validation dataloader"
                    )
                    batch_limit_val = self._resolve_batch_limit(
                        batch_limit, val_data_len
                    )
                train_data, val_data = self.fabric.setup_dataloaders(
                    train_data, val_data
                )

            self.accumulate_grad_batches = accumulate_grad_batches
            self.augment_targets = augment_targets
            self.gradients_clip_val = gradients_clip_val
            self.gradients_max_norm = gradients_max_norm

            self.resuming = self.epoch_count > 0
            self._print_start_message(epochs, steps)
            step_target: int | None = None
            epoch = self.epoch_count
            epochs_to_train_for = epochs
            if steps is not None:
                if steps <= 0:
                    raise ValueError("steps must be positive")
                step_target = self.step_count + steps
            elif epochs_to_train_for is None:
                raise ValueError("You must provide a value for epochs or steps.")
            if epochs_to_train_for is not None and epochs_to_train_for < 0:
                raise ValueError("epochs must be non-negative")

            self.model.train()
            # main loop
            self._trigger_callbacks(
                callbacks, "on_train_start", self.epoch_count, self.step_count
            )
            epoch_iter = (
                range(self.epoch_count, self.epoch_count + epochs_to_train_for)
                if epochs_to_train_for is not None
                else count(self.epoch_count)
            )
            for epoch in epoch_iter:
                if not self.should_train and not force_train:
                    if self.fabric.is_global_zero:
                        print("Training stopped (early stopping).\n")
                    self.fabric.barrier()
                    break

                self._trigger_callbacks(
                    callbacks, "on_epoch_start", epoch, self.step_count
                )
                self._trigger_callbacks(
                    callbacks, "on_train_epoch_start", epoch, self.step_count
                )
                if self.fabric.is_global_zero and show_metrics:
                    print()
                epoch_desc = f"Epoch: {epoch}"
                if step_target is None:
                    final_epoch = (
                        self.epoch_count + cast("int", epochs_to_train_for) - 1
                    )
                    epoch_desc = f"Epoch: {epoch}/{final_epoch}"
                train_progress_total = (
                    batch_limit_train if batch_limit_train > 0 else len(train_data)
                )
                # start of training loop
                with self._progress_bar(
                    total=train_progress_total,
                    desc=epoch_desc,
                    show_progress_bar=show_progress_bar,
                ) as progress_bar:
                    step_count_at_epoch_start = self.step_count
                    for batch_idx, batch in enumerate(train_data):
                        if (
                            batch_idx >= batch_limit_train and batch_limit_train > 0
                        ) or (
                            step_target is not None
                            and self.step_count >= step_target
                        ):
                            break
                        self._trigger_callbacks(
                            callbacks,
                            "on_train_batch_start",
                            epoch,
                            self.step_count,
                            batch_idx,
                        )
                        train_step_out = self.train_step(batch, batch_idx, epoch)
                        completed_step = self.step_count + 1
                        if train_step_out is not None:
                            loss, y_pred = train_step_out
                            self.tracker.update(
                                "train_loss",
                                loss.detach(),
                                batch_size=y_pred.shape[0],
                                step=completed_step,
                            )
                            for metric in self.metrics:
                                self.tracker.update(
                                    f"train_{metric._get_name()}",
                                    self.calculate_metric(metric, batch, y_pred),
                                    batch_size=y_pred.shape[0],
                                    step=completed_step,
                                )
                        if self.log_lr:
                            for i, opt in enumerate(self.optimizers):
                                self.tracker.update(
                                    f"lr_{i}",
                                    torch.tensor(
                                        opt.param_groups[0]["lr"],
                                        device=self.fabric.device,
                                    ),
                                )
                        self.step_count = completed_step
                        self._trigger_callbacks(
                            callbacks,
                            "on_train_batch_end",
                            epoch,
                            self.step_count,
                            batch_idx,
                        )
                        if progress_bar is not None:
                            progress_bar.update()
                    if (
                        step_target is not None
                        and self.step_count < step_target
                        and self.step_count == step_count_at_epoch_start
                    ):
                        raise ValueError("No training batches were processed")

                self._flush_pending_gradients()
                self.tracker.sync_epoch(key_type="train", step=self.step_count)
                self.history = self.tracker.get_history_items()
                if show_metrics:
                    self._print_metric_values(key_type="train")
                self._trigger_callbacks(
                    callbacks, "on_train_epoch_end", epoch, self.step_count
                )

                # commence validation loop
                self.model.eval()
                # val metrics are intentionally not synced when skipping validation
                if skip_validation or self.no_validation:
                    self.history = self.tracker.get_history_items()
                # otherwise run the validation loop normally
                else:
                    active_val_data = cast("DataLoader", val_data)
                    self._trigger_callbacks(
                        callbacks, "on_validation_epoch_start", epoch, self.step_count
                    )
                    val_progress_total = (
                        batch_limit_val if batch_limit_val > 0 else len(active_val_data)
                    )
                    with self._progress_bar(
                        total=val_progress_total,
                        desc="Validation",
                        show_progress_bar=show_progress_bar,
                    ) as progress_bar, torch.inference_mode():
                        for batch_idx, batch in enumerate(active_val_data):
                            if batch_idx >= batch_limit_val and batch_limit_val > 0:
                                break
                            self._trigger_callbacks(
                                callbacks,
                                "on_validation_batch_start",
                                epoch,
                                self.step_count,
                                batch_idx,
                            )
                            val_step_out = self.val_step(batch, batch_idx, epoch)
                            self._trigger_callbacks(
                                callbacks,
                                "on_validation_batch_end",
                                epoch,
                                self.step_count,
                                batch_idx,
                            )
                            if val_step_out is not None:
                                loss, y_pred = val_step_out
                                self.tracker.update(
                                    "val_loss", loss, batch_size=y_pred.shape[0]
                                )
                                for metric in self.metrics:
                                    metric_value = self.calculate_metric(
                                        metric, batch, y_pred
                                    )
                                    self.tracker.update(
                                        f"val_{metric._get_name()}",
                                        metric_value,
                                        batch_size=y_pred.shape[0],
                                    )
                            if progress_bar is not None:
                                progress_bar.update()

                    self.tracker.sync_epoch(key_type="val", step=self.step_count)
                    self.history = self.tracker.get_history_items()
                    if show_metrics:
                        self._print_metric_values(key_type="val")
                    # end of validation loop
                    self._trigger_callbacks(
                        callbacks, "on_validation_epoch_end", epoch, self.step_count
                    )

                # end of epoch
                self.model.train()
                self.scheduler_step()
                self.tracker.log("epoch", epoch, self.step_count)
                self._trigger_callbacks(
                    callbacks, "on_epoch_end", epoch, self.step_count
                )
                if step_target is not None and self.step_count >= step_target:
                    if self.fabric.is_global_zero:
                        print("Stopping training (step limit reached)")
                    break

            self._trigger_callbacks(callbacks, "on_train_end", epoch, self.step_count)
            self.epoch_count = self.tracker.get_history_length()

        except BaseException:
            stage_failed = True
            raise
        finally:
            if isinstance(emberdata, EmberData):
                if stage_failed:
                    with contextlib.suppress(Exception):
                        emberdata.cleanup(stage="fit")
                else:
                    emberdata.cleanup(stage="fit")

    @final
    def evaluate(
        self,
        eval_data: DataLoader | None = None,
        metrics: list[torchmetrics.Metric] | None = None,
        emberdata: EmberData | None = None,
        setup_dataloader_fabric: bool = True,
        show_metrics: bool = True,
        show_progress_bar: bool = False,
    ) -> dict[str, list[float]]:
        """
        Evaluates the model on the provided data and returns the computed metrics.

        Args:
            eval_data: `DataLoader` for evaluation.
            metrics: Optional list of torchmetrics Metrics to compute. Defaults to
                the metrics registered at construction time.
            emberdata: Optional `EmberData` instance. Takes priority over
                `eval_data`.
            setup_dataloader_fabric: Whether to set up the dataloader with Fabric.
                Set to `False` if already initialized with
                `fabric.setup_dataloaders()`. Default is True.
            show_metrics: Whether to print metric values after evaluation.
                Default is True.
            show_progress_bar: Whether to show a progress bar. Default is False.

        Returns:
            Dict mapping metric name to list of computed values.

        Raises:
            ValueError: If no evaluation data is provided.
        """
        stage_failed = False
        try:
            if isinstance(emberdata, EmberData):
                emberdata._setup_emberdata(self.fabric, stage="eval")
                eval_data = emberdata.eval_dataloader()
                if eval_data is None:
                    raise ValueError("EmberData instance did not provide a DataLoader")
            if eval_data is None:
                raise ValueError("No evaluation data provided")
            if setup_dataloader_fabric:
                eval_data = self._setup_dataloader_with_fabric(eval_data)
            eval_data_len = self._require_non_empty_dataloader(
                eval_data, "Evaluation dataloader"
            )
            active_metrics = (
                self.fabric.to_device(metrics) if metrics is not None else self.metrics
            )

            eval_tracker = EmberTracker(fabric=self.fabric, fabric_logging=False)
            eval_tracker.register(
                self._get_evaluation_tracker_keys(active_metrics), key_type="val"
            )

            self.model.eval()
            with self._progress_bar(
                total=eval_data_len,
                desc="Evaluation",
                show_progress_bar=show_progress_bar,
            ) as progress_bar, torch.inference_mode():
                for batch_idx, batch in enumerate(eval_data):
                    eval_step_out = self.eval_step(batch, batch_idx, eval_tracker)
                    if eval_step_out is not None:
                        loss, y_pred = eval_step_out
                        eval_tracker.update(
                            "val_loss", loss, batch_size=y_pred.shape[0]
                        )
                        for metric in active_metrics:
                            eval_tracker.update(
                                f"val_{metric._get_name()}",
                                self.calculate_metric(metric, batch, y_pred),
                                batch_size=y_pred.shape[0],
                            )
                    if progress_bar is not None:
                        progress_bar.update()

            eval_tracker.sync_epoch(key_type="val")
            metric_dict = eval_tracker.get_history_items()

            if show_metrics and self.fabric.is_global_zero:
                for k, v in metric_dict.items():
                    print(k, v)
                print()
            return metric_dict

        except BaseException:
            stage_failed = True
            raise
        finally:
            if isinstance(emberdata, EmberData):
                if stage_failed:
                    with contextlib.suppress(Exception):
                        emberdata.cleanup(stage="eval")
                else:
                    emberdata.cleanup(stage="eval")

    @final
    def predict(
        self,
        data: DataLoader | None = None,
        emberdata: EmberData | None = None,
        setup_dataloader_fabric: bool = True,
        concatenate_preds: bool = True,
        show_progress_bar: bool = True,
    ) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
        """
        Returns model predictions for the given input data.

        Args:
            data: `DataLoader` with model inputs for inference.
            emberdata: An `EmberData` instance. Takes precedence over `data`.
            setup_dataloader_fabric: Whether to setup the dataloaders with Fabric.
                Set this to False if you have already initialized your `DataLoader` with
                `fabric.setup_dataloaders`. Default is True.
            concatenate_preds: Whether to concatenate the final predictions into a
                single tensor with a length equal to number of total samples in the
                `DataLoader`'s dataset. If set to False, will instead return a list of
                tensors corresponding to the predictions for each batch.
                Default is True.
            show_progress_bar: Whether to show a progress bar. Default is True.

        Raises:
            ValueError: If no data is provided.
        """
        stage_failed = False
        try:
            if isinstance(emberdata, EmberData):
                emberdata._setup_emberdata(self.fabric, stage="predict")
                data = emberdata.predict_dataloader()
            if data is None:
                raise ValueError("No data provided.")
            if setup_dataloader_fabric:
                data = self._setup_dataloader_with_fabric(data)
            data_len = self._require_non_empty_dataloader(data, "Prediction dataloader")

            self.model.eval()
            batch_preds: list[tuple[torch.Tensor, ...]] = []
            with self._progress_bar(
                total=data_len,
                desc="Predict",
                show_progress_bar=show_progress_bar,
            ) as progress_bar, torch.inference_mode():
                for batch_idx, batch in enumerate(data):
                    pred = self.predict_step(batch, batch_idx)
                    batch_preds.append(pred if isinstance(pred, tuple) else (pred,))
                    if progress_bar is not None:
                        progress_bar.update()
            gathered_preds = cast(
                "list[tuple[torch.Tensor, ...]]", self.fabric.all_gather(batch_preds)
            )
            # unfurl extra rank dimension
            prediction_batches: list[Sequence[torch.Tensor]]
            if self.fabric.world_size > 1:
                prediction_batches = [
                    [output.transpose(1, 0).flatten(0, 1) for output in batch]
                    for batch in gathered_preds
                ]
            else:
                prediction_batches = [batch for batch in gathered_preds]

            self.fabric.barrier()
            preds: list[torch.Tensor | tuple[torch.Tensor, ...]]
            if concatenate_preds:
                preds = [
                    torch.cat(pred) for pred in zip(*prediction_batches, strict=False)
                ]
            else:
                preds = cast(
                    "list[torch.Tensor | tuple[torch.Tensor, ...]]", prediction_batches
                )
            preds = self.fabric.broadcast(preds)
            return preds

        except BaseException:
            stage_failed = True
            raise
        finally:
            if isinstance(emberdata, EmberData):
                if stage_failed:
                    with contextlib.suppress(Exception):
                        emberdata.cleanup(stage="predict")
                else:
                    emberdata.cleanup(stage="predict")

    def freeze(
        self,
        layer_names: list[str] | None = None,
        layer_indices: list[int] | None = None,
        recurse: bool = False,
    ) -> None:
        set_requires_grad(
            self.model, False, layer_names, layer_indices, recurse=recurse
        )

    def unfreeze(
        self,
        layer_names: list[str] | None = None,
        layer_indices: list[int] | None = None,
        recurse: bool = False,
    ) -> None:
        set_requires_grad(self.model, True, layer_names, layer_indices, recurse=recurse)

    def save_checkpoint(self, path: Path | str) -> None:
        """
        Saves a checkpoint to file using `fabric.save()`.
        Stores the model state, optimizer state(s), scheduler state(s), tracker
        history, and step count.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "model": self.model,
            "tracker": self.tracker,
            "step_count": self.step_count,
        }
        for i, opt in enumerate(self.optimizers):
            state[f"optimizer_{i}"] = opt
        if self.schedulers:
            for i, scheduler in enumerate(self.schedulers):
                state[f"scheduler_{i}"] = scheduler
        self.fabric.save(path, state)

    def load_checkpoint(self, path: Path | str) -> None:
        """
        Loads a checkpoint with `fabric.load()`.
        Restores the model state, optimizer state(s), scheduler state(s), tracker
        history, and step count.
        """
        checkpoint = self.fabric.load(path)
        state: dict[str, Any] = {
            "model": self.model,
            "tracker": self.tracker,
            "step_count": self.step_count,
        }
        for i, opt in enumerate(self.optimizers):
            state[f"optimizer_{i}"] = opt
        if self.schedulers:
            for i, scheduler in enumerate(self.schedulers):
                key = f"scheduler_{i}"
                if key in checkpoint:
                    state[key] = scheduler
        self.fabric.load(path, state)
        self.step_count = state["step_count"]
        self.history = self.tracker.get_history_items()
        self.epoch_count = self.tracker.get_history_length()
        self.resuming = True
        self.should_train = True
        if self.fabric.is_global_zero:
            print(f"Loaded checkpoint {path}")

    def load_best_checkpoint(self) -> None:
        """
        Loads the best checkpoint (as saved via a ModelCheckpoint callback).

        Shorthand for `load_checkpoint(embermodel.best_ckpt_path)`
        """
        if self.best_ckpt_path is None:
            raise RuntimeError("No best checkpoint found.")
        self.load_checkpoint(self.best_ckpt_path)

    def load_last_checkpoint(self) -> None:
        """
        Loads the last checkpoint (as saved via a ModelCheckpoint callback).

        Shorthand for `load_checkpoint(embermodel.last_ckpt_path)`
        """
        if self.last_ckpt_path is None:
            raise RuntimeError("No last checkpoint found.")
        self.load_checkpoint(self.last_ckpt_path)

    @rank_zero_only
    def save_weights(self, path: Path | str) -> None:
        """Saves the weights of `self.model` with `torch.save()`"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f=path)

    def load_weights(self, path: Path | str) -> None:
        """Loads saved weights for `self.model` with `fabric.load_raw()`"""
        self.fabric.load_raw(path, self.model)

    def summary(
        self,
        input_size: tuple[int, ...]
        | torch.Size
        | list[tuple[int, ...] | torch.Size]
        | None = None,
        input_data: torch.Tensor | list[torch.Tensor] | None = None,
        depth: int = 3,
        **kwargs,
    ) -> Any:
        """Display a model summary with torchinfo"""
        return model_summary(
            model=self.model,
            input_size=input_size,
            input_data=input_data,
            depth=depth,
            **kwargs,
        )

    def _trigger_callbacks(
        self,
        callbacks: list["EmberCallback"] | None,
        callback_type: str,
        epoch: int | None = None,
        step: int | None = None,
        batch_idx: int | None = None,
    ) -> None:
        if callbacks is None:
            return
        for callback in callbacks:
            callback._trigger(
                self, callback_type, epoch=epoch, step=step, batch_idx=batch_idx
            )

    def __repr__(self) -> str:
        """Print model structure"""
        return str(self.model)
