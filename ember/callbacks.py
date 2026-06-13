from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import optuna

    from ember.models import EmberModel

from pathlib import Path

import numpy as np
from lightning.fabric.utilities.rank_zero import rank_zero_only


def _validate_mode(mode: str) -> None:
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")


class EmberCallback:
    """Base class for custom callbacks to be used with EmberModel"""

    def __init__(self) -> None:
        """Base class for custom callbacks to be used with EmberModel"""
        pass

    def on_train_start(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the start of `EmberModel.fit()`.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_train_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the end of `EmberModel.fit()`.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_epoch_start(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the start of each epoch.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_epoch_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the end of each epoch.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_train_epoch_start(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the start of the training loop in each epoch.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_train_epoch_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the end of the training loop in each epoch.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_train_batch_start(
        self,
        embermodel: "EmberModel",
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        """
        Called at the beginning of each `train_step`.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
            batch_idx: The current batch index of the training `DataLoader`.
        """
        pass

    def on_train_batch_end(
        self,
        embermodel: "EmberModel",
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        """
        Called at the completion of each `train_step`.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
            batch_idx: The current batch index of the training `DataLoader`.
        """
        pass

    def on_validation_epoch_start(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the start of the validation loop in each epoch.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_validation_epoch_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Called at the end of the validation loop in each epoch.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
        """
        pass

    def on_validation_batch_start(
        self,
        embermodel: "EmberModel",
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        """
        Called at the beginning of each `val_step`.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
            batch_idx: The current batch index of the validation `DataLoader`.
        """
        pass

    def on_validation_batch_end(
        self,
        embermodel: "EmberModel",
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        """
        Called at the completion of each `val_step`.

        Args:
            embermodel: The `EmberModel` instance.
            epoch: The current epoch.
            step: The current step count.
            batch_idx: The current batch index of the validation `DataLoader`.
        """
        pass

    def _trigger(
        self,
        embermodel: "EmberModel",
        callback_type: str,
        epoch: int | None = 0,
        step: int | None = 0,
        batch_idx: int | None = 0,
    ) -> None:
        callback_function = getattr(self, callback_type, None)
        if callable(callback_function):
            args = (
                (embermodel, epoch, step, batch_idx)
                if "_batch" in callback_type
                else (embermodel, epoch, step)
            )
            callback_function(*args)


class ModelCheckpoint(EmberCallback):
    def __init__(
        self,
        monitor: str,
        mode: Literal["min", "max"] = "min",
        save_dir: Path | str = "./ckpts",
        save_prefix: str = "model",
        save_best: bool = True,
        save_top_k: int = -1,
        save_last: bool = True,
        save_every_n_epochs: int = 0,
        save_every_n_steps: int = 0,
        warmup_epochs: int = 0,
        verbose: bool = True,
    ) -> None:
        """
        Callback for saving model checkpoints.

        By default this will save one "best" checkpoint based on the specified mode and
        metric to monitor, and one "last" checkpoint at the end of training.

        Args:
            monitor: The metric to monitor.
            mode: Whether to treat decreasing (min) or increasing (max) metric values
                as better.
            save_dir: The directory in which to store checkpoints.
            save_prefix: String to prepend to each checkpoint. Default is "model".
            save_best: If True, saves a single best checkpoint. Equivalent to
                save_top_k = 1. Default is True.
            save_top_k: Saves and retains the top K best checkpoints.
                Set to 0 to disable. If -1, its value is determined by 'save_best'.
                Default is -1. Checkpoints will be named
                `{self.save_prefix}-epoch_{epoch:05d}.ckpt`
            save_last: Whether to save a checkpoint at the end of the train loop.
                Useful for resuming. Default is True. The checkpoint will be named
                `{self.save_prefix}-epoch_{epoch:05d}_last.ckpt`
            save_every_n_epochs: If > 0, saves a checkpoint every N epochs.
                This is independent from `save_top_k`. Checkpoints will be named
                `{self.save_prefix}-epoch_{epoch:05d}_periodic.ckpt`.
            save_every_n_steps: If > 0, saves a checkpoint every N steps.
                This is independent from `save_top_k`. Checkpoints will be named
                `{self.save_prefix}-step_{step:07d}.ckpt`.
            warmup_epochs: Number of training epochs to wait before saving checkpoints.
            verbose: Include print statements. Default is True.
        """
        _validate_mode(mode)
        if save_top_k < -1:
            raise ValueError("save_top_k must be >= -1")
        if save_every_n_epochs < 0:
            raise ValueError("save_every_n_epochs must be >= 0")
        if save_every_n_steps < 0:
            raise ValueError("save_every_n_steps must be >= 0")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")

        self.monitor = monitor
        self.mode = mode
        self.save_dir = Path(save_dir)
        self.save_prefix = save_prefix
        if save_top_k > 0:
            self.save_top_k = save_top_k
        elif save_top_k == 0:
            self.save_top_k = 0
        else:  # -1: derive from save_best
            self.save_top_k = 1 if save_best else 0
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        self.best_checkpoints: list[tuple[float, int]] = []  # (metric_value, epoch)
        self.best_ckpt_path: Path | None = None
        self.last_ckpt_path: Path | None = None

    @rank_zero_only
    def _delete_checkpoint(self, path: Path, verbose: bool = True) -> None:
        if verbose:
            print(f"Deleting checkpoint {path}")
        path.unlink()

    def _save_best_checkpoint(
        self,
        latest_metric: float,
        epoch: int,
        ckpt_path: Path,
        embermodel: "EmberModel",
    ) -> None:
        self.best_checkpoints.append((latest_metric, epoch))
        embermodel.save_checkpoint(path=ckpt_path)
        if self.verbose and embermodel.fabric.is_global_zero:
            print(f"Saving checkpoint to {ckpt_path}")

    def on_train_start(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        # reset best value trackers
        self.best_checkpoints = []
        self.best_ckpt_path = None
        self.last_ckpt_path = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.monitor not in embermodel.history:
            raise ValueError(f"{self.monitor} is not a registered metric.")
        history_values = embermodel.history[self.monitor]
        if not history_values:  # add dummy initial values for first training run
            self.best_checkpoints.append(
                (np.inf if self.mode == "min" else -np.inf, -1)
            )
        else:  # i.e. resuming
            best = min(history_values) if self.mode == "min" else max(history_values)
            self.best_checkpoints = [(best, -1)]

    def on_epoch_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        # skip epoch if warming up
        if epoch < self.warmup_epochs:
            return

        # Save checkpoints at the required frequency (independent of save_top_k)
        if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
            periodic_path = Path(
                self.save_dir,
                f"{self.save_prefix}-epoch_{epoch:05d}_periodic.ckpt",
            )
            embermodel.save_checkpoint(path=periodic_path)

        if not embermodel.history.get(self.monitor):
            return  # monitored metric not available (e.g. validation was skipped)

        latest_metric = embermodel.history[self.monitor][-1]
        ckpt_path = Path(self.save_dir, f"{self.save_prefix}-epoch_{epoch:05d}.ckpt")

        if self.save_top_k > 0:
            best_values = [v for v, _ in self.best_checkpoints]
            for bv in best_values:
                if (self.mode == "min" and latest_metric < bv) or (
                    self.mode == "max" and latest_metric > bv
                ):
                    self._save_best_checkpoint(
                        latest_metric, epoch, ckpt_path, embermodel
                    )
                    break

            # sort so the worst checkpoint is first (desc for min, asc for max)
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "min"))
            while len(self.best_checkpoints) > self.save_top_k:
                _, epoch_for_deletion = self.best_checkpoints.pop(0)
                if epoch_for_deletion >= 0:
                    self._delete_checkpoint(
                        path=Path(
                            self.save_dir,
                            f"{self.save_prefix}-epoch_{epoch_for_deletion:05d}.ckpt",
                        ),
                        verbose=self.verbose,
                    )
                embermodel.fabric.barrier()
            if len(self.best_checkpoints) > self.save_top_k:
                raise RuntimeError(
                    f"Checkpoint tracking invariant violated: "
                    f"best_checkpoints={len(self.best_checkpoints)}, "
                    f"save_top_k={self.save_top_k}"
                )

            # best_checkpoints is sorted worst-first; the last entry is the true best.
            # Update best_ckpt_path here (not in _save_best_checkpoint) so it always
            # reflects the overall best even when save_top_k > 1.
            _, best_epoch = self.best_checkpoints[-1]
            if best_epoch >= 0:
                self.best_ckpt_path = Path(
                    self.save_dir,
                    f"{self.save_prefix}-epoch_{best_epoch:05d}.ckpt",
                )
                embermodel.best_ckpt_path = self.best_ckpt_path

    def on_train_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        if self.save_last:
            ckpt_path = Path(
                self.save_dir, f"{self.save_prefix}-epoch_{epoch:05d}_last.ckpt"
            )
            embermodel.save_checkpoint(path=ckpt_path)
            self.last_ckpt_path = ckpt_path
            embermodel.last_ckpt_path = ckpt_path

    def on_train_batch_end(
        self,
        embermodel: "EmberModel",
        epoch: int = 0,
        step: int = 0,
        batch_idx: int = 0,
    ) -> None:
        if (
            self.save_every_n_steps > 0
            and step > 0
            and step % self.save_every_n_steps == 0
        ):
            ckpt_path = Path(self.save_dir, f"{self.save_prefix}-step_{step:07d}.ckpt")
            embermodel.save_checkpoint(path=ckpt_path)


class EarlyStopping(EmberCallback):
    def __init__(
        self,
        monitor: str,
        patience: int = 5,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
        threshold: float | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Callback to stop training once the monitored metric has stopped improving.

        Args:
            monitor: The metric name to monitor: "val_loss",
                "val_{torchmetric_name}", etc.
            patience: The number of epochs to count improvements over. If the metric
                has not improved after this many epochs, training is stopped.
            mode: One of "min" or "max"; whether to treat smaller or larger values
                as an improvement.
            min_delta: The minimum change to count as a valid improvement.
            threshold: An optional metric threshold. In "min" mode, values below the
                threshold are always counted as non-improved. Improvements are only
                considered for values above the threshold. In "max" mode, the reverse
                is true. This acts as a floor/ceiling for metric improvement and is
                useful for noisy loss curves.
            verbose: Whether to include print statements.
        """
        super().__init__()
        _validate_mode(mode)
        if patience < 0:
            raise ValueError("patience must be >= 0")

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        if threshold is None:
            self.threshold = -np.inf if self.mode == "min" else np.inf
        else:
            self.threshold = threshold
        self.verbose = verbose

        self.best_value: float = 0.0
        self.best_value_epoch: int = 0

    def on_train_start(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        if self.monitor not in embermodel.history:
            raise ValueError(f"{self.monitor} is not a registered metric.")
        if not embermodel.history[self.monitor]:
            # initialize best value when commencing training
            self.best_value = np.inf if self.mode == "min" else -np.inf
        else:
            # recover best value when resuming training
            prev_values = embermodel.history[self.monitor][-self.patience :]
            prev_epoch = len(embermodel.history[self.monitor])
            self.best_value = (
                min(prev_values) if self.mode == "min" else max(prev_values)
            )
            self.best_value_epoch = prev_epoch - (
                len(prev_values) - prev_values.index(self.best_value)
            )

    def on_epoch_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        monitor_history = embermodel.history.get(self.monitor, [])
        if not monitor_history:
            return  # monitored metric not available (e.g. validation was skipped)
        latest_metric = monitor_history[-1]
        has_improved = False

        if np.isnan(latest_metric):
            if embermodel.fabric.is_global_zero:
                print(
                    f"nan encountered; stopping training beyond epoch {epoch}.\n"
                    f"Best metric was: {self.best_value} "
                    f"at epoch {self.best_value_epoch}"
                )
            embermodel.should_train = False
            return

        if (
            self.mode == "min"
            and latest_metric > self.threshold
            and latest_metric < self.best_value - self.min_delta
        ) or (
            self.mode == "max"
            and latest_metric < self.threshold
            and latest_metric > self.best_value + self.min_delta
        ):
            has_improved = True
            self.best_value = latest_metric
            self.best_value_epoch = epoch

        if not has_improved and (epoch - self.best_value_epoch) >= self.patience:
            embermodel.should_train = False
            if self.verbose and embermodel.fabric.is_global_zero:
                print(
                    f"Metric has stalled; stopping training beyond epoch {epoch}.\n"
                    f"Best metric was: {self.best_value} "
                    f"at epoch {self.best_value_epoch}"
                )
            return
        if self.verbose and embermodel.fabric.is_global_zero:
            if has_improved:
                print("Metric has improved!")
            print(f"Best metric: {self.best_value} at epoch {self.best_value_epoch}")


class OptunaTrialPruner(EmberCallback):
    def __init__(self, trial: "optuna.Trial | Any", monitor: str) -> None:
        """
        Callback to report intermediate values for an Optuna trial (and decide whether
        to prune) at the end of every epoch.

        Args:
            trial: The Optuna trial object.
            monitor: The metric value to report for pruning, e.g. val_loss.

        Raises:
            TrialPruned: When a run should be pruned.
        """
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_train_start(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        if self.monitor not in embermodel.history:
            raise ValueError(f"{self.monitor} is not a registered metric.")

    def on_epoch_end(
        self, embermodel: "EmberModel", epoch: int = 0, step: int = 0
    ) -> None:
        monitor_history = embermodel.history.get(self.monitor, [])
        if not monitor_history:
            return  # monitored metric not available (e.g. validation was skipped)
        self.trial.report(monitor_history[-1], step=epoch)
        if self.trial.should_prune():
            from optuna import TrialPruned

            raise TrialPruned()
