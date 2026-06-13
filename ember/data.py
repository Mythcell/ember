from typing import TYPE_CHECKING, Literal, final

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from lightning.fabric import Fabric


class EmberData:
    """A lightweight data management module inspired by `LightningDataModule`"""

    def __init__(self) -> None:
        self._initialize_trackers()

    def pre_setup(self) -> None:
        """
        Called once, on the rank-zero process ONLY, before `setup()`.

        This is intended for one-off operations such as downloading or
        preprocessing data. Any instance variables assigned here only affect the rank
        zero process. Any logic that must be shared across ranks ought to instead
        run in `setup()`.
        """
        pass

    def setup(
        self,
        stage: Literal["fit", "eval", "predict"] = "fit",
    ) -> None:
        """
        Sets up the data for training, evaluation and prediction. This function is run
        once for ALL processes at the start of model.{fit/evaluate/predict}, and ignored
        in subsequent calls. See `reset()` for resetting the EmberData
        state if you wish to run the setup function from scratch each time.

        Args:
            stage: One of "fit", "eval" or "predict".
        """
        pass

    def train_dataloader(self) -> DataLoader | None:
        """
        Returns a `DataLoader` for the training loop in `EmberModel.fit()`.

        This is typically called after `setup(stage="fit")`.
        """
        return None

    def val_dataloader(self) -> DataLoader | None:
        """
        Returns a `DataLoader` for the validation loop in `EmberModel.fit()`,
        or `None` to skip validation.

        This is typically called after `setup(stage="fit")`.
        """
        return None

    def eval_dataloader(self) -> DataLoader | None:
        """
        Returns a `DataLoader` for the evaluation loop in `EmberModel.evaluate()`.

        This is typically called after `setup(stage="eval")`.
        """
        return None

    def predict_dataloader(self) -> DataLoader | None:
        """
        Returns a `DataLoader` for the prediction loop in `EmberModel.predict()`.

        This is typically called after `setup(stage="predict")`.
        """
        return None

    def cleanup(
        self,
        stage: Literal["fit", "eval", "predict"] = "fit",
    ) -> None:
        """
        A teardown function automatically called at the end of `EmberModel.fit()`,
        `.evaluate()`, and `.predict()`. Use this to release resources that were
        acquired during `setup()` or the dataloader methods, for example closing file
        handles, clearing large in-memory datasets, or freeing GPU buffers.

        Args:
            stage: One of `"fit"`, `"eval"`, or `"predict"`.
        """
        pass

    def reset(self) -> None:
        """
        By default, `pre-setup` is only run once, and `setup` is run once for each stage
        (`fit`, `eval` and `predict`). This function forcibly resets the internal flags
        to allow `pre-setup` and `setup` to run again.

        If you are overriding this, you may also wish to override `cleanup()`.
        """
        self._initialize_trackers()

    @final
    def _initialize_trackers(self) -> None:
        self.pre_setup_complete: bool = False
        self.first_setup: dict[str, bool] = {
            "fit": True,
            "eval": True,
            "predict": True,
        }

    @final
    def _setup_emberdata(
        self, fabric: "Fabric", stage: Literal["fit", "eval", "predict"]
    ) -> None:
        if not self.pre_setup_complete:
            if fabric.is_global_zero:
                self.pre_setup()
            fabric.barrier()
            self.pre_setup_complete = True
        if self.first_setup[stage]:
            self.setup(stage=stage)
            self.first_setup[stage] = False
