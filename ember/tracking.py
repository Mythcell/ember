from collections.abc import Sequence
from typing import Any, Literal, cast

import torch
from lightning import Fabric
from torchmetrics import Metric


class EmberTracker:
    """Track weighted scalars and named TorchMetrics through Lightning Fabric."""

    def __init__(
        self,
        fabric: Fabric,
        fabric_logging: bool = True,
    ) -> None:
        """Initialize a tracker.

        Args:
            fabric: Fabric instance used for reduction, broadcast, and logging.
            fabric_logging: Whether tracker updates are sent to Fabric loggers.
        """
        self.fabric = fabric
        self.fabric_logging = fabric_logging
        self.history: dict[str, list[torch.Tensor]] = {}
        self._key_types: dict[str, str] = {}
        self._last_epochs: dict[str, int | None] = {}
        self._torchmetrics: dict[str, Metric] = {}
        self._increments: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}

    def register(
        self,
        keys: list[str] | str,
        key_type: Literal["train", "val"] | str,
    ) -> None:
        """Register one or more history keys.

        Args:
            keys: Key name or names to register. Registering an existing key resets it.
            key_type: Group synchronized together by `sync_epoch()`. `EmberModel`
                automatically synchronizes `"train"` and `"val"` groups.
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key in self._torchmetrics:
                raise ValueError(f"{key} is already registered as a TorchMetric")
        for key in keys:
            self.history[key] = []
            self._last_epochs.pop(key, None)
            self._key_types[key] = key_type
            self._increments[key] = torch.zeros(1, device=self.fabric.device)
            self._counts[key] = torch.zeros(1, device=self.fabric.device)

    def register_torchmetric(self, key: str, metric: Metric, key_type: str) -> None:
        """Own a metric instance under an explicit history key and stage group.

        The instance is reset and moved to the Fabric device. Each key must own a
        separate instance; do not share it with another tracker or update it directly.
        """
        if key in self.history:
            raise ValueError(f"{key} is already registered")
        if not isinstance(metric, Metric):
            raise TypeError(f"{key} requires a TorchMetrics Metric instance")
        if any(metric is existing for existing in self._torchmetrics.values()):
            raise ValueError("Each TorchMetric key requires a separate instance")
        metric = metric.to(self.fabric.device)
        metric.reset()
        self._torchmetrics[key] = metric
        self._key_types[key] = key_type
        self.history[key] = []

    def update_torchmetric(self, key: str, *args: Any, **kwargs: Any) -> None:
        """Forward inputs to a registered metric, discarding its state on failure."""
        if key not in self._torchmetrics:
            if key in self.history:
                raise TypeError(f"{key} is a scalar key; use update()")
            raise KeyError(key)
        metric = self._torchmetrics[key]
        try:
            metric.update(*args, **kwargs)
        except BaseException:
            metric.reset()
            raise

    def update(
        self,
        key: str,
        value: torch.Tensor,
        batch_size: int = 1,
        step: int | None = None,
    ) -> None:
        """Accumulate a weighted value for one key.

        Args:
            key: Registered key to update. Unregistered keys are ignored.
            value: Scalar tensor to accumulate.
            batch_size: Weight applied to `value` when calculating the epoch mean.
            step: Optional logger step. When provided, logs the raw value as
                `{key}_step` on the global-zero process.
        """
        if key not in self.history:
            return
        if key in self._torchmetrics:
            raise TypeError(f"{key} is a TorchMetric key; use update_torchmetric()")
        value = value.detach()
        self._increments[key] += value * batch_size
        self._counts[key] += batch_size
        if step is not None and self.fabric.is_global_zero and self.fabric_logging:
            self.fabric.log(f"{key}_step", value, step=step)

    def sync_epoch(
        self,
        key_type: Literal["train", "val"] | str | None = None,
        keys: list[str] | str | None = None,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """Finalize selected scalars and TorchMetrics across all processes.

        TorchMetrics compute once when updated on every rank, and record zero without
        computing when unused on every rank. Registration and synchronization order
        must agree across ranks. Selected pending state is reset even when
        finalization fails.

        Args:
            key_type: Key group to synchronize. When provided, it determines the keys.
            keys: Explicit key or keys to synchronize when `key_type` is omitted.
            epoch: Current epoch for freshness checks and the default logger step.
            step: Explicit logger step, which takes precedence over `epoch`.

        Raises:
            ValueError: If neither a group nor keys are supplied, or a metric result
                is not a single tensor value. Empty groups are allowed.
            RuntimeError: If a metric was updated on only some ranks.
        """
        log_step = step if step is not None else epoch
        if key_type is not None:
            keys = list(filter(lambda x: self._key_types[x] == key_type, self.history))
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise ValueError("No keys have been provided")
        if not keys:
            return
        for key in keys:
            if key not in self.history:
                raise KeyError(key)

        try:
            metric_keys = [key for key in keys if key in self._torchmetrics]
            active: set[str] = set()
            if metric_keys:
                activity = torch.tensor(
                    [self._torchmetrics[key].update_called for key in metric_keys],
                    device=self.fabric.device,
                    dtype=torch.long,
                )
                activity = cast(
                    "torch.Tensor", self.fabric.all_reduce(activity, reduce_op="sum")
                )
                for key, count in zip(metric_keys, activity.tolist(), strict=True):
                    if count not in (0, self.fabric.world_size):
                        raise RuntimeError(
                            f"{key} must be updated on all ranks or none"
                        )
                    if count:
                        active.add(key)

            values: dict[str, torch.Tensor] = {}
            for key in keys:
                if key in self._torchmetrics:
                    if key not in active:
                        values[key] = torch.zeros((), device=self.fabric.device)
                        continue
                    value = self._torchmetrics[key].compute()
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise ValueError(f"{key} must compute a single tensor value")
                    values[key] = value.detach().reshape(()).to(self.fabric.device)
                else:
                    total = cast(
                        "torch.Tensor",
                        self.fabric.all_reduce(self._increments[key], reduce_op="sum"),
                    )
                    count = cast(
                        "torch.Tensor",
                        self.fabric.all_reduce(self._counts[key], reduce_op="sum"),
                    )
                    values[key] = total / count.clamp_min(1)
            if self.fabric.is_global_zero:
                for key, value in values.items():
                    self.history[key].append(value)
                    if log_step is not None and self.fabric_logging:
                        self.fabric.log(key, value, step=log_step)
            self.history = self.fabric.broadcast(self.history)
            self._last_epochs.update(dict.fromkeys(keys, epoch))
        finally:
            self._reset_pending(keys)

    def _reset_pending(self, keys: Sequence[str] | None = None) -> None:
        for key in self.history if keys is None else keys:
            if key in self._torchmetrics:
                self._torchmetrics[key].reset()
            else:
                self._increments[key].zero_()
                self._counts[key].zero_()

    def _clone_for_evaluation(self, fabric: Fabric) -> "EmberTracker":
        tracker = EmberTracker(fabric, fabric_logging=False)
        for key in self.get_keys_with_type("val"):
            if key in self._torchmetrics:
                tracker.register_torchmetric(
                    key, self._torchmetrics[key].clone(), "val"
                )
            else:
                tracker.register(key, "val")
        return tracker

    def _reset_epoch(self) -> None:
        self._last_epochs.clear()

    def _has_epoch_value(self, key: str, epoch: int) -> bool:
        return self._last_epochs.get(key) == epoch

    def log(self, key: str, value: Any, step: int) -> None:
        """Log a value directly through Fabric without adding it to history.

        Args:
            key: Logger key.
            value: Value to log.
            step: Logger step.
        """
        self.fabric.log(key, value, step)

    def state_dict(self) -> dict:
        """Return completed reporting and registrations, without pending state."""
        return {
            "history": {key: list(values) for key, values in self.history.items()},
            "_key_types": dict(self._key_types),
            "_metric_keys": list(self._torchmetrics),
        }

    def _normalize_state_dict(self, state_dict: dict) -> dict:
        """Infer legacy metric kinds from current registrations."""
        required = {"history", "_key_types"}
        if not required <= state_dict.keys():
            raise ValueError("Checkpoint tracker state requires history and key groups")
        state_dict = dict(state_dict)
        if "_metric_keys" not in state_dict:
            state_dict["_metric_keys"] = [
                key for key in state_dict["history"] if key in self._torchmetrics
            ]
        metric_keys = set(state_dict["_metric_keys"])
        for key in self.history.keys() & state_dict["history"].keys():
            if (key in metric_keys) != (key in self._torchmetrics):
                raise ValueError(f"Checkpoint value kind conflicts for {key}")
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scalars and configured metrics, skipping unconfigured metrics.

        Current registrations absent from the checkpoint retain empty histories.
        Metric instances and their stage groups come from the receiving tracker.
        Pending state is reset for all registrations.
        """
        state_dict = self._normalize_state_dict(state_dict)
        metric_keys = set(state_dict["_metric_keys"])
        for key in self.history:
            self.history[key] = []
        for key, values in state_dict["history"].items():
            if key in metric_keys:
                if key not in self._torchmetrics:
                    continue
            else:
                self.register(key, state_dict["_key_types"][key])
            self.history[key] = list(values)
        for metric in self._torchmetrics.values():
            metric.to(self.fabric.device)
        self._reset_pending()
        self._reset_epoch()

    def get_keys_with_type(
        self, key_type: Literal["train", "val", "other"] | str
    ) -> list[str]:
        """Return keys in one group.

        Args:
            key_type: Group to select. `"other"` selects keys outside the `"train"`
                and `"val"` groups.
        """
        if key_type == "other":
            return list(
                filter(
                    lambda x: self._key_types[x] not in {"train", "val"}, self.history
                )
            )
        return list(filter(lambda x: self._key_types[x] == key_type, self.history))

    def get_history_items(self) -> dict[str, list[float]]:
        """Return finalized history values as Python floats."""
        return {
            key: [value.item() for value in self.history[key]] for key in self.history
        }

    def get_history_length(self, key: str | None = None) -> int:
        """Return the history length for one key or the longest registered key.

        Args:
            key: Optional key to inspect. Missing keys have length zero.
        """
        if key is not None:
            return len(self.history.get(key, []))
        values = list(self.history.values())
        if values:
            return max(len(v) for v in values)
        return 0
