from typing import Any, Literal, cast

import torch
from lightning import Fabric


class EmberTracker:
    """Track weighted epoch values and log them through Lightning Fabric."""

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
        self.history: dict[str, list] = {}
        self._key_types: dict[str, str] = {}
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
            self.history[key] = []
            self._key_types[key] = key_type
            self._increments[key] = torch.zeros(1, device=self.fabric.device)
            self._counts[key] = torch.zeros(1, device=self.fabric.device)

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
        """Finalize selected weighted means across all processes.

        This blocking operation appends each mean to history, optionally logs it,
        broadcasts the resulting history, and resets the selected accumulators.

        Args:
            key_type: Key group to synchronize. When provided, it determines the keys.
            keys: Explicit key or keys to synchronize when `key_type` is omitted.
            epoch: Epoch used as the logger step.
            step: Explicit logger step, which takes precedence over `epoch`.

        Raises:
            ValueError: If no matching or explicit keys are provided.
        """
        log_step = step if step is not None else epoch
        if key_type:
            keys = list(filter(lambda x: self._key_types[x] == key_type, self.history))
            if not keys:
                raise ValueError(f"No keys found for key type {key_type}")
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise ValueError("No keys have been provided")

        # sync values across all processes, then aggregate on the root process
        for key in keys:
            self._increments[key] = cast(
                "torch.Tensor", self.fabric.all_reduce(self._increments[key])
            )
            self._counts[key] = cast(
                "torch.Tensor", self.fabric.all_reduce(self._counts[key])
            )
        if self.fabric.is_global_zero:
            for key in keys:
                value = self._increments[key] / torch.max(
                    self._counts[key], torch.ones(1, device=self.fabric.device)
                )
                self.history[key].append(value)
                if log_step is not None and self.fabric_logging:
                    self.fabric.log(key, value, step=log_step)
        self.fabric.barrier()

        for key in keys:  # reset counters
            self._counts[key].zero_()
            self._increments[key].zero_()
        self.history = self.fabric.broadcast(self.history)

    def log(self, key: str, value: Any, step: int) -> None:
        """Log a value directly through Fabric without adding it to history.

        Args:
            key: Logger key.
            value: Value to log.
            step: Logger step.
        """
        self.fabric.log(key, value, step)

    def state_dict(self) -> dict:
        """Return tracker history, key groups, and pending accumulators."""
        return {
            "history": dict(self.history),
            "_key_types": dict(self._key_types),
            "_increments": dict(self._increments),
            "_counts": dict(self._counts),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore tracker state and move pending accumulators to the Fabric device."""
        self.history = dict(state_dict["history"])
        self._key_types = dict(state_dict["_key_types"])
        # Fabric checkpoint loading places tensors on the CPU by default.
        self._increments = self.fabric.to_device(state_dict["_increments"])
        self._counts = self.fabric.to_device(state_dict["_counts"])

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
