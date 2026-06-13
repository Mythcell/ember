from typing import Any, Literal, cast

import torch
from lightning import Fabric


class EmberTracker:
    """Standalone utility for tracking metric history and logging with Fabric."""

    def __init__(
        self,
        fabric: Fabric,
        fabric_logging: bool = True,
    ) -> None:
        """
        Standalone utility for tracking metric history and logging with Fabric.

        This class maintains an internal history dictionary to track custom metrics
        (per epoch) and also implements per-step logging with Fabric loggers.
        Metrics are automatically aggregated and broadcast between devices.

        Args:
            fabric: The Fabric object.
            fabric_logging: Whether to log keys using the Fabric logger. We recommend
                enabling this for training but disabling it for custom runs/evaluations.
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
        """
        Register a new key (or list of keys) for metric tracking and logging.

        Args:
            keys: List of key name strings, or a single key name string.
            key_type: A key type to register each key with. Keys with type "train" or
                "val" will be automatically synchronized at the end of each training
                and validation epoch respectively. If you use a custom key type, you
                must manually call "sync_epoch" via callbacks.
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
        """
        Update the given key with the corresponding value multiplied by the batch size.
        Provide an optional step count to also log the value (will automatically log to
        f"{key}_step").

        Args:
            key: The key to update. If the key has not been registered, this is a no-op.
            value: The value to increment with.
            batch_size: Optional multiplier to apply to the value. For proper weighted
                averaging, it is recommended to pass the batch size that was used when
                obtaining the value.
            step: If provided, will also log the value (root process only).
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
        """
        Synchronizes the mean metric value across all processes for all keys of the
        specified key type or the given key(s) to mark the end of an epoch. Values are
        logged to the Fabric logger if either a step or epoch value is given.
        Note: This is a blocking function and should be called sparingly!

        EmberModel automatically calls `sync_epoch` for key type "train" at the end of
        each training loop, and "val" at the end of each validation loop. If you provide
        values for both step and epoch, step takes precedence.

        Args:
            key_type: The type of keys to synchronize.
            keys: List of keys, or a single key, to synchronize.
            epoch: Epoch to log with `fabric.log()`.
            step: Step to log with `fabric.log()`.

        Raises:
            ValueError: If no keys are found / provided.
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
        """
        Wrapper for fabric.log(). Note you must provide a step-count.

        Args:
            key: Metric to log
            value: Value to log
            step: Current step count
        """
        self.fabric.log(key, value, step)

    def state_dict(self) -> dict:
        return {
            "history": dict(self.history),
            "_key_types": dict(self._key_types),
            "_increments": dict(self._increments),
            "_counts": dict(self._counts),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.history = dict(state_dict["history"])
        self._key_types = dict(state_dict["_key_types"])
        # ensure tensors are on the correct device (fabric.load defaults to cpu!)
        self._increments = self.fabric.to_device(state_dict["_increments"])
        self._counts = self.fabric.to_device(state_dict["_counts"])

    def get_keys_with_type(
        self, key_type: Literal["train", "val", "other"] | str
    ) -> list[str]:
        """
        Returns a list of keys with the given key type.

        Args:
            key_type: The key type. The special type "other" will return all keys with
                a key type other than "train" and/or "val".
        """
        if key_type == "other":
            return list(
                filter(
                    lambda x: self._key_types[x] not in {"train", "val"}, self.history
                )
            )
        return list(filter(lambda x: self._key_types[x] == key_type, self.history))

    def get_history_items(self) -> dict[str, list[float]]:
        """Returns the history dictionary as lists of floats rather than Tensors."""
        return {
            key: list(map(lambda x: x.item(), self.history[key]))
            for key in self.history
        }

    def get_history_length(self, key: str | None = None) -> int:
        """
        Returns the length of the history metrics lists.
        In other words, the number of epochs tracked.

        Args:
            key: Optional specific key to check. If not provided, returns the
                maximum length across all keys.
        """
        if key is not None:
            return len(self.history.get(key, []))
        values = list(self.history.values())
        if values:
            return max(len(v) for v in values)
        return 0
