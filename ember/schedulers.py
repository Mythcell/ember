from typing import Any, Literal

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _calculate_fraction(
    epoch: int, startup_epochs: torch.Tensor, interp_epochs: torch.Tensor
) -> torch.Tensor:
    if interp_epochs.item() < 0:
        raise ValueError("interp_epochs must be non-negative.")
    if interp_epochs.item() == 0:
        return torch.where(
            torch.as_tensor(epoch, device=startup_epochs.device) < startup_epochs,
            torch.zeros((), device=startup_epochs.device),
            torch.ones((), device=startup_epochs.device),
        )
    fraction = (epoch - startup_epochs) / interp_epochs
    return fraction.clamp(0.0, 1.0)


class EmberParamScheduler(nn.Module):
    """Base module for epoch-indexed Ember parameter schedulers."""

    def __init__(self):
        super().__init__()


class LinearParamScheduler(EmberParamScheduler):
    """Linearly schedule a scalar between two values."""

    start_value: torch.Tensor
    end_value: torch.Tensor
    startup_epochs: torch.Tensor
    interp_epochs: torch.Tensor

    def __init__(
        self,
        start_value: float,
        end_value: float,
        startup_epochs: int,
        interp_epochs: int,
    ) -> None:
        """Initialize a linear parameter schedule.

        Args:
            start_value: Value held during the startup period.
            end_value: Value returned after interpolation completes.
            startup_epochs: Number of epochs to hold `start_value`.
            interp_epochs: Number of epochs used for linear interpolation.
        """
        super().__init__()
        self.register_buffer("start_value", torch.tensor(start_value))
        self.register_buffer("end_value", torch.tensor(end_value))
        self.register_buffer("startup_epochs", torch.tensor(startup_epochs))
        self.register_buffer("interp_epochs", torch.tensor(interp_epochs))

    def forward(self, epoch: int) -> torch.Tensor:
        """Return the scheduled scalar for a zero-based epoch index."""
        fraction = _calculate_fraction(epoch, self.startup_epochs, self.interp_epochs)
        return self.start_value + fraction * (self.end_value - self.start_value)


class ExponentialParamScheduler(EmberParamScheduler):
    """Schedule a scalar with an exponential growth or decay profile."""

    start_value: torch.Tensor
    end_value: torch.Tensor
    startup_epochs: torch.Tensor
    interp_epochs: torch.Tensor

    def __init__(
        self,
        start_value: float,
        end_value: float,
        startup_epochs: int,
        interp_epochs: int,
        profile: Literal["growth", "decay"] = "decay",
    ) -> None:
        """Initialize an exponential parameter schedule.

        Args:
            start_value: Value held during the startup period.
            end_value: Value returned after interpolation completes.
            startup_epochs: Number of epochs to hold `start_value`.
            interp_epochs: Number of epochs used for exponential interpolation.
            profile: Curvature of the interpolation, either `"growth"` or `"decay"`.

        Raises:
            ValueError: If `profile` is not `"growth"` or `"decay"`.
        """
        super().__init__()
        if profile not in {"growth", "decay"}:
            raise ValueError("profile must be 'growth' or 'decay'.")
        self.register_buffer("start_value", torch.tensor(start_value))
        self.register_buffer("end_value", torch.tensor(end_value))
        self.register_buffer("startup_epochs", torch.tensor(startup_epochs))
        self.register_buffer("interp_epochs", torch.tensor(interp_epochs))
        self.profile = profile

    def forward(self, epoch: int) -> torch.Tensor:
        """Return the scheduled scalar for a zero-based epoch index."""
        fraction = _calculate_fraction(epoch, self.startup_epochs, self.interp_epochs)
        one = torch.ones((), device=fraction.device, dtype=fraction.dtype)
        if self.profile == "growth":
            fraction = torch.expm1(fraction) / torch.expm1(one)
        else:
            fraction = -torch.expm1(-fraction) / -torch.expm1(-one)
        return self.start_value + fraction * (self.end_value - self.start_value)


def _validate_milestones(
    start_val: float, milestones: list[tuple[float, int]]
) -> list[tuple[float, int]]:
    """Validate and sort piecewise schedule milestones."""
    if any(m[1] <= 0 for m in milestones):
        raise ValueError("Milestone epochs must be positive.")
    milestones = sorted(milestones, key=lambda x: x[1])
    epochs = [m[1] for m in milestones]
    if len(epochs) > len(set(epochs)):
        raise ValueError("Duplicate epochs. Milestone epochs should be unique.")
    return [(start_val, 0), *milestones]


class PiecewiseLinearParamScheduler(EmberParamScheduler):
    """Linearly interpolate a scalar across epoch milestones."""

    values: torch.Tensor

    def __init__(
        self,
        start_val: float,
        milestones: list[tuple[float, int]],
    ) -> None:
        """Initialize a piecewise linear parameter schedule.

        Args:
            start_val: Value at epoch zero.
            milestones: `(value, epoch)` targets. Milestones are sorted by epoch,
                linearly interpolated, and clamped at the final value.

        Raises:
            ValueError: If milestone epochs are non-positive or duplicated.
        """
        super().__init__()
        self.milestones = _validate_milestones(start_val, milestones)
        self.register_buffer("values", torch.tensor([m[0] for m in self.milestones]))

    def forward(self, epoch: int) -> torch.Tensor:
        """Return the interpolated scalar for a zero-based epoch index."""
        if epoch <= self.milestones[0][1]:
            return self.values[0]

        for i in range(1, len(self.milestones)):
            v_prev, e_prev = self.milestones[i - 1]
            v_curr, e_curr = self.milestones[i]

            if epoch <= e_curr:
                fraction = (epoch - e_prev) / (e_curr - e_prev)
                return self.values[i - 1] + fraction * (v_curr - v_prev)
        # Clamp epochs beyond the final milestone.
        return self.values[-1]


class PiecewiseLinearLRScheduler(LRScheduler):
    """Linearly interpolate an optimizer learning rate across milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        start_val: float,
        milestones: list[tuple[float, int]],
        last_epoch: int = -1,
    ) -> None:
        """Initialize a piecewise linear learning-rate schedule.

        Args:
            optimizer: Wrapped optimizer.
            start_val: Learning rate at epoch zero. This overrides the optimizer's
                initial learning rate when scheduling begins.
            milestones: `(learning_rate, epoch)` targets. Epochs must be positive
                and unique.
            last_epoch: Index of the last completed epoch.

        Raises:
            ValueError: If milestone epochs are non-positive or duplicated.
        """
        self.milestones = _validate_milestones(start_val, milestones)
        self.start_val = start_val
        super().__init__(optimizer, last_epoch)

    def _calculate_lr(self, epoch: int) -> float:
        """Calculate the learning rate for one epoch."""
        if epoch <= 0:
            return self.start_val

        for i in range(1, len(self.milestones)):
            lr_prev, e_prev = self.milestones[i - 1]
            lr_curr, e_curr = self.milestones[i]
            if epoch <= e_curr:
                fraction = (epoch - e_prev) / (e_curr - e_prev)
                return lr_prev + fraction * (lr_curr - lr_prev)

        # Clamp epochs beyond the final milestone.
        return self.milestones[-1][0]

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return the scheduled learning rate for every optimizer parameter group."""
        lr = self._calculate_lr(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state including the start value and milestones."""
        state = super().state_dict()
        state["start_val"] = self.start_val
        state["milestones"] = self.milestones
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state including the start value and milestones."""
        self.start_val = state_dict["start_val"]
        self.milestones = state_dict["milestones"]
        super().load_state_dict(state_dict)
