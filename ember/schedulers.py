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
    """Base class for all Ember parameter schedulers"""

    def __init__(self):
        super().__init__()


class LinearParamScheduler(EmberParamScheduler):
    """An epoch-based linear parameter scheduler."""

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
        """
        An epoch-based linear parameter scheduler.

        Returns a value for a parameter based on the current epoch. The value starts at
        [start_value] for [startup_epochs] epochs, then linearly transitions to
        [end_value] over [interp_epochs] epochs.

        Args:
            start_value: Initial starting value.
            end_value: Final end value.
            startup_epochs: Number of epochs to remain at the starting value.
            interp_epochs: Number of epochs over which to linearly interpolate from
                the starting value to the end value.

        Returns:
            The desired parameter value.
        """
        super().__init__()
        self.register_buffer("start_value", torch.tensor(start_value))
        self.register_buffer("end_value", torch.tensor(end_value))
        self.register_buffer("startup_epochs", torch.tensor(startup_epochs))
        self.register_buffer("interp_epochs", torch.tensor(interp_epochs))

    def forward(self, epoch: int) -> torch.Tensor:
        fraction = _calculate_fraction(epoch, self.startup_epochs, self.interp_epochs)
        return self.start_value + fraction * (self.end_value - self.start_value)


class ExponentialParamScheduler(EmberParamScheduler):
    """An epoch-based exponential parameter scheduler."""

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
        """
        An epoch-based exponential parameter scheduler.

        Returns a value for a parameter based on the current epoch. The value starts at
        [start_value] for [startup_epochs] epochs, then exponentially transitions to
        [end_value] over [interp_epochs] epochs.

        Args:
            start_value: Initial starting value.
            end_value: Final end value.
            startup_epochs: Number of epochs to remain at the starting value.
            interp_epochs: Number of epochs over which to linearly interpolate from
                the starting value to the end value.
            profile: Type of exponential profile (growth or decay).

        Returns:
            The desired parameter value.
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
    """Validates the list of milestones to be provided to PiecewiseLinear schedulers"""
    if any(m[1] <= 0 for m in milestones):
        raise ValueError("Milestone epochs must be positive.")
    milestones = sorted(milestones, key=lambda x: x[1])
    epochs = [m[1] for m in milestones]
    if len(epochs) > len(set(epochs)):
        raise ValueError("Duplicate epochs. Milestone epochs should be unique.")
    return [(start_val, 0)] + milestones


class PiecewiseLinearParamScheduler(EmberParamScheduler):
    """An epoch-based piecewise linear parameter scheduler."""

    values: torch.Tensor

    def __init__(
        self,
        start_val: float,
        milestones: list[tuple[float, int]],
    ) -> None:
        """
        An epoch-based piecewise linear parameter scheduler.

        Implements a schedule defined by a start value and a list of milestones of
        (value, epoch) tuples. The value is linearly interpolated between milestones
        based on the current epoch. For epochs beyond the last milestone, the value
        is clamped to the last milestone's value.

        Args:
            start_val: The starting value.
            milestones: List of tuples of (value, epoch) that define the boundaries
                between different line segments. The list is automatically sorted,
                however epochs must be unique.

        Raises:
            ValueError: If `milestones` contains duplicate or negative epoch values.

        Examples:
            >>> # Linear schedule from: 0.1 --> 1.0 (epoch 10) --> 5.0 (epoch 100).
            >>> PiecewiseLinearParamScheduler(
            >>>     start_val=0.1,
            >>>     milestones=[(1.0, 10), (5.0, 100)],
            >>> )
        """
        super().__init__()
        self.milestones = _validate_milestones(start_val, milestones)
        self.register_buffer("values", torch.tensor([m[0] for m in self.milestones]))

    def forward(self, epoch: int) -> torch.Tensor:
        if epoch <= self.milestones[0][1]:
            return self.values[0]

        for i in range(1, len(self.milestones)):
            v_prev, e_prev = self.milestones[i - 1]
            v_curr, e_curr = self.milestones[i]

            if epoch <= e_curr:
                fraction = (epoch - e_prev) / (e_curr - e_prev)
                return self.values[i - 1] + fraction * (v_curr - v_prev)
        # at this point, epoch is beyond the last milestone, hence we use final segment
        return self.values[-1]


class PiecewiseLinearLRScheduler(LRScheduler):
    """A learning rate scheduler with a piecewise linear schedule."""

    def __init__(
        self,
        optimizer: Optimizer,
        start_val: float,
        milestones: list[tuple[float, int]],
        last_epoch: int = -1,
    ) -> None:
        """
        A learning rate scheduler with a piecewise linear schedule

        The schedule is defined by a starting learning rate (`start_val`) at epoch 0
        and a list of milestones of format (lr, epoch). The learning rate linearly
        interpolates between the milestone values based on the current epoch. For epochs
        beyond the last milestone, the value is clamped to the last milestone's value.

        Args:
            optimizer: Wrapped optimizer.
            start_val: The initial learning rate value (at epoch 0). This will
                override the initial LR set in the optimizer from epoch 0 onwards.
            milestones: A list of tuples with the format (lr, epoch) specifying target
                learning rates at specific epochs. Epochs must be positive and without
                duplicates.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.milestones = _validate_milestones(start_val, milestones)
        self.start_val = start_val
        super().__init__(optimizer, last_epoch)

    def _calculate_lr(self, epoch: int) -> float:
        """Calculates lr for the current epoch"""
        if epoch <= 0:
            return self.start_val

        for i in range(1, len(self.milestones)):
            lr_prev, e_prev = self.milestones[i - 1]
            lr_curr, e_curr = self.milestones[i]
            if epoch <= e_curr:
                fraction = (epoch - e_prev) / (e_curr - e_prev)
                return lr_prev + fraction * (lr_curr - lr_prev)

        # at this point epoch is beyond final milestone, hence clamp to that value
        return self.milestones[-1][0]

    def get_lr(self) -> list[float | torch.Tensor]:
        lr = self._calculate_lr(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["start_val"] = self.start_val
        state["milestones"] = self.milestones
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.start_val = state_dict["start_val"]
        self.milestones = state_dict["milestones"]
        super().load_state_dict(state_dict)
