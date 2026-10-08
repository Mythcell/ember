from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from omegaconf import DictConfig, ListConfig, OmegaConf


class EmberRunner[ResultT = None](ABC):
    """Base class for trusted experiments run directly or through the Ember CLI.

    The final constructor receives configuration and runner metadata. Subclasses
    implement `run()` and keep experiment-specific initialization inside that method.
    The result type defaults to `None`; parameterize the base class for other results.

    Attributes:
        cfg: Supplied OmegaConf config retained by reference, a config loaded from
            `cfg_path`, or `None`.
        cfg_path: Supplied configuration path, if any.
        script_dir: Directory containing the runner script, or `None` when omitted.
        verbosity: CLI verbosity level.
    """

    @final
    def __init__(
        self,
        cfg_path: Path | None = None,
        script_dir: Path | None = None,
        verbosity: int = 0,
        *,
        cfg: DictConfig | ListConfig | None = None,
    ) -> None:
        """Initialize runner state from at most one configuration source.

        Raises:
            ValueError: If both `cfg` and `cfg_path` are supplied.
        """
        if cfg is not None and cfg_path is not None:
            raise ValueError("Supply either cfg or cfg_path")

        self.cfg_path = cfg_path
        self.cfg = (
            cfg
            if cfg is not None
            else OmegaConf.load(cfg_path)
            if cfg_path is not None
            else None
        )
        self.script_dir = script_dir
        self.verbosity = verbosity

    @abstractmethod
    def run(self) -> ResultT:
        """Execute the experiment and return its result; the CLI discards it."""
