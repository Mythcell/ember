from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from omegaconf import OmegaConf


class EmberRunner(ABC):
    """Base class for trusted experiment scripts executed by the Ember CLI.

    The final constructor receives CLI metadata. Subclasses implement `run()` and
    keep experiment-specific initialization inside that method.

    Attributes:
        cfg: OmegaConf config loaded from `cfg_path`, or `None`.
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
    ) -> None:
        """Initialize CLI-managed runner state."""
        self.cfg_path = cfg_path
        self.cfg = OmegaConf.load(cfg_path) if cfg_path else None
        self.script_dir = script_dir
        self.verbosity = verbosity

    @abstractmethod
    def run(self) -> None:
        """Execute the experiment after the CLI has injected runner state."""
