from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from omegaconf import OmegaConf


class EmberRunner(ABC):
    """
    Base class for all Ember runners. Subclasses must implement `run()`.

    `__init__()` is intentionally `@final` since the CLI automatically injects
    config and metadata. Instead, all custom logic should live in
    `run()`, which is treated purely as a runnable script.

    Attributes:
        cfg: OmegaConf DictConfig loaded from the config file, or None if no
            config was provided. Use this to access training hyperparameters,
            model architecture, and other settings.
        cfg_path: Path to the configuration file (if any).
        script_dir: Directory containing the runner script. Useful for resolving
            relative paths to data, models, or other assets.
        project_root: Nearest parent directory containing a project marker such as
            `pyproject.toml` or `.git`, or `script_dir` when no marker is found.
        verbosity: Verbosity level (from cli `--verbose` flag).

    Runners are automatically discovered via the following resolution order:

    1. Define a single `EmberRunner` subclass:

        .. code-block:: python

            class MyRunner(EmberRunner):
                def run(self) -> None:
                    lr = self.cfg.learning_rate
                    model = instantiate(
                        self.cfg.model.type,
                        params=self.cfg.model.params,
                        local_path=self.script_dir,
                    )
                    train(model, lr=lr)
            # The cli auto-instantiates it.

    2. Export a module-level `runner` variable when construction needs to be
       explicit:

        .. code-block:: python

            class MyRunner(EmberRunner):
                def run(self) -> None:
                    model = load_model(self.cfg)
                    train(model)
            runner = MyRunner()  # cli auto-injects cfg, script_dir, verbosity
    """

    @final
    def __init__(
        self,
        cfg_path: Path | None = None,
        script_dir: Path | None = None,
        verbosity: int = 0,
        project_root: Path | None = None,
    ) -> None:
        self.cfg_path = cfg_path
        self.cfg = OmegaConf.load(cfg_path) if cfg_path else None
        self.script_dir = script_dir
        self.verbosity = verbosity
        self.project_root = project_root

    @abstractmethod
    def run(self) -> None:
        """
        Execute the runner's main logic.

        This function is automatically run by the Ember CLI. It should contain your
        training loop or whatever pipeline you wish to run.
        Note that, due to the potential for arbitrary code execution, you should only
        run the Ember CLI on scripts that you trust.

        Typical usage involves accessing config parameters and instantiating components:

        .. code-block:: python

            from ember.utils import instantiate

            def run(self) -> None:
                # Access config hyperparameters
                batch_size = self.cfg.batch_size
                num_epochs = self.cfg.training.epochs

                # instantiate model from config spec (uses importlib.import_module)
                model = instantiate(
                    self.cfg.model.type,
                    params=self.cfg.model.params,
                    local_path=self.script_dir,
                    expected_type=nn.Module,
                )

        The above example assumes a config styled like this:

        .. code-block:: yaml

            batch_size: 32
            training:
              epochs: 100
            model:
              type: mymodule.MyModule
              params:
                hidden_dim: 256
        """
