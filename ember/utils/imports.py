import importlib.util
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ember.run import EmberRunner

_SCRIPT_MODULE_NAME = "__script_module__"
_PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")


def load_script_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(_SCRIPT_MODULE_NAME, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script; no loader found for: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SCRIPT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def find_project_root(script_dir: Path) -> Path:
    script_dir = script_dir.resolve()
    for candidate in (script_dir, *script_dir.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_ROOT_MARKERS):
            return candidate
    return script_dir


def detect_runner_instance(
    module: Any,
    script_dir: Path,
    cfg_path: Path | None = None,
    verbosity: int = 0,
    project_root: Path | None = None,
) -> EmberRunner | None:
    candidate = getattr(module, "runner", None)
    if candidate is None:
        return None
    if not isinstance(candidate, EmberRunner):
        raise TypeError("Variable `runner` must be an EmberRunner instance")

    # auto-inject CLI args that the user may not have known at construction time
    if candidate.cfg is None:
        candidate.cfg_path = cfg_path
        candidate.cfg = OmegaConf.load(cfg_path) if cfg_path else None
    if candidate.script_dir is None:
        candidate.script_dir = script_dir
    if candidate.project_root is None:
        candidate.project_root = project_root
    candidate.verbosity = verbosity
    return candidate


def discover_runner_subclass(
    module: Any,
    script_dir: Path,
    cfg_path: Path | None = None,
    verbosity: int = 0,
    project_root: Path | None = None,
) -> EmberRunner:
    subclasses = [
        obj
        for obj in module.__dict__.values()
        if isinstance(obj, type)
        and issubclass(obj, EmberRunner)
        and obj is not EmberRunner
        and getattr(obj, "__module__", None) == _SCRIPT_MODULE_NAME
    ]
    if not subclasses:
        raise RuntimeError("No Runner subclass found (and no runner provided)")
    if len(subclasses) > 1:
        raise RuntimeError("Multiple Runner subclasses found; please expose `runner`")
    runner_cls: type[EmberRunner] = subclasses[0]
    return runner_cls(
        cfg_path=cfg_path,
        script_dir=script_dir,
        verbosity=verbosity,
        project_root=project_root,
    )
