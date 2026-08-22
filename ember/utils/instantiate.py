import contextlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def instantiate(
    class_spec: str,
    params: dict[str, Any] | None = None,
    local_path: Path | None = None,
    expected_type: type | None = None,
) -> Any:
    """Instantiate a class from a `module.Class` specification.

    Importing the module and constructing the class can execute arbitrary code. Only
    use trusted specifications and parameters.

    Args:
        class_spec: Dotted class specification, such as `torch.nn.ReLU`.
        params: Keyword arguments passed to the constructor.
        local_path: Optional directory used to discover a local module.
        expected_type: Optional required base type for the resolved class.

    Raises:
        ValueError: If `class_spec` does not contain a module and class name.
        ImportError: If the attribute does not resolve to a class.
        TypeError: If the class fails the type guard or its constructor rejects the
            supplied parameters.
    """
    params = {} if params is None else params
    module_name, sep, attr = class_spec.rpartition(".")
    if not all((module_name, sep, attr)):
        raise ValueError(f"Invalid class_spec: {class_spec}")

    module = _import_module(module_name, local_path)
    cls = getattr(module, attr, None)
    if cls is None or not inspect.isclass(cls):
        raise ImportError(f"{class_spec} is not a class")

    if expected_type and not issubclass(cls, expected_type):
        raise TypeError(f"{class_spec} must inherit from {expected_type.__name__}")

    try:
        return cls(**params)
    except TypeError as exc:
        raise TypeError(f"Failed to instantiate {class_spec}: {exc}") from exc


@contextlib.contextmanager
def _temp_sys_path(path: Path):
    """Temporarily add the given path to sys.path"""
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path_str)


def _import_module(name: str, local_path: Path | None) -> ModuleType:
    """
    Import the module with the given name. Checks sys.modules and any import aliases.
    Supports local modules imports with `local_path`.

    Args:
        name: Module name
        local_path: Optional directory path for local module discovery.
    """
    if name.startswith("."):
        return importlib.import_module(name)
    if local_path:
        local_module_path = _resolve_local_module_path(local_path, name)
        if local_module_path is not None:
            cached_module = sys.modules.get(name)
            if cached_module is not None and not _module_matches_path(
                cached_module, local_module_path
            ):
                return _load_local_module_with_parents(
                    name, local_path, local_module_path
                )
            try:
                with _temp_sys_path(local_path):
                    module = importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if not _missing_requested_module(exc, name):
                    raise
                return _load_local_module_with_parents(
                    name, local_path, local_module_path
                )
            if _module_matches_path(module, local_module_path):
                return module
            return _load_local_module_with_parents(name, local_path, local_module_path)
    # Check for names already registered under this name (e.g. `import torch.nn`)
    if name in sys.modules:
        return sys.modules[name]
    # Explicitly check the stack to find aliases (e.g. `import torch.nn as nn`)
    frame = inspect.currentframe()
    try:
        while frame is not None:
            for ns in (frame.f_globals, frame.f_locals):
                candidate = ns.get(name)
                if isinstance(candidate, ModuleType):
                    return candidate
            frame = frame.f_back
    finally:
        del frame
    if not local_path:
        return importlib.import_module(name)
    # Temporarily add local_path to sys.path so local modules are discoverable.
    with _temp_sys_path(local_path):
        return importlib.import_module(name)


def _resolve_local_module_path(local_path: Path, name: str) -> Path | None:
    """Resolve a local module to `module.py` or `module/__init__.py`."""
    module_root = Path(local_path).resolve().joinpath(*name.split("."))
    module_file = module_root.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_init = module_root / "__init__.py"
    if package_init.is_file():
        return package_init
    return None


def _module_matches_path(module: ModuleType, module_path: Path) -> bool:
    """Return True when a loaded module came from the expected local path."""
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    return Path(module_file).resolve() == module_path.resolve()


def _missing_requested_module(exc: ModuleNotFoundError, requested_name: str) -> bool:
    """Return True when importlib failed on the module Ember is trying to load."""
    missing_name = exc.name
    if missing_name is None:
        return False
    return (
        requested_name == missing_name
        or requested_name.startswith(f"{missing_name}.")
        or missing_name.startswith(f"{requested_name}.")
    )


def _load_local_module_with_parents(
    name: str,
    local_path: Path,
    module_path: Path,
) -> ModuleType:
    """Load a local module after loading any local parent packages."""
    parent_name, _, _ = name.rpartition(".")
    if parent_name:
        parent_module_path = _resolve_local_module_path(local_path, parent_name)
        if parent_module_path is not None:
            _load_local_module_with_parents(parent_name, local_path, parent_module_path)
    return _load_module_from_path(name, module_path)


def _load_module_from_path(name: str, module_path: Path) -> ModuleType:
    """Load a module directly from disk under the requested module name."""
    module_path = module_path.resolve()
    is_package = module_path.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name,
        module_path,
        submodule_search_locations=[str(module_path.parent)] if is_package else None,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module
        raise
    return module
