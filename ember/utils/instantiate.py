import contextlib
import importlib
import importlib.util
import inspect
import keyword
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, overload


@overload
def instantiate[T](
    target: str | Callable[..., Any],
    *args: Any,
    params: Mapping[str, Any] | None = None,
    local_path: Path | None = None,
    expected_type: type[T],
) -> T: ...


@overload
def instantiate[T](
    target: Callable[..., T],
    *args: Any,
    params: Mapping[str, Any] | None = None,
    local_path: Path | None = None,
    expected_type: None = None,
) -> T: ...


@overload
def instantiate(
    target: str,
    *args: Any,
    params: Mapping[str, Any] | None = None,
    local_path: Path | None = None,
    expected_type: None = None,
) -> Any: ...


def instantiate(
    target: str | Callable[..., Any],
    *args: Any,
    params: Mapping[str, Any] | None = None,
    local_path: Path | None = None,
    expected_type: type | None = None,
) -> Any:
    """Call a target supplied directly or as an absolute `module.attribute` path.

    Importing the module and calling the target can execute arbitrary code. Only
    use trusted specifications and parameters.

    Args:
        target: Callable or dotted path, such as `torch.nn.ReLU`.
        *args: Positional arguments passed to the target.
        params: Keyword arguments passed to the target without converting values.
        local_path: Optional directory for local string targets; ignored for callables.
        expected_type: Optional runtime-checkable base type required of the result.

    Raises:
        ValueError: If the string target is not an absolute dotted identifier path.
        ImportError: If importing fails or the target attribute is missing.
        TypeError: If the target is not callable, a type guard fails, or the call
            raises TypeError.
    """
    if isinstance(target, str):
        parts = target.split(".")
        if len(parts) < 2 or any(
            not part.isidentifier() or keyword.iskeyword(part) for part in parts
        ):
            raise ValueError(f"Invalid target: {target}")
        module_name, _, attr = target.rpartition(".")
        module = _import_module(module_name, local_path)
        try:
            resolved_target = getattr(module, attr)
        except AttributeError as exc:
            raise ImportError(f"Target attribute not found: {target}") from exc
        target_name = target
    else:
        resolved_target = target
        target_name = getattr(target, "__qualname__", type(target).__qualname__)

    if not callable(resolved_target):
        raise TypeError(f"{target_name} is not callable")
    if (
        expected_type is not None
        and inspect.isclass(resolved_target)
        and not issubclass(resolved_target, expected_type)
    ):
        raise TypeError(f"{target_name} must inherit from {expected_type.__name__}")

    try:
        result = resolved_target(*args, **({} if params is None else params))
    except TypeError as exc:
        raise TypeError(f"Failed to instantiate {target_name}: {exc}") from exc

    if expected_type is not None and not isinstance(result, expected_type):
        raise TypeError(
            f"{target_name} must return an instance of {expected_type.__name__}; "
            f"got {type(result).__name__}"
        )
    return result


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
    """Import a module, preferring a matching file or package at local_path."""
    if local_path is None:
        return importlib.import_module(name)
    with _temp_sys_path(local_path):
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
