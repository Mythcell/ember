import subprocess
import sys


def run_isolated_import(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_data_lazy_imports() -> None:
    result = run_isolated_import(
        """
import sys
from ember.data import EmberData
loaded = [
    name
    for name in sys.modules
    if name == "torchvision" or name.startswith("torchvision.")
]
if loaded:
    raise SystemExit(f"torchvision modules loaded: {loaded[:5]}")
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_runner_lazy_imports() -> None:
    result = run_isolated_import(
        """
import sys
from ember import EmberRunner
loaded = [
    name
    for name in sys.modules
    if name == "torch" or name.startswith("torch.")
]
if loaded:
    raise SystemExit(f"torch modules loaded: {loaded[:5]}")
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_model_lazy_imports() -> None:
    result = run_isolated_import(
        """
import sys
from ember import EmberModel
loaded = [
    name
    for name in sys.modules
    if name == "torchinfo" or name.startswith("torchinfo.")
]
if loaded:
    raise SystemExit(f"torchinfo modules loaded: {loaded[:5]}")
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_callback_lazy_imports() -> None:
    result = run_isolated_import(
        """
import sys
from ember.callbacks import EarlyStopping, ModelCheckpoint, OptunaTrialPruner
loaded = [
    name
    for name in sys.modules
    if name == "optuna" or name.startswith("optuna.")
]
if loaded:
    raise SystemExit(f"optuna modules loaded: {loaded[:5]}")
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout
