import subprocess
import sys

import pytest
import torch
import torch.nn as nn
from lightning import Fabric

from ember import EmberModel


def test_top_level_core_imports() -> None:
    from ember import EmberData, EmberRunner, EmberTracker
    from ember.data import EmberData as DataEmberData
    from ember.models import EmberModel as ModelsEmberModel
    from ember.run import EmberRunner as RunEmberRunner
    from ember.tracking import EmberTracker as TrackingEmberTracker

    assert EmberData is DataEmberData
    assert EmberModel is ModelsEmberModel
    assert EmberRunner is RunEmberRunner
    assert EmberTracker is TrackingEmberTracker


def test_embermnist_imports_from_mnist_module() -> None:
    from ember.mnist import EmberMNIST

    assert EmberMNIST.__name__ == "EmberMNIST"


def test_embermnist_is_not_reexported_from_data() -> None:
    with pytest.raises(ImportError):
        exec("from ember.data import EmberMNIST", {})  # noqa: S102


def run_isolated_import(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_importing_emberdata_does_not_import_torchvision() -> None:
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


def test_importing_embermodel_does_not_import_torchinfo() -> None:
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


def test_importing_callbacks_does_not_import_optuna() -> None:
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


def test_embermodel_summary_delegates_to_torchinfo() -> None:
    module = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    model = EmberModel(
        fabric=Fabric(devices=1, accelerator="auto"),
        model=module,
        optimizers=[optimizer],
        loss_fn=nn.MSELoss(),
    )

    summary = model.summary(input_size=(1, 2), verbose=0)

    assert summary.total_params == 6
