import torch.nn as nn
import torchmetrics
from lightning import Fabric
from torch.optim import Adam

from ember import EmberModel, EmberRunner
from ember.callbacks import ModelCheckpoint
from ember.mnist import EmberMNIST
from ember.utils import instantiate


class MyRunner(EmberRunner):
    def run(self) -> None:
        model: nn.Module = instantiate(
            self.cfg.model_type,
            params=self.cfg.model_params,
            local_path=self.script_dir,
            expected_type=nn.Module,
        )
        loss_fn: nn.Module = instantiate(
            self.cfg.loss_fn,
            expected_type=nn.Module,
        )
        data = EmberMNIST(
            batch_size=self.cfg.batch_size,
            root=self.script_dir / "data",
        )
        metric: torchmetrics.Metric = instantiate(
            self.cfg.metric_type,
            params=self.cfg.metric_params,
            expected_type=torchmetrics.Metric,
        )
        fabric = Fabric(accelerator=self.cfg.accelerator)

        optimizers = [Adam(params=model.parameters(), lr=self.cfg.lr)]
        em = EmberModel(
            fabric=fabric,
            model=model,
            optimizers=optimizers,
            loss_fn=loss_fn,
            metrics=[metric],
        )
        em.fit(
            epochs=self.cfg.epochs,
            emberdata=data,
            callbacks=[ModelCheckpoint(**self.cfg.model_checkpoint_params)],
        )
