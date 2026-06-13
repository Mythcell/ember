import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from torchvision.transforms import v2

from ember import EmberModel, EmberRunner
from ember.mnist import EmberMNIST


class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        return self.z_mean(x), self.z_log_var(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(
        self, z_mean: torch.Tensor, z_log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var


class VAELoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        reconstruction = F.binary_cross_entropy(x_recon, x, reduction="sum")
        reconstruction = reconstruction / batch_size
        kl_divergence = -0.5 * torch.sum(
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp()
        )
        kl_divergence = kl_divergence / batch_size
        return reconstruction + self.beta * kl_divergence


class VAEModel(EmberModel):
    def _step(
        self, batch: tuple[torch.Tensor, ...] | tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch[0]
        x_recon, z_mean, z_log_var = self.model(x)
        loss = self.loss_fn(x_recon, x, z_mean, z_log_var)
        return loss, x_recon

    def train_step(
        self, batch: tuple[torch.Tensor, ...] | tuple, batch_idx: int, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (optimizer, *_) = self.optimizers
        is_accumulating = (batch_idx + 1) % self.accumulate_grad_batches != 0

        with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
            loss, x_recon = self._step(batch)
            self.fabric.backward(loss / self.accumulate_grad_batches)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()

        return loss, x_recon

    def val_step(
        self, batch: tuple[torch.Tensor, ...] | tuple, batch_idx: int, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._step(batch)

    def eval_step(
        self,
        batch: tuple[torch.Tensor, ...] | tuple,
        batch_idx: int,
        eval_tracker=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._step(batch)

    def predict_step(
        self, batch: tuple[torch.Tensor, ...] | tuple, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch[0]
        return self.model(x)


class MNISTVAERunner(EmberRunner):
    def run(self) -> None:
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        data = EmberMNIST(
            transform=transform,
            batch_size=self.cfg.batch_size,
            val_fraction=0.1,
            root=self.script_dir / "data",
        )

        fabric = Fabric(accelerator="auto")
        model = VAE(latent_dim=self.cfg.latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        embermodel = VAEModel(
            fabric=fabric,
            model=model,
            optimizers=[optimizer],
            loss_fn=VAELoss(beta=self.cfg.beta),
        )

        embermodel.fit(
            emberdata=data,
            epochs=self.cfg.epochs,
            batch_limit=self.cfg.batch_limit,
        )

        metrics = embermodel.evaluate(emberdata=data)
        reconstructions, z_mean, z_log_var = embermodel.predict(
            emberdata=data,
            show_progress_bar=False,
        )
        print(metrics)
        print(reconstructions.shape, z_mean.shape, z_log_var.shape)
