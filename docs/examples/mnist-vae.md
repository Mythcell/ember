# MNIST VAE

The `examples/mnist_vae` directory contains a small standalone VAE example for
MNIST-style images:

```bash
ember examples/mnist_vae/run.py -c examples/mnist_vae/config.yaml
```

The example defines its own encoder, decoder, VAE loss, and `EmberModel`
subclass in the script. The runner reads the lightweight training parameters
from `config.yaml`, so the `EmberRunner` class is only a small wrapper around
the same logic that would otherwise live in `main()`.

This keeps the example self-contained while demonstrating the usual subclassing
pattern:

- use a normal `nn.Module` for the VAE;
- override `train_step()`, `val_step()`, `eval_step()`, and `predict_step()`;
- return `(loss, reconstruction)` from train and validation steps so the default
  tracker records `train_loss` and `val_loss`;
- omit classification metrics because MNIST labels are not the VAE target.

The script uses `EmberMNIST` with unnormalized `[0, 1]` image tensors so the
decoder's sigmoid output can be trained with binary cross-entropy. It sets
`root=self.script_dir / "data"` so downloaded MNIST files stay next to the
example rather than in the caller's working directory.
