# Runner Example

This example shows the intended `EmberRunner` workflow:

```bash
ember run.py -c config.yaml
```

`run.py` defines a single `EmberRunner` subclass. The CLI loads `config.yaml`
with OmegaConf and injects `self.cfg`, `self.cfg_path`, `self.script_dir`, and
`self.verbosity`.

The runner uses `self.script_dir` for local imports and data paths, so the
example can be launched from outside the example directory.

The example also demonstrates `ember.utils.instantiate()`:

- `models.SmallCNN` is resolved from `models.py` relative to the runner script.
- `nn.CrossEntropyLoss` and `torchmetrics.Accuracy` are resolved from imported
  modules.
- `expected_type` checks that config specs produce the expected object type.
- `EmberMNIST(root=...)` stores downloaded data in `examples/runner/data`.

Configs are optional; runners can also be plain script entrypoints with hardcoded
Python construction.
