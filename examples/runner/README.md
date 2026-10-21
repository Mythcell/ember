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

`EmberMNIST(root=...)` stores downloaded data in `examples/runner/data`.

Configs are optional; runners can also be plain script entrypoints with hardcoded
Python construction.

Programmatic callers can supply an existing OmegaConf `DictConfig` or `ListConfig`
with `cfg=` and call `.run()` directly. Supply either `cfg` or `cfg_path`;
supplying both raises `ValueError`. In-memory configs are retained by reference,
including empty configs, so caller and runner mutations are shared. With neither
source, `self.cfg` is `None`. Convert ordinary mappings with `OmegaConf.create`.
Supply `script_dir` explicitly when the runner needs local imports or data paths.

`EmberRunner` defaults to a `None` result. A runner that returns another type
declares `EmberRunner[ResultType]` and annotates `run() -> ResultType`; direct callers
receive that result with its static type. The CLI discards results. Exported runner
instances retain their existing config even when the CLI supplies `-c`; the CLI
injects a config only when `runner.cfg` is `None`.

## Reporting losses and TorchMetrics

`EmberModel(..., metrics=[...])` configures metrics for default supervised steps.
Ember registers independent training and validation instances with the tracker under
names such as `val_MulticlassAccuracy` for callback and scheduler monitors.

Custom training, validation and evaluation steps report through their tracker and
return `None`. Manual scalars accumulate weighted stage means; named TorchMetrics
use a separate API:

```python
tracker.register("custom_loss", key_type="train")
tracker.update("custom_loss", loss, batch_size=batch_size)

tracker.register_torchmetric("train_accuracy", metric, key_type="train")
tracker.update_torchmetric("train_accuracy", predictions, targets)
```

Use a separate metric instance for each key. The tracker computes scalar metric
results once per stage and resets pending state. Untouched keys record zero;
skipped validation records nothing. Distributed metric updates must occur on all
ranks or none.

Default steps call `report_step(tracker, key_type, loss, y_pred, y)` with the actual
loss targets, including paired augmentation. Override it to customise reporting;
`super().report_step(...)` records loss and updates constructor metrics. Custom
steps can update their tracker directly, as in the [VAE example](../mnist_vae/run.py).
Evaluation steps use the supplied `eval_tracker`; prediction steps return outputs.

### Evaluation and checkpoints

`evaluate()` clones the tracker's validation registrations into an isolated
tracker. Use `eval_tracker=` to supply another template; reporting calls must
match its keys. The template's history and pending state remain untouched.

Checkpoints restore scalar registrations automatically and metric histories for
configured TorchMetrics. Unconfigured metrics are skipped; new metrics start with
empty histories. Pending accumulators are discarded. Epoch progress is stored
explicitly. Checkpoints without an epoch count use the longest saved history.

## Instantiation

`ember.utils.instantiate()` calls a class or factory selected by Python code or
configuration. Projects choose their own config structure; Ember interprets no
reserved config keys and does not recursively instantiate values.

The runner supplies the model's live parameters to its configured optimiser:

```python
from torch.optim import Optimizer

from ember.utils import instantiate

optimizer = instantiate(
    self.cfg.optimizer_type,
    model.parameters(),
    params=self.cfg.optimizer_params,
    expected_type=Optimizer,
)
```

The target is called once with the supplied positional arguments and keyword
arguments from `params`. Ordinary mappings and OmegaConf `DictConfig` values are
accepted. The helper does not mutate the mapping or convert nested values; the
called target can mutate objects it receives through normal Python references.
Keyword-only dependencies can be supplied with ordinary dictionary construction,
such as `params={**cfg.decoder_params, "encoder": encoder}`. Duplicate positional
and keyword arguments raise the target's normal `TypeError`, with added context.
Constructor keywords named `params`, `local_path` or `expected_type` belong inside
the `params` mapping.

String targets must be absolute dotted Python identifiers, such as
`torch.nn.CrossEntropyLoss` or `torchmetrics.Accuracy`. Ember imports the module
before the final dot and retrieves the named attribute. Caller aliases, relative
paths, bare names and nested attribute traversal are not supported. Classes and
factories can also be passed directly:

```python
activation = instantiate(nn.ReLU, params={"inplace": True})
model = instantiate(build_model, params=cfg.model_params, expected_type=nn.Module)
```

`expected_type` requires a runtime-checkable class or base type, not a
parameterised typing expression. An incompatible class is rejected before its
constructor runs; every guarded result is checked after the call. A factory can
therefore run before its result is rejected. With `expected_type`, the static
result type is the required type. Otherwise, direct callables retain their
inferred return type, and string targets return `Any`. Constructor arguments are
validated by Python at runtime.

For local strings such as `models.SmallCNN`, the runner explicitly supplies
`local_path=self.script_dir`. A matching local module or package takes precedence
over a cached target from another directory. Matching-origin modules are reused;
without a local match, normal imports apply with the directory temporarily
available. Python's dependency cache remains process-wide: switching between
same-named packages can retain dependencies from the earlier directory.
`local_path` is ignored for direct callables.

Malformed strings raise `ValueError`; missing attributes raise `ImportError`;
non-callable targets and type mismatches raise `TypeError`. Import failures retain
their original exceptions. Call-time `TypeError` receives target context with the
original exception chained; other call exceptions propagate unchanged. Importing
and calling targets executes Python code, so specifications and arguments must
be trusted.
