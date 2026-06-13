# CLI

The Ember CLI is a thin entry point for running trusted local
`EmberRunner` scripts:

```bash
ember path/to/run.py
ember path/to/run.py --config path/to/config.yaml
```

It is intentionally small. The CLI loads a Python script, discovers an
`EmberRunner`, injects optional config metadata, and calls `runner.run()`.

!!! danger

    The CLI imports and executes local Python code. Runner scripts and
    config-driven calls to `instantiate()` can execute arbitrary code during
    module import or object construction. Only run scripts and configs that you
    trust.

## Usage

```bash
ember [OPTIONS] SCRIPT_PATH
```

| Argument | Purpose |
| --- | --- |
| `SCRIPT_PATH` | Python file containing an `EmberRunner` subclass or module-level `runner`. |

| Option | Purpose |
| --- | --- |
| `--config`, `-c` | Path to an optional YAML config file. |
| `--verbose`, `-v` | Increase logging verbosity. |
| `--version` | Print the installed Ember version and exit. |
| `--help` | Show CLI help. |

## Runner Discovery

The CLI loads `SCRIPT_PATH` as a Python module, then resolves the runner in this
order:

1. Use a module-level `runner = MyRunner()` object when present.
2. Otherwise, instantiate the single `EmberRunner` subclass found in the module.

If multiple runner subclasses are present, expose a module-level `runner` so the
entry point is explicit.

## Injected Runner State

The CLI passes runtime metadata through `EmberRunner.__init__()`:

| Attribute | Value |
| --- | --- |
| `self.cfg` | OmegaConf config loaded from `--config`, or `None`. |
| `self.cfg_path` | Path to the config file, if one was supplied. |
| `self.script_dir` | Directory containing `SCRIPT_PATH`. |
| `self.verbosity` | Verbosity value from `--verbose` / `-v`. |

Use `self.script_dir` when resolving local model modules, data files, or config
neighbors. See [Runners](core/runners.md) for runner patterns and
[Instantiation](utils/instantiation.md) for config-driven object construction.
