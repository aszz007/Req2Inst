# Reproducibility guidance

## Scope

The repository provides source code and a dependency specification. It does not provide an exact environment lock, datasets, model weights, trained checkpoints, generated predictions, or experiment caches. A source checkout alone is therefore not a complete reproduction package.

## Environment

Create an isolated Python environment from the repository root:

```bash
python -m venv .venv
```

Activate it with the command for your platform.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If an accelerator-specific PyTorch build is required, install the build appropriate for that platform.

The command-line entrypoints import the model runtime during startup. Their `--help` commands therefore require the runtime dependencies to be installed.

## Runtime resources

Before running inference or experiments, prepare the required resources locally:

- compatible base models;
- trained adapter weights or checkpoints;
- datasets and evaluation inputs;
- any credentials or network access required by the selected upstream resources.

Do not assume that a model, dataset, or checkpoint is redistributable merely because it is publicly downloadable. Record its identifier, revision, source, license, and checksum when available.

## Commands and records

For a run intended to support a result, record at least:

- the Git commit SHA and worktree state;
- operating system, Python version, accelerator, driver, and relevant library versions;
- model and checkpoint identifiers and checksums;
- dataset version, license, split, and preprocessing configuration;
- the exact command and configuration used;
- whether predictions were generated in that run or loaded from a local cache.

The experiment runner accepts cache-backed modes, but the repository does not distribute those caches. Generate the required predictions locally before using a cache-only path.

Do not describe a result as reproduced unless its environment, inputs, models, checkpoints, commands, and evaluation configuration match the intended experimental contract.
