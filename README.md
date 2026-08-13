# mmc-nirs

Companion package to [SimNIRS](https://github.com/fnirs-simbox/simnirs). It
stores precomputed light-transport Jacobians and provides helpers for turning
[MMC](https://github.com/fangq/mmc) (Mesh-based Monte Carlo) outputs into data
that SimNIRS can consume.

## What is MMC?

MMC simulates photon propagation through tissue on a tetrahedral mesh. This
mesh-based approach makes it possible to compute subject-specific sensitivity
(Jacobian) matrices for realistic head models—the forward models SimNIRS uses
to build synthetic fNIRS data.

## Installation

Install the package from PyPI once it is published:

```bash
pip install mmc-nirs
```

For local development with all test and pre-commit dependencies:

```bash
uv sync --all-extras
uv run pre-commit install
```

## Usage

Load a bundled experiment configuration:

```python
from mmc_nirs import load_default_config

config = load_default_config("pain")
```

Load the arrays required to initialize a SimNIRS simulator:

```python
from mmc_nirs import load_mmc_files

forward_model = load_mmc_files(config)
```

An experiment outside the package can be loaded from its configuration file.
By default, its data files are read from the directory containing `config.json`.
An optional `experiment_directory` can point to a different directory and is
resolved relative to the configuration file:

```python
from mmc_nirs import load_config, load_mmc_files

config = load_config("experiments/finger_tapping/config.json")
forward_model = load_mmc_files(config)
```

## Repository layout

- `mmc_nirs/experiments/<name>/` contains each experiment's `config.json`,
  mesh, registered probe, activation map, and Jacobian arrays.
- `mmc_nirs/loaders/` provides the public configuration and array loaders.
- `mmc_nirs/registration/` registers source and detector positions to a head
  mesh.
- `mmc_nirs/utils/` contains lower-level MMC serialization and output helpers.

> **Status:** early scaffolding. The current arrays are placeholders while the
> real MMC pipeline and file formats are finalized.
