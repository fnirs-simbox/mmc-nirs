# mmcnirs

`mmcnirs` provides light-transport preparation, execution, and loading tools
for [SimNIRS](https://github.com/fnirs-simbox/simnirs). It prepares head meshes
and fNIRS probes, runs [MMC](https://github.com/fangq/mmc) (Mesh-based Monte
Carlo) to generate Jacobians, and loads precomputed results for downstream
simulation.

Large example data, standard heads, and platform-specific MMC binaries are not
bundled in the Python package. They are downloaded anonymously from the public
[`nielsbracher/fnirs-simbox-assets`](https://huggingface.co/datasets/nielsbracher/fnirs-simbox-assets)
Hugging Face dataset when needed.

> **Status:** `mmcnirs` is pre-alpha software. APIs and data formats may still
> change before the first stable release.

## What is MMC?

MMC simulates photon propagation through tissue on a tetrahedral mesh. This
mesh-based approach makes it possible to compute subject-specific sensitivity
(Jacobian) matrices for realistic head models—the forward models SimNIRS uses
to build synthetic fNIRS data.

## Installation

`mmcnirs` requires Python 3.12–3.14. Install it from a local checkout:

```bash
python -m pip install .
```

For development with the test and pre-commit dependencies:

```bash
uv sync --all-extras
uv run pre-commit install
```

## Load a default experiment

Two precomputed example experiments are currently available: `pain` and
`pattern_cutting`.

```python
from mmcnirs import load_default_config, load_light_transport_results

config = load_default_config("pain")
light_transport_results = load_light_transport_results(config)
```

The first call downloads the public `experiments/pain` subtree to
`./mmcnirs-assets/experiments/pain`. Later calls synchronize against the public
dataset and reuse unchanged local files. A Hugging Face token or login is not
required.

Use `assets_root` to put the user-visible downloads somewhere else:

```python
config = load_default_config(
    "pattern_cutting",
    assets_root="/path/to/project-assets",
)
```

The supplied root mirrors the dataset layout, so this example is stored at
`/path/to/project-assets/experiments/pattern_cutting`.

## Load a local experiment

Use `load_config` for an experiment that is already stored locally. Relative
file paths are resolved from the directory containing `config.json`; an
optional relative `experiment_dir` in the configuration is resolved from that
same location.

```python
from mmcnirs import load_config, load_light_transport_results

config = load_config("/path/to/experiment/config.json")
light_transport_results = load_light_transport_results(config)
```

Pass `use_jacobian=False` to load only the mesh and registered probe data.

## Load a standard head

The current standard-head catalog contains `colin27`:

```python
from mmcnirs import load_standard_head

standard_head_directory = load_standard_head("colin27")
```

By default this returns
`./mmcnirs-assets/standard-heads/colin27`. To also copy the downloaded files to
another user-facing location, use `save=True` and `directory=...`.

## End-to-end Jacobian workflow

The Marimo workflow demonstrates the full path from public fNIRS inputs to a
prepared mesh, registered probe, wavelength-specific MMC inputs, and generated
Jacobians:

```bash
uv run marimo edit workflow_example/end_to_end_workflow.py
```

It downloads the `e2e-files` inputs and Colin27 standard head through the
central Hugging Face loader. When the workflow first runs MMC, the appropriate
runtime is downloaded, checksum-verified, and installed in the user's managed
cache.

## Hugging Face connection

All Hugging Face access is centralized in
[`mmcnirs/loaders/hf_loader.py`](mmcnirs/loaders/hf_loader.py). The connection
uses the public dataset's `main` branch and explicitly disables authentication;
neither callers nor users need to supply an HF token.

Higher-level loaders pass a category and keyword to this connection point:

| Caller or purpose | Category | Keyword | Dataset path |
|---|---|---|---|
| `load_default_config("pain")` | `experiment` | `pain` | `experiments/pain` |
| `load_default_config("pattern_cutting")` | `experiment` | `pattern_cutting` | `experiments/pattern_cutting` |
| `load_standard_head("colin27")` | `standard-head` | `colin27` | `standard-heads/colin27` |
| End-to-end workflow inputs | `workflow` | `e2e-files` | `e2e-files` |
| Managed MMC installation | `runtime` | `manifest` / `archive` | `mmc-runtime` |

The catalog can be inspected programmatically:

```python
from mmcnirs.loaders import HF_RESOURCE_KEYWORDS, download_hf_resource

print(HF_RESOURCE_KEYWORDS)
workflow_inputs = download_hf_resource("workflow", "e2e-files")
```

Experiment data, workflow inputs, and standard heads are stored beneath the
user-selectable asset root, which defaults to `./mmcnirs-assets` relative to the
current working directory. Hugging Face keeps small synchronization metadata
under `mmcnirs-assets/.cache/huggingface`; this is not part of the scientific
data and can be recreated.

The MMC runtime is intentionally separate from `mmcnirs-assets`. It is installed
under the platform's managed user cache:

- Linux: `$XDG_CACHE_HOME/mmcnirs`, or `~/.cache/mmcnirs`
- macOS: `~/Library/Caches/mmcnirs`
- Windows: `%LOCALAPPDATA%\mmcnirs\Cache`

Set `MMCNIRS_CACHE_DIR` to override the runtime cache location.

### Licensing

The [MIT license](LICENSE) covers the `mmcnirs` software, not the data or MMC
runtime downloaded from Hugging Face. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for a concise license map.

Every downloaded data directory includes a detailed `README.md` covering its
sources, attribution, citations, and redistribution terms. Keep that notice
with redistributed assets; the complete Hugging Face dataset is not MIT
licensed.

## Repository layout

- `mmcnirs/loaders/` contains configuration, standard-head, result, and Hugging
  Face loaders.
- `mmcnirs/light_transport/` prepares meshes, registers probes, and validates
  wavelength-specific Jacobian inputs.
- `mmcnirs/mmc/` manages the external MMC runtime and generates Jacobians.
- `mmcnirs/utils/` contains lower-level serialization and array helpers.
- `workflow_example/` contains the current end-to-end Marimo workflow.
- `mmcnirs-assets/` is the ignored, user-visible default download root created
  at runtime.

## Development

Run the test suite and lint checks with:

```bash
uv run pytest
uv run ruff check .
```
