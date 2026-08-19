# Third-Party Notices

The [MIT license](LICENSE) covers the `mmc-nirs` software and integration code.
It does not relicense the external data or MMC runtime described below.

## Public Hugging Face assets

`mmc-nirs` downloads optional resources from the public
[`nielsbracher/fnirs-simbox-assets`](https://huggingface.co/datasets/nielsbracher/fnirs-simbox-assets)
dataset. These resources are not included in the Python package and do not have
one common license:

| Resource | License summary |
|---|---|
| Default experiments | Mixed Colin27-derived, AAL3v1-derived, CC0-sourced, and project-generated content. |
| Colin27 standard head | Colin27-derived geometry and an AAL3v1-derived segmentation; the exact GPL version is not established by the AAL3v1 source archive used. |
| End-to-end inputs | The OpenNeuro `ds005776` version `1.0.1` source data are CC0 1.0; the folder notice defines the exact scope. |

Detailed provenance, attribution, citation, and redistribution information is
stored beside the data:

- [`experiments/pain/README.md`](https://huggingface.co/datasets/nielsbracher/fnirs-simbox-assets/blob/main/experiments/pain/README.md)
- [`experiments/pattern_cutting/README.md`](https://huggingface.co/datasets/nielsbracher/fnirs-simbox-assets/blob/main/experiments/pattern_cutting/README.md)
- [`standard-heads/colin27/README.md`](https://huggingface.co/datasets/nielsbracher/fnirs-simbox-assets/blob/main/standard-heads/colin27/README.md)
- [`e2e-files/README.md`](https://huggingface.co/datasets/nielsbracher/fnirs-simbox-assets/blob/main/e2e-files/README.md)

The loader keeps each folder's `README.md` with its files. Preserve that notice
and comply with the applicable upstream terms when redistributing an asset.

## Mesh-based Monte Carlo (MMC)

[MMC](https://github.com/fangq/mmc) is an optional external program downloaded
on demand from the Hugging Face dataset. It is not included in the `mmc-nirs`
Python package.

Copyright (C) 2010–2025 Qianqian Fang.

MMC is licensed under the GNU General Public License, version 3 or later. Each
runtime archive includes the upstream `LICENSE.txt`; the license is also
available in the [MMC repository](https://github.com/fangq/mmc/blob/master/LICENSE.txt).
Anyone redistributing MMC must comply with those terms.

These summaries are provided for provenance and orientation, not as legal
advice. Consult the accompanying notices and upstream licenses for the files
you use or redistribute.
