import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from scipy.io import loadmat

    from mmc_nirs import load_config
    from mmc_nirs.light_transport import (
        prepare_jacobian_inputs,
        prepare_mesh,
        prepare_probe,
    )
    from mmc_nirs.mmc.jacobian import generate_jacobian
    from mmc_nirs.utils.probe_utils import load_channel_pairs_from_snirf

    return (
        Path,
        generate_jacobian,
        json,
        load_channel_pairs_from_snirf,
        load_config,
        loadmat,
        mo,
        np,
        prepare_jacobian_inputs,
        prepare_mesh,
        prepare_probe,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Pattern-cutting light-transport workflow
    """)
    return


@app.cell
def _(Path, load_config):
    data_directory = Path(__file__).resolve().parent
    experiment_config = load_config(data_directory / "config.json")
    return data_directory, experiment_config


@app.cell
def _(data_directory, np):
    with np.load(data_directory / "input_mesh.npz", allow_pickle=False) as mesh_archive:
        mesh_nodes = mesh_archive["nodes"][:, :3].copy()
        mesh_elements = mesh_archive["elem"][:, :4].copy()
        mesh_tissue_ids = mesh_archive["elem"][:, -1].copy()
    return mesh_elements, mesh_nodes, mesh_tissue_ids


@app.cell
def _(mo):
    rebuild_inputs = mo.ui.checkbox(value=True, label="Rebuild prepared mesh and probe")
    rebuild_inputs
    return (rebuild_inputs,)


@app.cell
def _(
    experiment_config,
    mesh_elements,
    mesh_nodes,
    mesh_tissue_ids,
    prepare_mesh,
    rebuild_inputs,
):
    prepared_mesh = prepare_mesh(
        nodes=mesh_nodes,
        elements=mesh_elements,
        element_tissue_ids=mesh_tissue_ids,
        experiment_config=experiment_config,
        overwrite=rebuild_inputs.value,
    )
    return (prepared_mesh,)


@app.cell
def _(data_directory, load_channel_pairs_from_snirf, loadmat, np):
    probe_archive = loadmat(data_directory / "probe.mat", squeeze_me=True, struct_as_record=False)
    source_positions = np.asarray(probe_archive["sourcepos"], dtype=float)
    detector_positions = np.asarray(probe_archive["detpos"], dtype=float)
    channel_pairings = load_channel_pairs_from_snirf(data_directory / "NIRS-2019-08-10_006.snirf")
    return channel_pairings, detector_positions, source_positions


@app.cell
def _(
    channel_pairings,
    detector_positions,
    experiment_config,
    prepare_probe,
    prepared_mesh,
    rebuild_inputs,
    source_positions,
):
    prepared_probe = prepare_probe(
        source_positions=source_positions,
        detector_positions=detector_positions,
        prepared_mesh=prepared_mesh,
        channel_pairings=channel_pairings,
        experiment_config=experiment_config,
        plot=True,
        overwrite=rebuild_inputs.value,
    )
    return (prepared_probe,)


@app.cell
def _(experiment_config, json):
    optical_properties_path = (
        experiment_config["experiment_dir"] / experiment_config["filepaths"]["optical_properties"]
    )
    with optical_properties_path.open("r", encoding="utf-8") as optical_properties_file:
        optical_properties = json.load(optical_properties_file)

    wavelengths = tuple(experiment_config["wavelengths"])
    jacobian_paths = tuple(
        experiment_config["experiment_dir"] / filename
        for filename in experiment_config["filepaths"]["jacobians"]
    )
    if len(wavelengths) != len(jacobian_paths):
        raise ValueError("config wavelengths and Jacobian output paths must have the same length")
    return jacobian_paths, optical_properties, wavelengths


@app.cell
def _(mo):
    photon_count = mo.ui.number(
        start=1,
        step=100_000,
        value=5e8,
        label="Photons per MMC run",
    )
    photon_count
    return (photon_count,)


@app.cell
def _(
    experiment_config,
    optical_properties,
    photon_count,
    prepare_jacobian_inputs,
    prepared_mesh,
    prepared_probe,
    wavelengths,
):
    mmc_settings = dict(experiment_config["mmc_settings"])
    mmc_settings["nphoton"] = int(photon_count.value)
    jacobian_inputs = {
        wavelength: prepare_jacobian_inputs(
            prepared_mesh=prepared_mesh,
            prepared_probe=prepared_probe,
            optical_properties=optical_properties,
            mmc_settings=mmc_settings,
            wavelength=wavelength,
        )
        for wavelength in wavelengths
    }
    return jacobian_inputs, mmc_settings


@app.cell
def _(jacobian_inputs, mo, prepared_mesh, prepared_probe):
    input_rows = [
        {
            "wavelength_nm": wavelength,
            "photons": inputs.photon_count,
            "tissues": len(inputs.selected_properties),
        }
        for wavelength, inputs in jacobian_inputs.items()
    ]
    mo.vstack(
        [
            mo.md(
                f"Prepared **{len(prepared_mesh['nodes']):,} nodes**, "
                f"**{len(prepared_mesh['elements']):,} elements**, "
                f"**{len(prepared_probe['sourcepos'])} sources**, "
                f"**{len(prepared_probe['detpos'])} detectors**, and "
                f"**{len(prepared_probe['channel_pairings'])} channels**."
            ),
            mo.ui.table(input_rows, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    overwrite_jacobians = mo.ui.checkbox(value=True, label="Overwrite existing Jacobians")
    run_mmc = mo.ui.run_button(label="Run MMC")
    mo.hstack([overwrite_jacobians, run_mmc])
    return overwrite_jacobians, run_mmc


@app.cell
def _(
    generate_jacobian,
    jacobian_paths,
    mmc_settings,
    mo,
    optical_properties,
    overwrite_jacobians,
    prepared_mesh,
    prepared_probe,
    run_mmc,
    time,
    wavelengths,
):
    mo.stop(not run_mmc.value, mo.md("MMC has not been run."))

    jacobians = {}
    runtimes = {}
    for _wavelength, _save_path in zip(wavelengths, jacobian_paths, strict=True):
        _start = time.perf_counter()
        jacobians[_wavelength] = generate_jacobian(
            prepared_mesh=prepared_mesh,
            prepared_probe=prepared_probe,
            optical_properties=optical_properties,
            mmc_settings=mmc_settings,
            wavelength=_wavelength,
            save_path=_save_path,
            overwrite=overwrite_jacobians.value,
        )
        runtimes[_wavelength] = time.perf_counter() - _start
    return jacobians, runtimes


@app.cell
def _(jacobian_paths, jacobians, mo, np, runtimes, wavelengths):
    result_rows = []
    for _wavelength, _save_path in zip(wavelengths, jacobian_paths, strict=True):
        _jacobian = jacobians[_wavelength]["J"]
        result_rows.append(
            {
                "wavelength_nm": _wavelength,
                "shape": str(_jacobian.shape),
                "all_finite": bool(np.isfinite(_jacobian).all()),
                "seconds": round(runtimes[_wavelength], 1),
                "output": str(_save_path),
            }
        )
    mo.ui.table(result_rows, selection=None, pagination=False)
    return
if __name__ == "__main__":
    app.run()
