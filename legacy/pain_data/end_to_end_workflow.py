import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from huggingface_hub import snapshot_download
    from scipy.io import loadmat

    from mmc_nirs import load_config, load_standard_head
    from mmc_nirs.light_transport import (
        prepare_jacobian_inputs,
        prepare_jacobian_mesh,
        prepare_jacobian_probe,
    )
    from mmc_nirs.mmc.jacobian import generate_jacobian
    from mmc_nirs.utils.probe_utils import load_channel_pairs_from_snirf

    return (
        Path,
        generate_jacobian,
        json,
        load_channel_pairs_from_snirf,
        load_config,
        load_standard_head,
        loadmat,
        mo,
        np,
        plt,
        prepare_jacobian_inputs,
        prepare_jacobian_mesh,
        prepare_jacobian_probe,
        snapshot_download,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # End-to-end light-transport workflow

    This notebook follows the **pain-assessment example** from common fNIRS
    input files through a prepared head mesh, a registered probe, validated
    wavelength-specific MMC inputs, and generated Jacobians. Its purpose is
    both practical and diagnostic: it demonstrates how the core
    `mmc_nirs.light_transport` functions work together and helps expose any
    information or API steps still missing from the package.

    The workflow uses a Colin27 standard head, a MATLAB `probe.SD` file,
    channel pairings from a SNIRF file, optical properties from JSON, and the
    wavelengths and output paths declared in `config.json`.

    Run it with:

    ```bash
    uv run marimo edit workflow_example/end_to_end_workflow.py
    ```
    """)
    return


@app.cell
def _(Path):
    notebook_directory = Path(__file__).resolve().parent
    return (notebook_directory,)


@app.cell
def _(Path, snapshot_download):
    def download_workflow_inputs(destination: Path, *, overwrite: bool = False) -> Path:
        """Download every file in the private e2e-files dataset subtree."""
        token_path = destination.parent / "tokens" / "HF_TOKEN.txt"
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise RuntimeError(f"Hugging Face token file not found: {token_path}") from error
        if not token:
            raise RuntimeError(f"Hugging Face token file is empty: {token_path}")

        snapshot_download(
            repo_id="nielsbracher/fnirs-simbox-assets",
            repo_type="dataset",
            allow_patterns="e2e-files/**",
            local_dir=destination,
            token=token,
            force_download=overwrite,
        )

        input_directory = destination / "e2e-files"
        required_files = {
            "FingerTapping.snirf",
            "README.md",
            "config.json",
            "optical_properties.json",
            "probe.SD",
        }
        missing_files = sorted(filename for filename in required_files if not (input_directory / filename).is_file())
        if missing_files:
            raise FileNotFoundError(f"Downloaded e2e-files directory is missing: {', '.join(missing_files)}")
        return input_directory

    return (download_workflow_inputs,)


@app.cell
def _(mo):
    refresh_workflow_inputs = mo.ui.checkbox(
        value=False,
        label="Download the workflow inputs again and overwrite the saved files",
    )
    refresh_workflow_inputs
    return (refresh_workflow_inputs,)


@app.cell
def _(download_workflow_inputs, notebook_directory, refresh_workflow_inputs):
    input_directory = download_workflow_inputs(
        notebook_directory,
        overwrite=refresh_workflow_inputs.value,
    )
    input_files = tuple(sorted(path.name for path in input_directory.iterdir() if path.is_file()))
    return input_directory, input_files


@app.cell
def _(input_directory, input_files, mo):
    downloaded_input_files = "\n".join(f"- `{filename}`" for filename in input_files)
    mo.md(
        f"""
        ## 1. Download the workflow inputs

        Saved directory: `{input_directory}`

        {downloaded_input_files}

        The entire `e2e-files` subtree is synchronized so `README.md` remains
        beside the scientific inputs. It documents their source, license,
        access date, and provenance. The repository is private for now, so the
        helper uses the same project token file as the MMC runtime.
        """
    )
    return


@app.cell
def _(input_directory, load_config):
    config_path = input_directory / "config.json"
    experiment_config = load_config(config_path)
    return config_path, experiment_config


@app.cell
def _(config_path, experiment_config, mo):
    configured_wavelengths = ", ".join(str(value) for value in experiment_config["wavelengths"])
    configured_tissues = "\n".join(
        f"- `{tissue_id}`: `{tissue_name}`"
        for tissue_id, tissue_name in experiment_config["ordered_tissues"].items()
    )
    mo.md(
        f"""
        ## 2. Load the experiment configuration

        Configuration: `{config_path}`
        Resolved experiment directory: `{experiment_config["experiment_dir"]}`
        Wavelengths: **{configured_wavelengths} nm**

        Prepared meshes, probes, and Jacobians are written beneath
        `mmcnirs_outputs/`. That directory is intentionally ignored by Git;
        deleting it resets all generated light-transport outputs while leaving
        the reproducible inputs and this notebook untouched.

        The tissue IDs are positional MMC medium IDs and must stay contiguous
        from zero:

        {configured_tissues}
        """
    )
    return


@app.cell
def _(mo):
    refresh_standard_head = mo.ui.checkbox(
        value=False,
        label="Download Colin27 again and overwrite the saved files",
    )
    refresh_standard_head
    return (refresh_standard_head,)


@app.cell
def _(load_standard_head, notebook_directory, refresh_standard_head):
    standard_head_directory = load_standard_head(
        "colin27",
        save=True,
        directory=notebook_directory,
        overwrite=refresh_standard_head.value,
    )
    standard_head_files = tuple(sorted(path.name for path in standard_head_directory.iterdir() if path.is_file()))
    return standard_head_directory, standard_head_files


@app.cell
def _(mo, standard_head_directory, standard_head_files):
    downloaded_files = "\n".join(f"- `{filename}`" for filename in standard_head_files)
    mo.md(
        f"""
        ## 3. Download the Colin27 standard head

        Saved directory: `{standard_head_directory}`

        {downloaded_files}

        `README.md` carries the source and redistribution information and must
        remain with the data. `segmentation_map.npz` is not used for light
        transport here, but is retained for the downstream SimNIRS simulator.
        """
    )
    return


@app.cell
def _(np, standard_head_directory):
    standard_mesh_path = standard_head_directory / "colin27_mesh.npz"
    orientation_path = standard_head_directory / "orientation.txt"

    with np.load(standard_mesh_path, allow_pickle=False) as standard_mesh_archive:
        nodes_with_tissue_ids = standard_mesh_archive["nodes"].copy()
        elements_with_tissue_ids = standard_mesh_archive["elem"].copy()

    if nodes_with_tissue_ids.ndim != 2 or nodes_with_tissue_ids.shape[1] < 4:
        raise ValueError("Colin27 nodes must contain coordinates followed by a node tissue ID")
    if elements_with_tissue_ids.ndim != 2 or elements_with_tissue_ids.shape[1] < 5:
        raise ValueError("Colin27 elements must contain four node indices followed by a tissue ID")

    input_nodes = nodes_with_tissue_ids[:, :3]
    input_elements = elements_with_tissue_ids[:, :4]
    input_element_tissue_ids = elements_with_tissue_ids[:, -1]
    mesh_orientation = orientation_path.read_text(encoding="utf-8").strip()
    mesh_units = "mm"
    return (
        input_element_tissue_ids,
        input_elements,
        input_nodes,
        mesh_orientation,
        mesh_units,
        standard_mesh_path,
    )


@app.cell
def _(
    input_element_tissue_ids,
    input_elements,
    input_nodes,
    mesh_orientation,
    mesh_units,
    mo,
    np,
    standard_mesh_path,
):
    input_tissue_ids = ", ".join(str(int(value)) for value in np.unique(input_element_tissue_ids))
    mo.md(
        f"""
        ## 4. Prepare the mesh

        Input archive: `{standard_mesh_path}`
        Orientation: `{mesh_orientation}`
        Units: `{mesh_units}`
        Nodes: **{len(input_nodes):,}**
        Tetrahedra: **{len(input_elements):,}**
        Element tissue IDs present: **{input_tissue_ids}**

        The source archive uses `nodes` and `elem`. Node tissue IDs are stored
        in the last node column, while the last element column contains the MMC
        tissue ID needed here. `prepare_jacobian_mesh` keeps the geometry and
        element tissue IDs, normalizes indexing and units, converts orientation
        to RAS, and attaches the configured tissue order.
        """
    )
    return


@app.cell
def _(mo):
    rebuild_prepared_inputs = mo.ui.checkbox(
        value=False,
        label="Rebuild mesh.npz and probe.npz instead of reusing compatible saved files",
    )
    rebuild_prepared_inputs
    return (rebuild_prepared_inputs,)


@app.cell
def _(
    experiment_config,
    input_element_tissue_ids,
    input_elements,
    input_nodes,
    mesh_orientation,
    mesh_units,
    prepare_jacobian_mesh,
    rebuild_prepared_inputs,
):
    prepared_mesh = prepare_jacobian_mesh(
        nodes=input_nodes,
        elements=input_elements,
        element_tissue_ids=input_element_tissue_ids,
        orientation=mesh_orientation,
        units=mesh_units,
        experiment_config=experiment_config,
        overwrite=rebuild_prepared_inputs.value,
    )
    return (prepared_mesh,)


@app.cell
def _(experiment_config, mo, np, prepared_mesh):
    prepared_mesh_path = experiment_config["experiment_dir"] / experiment_config["filepaths"]["meshfile"]
    prepared_tissues = "\n".join(
        f"- `{int(tissue_id)}`: `{tissue_name}`"
        for tissue_id, tissue_name in zip(
            prepared_mesh["ordered_tissue_ids"],
            prepared_mesh["ordered_tissues"],
            strict=True,
        )
    )
    prepared_element_ids = ", ".join(
        str(int(value)) for value in np.unique(prepared_mesh["element_tissue_ids"])
    )
    mo.md(
        f"""
        Prepared mesh: `{prepared_mesh_path}`
        Prepared keys: `{", ".join(sorted(prepared_mesh))}`
        Element tissue IDs present: **{prepared_element_ids}**

        Tissue lookup carried by the prepared mesh:

        {prepared_tissues}
        """
    )
    return


@app.cell
def _(input_directory, load_channel_pairs_from_snirf, loadmat, np):
    sd_path = input_directory / "probe.SD"
    snirf_path = input_directory / "FingerTapping.snirf"

    sd_archive = loadmat(sd_path, squeeze_me=True, struct_as_record=False)
    sd_structure = sd_archive["SD"]
    source_positions = np.asarray(sd_structure.SrcPos, dtype=float)
    detector_positions = np.asarray(sd_structure.DetPos, dtype=float)
    channel_pairings = load_channel_pairs_from_snirf(snirf_path)
    return (
        channel_pairings,
        detector_positions,
        sd_path,
        snirf_path,
        source_positions,
    )


@app.cell
def _(
    channel_pairings,
    detector_positions,
    mo,
    sd_path,
    snirf_path,
    source_positions,
):
    mo.md(f"""
    ## 5. Prepare and register the probe

    Optode coordinates: `{sd_path.name}`
    Channel pairings: `{snirf_path.name}`
    Sources: **{len(source_positions)}**
    Detectors: **{len(detector_positions)}**
    Channels: **{len(channel_pairings)}**

    The probe coordinates use millimetres and `LIA` orientation. Channels
    with registered source-detector distance at most **20 mm** are treated
    as short separation.

    These units, orientation, and short-separation rules were recovered
    from the SD file and/or the associated study documentation. There is no
    sufficiently consistent standard representation of all three pieces of
    information to infer them automatically across common fNIRS datasets.
    """)
    return


@app.cell
def _(
    channel_pairings,
    detector_positions,
    experiment_config,
    plt,
    prepare_jacobian_probe,
    prepared_mesh,
    rebuild_prepared_inputs,
    source_positions,
):
    prepared_probe = prepare_jacobian_probe(
        source_positions=source_positions,
        detector_positions=detector_positions,
        prepared_mesh=prepared_mesh,
        units="mm",
        orientation="LIA",
        channel_pairings=channel_pairings,
        short_separation_flag="distance",
        short_separation_arg=20.0,
        experiment_config=experiment_config,
        plot=rebuild_prepared_inputs.value,
        overwrite=rebuild_prepared_inputs.value,
    )
    registration_figure = plt.gcf() if rebuild_prepared_inputs.value else None
    return prepared_probe, registration_figure


@app.cell
def _(experiment_config, mo, prepared_probe, registration_figure):
    prepared_probe_path = experiment_config["experiment_dir"] / experiment_config["filepaths"]["probefile"]
    registration_summary = mo.md(
        f"""
        Prepared probe: `{prepared_probe_path}`
        Registered sources: **{len(prepared_probe["sourcepos"])}**
        Registered detectors: **{len(prepared_probe["detpos"])}**
        Short-separation channels: **{len(prepared_probe["short_separation_indices"])}**
        Long-separation channels: **{len(prepared_probe["long_separation_indices"])}**

        Enable **Rebuild mesh.npz and probe.npz** above to perform registration
        again and display its diagnostic figure.
        """
    )
    mo.vstack(
        [registration_summary, registration_figure]
        if registration_figure is not None
        else [registration_summary]
    )
    return


@app.cell
def _(experiment_config, json):
    optical_properties_path = (
        experiment_config["experiment_dir"] / experiment_config["filepaths"]["optical_properties"]
    )
    with optical_properties_path.open("r", encoding="utf-8") as optical_properties_file:
        optical_properties = json.load(optical_properties_file)

    wavelengths = tuple(experiment_config["wavelengths"])
    jacobian_filenames = tuple(experiment_config["filepaths"]["jacobians"])
    if len(wavelengths) != len(jacobian_filenames):
        raise ValueError("config wavelengths and Jacobian output paths must have the same length")
    jacobian_paths = tuple(
        experiment_config["experiment_dir"] / filename for filename in jacobian_filenames
    )
    return (
        jacobian_paths,
        optical_properties,
        optical_properties_path,
        wavelengths,
    )


@app.cell
def _(mo):
    photon_count = mo.ui.number(
        start=1,
        step=100_000,
        value=1_000_000,
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
    notebook_mmc_settings = dict(experiment_config["mmc_settings"])
    notebook_mmc_settings["nphoton"] = int(photon_count.value)
    inputs_by_wavelength = {
        wavelength: prepare_jacobian_inputs(
            prepared_mesh=prepared_mesh,
            prepared_probe=prepared_probe,
            optical_properties=optical_properties,
            mmc_settings=notebook_mmc_settings,
            wavelength=wavelength,
        )
        for wavelength in wavelengths
    }
    return inputs_by_wavelength, notebook_mmc_settings


@app.cell
def _(inputs_by_wavelength, mo, optical_properties_path, prepared_mesh):
    tissue_names = prepared_mesh["ordered_tissues"].tolist()
    property_rows = []
    for _property_wavelength, prepared_inputs in inputs_by_wavelength.items():
        for tissue_name, values in zip(tissue_names, prepared_inputs.selected_properties, strict=True):
            property_rows.append(
                {
                    "wavelength_nm": _property_wavelength,
                    "tissue": tissue_name,
                    "mua": values[0],
                    "mus": values[1],
                    "g": values[2],
                    "n": values[3],
                }
            )

    mo.vstack(
        [
            mo.md(
                f"""
                ## 6. Prepare wavelength-specific Jacobian inputs

                Optical properties: `{optical_properties_path}`

                `prepare_jacobian_inputs` validates the complete prepared mesh
                and probe, orders the optical properties using the tissue map
                stored in the prepared mesh, normalizes channel indices, and
                resolves the photon count. Inspecting these values before MMC
                makes configuration problems fail early.
                """
            ),
            mo.ui.table(property_rows, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    overwrite_jacobians = mo.ui.checkbox(
        value=True,
        label="Overwrite existing Jacobian files",
    )
    run_mmc_button = mo.ui.run_button(label="Run MMC for every configured wavelength")
    mo.vstack(
        [
            mo.md(
                """
                ## 7. Generate the Jacobians with MMC

                This is the expensive step. One forward run is performed for
                every source and one adjoint run for every detector, at each
                configured wavelength. The button prevents Marimo reactivity
                from launching those simulations unintentionally.

                The reduced photon count is suitable for workflow iteration,
                not a production-quality forward model. Very small values may
                yield no detected photons for a source-detector pair.
                """
            ),
            overwrite_jacobians,
            run_mmc_button,
        ]
    )
    return overwrite_jacobians, run_mmc_button


@app.cell
def _(
    generate_jacobian,
    jacobian_paths,
    mo,
    notebook_mmc_settings,
    optical_properties,
    overwrite_jacobians,
    prepared_mesh,
    prepared_probe,
    run_mmc_button,
    time,
    wavelengths,
):
    mo.stop(
        not run_mmc_button.value,
        mo.md("MMC has not been started. Review the prepared inputs, then use the button above."),
    )

    generated_jacobians = {}
    generation_seconds = {}
    for _run_wavelength, _run_save_path in zip(wavelengths, jacobian_paths, strict=True):
        start_time = time.perf_counter()
        generated_jacobians[_run_wavelength] = generate_jacobian(
            prepared_mesh=prepared_mesh,
            prepared_probe=prepared_probe,
            optical_properties=optical_properties,
            mmc_settings=notebook_mmc_settings,
            wavelength=_run_wavelength,
            save_path=_run_save_path,
            overwrite=overwrite_jacobians.value,
        )
        generation_seconds[_run_wavelength] = time.perf_counter() - start_time
    return generated_jacobians, generation_seconds


@app.cell
def _(
    generated_jacobians,
    generation_seconds,
    jacobian_paths,
    mo,
    np,
    wavelengths,
):
    output_rows = []
    for _result_wavelength, _result_save_path in zip(wavelengths, jacobian_paths, strict=True):
        jacobian = generated_jacobians[_result_wavelength]["J"]
        output_rows.append(
            {
                "wavelength_nm": _result_wavelength,
                "output": str(_result_save_path),
                "shape": str(jacobian.shape),
                "configured_channels": len(generated_jacobians[_result_wavelength]["channelidx"]),
                "all_finite": bool(np.all(np.isfinite(jacobian))),
                "size_mib": round(_result_save_path.stat().st_size / 1024**2, 1),
                "seconds": round(generation_seconds[_result_wavelength], 1),
            }
        )

    mo.vstack(
        [
            mo.md(
                """
                ## 8. Inspect the generated outputs

                Both configured wavelengths have completed. The saved archives
                contain the source and detector Green's functions, the combined
                Jacobian, channel indices, baseline measurements, and registered
                optode geometry used by downstream simulation code.
                """
            ),
            mo.ui.table(output_rows, selection=None, pagination=False),
        ]
    )
    return
if __name__ == "__main__":
    app.run()
