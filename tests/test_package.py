import mmc_nirs


def test_package_exports_public_loaders() -> None:
    assert callable(mmc_nirs.load_config)
    assert callable(mmc_nirs.load_default_config)
    assert callable(mmc_nirs.load_light_transport_results)
    assert mmc_nirs.__version__
