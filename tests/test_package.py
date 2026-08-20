import mmcnirs


def test_package_exports_public_loaders() -> None:
    assert callable(mmcnirs.load_config)
    assert callable(mmcnirs.load_default_config)
    assert callable(mmcnirs.load_light_transport_results)
    assert mmcnirs.__version__
