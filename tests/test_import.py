def test_import():
    import medvis

    assert hasattr(medvis, "__version__")
