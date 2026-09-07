def test_planner_runtime_dependencies_import():
    import numpy
    import sklearn
    import pymoo
    from anchor import anchor_tabular, utils

    assert numpy.__version__
    assert sklearn.__version__
    assert pymoo.__version__
    assert anchor_tabular is not None
    assert utils is not None
