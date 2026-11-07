from pathlib import Path

import pytest

from util import resolve_experiment_config, validate_dataset_schema


@pytest.mark.parametrize(
    ("dataset", "canonical", "domain"),
    [
        ("drive", "drive", "AutonomousDriving_v1"),
        ("drive2.csv", "drivev2", "AutonomousDriving_v1"),
        ("drive3", "drivev3", "AutonomousDriving_v1"),
        ("uav", "uav", "UAV_v02"),
        ("uavv2.csv", "uavv2", "UAV_v02"),
        ("uavv3", "uavv3", "UAV_v02"),
        ("balanced_dataset", "balanced_dataset", "RescueRobot"),
        ("new_dataset.csv", "new_dataset", "RescueRobot"),
    ],
)
def test_dataset_selection_derives_domain(dataset, canonical, domain):
    config = resolve_experiment_config(dataset, "only_true")

    assert config.dataset_name == canonical
    assert config.domain == domain
    assert config.dataset_path == config.project_root / "datasets" / f"{canonical}.csv"


def test_autonomous_driving_has_only_car_speed_controllable():
    config = resolve_experiment_config("drive", "only_true")

    assert config.controllable_feature_names == ("car_speed",)
    assert config.controllable_feature_indices == (0,)
    assert config.observable_feature_names == (
        "p_x",
        "p_y",
        "orientation",
        "weather",
        "road_shape",
    )
    assert config.observable_feature_indices == (1, 2, 3, 4, 5)
    assert config.controllable_feature_domains == ((5.0, 50.0),)
    assert config.optimization_directions == (-1,)


def test_uav_uses_builder_domains_instead_of_universal_zero_to_hundred():
    config = resolve_experiment_config("uavv3", "hierarchical")

    assert config.controllable_feature_indices == (0, 1, 2)
    assert config.observable_feature_indices == (3, 4, 5, 6)
    assert config.controllable_feature_domains == ((0, 1), (5.0, 50.0), (0, 1))
    assert config.feature_domains[5] == (1000.0, 40000.0)
    assert config.feature_types[6] is int
    assert config.optimization_directions == (1, 1, -1)


def test_rescue_robot_partition_covers_every_feature_once():
    config = resolve_experiment_config("dataset15000_balanced_joint", "only_true")

    assert config.controllable_feature_indices == (0, 1, 2, 3)
    assert config.observable_feature_indices == (4, 5, 6, 7, 8)
    assert set(config.controllable_feature_indices).isdisjoint(config.observable_feature_indices)
    assert sorted(config.controllable_feature_indices + config.observable_feature_indices) == list(
        range(len(config.feature_names))
    )


def test_wip_mode_controls_threshold_and_result_namespace():
    only_true = resolve_experiment_config("drive", "only_true")
    hierarchical = resolve_experiment_config("drive", "hierarchical")

    assert only_true.wip_min_satisfied_requirements == 3
    assert hierarchical.wip_min_satisfied_requirements == 2
    assert only_true.results_dir.parts[-2:] == ("WIP_only_true", "drive")
    assert hierarchical.results_dir.parts[-2:] == ("WIP_hierarchical", "drive")


def test_unknown_dataset_is_rejected_without_domain_fallback():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        resolve_experiment_config("balnced_dataset", "only_true")


def test_unknown_wip_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported WIP mode"):
        resolve_experiment_config("balanced_dataset", "all")


@pytest.mark.parametrize(
    "dataset",
    [
        "drive",
        "drivev2",
        "drivev3",
        "uav",
        "uavv2",
        "uavv3",
        "balanced_dataset",
        "balanced_dataset2",
        "dataset5000",
        "dataset5000_balanced_joint",
        "dataset15000_generated",
        "dataset15000_balanced_joint",
        "new_dataset",
    ],
)
def test_every_registered_dataset_matches_derived_schema(dataset):
    config = resolve_experiment_config(dataset, "only_true")
    header = Path(config.dataset_path).read_text(encoding="utf-8").splitlines()[0].split(",")
    validate_dataset_schema(header, config)
