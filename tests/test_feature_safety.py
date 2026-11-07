from dataclasses import replace

import numpy as np
import pytest

from util import (
    assemble_candidate,
    optimization_score,
    point_to_polytopes_distances,
    resolve_experiment_config,
    validate_adaptation,
    validate_dataset_schema,
)


@pytest.fixture
def drive_config():
    return resolve_experiment_config("drive", "only_true")


@pytest.fixture
def non_contiguous_config(drive_config):
    return replace(
        drive_config,
        feature_names=("a", "b", "c", "d"),
        requirement_names=("req_0",),
        controllable_feature_names=("a", "c"),
        controllable_feature_indices=(0, 2),
        observable_feature_names=("b", "d"),
        observable_feature_indices=(1, 3),
        controllable_feature_domains=((0.0, 10.0), (20.0, 30.0)),
        controllable_feature_types=(float, float),
        optimization_directions=(1, -1),
    )


def test_schema_rejects_reordered_features(drive_config):
    columns = list(drive_config.feature_names)
    columns[0], columns[1] = columns[1], columns[0]
    with pytest.raises(ValueError, match="Dataset columns"):
        validate_dataset_schema(columns + list(drive_config.requirement_names), drive_config)


def test_assemble_candidate_preserves_non_controllable_positions(non_contiguous_config):
    original = np.array([10.0, 20.0, 30.0, 40.0])
    candidate = assemble_candidate(original, [11.0, 31.0], non_contiguous_config)
    np.testing.assert_array_equal(candidate, [11.0, 20.0, 31.0, 40.0])


def test_minimized_car_speed_scores_domain_minimum_highest(drive_config):
    assert optimization_score(np.array([5, 1, 1, 0, 0, 0]), drive_config) == 100
    assert optimization_score(np.array([50, 1, 1, 0, 0, 0]), drive_config) == 0


def test_mixed_directions_use_actual_domains():
    config = resolve_experiment_config("uav", "only_true")
    best = np.array([1, 50, 0, 1, 0, 1000, 1])
    assert optimization_score(best, config) == 300


def test_distance_uses_explicit_non_contiguous_indices(non_contiguous_config):
    sample = np.array([5.0, 50.0, 25.0, 80.0])
    polytopes = [
        {
            "a": (0, 10, False, False),
            "b": (0, 40, False, False),
            "c": (20, 30, False, False),
            "d": (-np.inf, np.inf, False, False),
        }
    ]
    controllable, observable = point_to_polytopes_distances(
        sample, polytopes, non_contiguous_config
    )
    np.testing.assert_array_equal(controllable, [0.0])
    np.testing.assert_array_equal(observable, [100.0])


def test_adaptation_rejects_observable_mutation(drive_config):
    original = np.array([20.0, 1, 2, 0, 0, 1])
    changed = original.copy()
    changed[2] = 3
    with pytest.raises(ValueError, match="observable"):
        validate_adaptation(original, changed, drive_config)


def test_adaptation_rejects_non_discrete_value():
    config = resolve_experiment_config("uav", "only_true")
    original = np.array([0, 20.0, 1, 2, 12, 5000.0, 4])
    changed = original.copy()
    changed[0] = 0.5
    with pytest.raises(ValueError, match="discrete"):
        validate_adaptation(original, changed, config)
