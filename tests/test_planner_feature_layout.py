import numpy as np

from AnchorsPlanner import AnchorsPlanner
from NSGA3Planner import Adaptation


def test_nsga_builds_full_candidates_with_non_contiguous_controllables():
    problem = Adaptation(
        models=[],
        targetConfidence=np.array([]),
        popSize=2,
        controllableFeatureIndices=np.array([0, 2]),
        controllableFeatureDomains=np.array([[0, 10], [20, 30]]),
        optimizationDirections=np.array([1, -1]),
    )
    original = np.array([5.0, 50.0, 25.0, 80.0])
    problem.original_sample = original

    candidates = problem.build_full_candidates(np.array([[6.0, 21.0], [7.0, 22.0]]))

    np.testing.assert_array_equal(candidates[:, [0, 2]], [[6.0, 21.0], [7.0, 22.0]])
    np.testing.assert_array_equal(candidates[:, [1, 3]], [[50.0, 80.0], [50.0, 80.0]])


def test_nsga_rounds_discrete_controllable_values():
    problem = Adaptation(
        models=[],
        targetConfidence=np.array([]),
        popSize=1,
        controllableFeatureIndices=np.array([0, 1]),
        controllableFeatureDomains=np.array([[0, 1], [5, 50]]),
        optimizationDirections=np.array([1, 1]),
        controllableFeatureTypes=(int, float),
    )
    problem.original_sample = np.array([0.0, 20.0, 7.0])

    candidate = problem.build_full_candidates(np.array([[0.6, 21.25]]))

    np.testing.assert_array_equal(candidate, [[1.0, 21.25, 7.0]])


def test_anchors_returns_no_adaptation_when_no_full_success_polytope_exists():
    planner = AnchorsPlanner.__new__(AnchorsPlanner)
    planner.explanations = []

    assert planner.evaluate_sample(np.array([1.0, 2.0])) == (None, None, None, 0)
