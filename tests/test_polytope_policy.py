from pathlib import Path

from util import (
    filter_polytopes_for_mode,
    resolve_anchor_output_paths,
    resolve_experiment_config,
)


def test_only_true_rejects_partial_polytopes():
    levels = [{"n_satisfied_reqs": 2}, {"n_satisfied_reqs": 3}]
    selected = filter_polytopes_for_mode(levels, "only_true", 3)
    assert [polytope["n_satisfied_reqs"] for polytope in selected] == [3]


def test_hierarchical_preserves_all_eligible_levels():
    levels = [
        {"n_satisfied_reqs": 1},
        {"n_satisfied_reqs": 2},
        {"n_satisfied_reqs": 3},
    ]
    selected = filter_polytopes_for_mode(levels, "hierarchical", 3)
    assert [polytope["n_satisfied_reqs"] for polytope in selected] == [2, 3]


def test_anchor_write_paths_are_dataset_and_mode_specific():
    config = resolve_experiment_config("uavv2", "hierarchical")
    explanations, rules = resolve_anchor_output_paths(config, for_write=True)

    assert explanations == config.anchor_dir / "anchors_explanations.csv"
    assert rules == config.anchor_dir / "anchors_rules.csv"
    assert explanations.parts[-4:] == (
        "polytopes",
        "WIP_hierarchical",
        "uavv2",
        "anchors_explanations.csv",
    )


def test_missing_anchor_read_source_returns_canonical_paths(tmp_path):
    config = resolve_experiment_config("drive", "only_true", builder_dir=Path("MDP_Dataset_Builder"))
    explanations, rules = resolve_anchor_output_paths(config, for_write=False)
    assert explanations == config.anchor_explanations_path
    assert rules == config.anchor_rules_path
