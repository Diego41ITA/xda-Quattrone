# Centralized Experiment Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make two hard-coded selectors in `main/util.py` determine the dataset, domain, feature partition, WIP strategy, paths, and optimization semantics safely for every supported experiment.

**Architecture:** `util.py` owns an immutable resolved experiment configuration plus pure helpers for feature layout, scoring, paths, polytope distances, and adaptation validation. The evaluator and planners consume explicit full-vector indices and never infer layout from prefixes or offsets. Existing anchor CSVs remain readable through validated legacy fallbacks.

**Tech Stack:** Python 3, dataclasses, pathlib, NumPy, pandas, pytest, scikit-learn, pymoo, Anchor.

**Spec:** `docs/superpowers/specs/2026-09-07-centralized-experiment-configuration-design.md`

## Global Constraints

- Routine experiment selection is controlled only by `SELECTED_DATASET` and `WIP_MODE` in `main/util.py`.
- Supported WIP modes are exactly `only_true` and `hierarchical`.
- Feature names and domains come from the selected domain's builder configuration.
- Controllable indices are derived from configured names; observable indices are their ordered complement.
- Planner logic must support non-contiguous controllable indices.
- Test labels must not influence WIP planning.
- Existing result, polytope, and user-modified files must not be moved or deleted.
- Production changes follow red-green-refactor cycles.

---

### Task 0: Python 3.12 dependency baseline

**Files:**
- Create: `tests/test_runtime_dependencies.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: a Python 3.12-installable environment containing NumPy, pandas/scikit-learn compatibility, pymoo, and the `anchor` modules imported by both polytope planners.

- [ ] **Step 1: Add an import smoke test for planner dependencies**

```python
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
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_runtime_dependencies.py -v`

Expected: failure because `pymoo` and `anchor` are absent.

- [ ] **Step 3: Replace Python-3.12-incompatible pins**

Use bounded versions with Python 3.12 wheels, replace the unused `alibi` dependency with `anchor-exp`, and retain the local MDP simulator wheel.

- [ ] **Step 4: Install and verify GREEN**

Run: `python -m pip install -r requirements.txt` followed by `python -m pytest tests/test_runtime_dependencies.py -v`.

Expected: installation succeeds and the dependency smoke test passes.

---

### Task 1: Dataset selection and resolved configuration

**Files:**
- Create: `tests/test_experiment_config.py`
- Modify: `main/util.py:19-139`

**Interfaces:**
- Produces: `SELECTED_DATASET: str`, `WIP_MODE: str`, `ExperimentConfig`, `resolve_experiment_config(dataset_name=None, wip_mode=None, builder_dir=None) -> ExperimentConfig`.
- Preserves: `get_evaluator_domain_config(...)` as a compatibility wrapper during migration.

- [ ] **Step 1: Write failing mapping and feature-layout tests**

```python
@pytest.mark.parametrize(
    ("dataset", "canonical", "domain"),
    [
        ("drive", "drive", "AutonomousDriving_v1"),
        ("drive2.csv", "drivev2", "AutonomousDriving_v1"),
        ("drive3", "drivev3", "AutonomousDriving_v1"),
        ("uavv3", "uavv3", "UAV_v02"),
        ("balanced_dataset", "balanced_dataset", "RescueRobot"),
        ("new_dataset.csv", "new_dataset", "RescueRobot"),
    ],
)
def test_resolve_dataset_mapping(dataset, canonical, domain):
    config = resolve_experiment_config(dataset, "only_true")
    assert config.dataset_name == canonical
    assert config.domain == domain


def test_autonomous_driving_partition_and_direction():
    config = resolve_experiment_config("drive", "only_true")
    assert config.controllable_feature_names == ("car_speed",)
    assert config.controllable_feature_indices == (0,)
    assert config.observable_feature_indices == (1, 2, 3, 4, 5)
    assert config.optimization_directions == (-1,)
    assert config.controllable_feature_domains == ((5.0, 50.0),)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_experiment_config.py -v`

Expected: collection/import failure because `ExperimentConfig` and `resolve_experiment_config` do not exist.

- [ ] **Step 3: Implement the immutable configuration and explicit registries**

Add dataset aliases, the complete explicit dataset-to-domain mapping, domain profiles keyed by builder folder name, and a frozen `ExperimentConfig` dataclass. Normalize optional `.csv` suffixes, load the matching builder module, derive requirements and both feature partitions, and return repository-root-absolute paths.

- [ ] **Step 4: Add failing invalid-selection tests**

```python
def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        resolve_experiment_config("balnced_dataset", "only_true")


def test_unknown_wip_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported WIP mode"):
        resolve_experiment_config("balanced_dataset", "all")
```

- [ ] **Step 5: Implement selection validation and compatibility wrapper**

Make `get_evaluator_domain_config` delegate to the new resolver where possible and return the current dictionary shape only for legacy callers. Do not retain an independent domain registry.

- [ ] **Step 6: Run configuration tests and verify GREEN**

Run: `python -m pytest tests/test_experiment_config.py -v`

Expected: all Task 1 tests pass.

---

### Task 2: Dataset schema, score, feature mutation, and polytope helpers

**Files:**
- Create: `tests/test_feature_safety.py`
- Modify: `main/util.py`

**Interfaces:**
- Produces: `validate_dataset_schema(columns, config)`, `optimization_score(sample, config) -> float`, `assemble_candidate(original, controllable_values, config) -> np.ndarray`, `validate_adaptation(original, adaptation, config)`, and `point_to_polytopes_distances(sample, polytopes, config)`.
- Consumes: `ExperimentConfig` from Task 1.

- [ ] **Step 1: Write failing schema and non-contiguous-layout tests**

```python
def test_schema_rejects_reordered_features(config):
    columns = list(config.feature_names)
    columns[0], columns[1] = columns[1], columns[0]
    with pytest.raises(ValueError, match="Dataset columns"):
        validate_dataset_schema(columns + list(config.requirement_names), config)


def test_assemble_candidate_preserves_non_controllable_positions(non_contiguous_config):
    original = np.array([10.0, 20.0, 30.0, 40.0])
    candidate = assemble_candidate(original, [11.0, 31.0], non_contiguous_config)
    np.testing.assert_array_equal(candidate, [11.0, 20.0, 31.0, 40.0])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_feature_safety.py -v`

Expected: failures because the helper functions are absent.

- [ ] **Step 3: Implement schema, partition, and candidate helpers**

Compare the complete ordered CSV schema against `feature_names + requirement_names`; construct candidates by copying the full sample and assigning only at `controllable_feature_indices`.

- [ ] **Step 4: Write failing normalized-score tests**

```python
def test_minimized_car_speed_scores_domain_minimum_highest(drive_config):
    assert optimization_score(np.array([5, 1, 1, 0, 0, 0]), drive_config) == 100
    assert optimization_score(np.array([50, 1, 1, 0, 0, 0]), drive_config) == 0


def test_mixed_directions_use_actual_domains(uav_config):
    best = np.array([1, 50, 0, 1, 0, 1000, 1])
    assert optimization_score(best, uav_config) == 300
```

- [ ] **Step 5: Implement normalized scoring**

Normalize each controllable value with its actual lower and upper builder bounds. Add the normalized percentage for direction `1` and its complement for direction `-1`.

- [ ] **Step 6: Write failing distance and adaptation-invariant tests**

```python
def test_distance_uses_explicit_non_contiguous_indices(non_contiguous_config):
    sample = np.array([5.0, 50.0, 25.0, 80.0])
    polytopes = [{"a": (0, 10, False, False), "b": (0, 40, False, False),
                  "c": (20, 30, False, False), "d": (-np.inf, np.inf, False, False)}]
    controllable, observable = point_to_polytopes_distances(sample, polytopes, non_contiguous_config)
    np.testing.assert_array_equal(controllable, [0.0])
    np.testing.assert_array_equal(observable, [100.0])


def test_adaptation_rejects_observable_mutation(drive_config):
    original = np.array([20.0, 1, 2, 0, 0, 1])
    changed = original.copy()
    changed[2] = 3
    with pytest.raises(ValueError, match="observable"):
        validate_adaptation(original, changed, drive_config)
```

- [ ] **Step 7: Implement shared distances and adaptation validation**

Keep infinite bounds infinite, compute squared point-to-interval distance by the resolved feature index, validate vector length, observable equality, controllable bounds, and integer/discrete values.

- [ ] **Step 8: Run helper tests and verify GREEN**

Run: `python -m pytest tests/test_feature_safety.py -v`

Expected: all Task 2 tests pass.

---

### Task 3: Mode-aware paths and polytope selection

**Files:**
- Create: `tests/test_polytope_policy.py`
- Modify: `main/util.py`
- Modify: `main/WIP.py:70-440`
- Modify: `main/AnchorsPlanner.py:63-350`

**Interfaces:**
- Produces: `resolve_anchor_output_paths(config, for_write=False)`, `filter_polytopes_for_mode(polytopes, mode, requirement_count)`, and WIP construction driven by `ExperimentConfig`.
- Consumes: canonical paths and WIP policy from Task 1.

- [ ] **Step 1: Write failing mode and path tests**

```python
def test_only_true_uses_requirement_count(drive_config):
    assert drive_config.wip_min_satisfied_requirements == 3


def test_hierarchical_uses_two_as_generation_floor():
    config = resolve_experiment_config("drive", "hierarchical")
    assert config.wip_min_satisfied_requirements == 2


def test_result_path_is_namespaced_by_mode():
    config = resolve_experiment_config("uavv2", "hierarchical")
    assert config.results_dir.parts[-2:] == ("WIP_hierarchical", "uavv2")
```

- [ ] **Step 2: Run policy tests and verify RED**

Run: `python -m pytest tests/test_polytope_policy.py -v`

Expected: failures because mode-derived policies and canonical paths are incomplete.

- [ ] **Step 3: Implement canonical write paths and validated legacy read fallbacks**

Use `main/polytopes/WIP_<mode>/<dataset>` for writes and search canonical then domain-specific legacy directories for reads. Validate that a discovered CSV contains `n_satisfied_reqs` and every configured feature column before returning it.

- [ ] **Step 4: Write failing polytope-filter tests**

```python
def test_only_true_rejects_partial_polytopes():
    levels = [{"n_satisfied_reqs": 2}, {"n_satisfied_reqs": 3}]
    assert [p["n_satisfied_reqs"] for p in filter_polytopes_for_mode(levels, "only_true", 3)] == [3]


def test_hierarchical_preserves_all_eligible_levels():
    levels = [{"n_satisfied_reqs": 1}, {"n_satisfied_reqs": 2}, {"n_satisfied_reqs": 3}]
    assert [p["n_satisfied_reqs"] for p in filter_polytopes_for_mode(levels, "hierarchical", 3)] == [2, 3]
```

- [ ] **Step 5: Implement WIP generation/loading policy**

Replace `K = 12` and the runtime `nreqs` parameter with mode-derived configuration. Generate rules from predictions only, preserve `n_satisfied_reqs`, use `ast.literal_eval`-compatible interval parsing instead of unrestricted `eval`, and handle an empty set by using the PDP fallback.

- [ ] **Step 6: Make Anchors load fully positive polytopes through the same resolver**

Anchors remains the all-requirements baseline regardless of WIP mode, but it uses the selected dataset's validated anchor source and the common filtering helper.

- [ ] **Step 7: Run policy tests and verify GREEN**

Run: `python -m pytest tests/test_polytope_policy.py -v`

Expected: all Task 3 tests pass.

---

### Task 4: Refactor every planner to preserve observable features

**Files:**
- Create: `tests/test_planner_feature_layout.py`
- Modify: `main/WIP.py:630-1250`
- Modify: `main/AnchorsPlanner.py:493-890`
- Modify: `main/CustomPlanner.py:119-323`
- Modify: `main/NSGA3Planner.py:10-92`

**Interfaces:**
- Consumes: `ExperimentConfig`, `assemble_candidate`, `point_to_polytopes_distances`, and `validate_adaptation`.
- Produces: planners that accept and return full feature vectors with unchanged observable positions.

- [ ] **Step 1: Write failing projection and NSGA assembly tests**

```python
def test_polytope_projection_changes_only_configured_controllables(planner, config):
    original = np.array([20.0, 2.0, 3.0, 0, 1, 2])
    projected = planner.go_inside_CF_given_polytope(original.copy(), polytope)
    np.testing.assert_array_equal(projected[list(config.observable_feature_indices)],
                                  original[list(config.observable_feature_indices)])


def test_nsga_full_vector_preserves_observables(problem, config):
    candidate = problem.build_full_candidates(np.array([[10.0]]))
    np.testing.assert_array_equal(candidate[0, list(config.observable_feature_indices)],
                                  problem.external_sample[list(config.observable_feature_indices)])
```

- [ ] **Step 2: Run planner-layout tests and verify RED**

Run: `python -m pytest tests/test_planner_feature_layout.py -v`

Expected: WIP/Anchors fail on hard-coded offsets and NSGA lacks a reusable full-vector assembly interface.

- [ ] **Step 3: Refactor WIP and Anchors distance/projection/local search**

Delegate distance calculations to the shared helper. Replace every `sample[i]`, `i + 2`, and `i + 3` layout assumption with resolved indices. Intersect polytope bounds with each controllable feature's actual domain and make local deltas operate on full-vector positions.

- [ ] **Step 4: Refactor CustomPlanner candidate combinations**

Replace concatenation of controllable possibilities and external values with full-sample copying plus explicit assignments. Ensure KNN/PDP lookups retain the dataset's original feature order.

- [ ] **Step 5: Refactor NSGA-III full candidate assembly**

Store the original full sample for each optimization, evolve only controllable variables, and reconstruct each evaluated candidate by assigning at explicit indices. Remove `externalFeatures` concatenation and make `findAdaptation` accept a full sample.

- [ ] **Step 6: Run planner-layout tests and verify GREEN**

Run: `python -m pytest tests/test_planner_feature_layout.py -v`

Expected: all Task 4 tests pass, including a non-contiguous synthetic partition.

---

### Task 5: Migrate evaluator and analyzer to the central configuration

**Files:**
- Create: `tests/test_evaluator_configuration.py`
- Modify: `main/evaluator.py:1-320`
- Modify: `main/polytopes/AnchorsAnalyzer.py:135-180`
- Modify: `main/util.py:139-end`

**Interfaces:**
- Consumes: `resolve_experiment_config()`, `optimization_score`, and `validate_adaptation`.
- Produces: one supported evaluator flow with no local dataset/domain/layout constants.

- [ ] **Step 1: Write failing source-level regression tests**

```python
def test_evaluator_has_no_ground_truth_planner_threshold():
    source = Path("main/evaluator.py").read_text(encoding="utf-8")
    assert "num_reqs_satisfied = 4" not in source
    assert "evaluate_sample(row, num_reqs_satisfied)" not in source


def test_evaluator_does_not_slice_first_n_features_for_nsga():
    source = Path("main/evaluator.py").read_text(encoding="utf-8")
    assert "row[n_controllableFeatures:]" not in source
```

- [ ] **Step 2: Run evaluator regression tests and verify RED**

Run: `python -m pytest tests/test_evaluator_configuration.py -v`

Expected: assertions fail on the current hard-coded threshold and first-`N` slicing.

- [ ] **Step 3: Migrate evaluator construction and execution**

Load the active config once, validate the real CSV before splitting, use builder domains, call the shared score, pass full samples to every planner, validate each returned adaptation, and format controllable values by explicit indices.

- [ ] **Step 4: Align builder evaluation and AnchorsAnalyzer**

Pass the selected domain/config to builder evaluation functions without relying on the root `MDP_Dataset_Builder/config.py`. Make AnchorsAnalyzer resolve its dataset, features, partition, and files through the central configuration.

- [ ] **Step 5: Run evaluator tests and verify GREEN**

Run: `python -m pytest tests/test_evaluator_configuration.py -v`

Expected: all Task 5 tests pass.

---

### Task 6: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-09-07-centralized-experiment-configuration.md`
- Test: `tests/test_experiment_config.py`
- Test: `tests/test_feature_safety.py`
- Test: `tests/test_polytope_policy.py`
- Test: `tests/test_planner_feature_layout.py`
- Test: `tests/test_evaluator_configuration.py`

**Interfaces:**
- Documents: the final selectors, supported mappings, WIP semantics, output paths, and validation failures.

- [ ] **Step 1: Update README usage instructions**

Document editing `SELECTED_DATASET` and `WIP_MODE`, list aliases and domain mappings, state that observables are immutable, and show canonical result/polytope paths.

- [ ] **Step 2: Run the focused suite**

Run: `python -m pytest tests/test_experiment_config.py tests/test_feature_safety.py tests/test_polytope_policy.py tests/test_planner_feature_layout.py tests/test_evaluator_configuration.py -v`

Expected: all focused tests pass without warnings originating from the new code.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -v`

Expected: all tests pass.

- [ ] **Step 4: Compile the modified Python modules**

Run: `python -m py_compile main/util.py main/evaluator.py main/WIP.py main/AnchorsPlanner.py main/CustomPlanner.py main/NSGA3Planner.py main/polytopes/AnchorsAnalyzer.py`

Expected: exit code zero and no output.

- [ ] **Step 5: Run real-dataset configuration smoke checks**

Run a short script that calls `resolve_experiment_config` for every canonical dataset and validates only its CSV header. It must not train models or invoke the MDP simulator.

Expected: every existing CSV matches its inferred domain, features, and requirements.

- [ ] **Step 6: Inspect the final diff and repository status**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; pre-existing user changes remain untouched and are reported separately from implementation files.
