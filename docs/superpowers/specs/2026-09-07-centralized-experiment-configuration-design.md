# Centralized Experiment Configuration Design

## Objective

Make `main/util.py` the single control point for selecting an experiment dataset and the WIP polytope strategy. Every other domain-specific value must be derived from those two hard-coded choices, validated before planner construction, and consumed without positional assumptions about controllable or observable features.

## User-facing configuration

`main/util.py` exposes exactly two values intended for routine manual editing:

```python
SELECTED_DATASET = "balanced_dataset"
WIP_MODE = "only_true"  # "only_true" or "hierarchical"
```

Dataset names may be written with or without the `.csv` suffix. The aliases `drive2` and `drive3` resolve to the existing files `drivev2.csv` and `drivev3.csv`. Unsupported dataset names and WIP modes fail immediately with an error that lists the supported values.

The evaluator does not provide command-line or local overrides for these settings. This prevents two competing sources of experiment configuration.

## Dataset-to-domain mapping

The selected dataset determines the MDP domain and builder configuration:

| Canonical datasets | Domain |
| --- | --- |
| `drive`, `drivev2`, `drivev3` | `AutonomousDriving_v1` |
| `uav`, `uavv2`, `uavv3` | `UAV_v02` |
| `balanced_dataset`, `balanced_dataset2`, `dataset5000`, `dataset5000_balanced_joint`, `dataset15000_generated`, `dataset15000_balanced_joint`, `new_dataset` | `RescueRobot` |

The registry is explicit so a misspelled or newly added dataset cannot silently use the wrong domain. Adding a dataset requires adding it to the registry or alias map.

## Domain profiles

Each domain profile declares controllable feature names and optimization directions. Feature order, feature types, numeric domains, and requirement count are loaded from that domain's `MDP_Dataset_Builder/<domain>/config.py`.

### RescueRobot

- Controllable features: `cruise speed`, `image resolution`, `illuminance`, `controls responsiveness`
- Observable features: `power`, `smoke intensity`, `obstacle size`, `obstacle distance`, `firm obstacle`
- Optimization directions: `[1, -1, -1, -1]`

### UAV_v02

- Controllable features: `formation`, `flying_speed`, `countermeasure`
- Observable features: `weather`, `day_time`, `threat_range`, `#threats`
- Optimization directions: `[1, 1, -1]`

### AutonomousDriving_v1

- Controllable feature: `car_speed`
- Observable features: `p_x`, `p_y`, `orientation`, `weather`, `road_shape`
- Optimization direction: `[-1]`

For AutonomousDriving, speed is minimized subject to satisfying every requirement. Thus the classifiers determine the feasible speed range and the optimization score selects the lowest feasible speed.

## Derived experiment configuration

A single configuration builder in `util.py` resolves the active selection into one validated configuration object. It includes at least:

- canonical dataset name and absolute dataset path;
- domain name and builder configuration path;
- ordered feature and requirement names;
- ordered controllable and observable feature names;
- controllable and observable feature indices;
- controllable feature domains and types;
- optimization directions;
- target confidence and anchor confidence;
- WIP mode and minimum satisfied-requirement threshold;
- anchor/rules paths, results directory, and explainability directory.

Callers receive this resolved object and do not rebuild domain metadata locally.

## Feature-layout invariants

Feature identity is always name-based at configuration boundaries and index-based only after validation. The following invariants are checked before training or planner construction:

1. Dataset feature columns exactly equal the builder's `SS_VARIABLES` keys in the same order.
2. Dataset requirement columns exactly equal `req_0` through `req_N`, where `N + 1` is the number of builder constraints.
3. Every configured controllable feature exists exactly once in `SS_VARIABLES`.
4. Controllable indices are derived from names rather than hand-written.
5. Observable indices are the ordered complement of controllable indices.
6. Controllable and observable index sets are disjoint and their union covers every feature.
7. Optimization directions contain only `1` or `-1` and match the controllable-feature count.
8. Every controllable domain comes from the matching builder `SS_VARIABLES` entry and has a valid lower and upper bound.
9. Every candidate adaptation has the expected full feature-vector length.
10. Observable values are unchanged after every planner adaptation, allowing only an explicit numeric equality tolerance.

No planner may assume that controllable features occupy the first positions or are contiguous. Display, clamping, scoring, distance calculation, candidate assembly, and dataset export all use the resolved indices.

## WIP modes

### `only_true`

- Anchor generation selects only samples predicted to satisfy every requirement.
- Anchor loading retains only polytopes whose `n_satisfied_reqs` equals the total requirement count.
- The WIP search operates only on those fully positive polytopes.

### `hierarchical`

- Anchor generation selects samples satisfying at least two requirements, capped safely when a domain has fewer than two requirements.
- Each polytope retains its `n_satisfied_reqs` level.
- Search examines levels from highest to lowest and stops at the first level containing a polytope compatible with the sample's observable projection.
- The evaluator never derives the WIP threshold from `y_test`; test labels are used only for post-hoc evaluation.

The selected mode determines both the behavior and output namespace, preventing results from different strategies from being mixed.

## Paths and compatibility

Canonical output paths are repository-root-relative and independent of the process working directory:

```text
results/WIP_only_true/<dataset>/
results/WIP_hierarchical/<dataset>/
main/polytopes/WIP_only_true/<dataset>/
main/polytopes/WIP_hierarchical/<dataset>/
```

Newly generated anchors are written only to the canonical path. When loading existing anchors, the resolver first checks the canonical location and may fall back to the legacy domain directories `main/polytopes/rescue_robot`, `main/polytopes/uav`, and `main/polytopes/uavv2`. A fallback is accepted only after its CSV schema and feature names pass validation. Existing files are not moved, renamed, or deleted.

## Shared point-to-polytope distance

The duplicate WIP and Anchors distance implementations are replaced by one shared helper. It receives:

- a full sample vector;
- the ordered feature names;
- explicit controllable and observable indices;
- interval-valued polytopes.

For each feature, distance is zero inside its interval and the squared distance to the closest bound outside it. Infinite bounds remain mathematically unbounded; they are not replaced by `0` or `100`. Controllable and observable distances are returned separately.

WIP adds hierarchical level selection around this shared calculation. Anchors applies the same calculation to its fully positive polytope set.

## Adaptation safety

All planner mutations target `sample[controllable_index]`. Projection into a polytope uses the builder-provided domain and respects open or closed interval bounds. Integer controllable variables are coerced according to their builder type before evaluation.

NSGA-III reconstructs the full candidate vector by assigning evolved values to controllable indices and copying every observable index from the original sample. CustomPlanner and the WIP PDP fallback likewise construct combinations by assigning through controllable indices rather than appending observable values.

After a planner returns, the evaluator validates that:

- the vector has the expected length;
- all controllable values are within their configured domains;
- all observable values equal the starting sample values;
- discrete and integer variables satisfy their type/domain constraints.

Invalid adaptations fail the experiment row with a descriptive error instead of being silently clamped or exported.

## Optimization score

The shared optimization score normalizes each controllable feature to `[0, 100]` using its actual builder domain:

- direction `1`: normalized value is the contribution;
- direction `-1`: `100 - normalized value` is the contribution.

The total score ranges from zero to `100 * number_of_controllable_features`. This is consistent with NSGA-III's objective directions and works for domains such as AutonomousDriving `[5, 50]` and UAV categorical/bounded variables.

## Evaluator changes

`main/evaluator.py` requests the active configuration from `util.py` and contains no dataset or domain selection. It removes:

- the hard-coded `mdpDomain`;
- the universal `[0, 100]` controllable domains;
- `num_reqs_satisfied = 4`;
- the locally defined optimization score;
- first-`N` slicing for display, clamping, or NSGA-III external features.

Planner construction uses the resolved configuration. WIP receives its mode and anchor threshold at construction, and `evaluate_sample` receives only the sample and confidence threshold.

## Error handling

Configuration and input errors are raised before expensive model training. Messages identify the selected dataset, inferred domain, offending feature or column, expected value, and actual value. Missing canonical anchor files report all checked compatibility paths and instruct the user to enable anchor generation.

An empty polytope set is handled explicitly: the Anchors baseline returns no adaptation, while WIP invokes its PDP fallback if available. Neither implementation indexes an empty distance array.

## Testing strategy

Tests use small synthetic datasets and configurations and avoid running full MDP simulations. Coverage includes:

- every dataset-to-domain mapping and alias;
- rejection of unknown datasets and WIP modes;
- feature and requirement header validation;
- derived domains, indices, observable complements, and directions for all three domains;
- non-contiguous controllable indices;
- score normalization for maximize and minimize directions;
- point-to-polytope distance with finite, infinite, open, and closed bounds;
- preservation of observable features in candidate assembly and adaptation validation;
- WIP `only_true` and hierarchical filtering;
- empty-polytope behavior;
- canonical path selection and legacy read fallback.

A lightweight smoke test loads each real dataset header against its mapped builder configuration. Full evaluator execution remains an explicit integration test because it trains models and invokes the MDP builder.

## Files in scope

- `main/util.py`: active selectors, registry, configuration resolution, validation, scoring, paths, shared feature and distance helpers.
- `main/evaluator.py`: consume resolved configuration and enforce adaptation invariants.
- `main/WIP.py`: consume explicit indices/mode, remove hard-coded thresholds and offsets, use shared distances.
- `main/AnchorsPlanner.py`: consume explicit indices and shared distances; preserve observable values.
- `main/CustomPlanner.py`: ensure Cartesian candidates are assembled through explicit feature indices.
- `main/NSGA3Planner.py`: preserve the full original feature layout for arbitrary controllable indices.
- `main/polytopes/AnchorsAnalyzer.py`: consume the same active feature partition where applicable.
- `tests/`: focused configuration and planner regression tests.
- `README.md`: document the two hard-coded selectors and generated paths.

`main/main.py` remains a legacy entry point and is not extended. It may be marked as legacy in documentation, while `main/evaluator.py` remains the supported experiment driver.
