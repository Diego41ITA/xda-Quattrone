# Explanation-Driven Self-Adaptations (XDA) Replication Package

## Papers

- [Efficient Self-Adaptation through Explanation-Driven White-Box Optimization](https://dl.acm.org/doi/10.1145/3737648)
- [Explanation-driven Self-adaptation using Model-agnostic Interpretable Machine Learning](https://dl.acm.org/doi/10.1145/3643915.3644085)

## Installation

```bash
python -m pip install -r requirements.txt
```

The bundled `mdp_simulator` wheel declares Python `<3.12`. It is pure Python;
when using Python 3.12, install it after the requirements without dependency
resolution:

```bash
python -m pip install --no-deps --ignore-requires-python ./MDP_Dataset_Builder/MDP_Simulator/mdp_simulator-0.1.12-py3-none-any.whl
```

## Select an experiment

The central configuration is at the top of `main/util.py`. Routine experiment
selection requires changing only these two strings:

```python
SELECTED_DATASET = "balanced_dataset"
WIP_MODE = "only_true"
```

`WIP_MODE` accepts:

- `only_true`: use only polytopes whose training sample is predicted to satisfy
  every requirement;
- `hierarchical`: navigate all eligible levels starting at two satisfied
  requirements, preferring higher levels when the observable state belongs to
  multiple polytopes.

The selected dataset determines the rest of the configuration automatically:

| Dataset names | Domain | Controllable features | Direction |
| --- | --- | --- | --- |
| `drive`, `drivev2`, `drivev3` (aliases: `drive2`, `drive3`) | `AutonomousDriving_v1` | `car_speed` | minimize |
| `uav`, `uavv2`, `uavv3` | `UAV_v02` | `formation`, `flying_speed`, `countermeasure` | maximize, maximize, minimize |
| all `balanced_*`, `dataset5000*`, `dataset15000*`, `new_dataset` datasets registered in `util.py` | `RescueRobot` | `cruise speed`, `image resolution`, `illuminance`, `controls responsiveness` | maximize, minimize, minimize, minimize |

Feature names, numeric domains, integer types, and requirement count are loaded
from the matching `MDP_Dataset_Builder/<domain>/config.py`. The evaluator fails
early if the CSV header does not match that schema exactly. Observable features
are the ordered complement of the controllable indices and planners are not
allowed to modify them.

## Run

```bash
python main/evaluator.py
```

`python main/main.py` is retained as a compatibility entry point and invokes the
same evaluator. There is no separate dataset or domain configuration in that
file.

The evaluator trains one classifier per requirement and compares
`CustomPlanner`, `NSGA3Planner`, `WIPPlanner`, and `AnchorsPlanner`. NSGA-III and
the polytope planners evolve only controllable values, then reconstruct a full
feature vector while preserving every observable value from the input sample.

Results are written to:

```text
results/WIP_<mode>/<dataset>/
```

Canonical polytope files are written to:

```text
main/polytopes/WIP_<mode>/<dataset>/
```

Validated domain-specific legacy polytope folders remain readable. New files
are always written to the canonical dataset- and mode-specific location.

When builder evaluation is enabled in `main/evaluator.py`, generated adaptations
are checked against the selected domain schema and evaluated through the
matching domain configuration. The resulting folder contains `X_train.csv`,
`X_test_evaluated.csv`, `results.csv`, and the builder-evaluated planner CSVs.

## Tests

```bash
python -m pytest -q
```

The suite verifies every registered dataset header, feature partitions,
domain-derived bounds, score direction, observable immutability, WIP policy,
and non-contiguous planner layouts.
