import os
import shutil
import subprocess
import sys
import threading
import time as time_module
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if os.name == "nt":
    import msvcrt
else:
    import fcntl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILDER_DIR = PROJECT_ROOT / "MDP_Dataset_Builder"

# These are the only two experiment switches that should normally be edited.
SELECTED_DATASET = "drive"
WIP_MODE = "hierarchical"

_DATASET_ALIASES = {
    "drive2": "drivev2",
    "drive3": "drivev3",
}

_DATASET_DOMAINS = {
    "drive": "AutonomousDriving_v1",
    "drivev2": "AutonomousDriving_v1",
    "drivev3": "AutonomousDriving_v1",
    "uav": "UAV_v02",
    "uavv2": "UAV_v02",
    "uavv3": "UAV_v02",
    "balanced_dataset": "RescueRobot",
    "balanced_dataset2": "RescueRobot",
    "dataset5000": "RescueRobot",
    "dataset5000_balanced_joint": "RescueRobot",
    "dataset15000_generated": "RescueRobot",
    "dataset15000_balanced_joint": "RescueRobot",
    "new_dataset": "RescueRobot",
}

_DOMAIN_CONTROLLABLES = {
    "AutonomousDriving_v1": (("car_speed",), (-1,)),
    "UAV_v02": (("formation", "flying_speed", "countermeasure"), (1, 1, -1)),
    "RescueRobot": (
        ("cruise speed", "image resolution", "illuminance", "controls responsiveness"),
        (1, -1, -1, -1),
    ),
}

_SUPPORTED_WIP_MODES = {"only_true", "hierarchical"}


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    dataset_name: str
    dataset_path: Path
    domain: str
    feature_names: tuple[str, ...]
    feature_domains: tuple[tuple[float | int, float | int], ...]
    feature_types: tuple[type, ...]
    requirement_names: tuple[str, ...]
    controllable_feature_names: tuple[str, ...]
    controllable_feature_indices: tuple[int, ...]
    observable_feature_names: tuple[str, ...]
    observable_feature_indices: tuple[int, ...]
    controllable_feature_domains: tuple[tuple[float | int, float | int], ...]
    controllable_feature_types: tuple[type, ...]
    optimization_directions: tuple[int, ...]
    wip_mode: str
    wip_min_satisfied_requirements: int
    results_dir: Path
    anchor_dir: Path
    anchor_explanations_path: Path
    anchor_rules_path: Path
    target_confidence: float = 0.8
    anchor_confidence: float = 0.95


def _canonical_dataset_name(dataset_name: str | Path) -> str:
    name = Path(dataset_name).stem.lower()
    return _DATASET_ALIASES.get(name, name)


def resolve_experiment_config(
    dataset_name: str | Path | None = None,
    wip_mode: str | None = None,
    builder_dir: str | Path | None = None,
) -> ExperimentConfig:
    dataset_name = _canonical_dataset_name(dataset_name or SELECTED_DATASET)
    wip_mode = (wip_mode or WIP_MODE).lower()
    builder_dir = Path(builder_dir) if builder_dir is not None else BUILDER_DIR

    if dataset_name not in _DATASET_DOMAINS:
        supported = ", ".join(sorted(_DATASET_DOMAINS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
    if wip_mode not in _SUPPORTED_WIP_MODES:
        supported = ", ".join(sorted(_SUPPORTED_WIP_MODES))
        raise ValueError(f"Unsupported WIP mode '{wip_mode}'. Supported modes: {supported}")

    domain = _DATASET_DOMAINS[dataset_name]
    module = _load_builder_domain_module(builder_dir, domain)
    ss_variables = getattr(module, "SS_VARIABLES", None)
    constraints = getattr(module, "CONSTRAINTS", None)
    if not isinstance(ss_variables, dict) or constraints is None:
        raise ValueError(f"Builder config for {domain} must define SS_VARIABLES and CONSTRAINTS")

    feature_names = tuple(ss_variables)
    feature_domains = tuple(tuple(variable["range"]) for variable in ss_variables.values())
    feature_types = tuple(variable["domain"] for variable in ss_variables.values())
    controllable_names, directions = _DOMAIN_CONTROLLABLES[domain]
    missing = [name for name in controllable_names if name not in ss_variables]
    if missing:
        raise ValueError(f"Controllable features missing from {domain} SS_VARIABLES: {missing}")

    controllable_indices = tuple(feature_names.index(name) for name in controllable_names)
    observable_indices = tuple(
        index for index in range(len(feature_names)) if index not in controllable_indices
    )
    observable_names = tuple(feature_names[index] for index in observable_indices)
    domains = tuple(tuple(ss_variables[name]["range"]) for name in controllable_names)
    controllable_types = tuple(ss_variables[name]["domain"] for name in controllable_names)
    requirement_names = tuple(f"req_{index}" for index in range(len(constraints)))
    min_satisfied = len(requirement_names) if wip_mode == "only_true" else min(2, len(requirement_names))

    result_namespace = "WIP_only_true" if wip_mode == "only_true" else "WIP_hierarchical"
    results_dir = PROJECT_ROOT / "results" / result_namespace / dataset_name
    anchor_dir = PROJECT_ROOT / "main" / "polytopes" / result_namespace / dataset_name
    config = ExperimentConfig(
        project_root=PROJECT_ROOT,
        dataset_name=dataset_name,
        dataset_path=PROJECT_ROOT / "datasets" / f"{dataset_name}.csv",
        domain=domain,
        feature_names=feature_names,
        feature_domains=feature_domains,
        feature_types=feature_types,
        requirement_names=requirement_names,
        controllable_feature_names=controllable_names,
        controllable_feature_indices=controllable_indices,
        observable_feature_names=observable_names,
        observable_feature_indices=observable_indices,
        controllable_feature_domains=domains,
        controllable_feature_types=controllable_types,
        optimization_directions=directions,
        wip_mode=wip_mode,
        wip_min_satisfied_requirements=min_satisfied,
        results_dir=results_dir,
        anchor_dir=anchor_dir,
        anchor_explanations_path=anchor_dir / "anchors_explanations.csv",
        anchor_rules_path=anchor_dir / "anchors_rules.csv",
    )
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    all_indices = config.controllable_feature_indices + config.observable_feature_indices
    if len(set(all_indices)) != len(all_indices) or sorted(all_indices) != list(
        range(len(config.feature_names))
    ):
        raise ValueError("Controllable and observable feature indices must partition all features")
    if len(config.controllable_feature_indices) != len(config.controllable_feature_domains):
        raise ValueError("Every controllable feature must have a builder domain")
    if len(config.controllable_feature_indices) != len(config.optimization_directions):
        raise ValueError("Every controllable feature must have an optimization direction")
    if any(direction not in {-1, 1} for direction in config.optimization_directions):
        raise ValueError("Optimization directions must be either -1 or 1")


def validate_dataset_schema(columns, config: ExperimentConfig) -> None:
    actual = tuple(columns)
    expected = config.feature_names + config.requirement_names
    if actual != expected:
        raise ValueError(
            "Dataset columns do not match the selected experiment schema. "
            f"Expected {list(expected)}, received {list(actual)}"
        )


def assemble_candidate(
    original,
    controllable_values,
    config: ExperimentConfig,
) -> np.ndarray:
    candidate = np.asarray(original).copy()
    values = np.asarray(controllable_values)
    if candidate.ndim != 1 or len(candidate) != len(config.feature_names):
        raise ValueError("Original sample must be a full one-dimensional feature vector")
    if values.ndim != 1 or len(values) != len(config.controllable_feature_indices):
        raise ValueError("One value is required for every controllable feature")
    candidate[list(config.controllable_feature_indices)] = values
    return candidate


def optimization_score(sample, config: ExperimentConfig) -> float:
    sample = np.asarray(sample)
    if sample.ndim != 1 or len(sample) != len(config.feature_names):
        raise ValueError("Optimization score requires a full feature vector")

    score = 0.0
    for index, bounds, direction in zip(
        config.controllable_feature_indices,
        config.controllable_feature_domains,
        config.optimization_directions,
    ):
        lower, upper = bounds
        if upper <= lower:
            raise ValueError(f"Invalid controllable domain {bounds}")
        normalized = (float(sample[index]) - lower) / (upper - lower)
        normalized = min(1.0, max(0.0, normalized))
        score += 100.0 * (normalized if direction == 1 else 1.0 - normalized)
    return score


def _squared_interval_distance(value, interval) -> float:
    lower, upper = interval[:2]
    if value < lower:
        return float(lower - value) ** 2
    if value > upper:
        return float(value - upper) ** 2
    return 0.0


def point_to_polytopes_distances(
    sample,
    polytopes,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sample = np.asarray(sample)
    if sample.ndim != 1 or len(sample) != len(config.feature_names):
        raise ValueError("Polytope distance requires a full feature vector")

    controllable_distances = np.zeros(len(polytopes), dtype=float)
    observable_distances = np.zeros(len(polytopes), dtype=float)
    for polytope_index, polytope in enumerate(polytopes):
        for feature_index in config.controllable_feature_indices:
            feature_name = config.feature_names[feature_index]
            interval = polytope.get(feature_name, (-np.inf, np.inf))
            controllable_distances[polytope_index] += _squared_interval_distance(
                sample[feature_index], interval
            )
        for feature_index in config.observable_feature_indices:
            feature_name = config.feature_names[feature_index]
            interval = polytope.get(feature_name, (-np.inf, np.inf))
            observable_distances[polytope_index] += _squared_interval_distance(
                sample[feature_index], interval
            )
    return controllable_distances, observable_distances


def validate_adaptation(original, adaptation, config: ExperimentConfig) -> None:
    original = np.asarray(original)
    adaptation = np.asarray(adaptation)
    expected_shape = (len(config.feature_names),)
    if original.shape != expected_shape or adaptation.shape != expected_shape:
        raise ValueError(f"Samples must have shape {expected_shape}")

    observable_indices = list(config.observable_feature_indices)
    if not np.array_equal(original[observable_indices], adaptation[observable_indices]):
        raise ValueError("An adaptation cannot modify observable features")

    for index, bounds, value_type in zip(
        config.controllable_feature_indices,
        config.controllable_feature_domains,
        config.controllable_feature_types,
    ):
        value = adaptation[index]
        lower, upper = bounds
        if not np.isfinite(value) or value < lower or value > upper:
            raise ValueError(
                f"Controllable feature '{config.feature_names[index]}' is outside {bounds}"
            )
        if value_type is int and not float(value).is_integer():
            raise ValueError(
                f"Controllable feature '{config.feature_names[index]}' must be discrete"
            )


def get_active_experiment_config() -> ExperimentConfig:
    return resolve_experiment_config()


def build_results_dir(
    dataset_path: str | Path | None = None,
    wip_mode: str | None = None,
) -> Path:
    dataset_name = dataset_path if dataset_path is not None else SELECTED_DATASET
    return resolve_experiment_config(dataset_name, wip_mode).results_dir


def resolve_anchor_output_paths(
    training_dataset: ExperimentConfig | str | Path | None = None,
    wip_mode: str | None = None,
    *,
    for_write: bool = False,
) -> tuple[Path, Path]:
    if isinstance(training_dataset, ExperimentConfig):
        config = training_dataset
    else:
        dataset_name = training_dataset if training_dataset is not None else SELECTED_DATASET
        training_path = Path(dataset_name)
        if training_path.stem.lower() in {"x_train", "train"} and training_path.parent.name:
            dataset_name = training_path.parent.name
        config = resolve_experiment_config(dataset_name, wip_mode)

    canonical = (config.anchor_explanations_path, config.anchor_rules_path)
    if for_write or all(path.exists() for path in canonical):
        return canonical

    legacy_names = [config.dataset_name]
    if config.domain == "RescueRobot":
        legacy_names.append("rescue_robot")
    elif config.domain == "UAV_v02":
        legacy_names.extend(("uavv2", "uav"))
    elif config.domain == "AutonomousDriving_v1":
        legacy_names.append("autonomous_driving")

    for legacy_name in dict.fromkeys(legacy_names):
        base_dir = PROJECT_ROOT / "main" / "polytopes" / legacy_name
        paths = (
            base_dir / "anchors_explanations.csv",
            base_dir / "anchors_rules.csv",
        )
        if all(path.exists() for path in paths) and _anchor_files_match_config(paths, config):
            return paths
    return canonical


def _anchor_files_match_config(paths, config: ExperimentConfig) -> bool:
    try:
        explanation_columns = tuple(pd.read_csv(paths[0], nrows=0).columns)
        rule_columns = tuple(pd.read_csv(paths[1], nrows=0).columns)
    except (OSError, pd.errors.ParserError):
        return False
    return (
        "n_satisfied_reqs" in explanation_columns
        and "n_satisfied_reqs" in rule_columns
        and all(name in explanation_columns for name in config.feature_names)
    )


def filter_polytopes_for_mode(polytopes, mode: str, requirement_count: int):
    if mode not in _SUPPORTED_WIP_MODES:
        raise ValueError(f"Unsupported WIP mode '{mode}'")
    minimum = requirement_count if mode == "only_true" else min(2, requirement_count)
    return [
        polytope
        for polytope in polytopes
        if int(polytope["n_satisfied_reqs"]) >= minimum
    ]


def _load_builder_domain_module(builder_dir, mdp_domain=None):
    builder_dir = Path(builder_dir)
    config_path = _builder_config_path(builder_dir, mdp_domain)
    spec = importlib.util.spec_from_file_location("_xda_mdp_builder_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load MDP builder config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    builder_path = str(builder_dir.resolve())
    project_root_path = str(builder_dir.resolve().parent)
    added_paths = []
    if builder_path not in sys.path:
        sys.path.insert(0, builder_path)
        added_paths.append(builder_path)
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
        added_paths.append(project_root_path)
    try:
        spec.loader.exec_module(module)
    finally:
        for path in reversed(added_paths):
            if path in sys.path:
                sys.path.remove(path)

    return module


def get_evaluator_domain_config(mdp_domain, builder_dir=None):
    supported_domains = set(_DATASET_DOMAINS.values())
    if mdp_domain not in supported_domains:
        raise ValueError(
            f"Unsupported evaluator domain '{mdp_domain}'. Supported domains: "
            f"{', '.join(sorted(supported_domains))}"
        )

    active_name = _canonical_dataset_name(SELECTED_DATASET)
    if _DATASET_DOMAINS.get(active_name) == mdp_domain:
        dataset_name = active_name
    else:
        dataset_name = next(
            name for name, domain in _DATASET_DOMAINS.items() if domain == mdp_domain
        )
    resolved = resolve_experiment_config(dataset_name, builder_dir=builder_dir)

    config = {
        "dataset_path": resolved.dataset_path,
        "feature_names": list(resolved.feature_names),
        "controllable_feature_indices": list(resolved.controllable_feature_indices),
        "controllable_feature_names": list(resolved.controllable_feature_names),
        "external_feature_names": list(resolved.observable_feature_names),
        "controllable_feature_domains": list(resolved.controllable_feature_domains),
        "optimization_directions": list(resolved.optimization_directions),
        "reqs": list(resolved.requirement_names),
    }
    validate_domain_planner_config(config)
    return config


def validate_domain_planner_config(config):
    controllable_feature_indices = config["controllable_feature_indices"]
    controllable_feature_names = config["controllable_feature_names"]
    optimization_directions = config["optimization_directions"]
    feature_names = config["feature_names"]

    if len(controllable_feature_indices) != len(controllable_feature_names):
        raise ValueError(
            "controllable_feature_indices and controllable_feature_names must have the same length"
        )

    if len(controllable_feature_indices) != len(optimization_directions):
        raise ValueError(
            "optimization_directions must match the number of controllable features"
        )

    for index, feature_name in zip(controllable_feature_indices, controllable_feature_names):
        if feature_names[index] != feature_name:
            raise ValueError(
                "controllable_feature_names must match feature_names at controllable_feature_indices"
            )


def build_explainability_paths(results_dir: str | Path, enabled: bool):
    if not enabled:
        return None, None

    plots_dir = Path(results_dir) / "explainability_plots"
    adaptations_dir = plots_dir / "adaptations"
    return plots_dir, adaptations_dir


def build_evaluated_input_dataset(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_names,
    req_names,
    test_num: int,
) -> pd.DataFrame:
    evaluated_features = x_test.loc[:, feature_names].head(test_num).reset_index(drop=True)
    evaluated_requirements = y_test.loc[:, req_names].head(test_num).reset_index(drop=True)
    evaluated_dataset = pd.concat([evaluated_features, evaluated_requirements], axis=1)
    evaluated_dataset.insert(0, "row_index", pd.Series(evaluated_dataset.index, dtype="int64"))
    return evaluated_dataset


RESULTS_DIR = PROJECT_ROOT / "results" / (
    "WIP_only_true" if WIP_MODE == "only_true" else "WIP_hierarchical"
) / _canonical_dataset_name(SELECTED_DATASET)
DEFAULT_TOTAL_THREADS = 8
_EVALUATION_LOCK = threading.Lock()
TIME_EPSILON = 1e-10


def guard_time_value(time_value, epsilon=TIME_EPSILON):
    if time_value == 0:
        return epsilon
    return time_value

def vecPredictProba(models, X):
    if type(X) is list:
        X = np.array(X)

    probas = np.empty((X.shape[0], len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict_proba(X)[:, 1]
    return probas


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


@contextmanager
def _builder_workspace_lock(lock_path):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with _EVALUATION_LOCK:
        with lock_path.open("a+") as lock_file:
            _lock_file(lock_file)
            try:
                yield
            finally:
                _unlock_file(lock_file)


def _lock_file(lock_file):
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write("0")
        lock_file.flush()
    lock_file.seek(0)

    if os.name == "nt":
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time_module.sleep(0.1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_file(lock_file):
    lock_file.seek(0)
    if os.name == "nt":
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cleanup_builder_outputs(builder_dir):
    builder_dir = Path(builder_dir)
    for pattern in ("*-*.csv", "merge.csv"):
        for output_file in builder_dir.glob(pattern):
            if output_file.is_file():
                output_file.unlink()


def _run_builder_jobs(builder_dir, dataset_path, total_threads=DEFAULT_TOTAL_THREADS, mdp_domain=None):
    builder_dir = Path(builder_dir)
    dataset_path = Path(dataset_path)
    processes = []

    for index in range(total_threads):
        command = [
            sys.executable,
            "main.py",
            "--index-to-run",
            str(index),
            "--total-executions",
            str(total_threads),
            "--path-to-dataset",
            str(dataset_path),
        ]
        if mdp_domain is not None:
            command.extend(["--domain", str(mdp_domain)])
        processes.append(subprocess.Popen(command, cwd=builder_dir))

    failed_processes = []
    for index, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failed_processes.append((index, return_code))

    if failed_processes:
        failures = ", ".join(
            f"worker {index} exited with code {return_code}"
            for index, return_code in failed_processes
        )
        raise RuntimeError(f"Dataset evaluation failed: {failures}")

    subprocess.run([sys.executable, "merge_csvs.py"], cwd=builder_dir, check=True)


def _evaluate_dataset_once(dataset, name, builder_dir, results_dir, total_threads, mdp_domain=None):
    builder_dir = Path(builder_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = builder_dir / "starting_combinations.npy"
    source_file = builder_dir / "merge.csv"
    destination_file = results_dir / f"{name}.csv"

    _cleanup_builder_outputs(builder_dir)
    np.save(dataset_path, dataset)
    _run_builder_jobs(builder_dir, dataset_path, total_threads, mdp_domain=mdp_domain)

    if destination_file.exists():
        destination_file.unlink()
    shutil.move(str(source_file), str(destination_file))
    _cleanup_builder_outputs(builder_dir)


def _builder_config_path(builder_dir, mdp_domain=None):
    builder_dir = Path(builder_dir)
    if mdp_domain is not None:
        domain_config_path = builder_dir / str(mdp_domain) / "config.py"
        if domain_config_path.exists():
            return domain_config_path
    return builder_dir / "config.py"


def _load_builder_ss_variables(builder_dir, mdp_domain=None):
    module = _load_builder_domain_module(builder_dir, mdp_domain)
    ss_variables = getattr(module, "SS_VARIABLES", None)
    if not isinstance(ss_variables, dict):
        config_path = _builder_config_path(builder_dir, mdp_domain)
        raise ValueError(f"{config_path} must define SS_VARIABLES as a dict")
    return ss_variables


def validateBuilderFeatureNames(feature_names, builder_dir=None, mdp_domain=None):
    builder_dir = Path(builder_dir) if builder_dir is not None else BUILDER_DIR
    expected_features = list(_load_builder_ss_variables(builder_dir, mdp_domain).keys())
    actual_features = list(feature_names)

    if actual_features != expected_features:
        raise ValueError(
            "Dataset evaluation feature names do not match MDP builder "
            f"SS_VARIABLES. Received {len(actual_features)} feature names, but "
            f"{Path(builder_dir) / 'config.py'} expects {len(expected_features)}. "
            f"Actual feature names: {actual_features}. "
            f"Expected variables: {expected_features}. "
            "Use a dataset and MDP builder config for the same domain, or set "
            "evaluate = False when only planner metrics are needed."
        )


def _validate_dataset_matches_builder(dataset, builder_dir, mdp_domain=None):
    ss_variables = _load_builder_ss_variables(builder_dir, mdp_domain)
    expected_features = list(ss_variables.keys())
    expected_count = len(expected_features)

    dataset_array = np.asarray(dataset)
    if dataset_array.ndim != 2:
        raise ValueError(
            f"Dataset evaluation expected a 2D dataset, got shape {dataset_array.shape}"
        )

    actual_count = dataset_array.shape[1]
    if actual_count != expected_count:
        actual_features = list(dataset.columns) if hasattr(dataset, "columns") else None
        details = (
            f" Actual columns: {actual_features}."
            if actual_features is not None
            else ""
        )
        raise ValueError(
            f"Dataset evaluation received {actual_count} feature columns, but "
            f"{Path(builder_dir) / 'config.py'} expects {expected_count} SS_VARIABLES."
            f"{details} Expected variables: {expected_features}. "
            "Use a dataset and MDP builder config for the same domain, or set "
            "evaluate = False when only planner metrics are needed."
        )

    if hasattr(dataset, "columns") and list(dataset.columns) != expected_features:
        raise ValueError(
            "Dataset evaluation columns do not match MDP builder SS_VARIABLES. "
            f"Actual columns: {list(dataset.columns)}. "
            f"Expected variables: {expected_features}."
        )


def _coerce_value_to_builder_domain(value, variable_config):
    if pd.isna(value):
        raise ValueError("Dataset evaluation received NaN values")

    min_value, max_value = variable_config["range"]
    domain = variable_config["domain"]

    clipped_value = min(max(value, min_value), max_value)
    if domain is int:
        return int(round(clipped_value))
    if domain is float:
        return float(clipped_value)
    return domain(clipped_value)


def _coerce_dataset_to_builder_domain(dataset, builder_dir, mdp_domain=None):
    ss_variables = _load_builder_ss_variables(builder_dir, mdp_domain)
    expected_features = list(ss_variables.keys())

    if hasattr(dataset, "copy"):
        coerced_dataset = dataset.copy()
        for feature_name in expected_features:
            coerced_dataset[feature_name] = coerced_dataset[feature_name].apply(
                lambda value: _coerce_value_to_builder_domain(value, ss_variables[feature_name])
            )
        return coerced_dataset

    dataset_array = np.asarray(dataset).copy()
    for index, feature_name in enumerate(expected_features):
        vectorized_coercion = np.vectorize(
            lambda value: _coerce_value_to_builder_domain(value, ss_variables[feature_name])
        )
        dataset_array[:, index] = vectorized_coercion(dataset_array[:, index])
    return dataset_array


def evaluateDataset(dataset, name, builder_dir=None, results_dir=None, total_threads=DEFAULT_TOTAL_THREADS, mdp_domain=None):
    builder_dir = Path(builder_dir) if builder_dir is not None else BUILDER_DIR
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR

    _validate_dataset_matches_builder(dataset, builder_dir, mdp_domain)
    dataset = _coerce_dataset_to_builder_domain(dataset, builder_dir, mdp_domain)

    with _builder_workspace_lock(builder_dir / ".evaluation.lock"):
        _evaluate_dataset_once(dataset, name, builder_dir, results_dir, total_threads, mdp_domain=mdp_domain)


def evaluateAdaptations(dataset, featureNames, results_dir=None, mdp_domain=None):

    customAdaptations = pd.DataFrame(dataset['custom_adaptation'].to_list(), columns=featureNames)
    nsga3Adaptations = pd.DataFrame(dataset['nsga3_adaptation'].to_list(), columns=featureNames)
    customAdaptations_anchors = pd.DataFrame(dataset['anchors_adaptation'].to_list(), columns=featureNames)
    customAdaptations_WIP = pd.DataFrame(dataset['wip_adaptation'].to_list(), columns=featureNames)

    evaluateDataset(
        nsga3Adaptations,
        "nsga3Dataset",
        results_dir=results_dir,
        mdp_domain=mdp_domain,
    )
    evaluateDataset(
        customAdaptations,
        "customDataset",
        results_dir=results_dir,
        mdp_domain=mdp_domain,
    )
    evaluateDataset(
        customAdaptations_anchors,
        "anchorsDataset",
        results_dir=results_dir,
        mdp_domain=mdp_domain,
    )
    evaluateDataset(
        customAdaptations_WIP,
        "wipDataset",
        results_dir=results_dir,
        mdp_domain=mdp_domain,
    )

def readFromCsv(path):
    results = pd.read_csv(path)
    columns = ["nsga3_adaptation", "custom_adaptation", "anchors_adaptation", "wip_adaptation", "nsga3_confidence", "custom_confidence", "anchors_confidence", "wip_confidence"]

    # numpy arrays are read as strings, must convert them back in arrays
    for c in columns:
        results[c] = results[c].apply(lambda x:  np.fromstring(x[1:-1], dtype=float, sep=' ')if x[1]!='[' else np.fromstring(x[2:-2], dtype=float, sep=' '))


    return results
