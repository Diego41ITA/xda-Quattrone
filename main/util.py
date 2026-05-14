import os
import shutil
import subprocess
import sys
import threading
import time as time_module
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from pathlib import Path


def build_results_dir(dataset_path: str | Path) -> Path:
    dataset_stem = Path(dataset_path).stem
    return Path("../results") / dataset_stem


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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILDER_DIR = PROJECT_ROOT / "MDP_Dataset_Builder"
RESULTS_DIR = PROJECT_ROOT / "results"
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
    builder_dir = Path(builder_dir)
    config_path = _builder_config_path(builder_dir, mdp_domain)
    spec = importlib.util.spec_from_file_location("_xda_mdp_builder_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load MDP builder config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    builder_path = str(builder_dir.resolve())
    added_builder_path = builder_path not in sys.path
    if added_builder_path:
        sys.path.insert(0, builder_path)
    try:
        spec.loader.exec_module(module)
    finally:
        if added_builder_path:
            sys.path.remove(builder_path)

    ss_variables = getattr(module, "SS_VARIABLES", None)
    if not isinstance(ss_variables, dict):
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
