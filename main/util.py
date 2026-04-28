import os
import shutil
import subprocess
import sys
import threading
import time as time_module
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

if os.name == "nt":
    import msvcrt
else:
    import fcntl


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


def _run_builder_jobs(builder_dir, dataset_path, total_threads=DEFAULT_TOTAL_THREADS):
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


def _evaluate_dataset_once(dataset, name, builder_dir, results_dir, total_threads):
    builder_dir = Path(builder_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = builder_dir / "starting_combinations.npy"
    source_file = builder_dir / "merge.csv"
    destination_file = results_dir / f"{name}.csv"

    _cleanup_builder_outputs(builder_dir)
    np.save(dataset_path, dataset)
    _run_builder_jobs(builder_dir, dataset_path, total_threads)

    if destination_file.exists():
        destination_file.unlink()
    shutil.move(str(source_file), str(destination_file))
    _cleanup_builder_outputs(builder_dir)


def evaluateDataset(dataset, name, builder_dir=None, results_dir=None, total_threads=DEFAULT_TOTAL_THREADS):
    builder_dir = Path(builder_dir) if builder_dir is not None else BUILDER_DIR
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR

    with _builder_workspace_lock(builder_dir / ".evaluation.lock"):
        _evaluate_dataset_once(dataset, name, builder_dir, results_dir, total_threads)


def evaluateAdaptations(dataset, featureNames):

    customAdaptations = pd.DataFrame(dataset['custom_adaptation'].to_list(), columns=featureNames)
    nsga3Adaptations = pd.DataFrame(dataset['nsga3_adaptation'].to_list(), columns=featureNames)
    customAdaptations_anchors = pd.DataFrame(dataset['anchors_adaptation'].to_list(), columns=featureNames)
    customAdaptations_WIP = pd.DataFrame(dataset['wip_adaptation'].to_list(), columns=featureNames)

    evaluateDataset(nsga3Adaptations, "nsga3Dataset")
    evaluateDataset(customAdaptations, "customDataset")
    evaluateDataset(customAdaptations_anchors, "anchorsDataset")
    evaluateDataset(customAdaptations_WIP, "wipDataset")

def readFromCsv(path):
    results = pd.read_csv(path)
    columns = ["nsga3_adaptation", "custom_adaptation", "anchors_adaptation", "wip_adaptation", "nsga3_confidence", "custom_confidence", "anchors_confidence", "wip_confidence"]

    # numpy arrays are read as strings, must convert them back in arrays
    for c in columns:
        results[c] = results[c].apply(lambda x:  np.fromstring(x[1:-1], dtype=float, sep=' ')if x[1]!='[' else np.fromstring(x[2:-2], dtype=float, sep=' '))


    return results
