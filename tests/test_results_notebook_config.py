import json
import os
from pathlib import Path


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = WORKTREE_ROOT / "results" / "ResultsNB.ipynb"


def test_results_notebook_initial_cell_uses_active_experiment_config(monkeypatch):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    first_code_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    namespace = {"__name__": "__main__"}

    monkeypatch.chdir(NOTEBOOK_PATH.parent)
    exec("".join(first_code_cell["source"]), namespace)

    experiment = namespace["experiment"]
    assert namespace["pathToResults"] == f"{experiment.results_dir}{os.sep}"
    assert namespace["featureNames"] == list(experiment.feature_names)
    assert namespace["reqs"] == list(experiment.requirement_names)
    assert namespace["reqsNamesInGraphs"] == [
        f"R{index}" for index in range(1, len(experiment.requirement_names) + 1)
    ]
