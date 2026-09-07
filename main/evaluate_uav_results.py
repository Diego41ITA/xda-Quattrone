import argparse
from pathlib import Path

from util import (
    build_results_dir,
    evaluateAdaptations,
    get_evaluator_domain_config,
    readFromCsv,
    validateBuilderFeatureNames,
    validate_domain_planner_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MDP_DOMAIN = "RescueRobot"


def _default_results_csv_for_domain(mdp_domain):
    domain_config = get_evaluator_domain_config(mdp_domain)
    validate_domain_planner_config(domain_config)
    return (Path(__file__).resolve().parent / build_results_dir(domain_config["dataset_path"]) / "results.csv").resolve()


def evaluate_uav_results(results_csv=None, mdp_domain=DEFAULT_MDP_DOMAIN):
    domain_config = get_evaluator_domain_config(mdp_domain)
    validate_domain_planner_config(domain_config)
    feature_names = domain_config["feature_names"]

    if results_csv is None:
        results_csv = _default_results_csv_for_domain(mdp_domain)
    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found for domain {mdp_domain}: {results_csv}")

    validateBuilderFeatureNames(feature_names, mdp_domain=mdp_domain)
    results = readFromCsv(results_csv)
    evaluateAdaptations(
        results,
        feature_names,
        results_dir=results_csv.parent,
        mdp_domain=mdp_domain,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=DEFAULT_MDP_DOMAIN)
    parser.add_argument("--results-csv", default=None)
    args = parser.parse_args()
    evaluate_uav_results(results_csv=args.results_csv, mdp_domain=args.domain)
