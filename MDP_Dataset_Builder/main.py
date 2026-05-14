from utils import build_sequences
from utils.file_proxy import build_output_csv, dump_sets_to_csv
import argparse
import importlib.util
from pathlib import Path

try:
    import mdp_simulator
except ModuleNotFoundError:
    mdp_simulator = None


DEFAULT_DOMAIN = "RescueRobot"


def parse_domain_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    args, _ = parser.parse_known_args()
    return args.domain


def load_domain_config(domain=None, builder_dir=None):
    builder_dir = Path(builder_dir) if builder_dir is not None else Path(__file__).resolve().parent
    config_candidates = []
    if domain is not None:
        config_candidates.append(builder_dir / str(domain) / "config.py")
    config_candidates.append(builder_dir / "config.py")

    for config_path in config_candidates:
        if not config_path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            f"_xda_mdp_builder_config_{config_path.parent.name}",
            config_path,
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise FileNotFoundError(f"Could not find a builder config for domain {domain!r}")


def configure_mdp_simulator(domain, simulator=None, enums_module=None):
    simulator = simulator if simulator is not None else mdp_simulator
    if simulator is None:
        raise RuntimeError("mdp_simulator is not installed in this Python environment")

    if enums_module is None:
        from mdp_simulator.utils import enums as enums_module

    simulator.config.FOLDER_NAME = f"./{domain}"
    simulator.config.DEBUG_LEVEL = enums_module.LogTypes.ERROR


def create_dataset(ss_variables, max_expected, random_sampling=False):
    # Build the input dataset entries
    sets = build_sequences(ss_variables, max_expected, random_sampling)
    dump_sets_to_csv(sets)


def compute_results(ss_variables, index_to_run, total_to_train, constraints, path_to_dataset):
    import utils.executor as executor

    result = executor.run(ss_variables, index_to_run, total_to_train, constraints, path_to_dataset)
    build_output_csv(ss_variables, result['input_data'], result['output_data'], f"{index_to_run}-{total_to_train}.csv")


if __name__ == "__main__":
    domain = parse_domain_arg()
    config = load_domain_config(domain)
    configure_mdp_simulator(domain)

    if config.MAX_SAMPLES is not None:
        print(f"Creating {config.MAX_SAMPLES} samples")
        create_dataset(config.SS_VARIABLES, config.MAX_SAMPLES, random_sampling=True)
    if config.INDEX_TO_RUN is not None and config.TOTAL_TO_RUN is not None:
        print(f"Running {config.INDEX_TO_RUN} out of {config.TOTAL_TO_RUN}")
        compute_results(config.SS_VARIABLES, config.INDEX_TO_RUN, config.TOTAL_TO_RUN, config.CONSTRAINTS, config.PATH_TO_DATASET)
