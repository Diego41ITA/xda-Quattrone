import argparse
from RescueRobot.robot import SemanticSpaceVariable

MAX_SAMPLES = None  # 1000
INDEX_TO_RUN = None  # 0
TOTAL_TO_RUN = None  # 1
PATH_TO_DATASET = None  # "./starting_combinations.npy"

SS_VARIABLES = {
    "cruise speed": {"domain": float, "range": [0, 100]},
    "image resolution": {"domain": float, "range": [0, 100]},
    "illuminance": {"domain": float, "range": [0, 100]},
    "controls responsiveness": {"domain": float, "range": [0, 100]},
    "power": {"domain": int, "range": [0, 100]},
    "smoke intensity": {"domain": float, "range": [0, 100]},
    "obstacle size": {"domain": float, "range": [0, 100]},
    "obstacle distance": {"domain": float, "range": [0, 100]},
    "firm obstacle": {"domain": int, "range": [0, 1]},
}

CONSTRAINT_EXAMPLE = {
    "S0": {
        "a": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    },
    "S5": {
        "g": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    },
    "S10": {
        "l": [[0.0, 1.0], [0.0, 1.0]],
        "m": [[0.0, 1.0], [0.0, 1.0]],
    }
}

low_missclassification = {
    "S0": {
        "a": [[0.0, 1.0], [0.0, 1.0], [0.0, 0.15]],
        "b": [[0.0, 0.15], [0.0, 1.0]]
    }
}

low_contact = {
    "S0": {
        "a": [[0.0, 0.05], [0.0, 1.0], [0.0, 1.0]]
    },
    "S5": {
        "g": [[0.0, 0.05], [0.0, 1.0], [0.0, 1.0]]
    },
    "S6": {
        "h": [[0.0, 0.05], [0.0, 1.0]]
    }
}

low_crash = {
    "S10": {
        "l": [[0.0, 1.0], [0.0, 0.1]],
        "m": [[0.0, 0.1], [0.0, 1.0]]
    }
}

safe = {
    "S5": {
        "g": [[0.0, 1.0], [0.0, 0.15], [0.0, 1.0]]
    },
    "S8": {
        "j": [[0.0, 1.0], [0.0, 0.15]]
    }
}

test = {
    "S6": {
        "h": [[0.0, 0.03], [0.0, 1.0]]
    }
}

low_missclassification_v2 = {
    "S0": {
        "a": [[0.0, 0.9], [0.0, 0.9], [0.0, 0.12]],
        "b": [[0.0, 0.12], [0.0, 0.9]]
    }
}

low_contact_v2 = {
    "S0": {
        "a": [[0.0, 0.04], [0.0, 0.9], [0.0, 0.9]]
    },
    "S5": {
        "g": [[0.0, 0.04], [0.0, 0.9], [0.0, 0.9]]
    },
    "S6": {
        "h": [[0.0, 0.04], [0.0, 0.9]]
    }
}

low_crash_v2 = {
    "S10": {
        "l": [[0.0, 0.9], [0.0, 0.08]],
        "m": [[0.0, 0.08], [0.0, 0.9]]
    }
}

safe_v2 = {
    "S5": {
        "g": [[0.0, 0.9], [0.0, 0.12], [0.0, 0.9]]
    },
    "S8": {
        "j": [[0.0, 0.9], [0.0, 0.12]]
    }
}

low_missclassification_v3 = {
    "S0": {
        "a": [[0.0, 0.85], [0.0, 0.85], [0.0, 0.1]],
        "b": [[0.0, 0.1], [0.0, 0.85]]
    }
}

low_contact_v3 = {
    "S0": {
        "a": [[0.0, 0.03], [0.0, 0.85], [0.0, 0.85]]
    },
    "S5": {
        "g": [[0.0, 0.03], [0.0, 0.85], [0.0, 0.85]]
    },
    "S6": {
        "h": [[0.0, 0.03], [0.0, 0.85]]
    }
}

low_crash_v3 = {
    "S10": {
        "l": [[0.0, 0.85], [0.0, 0.06]],
        "m": [[0.0, 0.06], [0.0, 0.85]]
    }
}

safe_v3 = {
    "S5": {
        "g": [[0.0, 0.85], [0.0, 0.1], [0.0, 0.85]]
    },
    "S8": {
        "j": [[0.0, 0.85], [0.0, 0.1]]
    }
}

constraints_v3 = [
    #test,
    low_missclassification_v3,
    low_contact_v3,
    low_crash_v3,
    safe_v3
]

constraints_v2 = [
    #test,
    low_missclassification_v2,
    low_contact_v2,
    low_crash_v2,
    safe_v2
]

constraints_v1 = [
    #test,
    low_missclassification_v3,
    low_contact_v3,
    low_crash_v3,
    safe_v3
]

CONSTRAINTS = constraints_v3

# Load arguments from cli

all_args = argparse.ArgumentParser()

all_args.add_argument("-m", "--max-samples", required=False, help="Max Samples to generate", type=int, nargs="?")
all_args.add_argument("-i", "--index-to-run", required=False, help="Index of this execution", type=int)
all_args.add_argument("-t", "--total-executions", required=False, help="Index of this execution", type=int)
all_args.add_argument("-p", "--path-to-dataset", required=False, help="Path to the dataset", type=str)

args, _ = all_args.parse_known_args()
args = vars(args)

# Update default values

if args.get("max_samples") is not None:
    print(f"Max Samples: {args.get('max_samples')}")
    MAX_SAMPLES = args.get("max_samples")

if args.get("index_to_run") is not None:
    print(f"Index to run: {args.get('index_to_run')}")
    INDEX_TO_RUN = args.get("index_to_run")

if args.get("total_executions") is not None:
    print(f"Total executions: {args.get('total_executions')}")
    TOTAL_TO_RUN = args.get("total_executions")

if args.get("path-to-dataset") is not None:
    print(f"Using dataset: {args.get('path-to-dataset')}")
    PATH_TO_DATASET = args.get("path-to-dataset")
