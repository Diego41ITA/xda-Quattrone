from utils.constraints_builder import compute_constraints
import argparse

MAX_SAMPLES = None  # 1000
INDEX_TO_RUN = None  # 0
TOTAL_TO_RUN = None  # 1
PATH_TO_DATASET = None  # "./starting_combinations.npy"

SS_VARIABLES = {
    "formation": { "domain": int, "range": [0, 1], },
    "flying_speed": { "domain": float, "range": [5., 50.], },
    "countermeasure": { "domain": int, "range": [0, 1], },
    "weather": { "domain": int, "range": [1, 4], },
    "day_time": { "domain": int, "range": [0, 23], },
    "threat_range": { "domain": float, "range": [1000.0, 40000.0], },
    "#threats": { "domain": int, "range": [1, 100], }
}


# Constraints definition

IDEAL_SPOTS = {
    "S0": {
        "sTrt": [0.8, 0.05, 0.15]
    },
    "S10": {
        "sTrt": [0.44, 0.44, 0.02, 0.1]
    },
    "S20": {
        "sTrt": [0.95, 0.005, 0.045],
    }
}

#constraints = compute_constraints([.035, .04, .045, .055], IDEAL_SPOTS) #constraintsv1
#constraints = compute_constraints([0.025, 0.03, 0.035, 0.045], IDEAL_SPOTS) #constraintsv2
constraints = compute_constraints([0.02, 0.025, 0.03, 0.04], IDEAL_SPOTS) #constraintsv3

CONSTRAINTS = [
    # SINGLE CONSTRAINTS
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        }
    },
    {
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        }
    },
    {
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0],
        }
    },
    # DUAL CONSTRAINTS
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        }
    },
    {
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0]
        }
    },
    {
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][3]
        }
    },
    # TRIPLE CONSTRAINTS
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0]
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][1]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][1],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][1]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][1],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][2]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][2],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][2],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][3]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][3],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][3],
        }
    },
]

MINIMAL_CONSTRAINTS = {
    "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
    },
    "S10": {
        "sTrt": constraints["S10"]["sTrt"][0],
    },
    "S20": {
        "sTrt": constraints["S20"]["sTrt"][0],
    }
}

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

# _template = {
#     "S0": {
#         "sTrt": constraints["S0"]["sTrt"][0]
#     },
#     "S7": {
#         "sTrt": constraints["S7"]["sTrt"][0]
#     },
#     "S14": {
#         "sTrt": constraints["S14"]["sTrt"][0],
#         "sTrt": constraints["S14"]["sTrt"][0]
#     }
# }
