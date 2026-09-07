from utils.constraints_builder import compute_constraints
import argparse

MAX_SAMPLES = None  # 1000
INDEX_TO_RUN = None  # 0
TOTAL_TO_RUN = None  # 1
PATH_TO_DATASET = None  # "./starting_combinations.npy"


SS_VARIABLES = {
    "car_speed": {"domain": float, "range": [5.0, 50.0]},
    "p_x": {"domain": float, "range": [0.0, 10.0]},
    "p_y": {"domain": float, "range": [0.0, 10.0]},
    "orientation": {"domain": int, "range": [-30, 30]},
    "weather": {"domain": int, "range": [0, 2]},
    "road_shape": {"domain": int, "range": [0, 2]},
}

# Constraints definition

IDEAL_SPOTS = {
    "S0": {
        "a": [0.001, 0.999]
    },
    "S2": {
        "b": [0.98, 0.015, 0.005],
    }
}

#constraints = compute_constraints([.025, .03, .04, .045], IDEAL_SPOTS)
#constraints = compute_constraints([.001, .002], IDEAL_SPOTS) #constraints v1
# constraints = compute_constraints([0.0008, 0.0015], IDEAL_SPOTS) #constraintsv2
constraints = compute_constraints([0.0005, 0.001], IDEAL_SPOTS) #constraintsv3
CONSTRAINTS = [
    # SINGLE CONSTRAINTS
    {
        "S0": {
            "a": constraints["S0"]["a"][0]
        }
    },
    {
        "S2": {
            "b": constraints["S2"]["b"][1]
        }
    },
    {
        "S0": {
            "a": constraints["S0"]["a"][1]
        },
        "S2": {
            "b": constraints["S2"]["b"][1]
        }
    }
]

MINIMAL_CONSTRAINTS = {
    "S0": {
        "a": constraints["S0"]["a"][0]
    },
    "S2": {
        "b": constraints["S2"]["b"][1]
    }
}

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

# _template = {
#     "S0": {
#         "a": constraints["S0"]["a"][0]
#     },
#     "S5": {
#         "g": constraints["S5"]["g"][0]
#     },
#     "S10": {
#         "l": constraints["S10"]["l"][0],
#         "m": constraints["S10"]["m"][0]
#     }
# }
