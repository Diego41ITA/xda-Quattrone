# Create a dataset starting from a series of semantic space variables ranges
import numpy as np
import random

FLOAT_MULTIPLIER = 2


def __number_of_max_steps(variable: dict) -> int:
    global FLOAT_MULTIPLIER
    # {"domain": int, "range": [0, 100]}
    steps = variable['range'][1] - variable['range'][0] + 1
    if variable['domain'] is float:
        steps *= FLOAT_MULTIPLIER
    return steps


def __optimal_steps(ss_variables: dict, expected_maximum) -> dict:
    max_steps = dict()

    for key, variable in ss_variables.items():
        max_steps.update({key: __number_of_max_steps(variable)})

    computed_steps = np.array(list(max_steps.values()), float)

    while True:
        if computed_steps.prod() <= expected_maximum:
            break

        while True:
            computed_steps[computed_steps.argmax()] = 1

            new_max = computed_steps.max(initial=None)

            if new_max == 1:
                raise Exception("All steps reduced to 1.")

            free_steps = expected_maximum / computed_steps.prod()

            free_variables = np.where(computed_steps == 1)[0]
            free_variables_len = len(free_variables)

            new_step = free_steps ** (1. / free_variables_len)

            if new_step >= new_max:
                computed_steps[free_variables] = new_step
                break

    print(computed_steps)
    computed_steps = list(map(lambda x: int(round(x, 0)), computed_steps))

    for key, new_value in zip(max_steps.keys(), computed_steps):
        max_steps.update({key: new_value})

    return max_steps


def __build_single_combination(bound, n_steps, domain):
    result = []

    step_len = (bound[1] - bound[0] + 1) / n_steps
    first_value = bound[0] + (step_len / 2)
    result.append(first_value)

    for _ in range(1, n_steps):
        value = result[-1] + step_len
        result.append(value)

    result = list(map(lambda x: round(x, 3), result))
    result = list(map(lambda x: domain(x), result))

    return result


def __build_multiple_combinations(ss_variables: dict, optimal_steps: dict) -> dict:
    result = dict()
    for key in ss_variables.keys():
        bound = ss_variables[key]['range']
        domain = ss_variables[key]['domain']
        n_steps = optimal_steps[key]

        single_combination = __build_single_combination(bound, n_steps, domain)

        result.update({key: single_combination})
    return result


def __set_of_combinations(combinations_per_variable: dict):
    import itertools

    domains = list(combinations_per_variable.values())

    combinations = set(itertools.product(*domains))

    return combinations


def build_sequences(ss_variables: dict, expected_maximum: int, random_sampling=False):
    if random_sampling:
        sets = __build_random_combinations(ss_variables, expected_maximum)
    else:
        computed_optimal_steps = __optimal_steps(ss_variables, expected_maximum)
        computed_combinations = __build_multiple_combinations(ss_variables, computed_optimal_steps)
        sets = __set_of_combinations(computed_combinations)

    return sets


def __build_random_combinations(ss_variables: dict, expected_maximum):
    combinations = set()

    while len(combinations) < expected_maximum:
        new_combination = list()
        for key, variable in ss_variables.items():
            bound = variable['range']
            domain = variable['domain']
            if domain is int:
                new_value = random.randint(bound[0], bound[1])
            else:
                new_value = round(random.uniform(bound[0], bound[1]), 4)
            new_combination.append(new_value)
        combinations.add(tuple(new_combination))

    return combinations


if __name__ == "__main__":
    try:
        import MDP_Dataset_Builder.config as config
    except ModuleNotFoundError:
        import config
    build_sequences(config.SS_VARIABLES, config.MAX_SAMPLES)
