import numpy as np

MULTIPLIER = 4


def do_nothing(old_probability: np.ndarray, variable_value) -> np.ndarray:
    return old_probability


# S0 ###########################################################################################


def formation_s0_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S0; u_00;OBSERVABLE;sTrt, 0.8, S1;sTrt, 0.05, S21;sTrt, 0.15, S22;"

    clamp_max = 0.001
    clamp_min = 0.0
    formula = 0.001 * variable_value

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - increment
    old_probability[1] = old_probability[1] + increment

    return old_probability


def flying_speed_s0_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S0; u_00;OBSERVABLE;sTrt, 0.8, S1;sTrt, 0.05, S21;sTrt, 0.15, S22;"

    clamp_max = 0.01
    clamp_min = 0.0
    formula = ((variable_value - 5) / ((variable_value - 5) + 1.0)) * 0.01

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + (increment / 2)
    old_probability[1] = old_probability[1] - increment
    old_probability[2] = old_probability[2] + (increment / 2)

    return old_probability


def countermeasure_s0_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.005
    clamp_min = 0.0
    formula = 0.005 * variable_value

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + increment
    old_probability[1] = old_probability[1] - increment
    old_probability[2] = old_probability[2]

    return old_probability


def threat_range_s0_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S0; u_00;OBSERVABLE;sTrt, 0.8, S1;sTrt, 0.05, S21;sTrt, 0.15, S22;"

    x = variable_value
    clamp_max = 0.025
    clamp_min = 0.0
    formula = 0.025 - (((4000 - x) / ((4000 - x) + 25)) * 0.025)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - increment
    old_probability[1] = old_probability[1] + increment
    old_probability[2] = old_probability[2]

    return old_probability


def threats_s0_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S0; u_00;OBSERVABLE;sTrt, 0.8, S1;sTrt, 0.05, S21;sTrt, 0.15, S22;"

    x = variable_value
    clamp_max = 0.02
    clamp_min = 0.0
    formula = 0.02 - (((100 - x) / ((100 - x) + 1.0)) * 0.02)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - increment
    old_probability[1] = old_probability[1] + increment
    old_probability[2] = old_probability[2]

    return old_probability


# S10 ###########################################################################################


def formation_s10_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S7; u_07;OBSERVABLE;sTrt, 0.44, S6;sTrt, 0.44, S8;sTrt, 0.02, S21;sTrt, 0.1, S22;"

    clamp_max = 0.0015
    clamp_min = 0.0
    formula = 0.0015 * variable_value

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] - (increment / 2)
    old_probability[2] = old_probability[2] + increment

    return old_probability


def flying_speed_s10_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S7; u_07;OBSERVABLE;sTrt, 0.44, S6;sTrt, 0.44, S8;sTrt, 0.02, S21;sTrt, 0.1, S22;"

    clamp_max = 0.01
    clamp_min = 0.0
    formula = ((variable_value - 5) / ((variable_value - 5) + 1.0)) * 0.01

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] + (increment / 2)
    old_probability[2] = old_probability[2] - increment
    old_probability[3] = old_probability[3] + increment

    return old_probability


def countermeasure_s10_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S7; u_07;OBSERVABLE;sTrt, 0.44, S6;sTrt, 0.44, S8;sTrt, 0.02, S21;sTrt, 0.1, S22;"

    clamp_max = 0.005
    clamp_min = 0.0
    formula = 0.005 * variable_value

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + (increment / 2)
    old_probability[1] = old_probability[1] + (increment / 2)
    old_probability[2] = old_probability[2] - increment
    old_probability[3] = old_probability[3]

    return old_probability


def threat_range_s10_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S7; u_07;OBSERVABLE;sTrt, 0.44, S6;sTrt, 0.44, S8;sTrt, 0.02, S21;sTrt, 0.1, S22;"

    clamp_max = 0.025
    clamp_min = 0.0
    formula = 0.025 - \
              (((4000 - variable_value) / ((4000 - variable_value) + 25)) * 0.025)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] - (increment / 2)
    old_probability[2] = old_probability[2] + increment
    old_probability[3] = old_probability[3]

    return old_probability


def threats_s10_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S7; u_07;OBSERVABLE;sTrt, 0.44, S6;sTrt, 0.44, S8;sTrt, 0.02, S21;sTrt, 0.1, S22;"

    clamp_max = 0.02
    clamp_min = 0.0
    formula = 0.02 - \
              (((100 - variable_value) / ((100 - variable_value) + 1.0)) * 0.02)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] - (increment / 2)
    old_probability[2] = old_probability[2] + increment
    old_probability[3] = old_probability[3]

    return old_probability

def weather_s10_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S14;u_14;OBSERVABLE;sTrt, 0.47, S13;sTrt, 0.47, S15;sTrt, 0.01, S21;sTrt, 0.05, S22;"

    clamp_max = 0.002
    clamp_min = 0.0
    x = variable_value
    formula = 0.0005 * x

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + (increment / 2)
    old_probability[1] = old_probability[1] + (increment / 2)
    old_probability[2] = old_probability[2] - increment
    old_probability[3] = old_probability[3]

    return old_probability


# S14 ##########################################################################################


# def flying_speed_s14_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
#     "S14;u_14;OBSERVABLE;sTrt, 0.47, S13;sTrt, 0.47, S15;sTrt, 0.01, S21;sTrt, 0.05, S22;"

#     clamp_max = 0.01
#     clamp_min = 0.0
#     x = variable_value
#     formula = ((x - 5) / ((x - 5) + 1.0)) * 0.01

#     increment = max(min(clamp_max, formula), clamp_min)
#     increment *= MULTIPLIER

#     old_probability[0] = old_probability[0] - (increment / 2)
#     old_probability[1] = old_probability[1] + (increment / 2)
#     old_probability[2] = old_probability[2] - increment
#     old_probability[3] = old_probability[3] + increment

#     return old_probability


# def countermeasure_s14_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
#     "S14;u_14;OBSERVABLE;sTrt, 0.47, S13;sTrt, 0.47, S15;sTrt, 0.01, S21;sTrt, 0.05, S22;"

#     clamp_max = 0.005
#     clamp_min = 0.0
#     x = variable_value
#     formula = 0.005 * x

#     increment = max(min(clamp_max, formula), clamp_min)
#     increment *= MULTIPLIER

#     old_probability[0] = old_probability[0] + (increment / 2)
#     old_probability[1] = old_probability[1] + (increment / 2)
#     old_probability[2] = old_probability[2] - increment
#     old_probability[3] = old_probability[3]

#     return old_probability


# def weather_s14_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
#     "S14;u_14;OBSERVABLE;sTrt, 0.47, S13;sTrt, 0.47, S15;sTrt, 0.01, S21;sTrt, 0.05, S22;"

#     clamp_max = 0.002
#     clamp_min = 0.0
#     x = variable_value
#     formula = 0.0005 * x

#     increment = max(min(clamp_max, formula), clamp_min)
#     increment *= MULTIPLIER

#     old_probability[0] = old_probability[0] + (increment / 2)
#     old_probability[1] = old_probability[1] + (increment / 2)
#     old_probability[2] = old_probability[2] - increment
#     old_probability[3] = old_probability[3]

#     return old_probability


# def threat_range_s14_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
#     "S14;u_14;OBSERVABLE;sTrt, 0.47, S13;sTrt, 0.47, S15;sTrt, 0.01, S21;sTrt, 0.05, S22;"

#     clamp_max = 0.025
#     clamp_min = 0.0
#     x = variable_value
#     formula = 0.025 - (((4000 - x) / ((4000 - x) + 25)) * 0.025)

#     increment = max(min(clamp_max, formula), clamp_min)
#     increment *= MULTIPLIER

#     old_probability[0] = old_probability[0] - (increment / 2)
#     old_probability[1] = old_probability[1] - (increment / 2)
#     old_probability[2] = old_probability[2] + increment
#     old_probability[3] = old_probability[3]

#     return old_probability


# def threats_s14_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
#     "S14;u_14;OBSERVABLE;sTrt, 0.47, S13;sTrt, 0.47, S15;sTrt, 0.01, S21;sTrt, 0.05, S22;"

#     clamp_max = 0.02
#     clamp_min = 0.0
#     x = variable_value
#     formula = 0.02 - (((100 - x) / ((100 - x) + 1.0)) * 0.02)

#     increment = max(min(clamp_max, formula), clamp_min)
#     increment *= MULTIPLIER

#     old_probability[0] = old_probability[0] - (increment / 2)
#     old_probability[1] = old_probability[1] - (increment / 2)
#     old_probability[2] = old_probability[2] + increment
#     old_probability[3] = old_probability[3]

#     return old_probability


# S20 ##########################################################################################


def flying_speed_s20_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S20;u_20;OBSERVABLE;sTrt, 0.95, S19;sTrt, 0.005, S21;sTrt, 0.045, S22;"

    clamp_max = 0.01
    clamp_min = 0.0
    x = variable_value
    formula = ((x - 5) / ((x - 5) + 1.0)) * 0.01

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + (increment / 2)
    old_probability[1] = old_probability[1] - increment
    old_probability[2] = old_probability[2] + (increment / 2)

    return old_probability


def countermeasure_s20_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S20;u_20;OBSERVABLE;sTrt, 0.95, S19;sTrt, 0.005, S21;sTrt, 0.045, S22;"

    clamp_max = 0.005
    clamp_min = 0.0
    x = variable_value
    formula = 0.005 * x

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + increment
    old_probability[1] = old_probability[1] - increment
    old_probability[2] = old_probability[2]

    return old_probability


def weather_s20_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S20;u_20;OBSERVABLE;sTrt, 0.95, S19;sTrt, 0.005, S21;sTrt, 0.045, S22;"

    clamp_max = 0.002
    clamp_min = 0.0
    x = variable_value
    formula = 0.0005 * x

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + increment
    old_probability[1] = old_probability[1] - increment
    old_probability[2] = old_probability[2]

    return old_probability


def threat_range_s20_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    "S20;u_20;OBSERVABLE;sTrt, 0.95, S19;sTrt, 0.005, S21;sTrt, 0.045, S22;"

    clamp_max = 0.025
    clamp_min = 0.0
    x = variable_value
    formula = 0.025 - (((4000 - x) / ((4000 - x) + 25)) * 0.025)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - increment
    old_probability[1] = old_probability[1] + increment
    old_probability[2] = old_probability[2]

    return old_probability


def threats_s20_sTrt(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.02
    clamp_min = 0.0
    x = variable_value
    formula = 0.02 - (((100 - x) / ((100 - x) + 1.0)) * 0.02)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - increment
    old_probability[1] = old_probability[1] + increment
    old_probability[2] = old_probability[2]

    return old_probability


SemanticSpaceVariable = [
    {
        "Name": "formation",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 1],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "sTrt",
                "Method": formation_s0_sTrt
            },
            {
                "StateId": "S10",
                "ActionId": "sTrt",
                "Method": do_nothing
            },
            {
                "StateId": "S20",
                "ActionId": "sTrt",
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "flying_speed",
        "Type": "SYS",
        "Domain": float,
        "Range": [5., 50.],
        "Default": 5.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "sTrt",
                "Method": flying_speed_s0_sTrt
            },
            {
                "StateId": "S10",
                "ActionId": "sTrt",
                "Method": flying_speed_s10_sTrt
            },
            {
                "StateId": "S20",
                "ActionId": "sTrt",
                "Method": flying_speed_s20_sTrt
            },
        ],
    },
    {
        "Name": "countermeasure",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 1],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "sTrt",
                "Method": countermeasure_s0_sTrt
            },
            {
                "StateId": "S10",
                "ActionId": "sTrt",
                "Method": countermeasure_s10_sTrt
            },
            {
                "StateId": "S20",
                "ActionId": "sTrt",
                "Method": countermeasure_s20_sTrt
            },
        ],
    },
    {
        "Name": "weather",
        "Type": "SYS",
        "Domain": int,
        "Range": [1, 4],
        "Default": 1,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "sTrt",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "sTrt",
                "Method": weather_s10_sTrt
            },
            {
                "StateId": "S20",
                "ActionId": "sTrt",
                "Method": weather_s20_sTrt
            },
        ],
    },
    {
        "Name": "day_time",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 23],
        "Default": 1,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "sTrt",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "sTrt",
                "Method": do_nothing
            },
            {
                "StateId": "S20",
                "ActionId": "sTrt",
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "threat_range",
        "Type": "SYS",
        "Domain": float,
        "Range": [1000.0, 40000.0],
        "Default": 1000,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "sTrt",
                "Method": threat_range_s0_sTrt
            },
            {
                "StateId": "S10",
                "ActionId": "sTrt",
                "Method": threat_range_s10_sTrt
            },
            {
                "StateId": "S20",
                "ActionId": "sTrt",
                "Method": threat_range_s20_sTrt
            },
        ],
    },
    {
        "Name": "#threats",
        "Type": "SYS",
        "Domain": int,
        "Range": [1, 100],
        "Default": 1,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "sTrt",
                "Method": threats_s0_sTrt
            },
            {
                "StateId": "S10",
                "ActionId": "sTrt",
                "Method": threats_s10_sTrt
            },
            {
                "StateId": "S20",
                "ActionId": "sTrt",
                "Method": threats_s20_sTrt
            },
        ],
    }
]
