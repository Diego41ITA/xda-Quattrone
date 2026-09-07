import numpy as np
from math import sin, pi, isclose

MULTIPLIER = 1

def do_nothing(old_probability: np.ndarray, variable_value) -> np.ndarray:
    return old_probability


def car_speed_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.002
    clamp_min = 0.0
    formula = 0.00005 * variable_value

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + increment
    old_probability[1] = old_probability[1] - increment

    return old_probability

def car_speed_s2_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.01
    clamp_min = 0.0
    formula = ((variable_value - 5) / ((variable_value - 5) + 1.0)) * 0.011

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] + increment
    old_probability[2] = old_probability[2] - (increment / 2)

    return old_probability

def p_y_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.02
    clamp_min = 0.0
    formula = 0.1-(((10-variable_value)/((10-variable_value)+0.2))*0.1)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + increment
    old_probability[1] = old_probability[1] - increment

    return old_probability

def p_x_s2_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.02
    clamp_min = 0.0
    formula = 0.1-(((10-variable_value)/((10-variable_value)+0.2))*0.1)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] - (increment / 2)
    old_probability[2] = old_probability[2] + increment

    return old_probability

def weather_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.001
    clamp_min = 0.0
    formula = 0.0
    if variable_value > 0:
        formula = 0.0005 * variable_value

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] + increment
    old_probability[1] = old_probability[1] - increment

    return old_probability

def weather_s2_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.01
    clamp_min = 0.0
    formula = 0.0
    if variable_value > 0:
        formula = 0.005 - ((variable_value / (variable_value + 1.0)) * 0.009)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - increment
    old_probability[1] = old_probability[1] + (increment / 2)
    old_probability[2] = old_probability[2] + (increment / 2)

    return old_probability

def orientation_s2_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.009
    clamp_min = 0.0
    formula = 0.0
    if variable_value > 0:
        formula = ((variable_value / (variable_value + 1.0)) * 0.009)

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] - (increment / 2)
    old_probability[2] = old_probability[2] + increment

    return old_probability

def shape_s2_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    clamp_max = 0.001
    clamp_min = 0.0
    formula = 0.0
    if variable_value == 1:
        formula = 0.001

    increment = max(min(clamp_max, formula), clamp_min)
    increment *= MULTIPLIER

    old_probability[0] = old_probability[0] - (increment / 2)
    old_probability[1] = old_probability[1] + increment
    old_probability[2] = old_probability[2] - (increment / 2)

    return old_probability

SemanticSpaceVariable = [
    {
        "Name": "car_speed",
        "Type": "SYS",
        "Domain": float,
        "Range": [5.0, 50.0],
        "Default": 5.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": car_speed_s0_a
            },
            {
                "StateId": "S2",
                "ActionId": "b",
                "Method": car_speed_s2_b
            }
        ],
    },
    {
        "Name": "p_x",
        "Type": "SYS",
        "Domain": float,
        "Range": [0.0, 10.0],
        "Default": 0.5,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": do_nothing
            },
            {
                "StateId": "S2",
                "ActionId": "b",
                "Method": p_x_s2_b
            },
        ],
    },
    {
        "Name": "p_y",
        "Type": "SYS",
        "Domain": float,
        "Range": [0.0, 10.0],
        "Default": 0.5,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": p_y_s0_a
            },
            {
                "StateId": "S2",
                "ActionId": "b",
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "orientation",
        "Type": "SYS",
        "Domain": int,
        "Range": [-30, 30],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": do_nothing
            },
            {
                "StateId": "S2",
                "ActionId": "b",
                "Method": orientation_s2_b
            },
        ],
    },
    {
        "Name": "weather",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 2],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": weather_s0_a
            },
            {
                "StateId": "S2",
                "ActionId": "b",
                "Method": weather_s2_b
            },
        ],
    },
    {
        "Name": "road_shape",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 2],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": do_nothing
            },
            {
                "StateId": "S2",
                "ActionId": "b",
                "Method": shape_s2_b
            },
        ],
    },
]
