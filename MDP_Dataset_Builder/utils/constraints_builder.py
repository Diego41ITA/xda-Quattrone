# import config as conf
# import json


def compute_constraints(constraints_offsets: list, spots):
    # constraints_offsets = [.03, .06, .09]

    # spots = conf.IDEAL_SPOTS

    constraints = dict()

    for state_id in spots.keys():
        constraints[state_id] = dict()
        for action_id in spots[state_id].keys():
            local_constraints = []

            # for action a we have: spot = [0.05, 0.85, 0.10]
            spot = spots[state_id][action_id]

            # For the offsets compute the bounds
            for offset in constraints_offsets:
                local_constraint = []

                for elem in spot:
                    low_bound = max(0.0, round(elem - offset, 4))
                    high_bound = min(1.0, round(elem + offset, 4))
                    # if elem > 0.5:
                    #     high_bound = 1.0
                    # else:
                    #     low_bound = 0.0
                    local_constraint.append([low_bound, high_bound])

                local_constraints.append(local_constraint)

            constraints[state_id][action_id] = local_constraints

    return constraints


# import json
#
# IDEAL_SPOTS = {
#     "S0": {
#         "a": [0.05, 0.85, 0.10]
#     },
#     "S5": {
#         "g": [0.05, 0.1, 0.85]
#     },
#     "S10": {
#         "l": [0.90, .10],
#         "m": [0.10, .90],
#     }
# }
# computed_constraints = compute_constraints([.03, .07], IDEAL_SPOTS)
#
# with open('constraints.json', 'w') as f:
#     json.dump(computed_constraints, f)
