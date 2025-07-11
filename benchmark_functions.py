import math
from copy import deepcopy

import numpy as np


# this is a toy example of a benchmark function library
def get_benchmark_functions_with_constraints(function_names, constraints_names):
    """
    Returns a list of benchmark functions.
    """
    function_names = function_names.split(",")
    constraints_names = constraints_names.split(",")
    available_functions = {
        "branin": branin,
        "Currin": Currin,
        # Add other functions here as needed
    }
    available_constraints = {
        "branin_constraints": branin_constraints,
        "Currin_constraints": Currin_constraints,
        # Add other constraints here as needed
    }
    functions = []
    for name in function_names:
        if name in available_functions:
            functions.append(available_functions[name])
        else:
            raise ValueError(f"Function '{name}' is not available.")
    constraints = []
    for name in constraints_names:
        if name in available_constraints:
            constraints.append(available_constraints[name])
        else:
            raise ValueError(f"Constraint '{name}' is not available.")
    return functions, constraints


def Currin(x, d):
    return -1 * float(
        (
            (1 - math.exp(-0.5 * (1 / x[1])))
            * (
                (2300 * pow(x[0], 3) + 1900 * x[0] * x[0] + 2092 * x[0] + 60)
                / (100 * pow(x[0], 3) + 500 * x[0] * x[0] + 4 * x[0] + 20)
            )
        )
    )


def branin(x1, d):
    x = deepcopy(x1)
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    return -1 * float(
        np.square(
            x[1]
            - (5.1 / (4 * np.square(math.pi))) * np.square(x[0])
            + (5 / math.pi) * x[0]
            - 6
        )
        + 10 * (1 - (1.0 / (8 * math.pi))) * np.cos(x[0])
        + 10
    )


def Currin_constraints(x, d):
    return -1 * float(
        (
            (1 - math.exp(-0.5 * (1 / x[1])))
            * (
                (2300 * pow(x[0], 3) + 1900 * x[0] * x[0] + 2092 * x[0] + 60)
                / (100 * pow(x[0], 3) + 500 * x[0] * x[0] + 4 * x[0] + 20)
            )
        )
    )


def branin_constraints(x1, d):
    x = deepcopy(x1)
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    return -1 * float(
        np.square(
            x[1]
            - (5.1 / (4 * np.square(math.pi))) * np.square(x[0])
            + (5 / math.pi) * x[0]
            - 6
        )
        + 10 * (1 - (1.0 / (8 * math.pi))) * np.cos(x[0])
        + 10
    )
