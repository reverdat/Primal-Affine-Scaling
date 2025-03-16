import numpy as np


def heuristic_m(c: np.array):
    return 1000 * float(np.max(np.abs(c)))
