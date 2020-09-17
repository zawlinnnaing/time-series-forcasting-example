import numpy as np


def filter_index_and_null(item):
    if item is None or item == "":
        return False
    if item == "index":
        return False
    return True


def np_iter(np_array):
    return np.nditer(np_array)
