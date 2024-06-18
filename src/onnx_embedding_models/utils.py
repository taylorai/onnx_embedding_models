import numpy as np

def dict_slice(d, idx):
    """
    Slice a dictionary of jagged arrays, returning a dictionary of singleton numpy arrays.
    """
    return {k: np.array(v[idx]).reshape(1, -1) for k, v in d.items()}