from typing import Optional, Sequence, Union

import numpy as np


def reduce(
    x: Sequence[Union[int, float]],
    areas: Optional[Sequence[float]] = None,
    how: str = "sum",
) -> float:
    """Reduce a numeric sequence.

    Note:
        Optionally can weight the input values based on area.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )
        areas (Sequence, optional):
            The areas of the spatial objects. This is for weighting. Optional.
        how (str):
            The reduction method for the neighborhood. One of:
            "sum", "mean", "median", "min", "max", "std".

    Raises:
        ValueError:
            If an illegal reduction method is given.

    Returns:
        float:
            The reduced value of the input array.
    """
    w = 1.0
    if areas is not None:
        w = areas / (np.sum(areas) + 1e-8)

    res = 0
    if how == "sum":
        res = np.sum(x * w)
    elif how == "mean":
        res = np.mean(x * w)
    elif how == "median":
        res = np.median(x * w)
    elif how == "max":
        res = np.max(x * w)
    elif how == "min":
        res = np.min(x * w)
    elif how == "std":
        res = np.std(x * w)
    elif how == "var":
        res = np.var(x * w)
    else:
        allowed = ("sum", "mean", "median", "min", "max", "std", "var")
        ValueError(f"Illegal param `how`. Got: {how}, Allowed: {allowed}")

    return float(res)
