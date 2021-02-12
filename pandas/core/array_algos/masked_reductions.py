
'\nmasked_reductions.py is for reduction algorithms using a mask-based approach\nfor missing values.\n'
from typing import Callable
import numpy as np
from pandas._libs import missing as libmissing
from pandas.compat.numpy import np_version_under1p17
from pandas.core.nanops import check_below_min_count

def _sumprod(func, values, mask, *, skipna=True, min_count=0):
    '\n    Sum or product for 1D masked array.\n\n    Parameters\n    ----------\n    func : np.sum or np.prod\n    values : np.ndarray\n        Numpy array with the values (can be of any dtype that support the\n        operation).\n    mask : np.ndarray\n        Boolean numpy array (True values indicate missing values).\n    skipna : bool, default True\n        Whether to skip NA.\n    min_count : int, default 0\n        The required number of valid values to perform the operation. If fewer than\n        ``min_count`` non-NA values are present the result will be NA.\n    '
    if (not skipna):
        if (mask.any() or check_below_min_count(values.shape, None, min_count)):
            return libmissing.NA
        else:
            return func(values)
    else:
        if check_below_min_count(values.shape, mask, min_count):
            return libmissing.NA
        if np_version_under1p17:
            return func(values[(~ mask)])
        else:
            return func(values, where=(~ mask))

def sum(values, mask, *, skipna=True, min_count=0):
    return _sumprod(np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count)

def prod(values, mask, *, skipna=True, min_count=0):
    return _sumprod(np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count)

def _minmax(func, values, mask, *, skipna=True):
    '\n    Reduction for 1D masked array.\n\n    Parameters\n    ----------\n    func : np.min or np.max\n    values : np.ndarray\n        Numpy array with the values (can be of any dtype that support the\n        operation).\n    mask : np.ndarray\n        Boolean numpy array (True values indicate missing values).\n    skipna : bool, default True\n        Whether to skip NA.\n    '
    if (not skipna):
        if (mask.any() or (not values.size)):
            return libmissing.NA
        else:
            return func(values)
    else:
        subset = values[(~ mask)]
        if subset.size:
            return func(subset)
        else:
            return libmissing.NA

def min(values, mask, *, skipna=True):
    return _minmax(np.min, values=values, mask=mask, skipna=skipna)

def max(values, mask, *, skipna=True):
    return _minmax(np.max, values=values, mask=mask, skipna=skipna)

def mean(values, mask, skipna=True):
    if ((not values.size) or mask.all()):
        return libmissing.NA
    _sum = _sumprod(np.sum, values=values, mask=mask, skipna=skipna)
    count = np.count_nonzero((~ mask))
    mean_value = (_sum / count)
    return mean_value
