
import numpy as np
from pandas.core.dtypes.common import is_list_like

def cartesian_product(X):
    "\n    Numpy version of itertools.product.\n    Sometimes faster (for large inputs)...\n\n    Parameters\n    ----------\n    X : list-like of list-likes\n\n    Returns\n    -------\n    product : list of ndarrays\n\n    Examples\n    --------\n    >>> cartesian_product([list('ABC'), [1, 2]])\n    [array(['A', 'A', 'B', 'B', 'C', 'C'], dtype='<U1'), array([1, 2, 1, 2, 1, 2])]\n\n    See Also\n    --------\n    itertools.product : Cartesian product of input iterables.  Equivalent to\n        nested for-loops.\n    "
    msg = 'Input must be a list-like of list-likes'
    if (not is_list_like(X)):
        raise TypeError(msg)
    for x in X:
        if (not is_list_like(x)):
            raise TypeError(msg)
    if (len(X) == 0):
        return []
    lenX = np.fromiter((len(x) for x in X), dtype=np.intp)
    cumprodX = np.cumproduct(lenX)
    if np.any((cumprodX < 0)):
        raise ValueError('Product space too large to allocate arrays!')
    a = np.roll(cumprodX, 1)
    a[0] = 1
    if (cumprodX[(- 1)] != 0):
        b = (cumprodX[(- 1)] / cumprodX)
    else:
        b = np.zeros_like(cumprodX)
    return [tile_compat(np.repeat(x, b[i]), np.product(a[i])) for (i, x) in enumerate(X)]

def tile_compat(arr, num):
    '\n    Index compat for np.tile.\n\n    Notes\n    -----\n    Does not support multi-dimensional `num`.\n    '
    if isinstance(arr, np.ndarray):
        return np.tile(arr, num)
    taker = np.tile(np.arange(len(arr)), num)
    return arr.take(taker)
