
'\nMethods used by Block.replace and related methods.\n'
import operator
import re
from typing import Optional, Pattern, Union
import numpy as np
from pandas._typing import ArrayLike, Scalar
from pandas.core.dtypes.common import is_datetimelike_v_numeric, is_numeric_v_string_like, is_re, is_scalar
from pandas.core.dtypes.missing import isna

def compare_or_regex_search(a, b, regex, mask):
    '\n    Compare two array_like inputs of the same shape or two scalar values\n\n    Calls operator.eq or re.search, depending on regex argument. If regex is\n    True, perform an element-wise regex matching.\n\n    Parameters\n    ----------\n    a : array_like\n    b : scalar or regex pattern\n    regex : bool\n    mask : array_like\n\n    Returns\n    -------\n    mask : array_like of bool\n    '

    def _check_comparison_types(result: Union[(ArrayLike, bool)], a: ArrayLike, b: Union[(Scalar, Pattern)]):
        '\n        Raises an error if the two arrays (a,b) cannot be compared.\n        Otherwise, returns the comparison result as expected.\n        '
        if (is_scalar(result) and isinstance(a, np.ndarray)):
            type_names = [type(a).__name__, type(b).__name__]
            if isinstance(a, np.ndarray):
                type_names[0] = f'ndarray(dtype={a.dtype})'
            raise TypeError(f'Cannot compare types {repr(type_names[0])} and {repr(type_names[1])}')
    if (not regex):
        op = (lambda x: operator.eq(x, b))
    else:
        op = np.vectorize((lambda x: (bool(re.search(b, x)) if (isinstance(x, str) and isinstance(b, (str, Pattern))) else False)))
    if isinstance(a, np.ndarray):
        a = a[mask]
    if is_numeric_v_string_like(a, b):
        return np.zeros(a.shape, dtype=bool)
    elif is_datetimelike_v_numeric(a, b):
        _check_comparison_types(False, a, b)
        return False
    result = op(a)
    if (isinstance(result, np.ndarray) and (mask is not None)):
        tmp = np.zeros(mask.shape, dtype=np.bool_)
        tmp[mask] = result
        result = tmp
    _check_comparison_types(result, a, b)
    return result

def replace_regex(values, rx, value, mask):
    '\n    Parameters\n    ----------\n    values : ArrayLike\n        Object dtype.\n    rx : re.Pattern\n    value : Any\n    mask : np.ndarray[bool], optional\n\n    Notes\n    -----\n    Alters values in-place.\n    '
    if (isna(value) or (not isinstance(value, str))):

        def re_replacer(s):
            if (is_re(rx) and isinstance(s, str)):
                return (value if (rx.search(s) is not None) else s)
            else:
                return s
    else:

        def re_replacer(s):
            if (is_re(rx) and isinstance(s, str)):
                return rx.sub(value, s)
            else:
                return s
    f = np.vectorize(re_replacer, otypes=[values.dtype])
    if (mask is None):
        values[:] = f(values)
    else:
        values[mask] = f(values[mask])
