
'\nOps for masked arrays.\n'
from typing import Optional, Union
import numpy as np
from pandas._libs import lib, missing as libmissing

def kleene_or(left, right, left_mask, right_mask):
    '\n    Boolean ``or`` using Kleene logic.\n\n    Values are NA where we have ``NA | NA`` or ``NA | False``.\n    ``NA | True`` is considered True.\n\n    Parameters\n    ----------\n    left, right : ndarray, NA, or bool\n        The values of the array.\n    left_mask, right_mask : ndarray, optional\n        The masks. Only one of these may be None, which implies that\n        the associated `left` or `right` value is a scalar.\n\n    Returns\n    -------\n    result, mask: ndarray[bool]\n        The result of the logical or, and the new mask.\n    '
    if (left_mask is None):
        return kleene_or(right, left, right_mask, left_mask)
    assert isinstance(left, np.ndarray)
    raise_for_nan(right, method='or')
    if (right is libmissing.NA):
        result = left.copy()
    else:
        result = (left | right)
    if (right_mask is not None):
        left_false = (~ (left | left_mask))
        right_false = (~ (right | right_mask))
        mask = (((left_false & right_mask) | (right_false & left_mask)) | (left_mask & right_mask))
    elif (right is True):
        mask = np.zeros_like(left_mask)
    elif (right is libmissing.NA):
        mask = (((~ left) & (~ left_mask)) | left_mask)
    else:
        mask = left_mask.copy()
    return (result, mask)

def kleene_xor(left, right, left_mask, right_mask):
    '\n    Boolean ``xor`` using Kleene logic.\n\n    This is the same as ``or``, with the following adjustments\n\n    * True, True -> False\n    * True, NA   -> NA\n\n    Parameters\n    ----------\n    left, right : ndarray, NA, or bool\n        The values of the array.\n    left_mask, right_mask : ndarray, optional\n        The masks. Only one of these may be None, which implies that\n        the associated `left` or `right` value is a scalar.\n\n    Returns\n    -------\n    result, mask: ndarray[bool]\n        The result of the logical xor, and the new mask.\n    '
    if (left_mask is None):
        return kleene_xor(right, left, right_mask, left_mask)
    raise_for_nan(right, method='xor')
    if (right is libmissing.NA):
        result = np.zeros_like(left)
    else:
        result = (left ^ right)
    if (right_mask is None):
        if (right is libmissing.NA):
            mask = np.ones_like(left_mask)
        else:
            mask = left_mask.copy()
    else:
        mask = (left_mask | right_mask)
    return (result, mask)

def kleene_and(left, right, left_mask, right_mask):
    '\n    Boolean ``and`` using Kleene logic.\n\n    Values are ``NA`` for ``NA & NA`` or ``True & NA``.\n\n    Parameters\n    ----------\n    left, right : ndarray, NA, or bool\n        The values of the array.\n    left_mask, right_mask : ndarray, optional\n        The masks. Only one of these may be None, which implies that\n        the associated `left` or `right` value is a scalar.\n\n    Returns\n    -------\n    result, mask: ndarray[bool]\n        The result of the logical xor, and the new mask.\n    '
    if (left_mask is None):
        return kleene_and(right, left, right_mask, left_mask)
    assert isinstance(left, np.ndarray)
    raise_for_nan(right, method='and')
    if (right is libmissing.NA):
        result = np.zeros_like(left)
    else:
        result = (left & right)
    if (right_mask is None):
        if (right is libmissing.NA):
            mask = ((left & (~ left_mask)) | left_mask)
        else:
            mask = left_mask.copy()
            if (right is False):
                mask[:] = False
    else:
        left_false = (~ (left | left_mask))
        right_false = (~ (right | right_mask))
        mask = ((left_mask & (~ right_false)) | (right_mask & (~ left_false)))
    return (result, mask)

def raise_for_nan(value, method):
    if (lib.is_float(value) and np.isnan(value)):
        raise ValueError(f"Cannot perform logical '{method}' with floating NaN")
