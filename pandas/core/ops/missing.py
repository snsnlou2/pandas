
'\nMissing data handling for arithmetic operations.\n\nIn particular, pandas conventions regarding division by zero differ\nfrom numpy in the following ways:\n    1) np.array([-1, 0, 1], dtype=dtype1) // np.array([0, 0, 0], dtype=dtype2)\n       gives [nan, nan, nan] for most dtype combinations, and [0, 0, 0] for\n       the remaining pairs\n       (the remaining being dtype1==dtype2==intN and dtype==dtype2==uintN).\n\n       pandas convention is to return [-inf, nan, inf] for all dtype\n       combinations.\n\n       Note: the numpy behavior described here is py3-specific.\n\n    2) np.array([-1, 0, 1], dtype=dtype1) % np.array([0, 0, 0], dtype=dtype2)\n       gives precisely the same results as the // operation.\n\n       pandas convention is to return [nan, nan, nan] for all dtype\n       combinations.\n\n    3) divmod behavior consistent with 1) and 2).\n'
import operator
import numpy as np
from pandas.core.dtypes.common import is_float_dtype, is_integer_dtype, is_scalar
from pandas.core.ops.roperator import rdivmod, rfloordiv, rmod

def fill_zeros(result, x, y):
    "\n    If this is a reversed op, then flip x,y\n\n    If we have an integer value (or array in y)\n    and we have 0's, fill them with np.nan,\n    return the result.\n\n    Mask the nan's from x.\n    "
    if is_float_dtype(result.dtype):
        return result
    is_variable_type = (hasattr(y, 'dtype') or hasattr(y, 'type'))
    is_scalar_type = is_scalar(y)
    if ((not is_variable_type) and (not is_scalar_type)):
        return result
    if is_scalar_type:
        y = np.array(y)
    if is_integer_dtype(y.dtype):
        if (y == 0).any():
            mask = ((y == 0) & (~ np.isnan(result))).ravel()
            shape = result.shape
            result = result.astype('float64', copy=False).ravel()
            np.putmask(result, mask, np.nan)
            result = result.reshape(shape)
    return result

def mask_zero_div_zero(x, y, result):
    '\n    Set results of  0 // 0 to np.nan, regardless of the dtypes\n    of the numerator or the denominator.\n\n    Parameters\n    ----------\n    x : ndarray\n    y : ndarray\n    result : ndarray\n\n    Returns\n    -------\n    ndarray\n        The filled result.\n\n    Examples\n    --------\n    >>> x = np.array([1, 0, -1], dtype=np.int64)\n    >>> x\n    array([ 1,  0, -1])\n    >>> y = 0       # int 0; numpy behavior is different with float\n    >>> result = x // y\n    >>> result      # raw numpy result does not fill division by zero\n    array([0, 0, 0])\n    >>> mask_zero_div_zero(x, y, result)\n    array([ inf,  nan, -inf])\n    '
    if (not isinstance(result, np.ndarray)):
        return result
    if is_scalar(y):
        y = np.array(y)
    zmask = (y == 0)
    if isinstance(zmask, bool):
        return result
    if zmask.any():
        zneg_mask = (zmask & np.signbit(y))
        zpos_mask = (zmask & (~ zneg_mask))
        nan_mask = (zmask & (x == 0))
        with np.errstate(invalid='ignore'):
            neginf_mask = ((zpos_mask & (x < 0)) | (zneg_mask & (x > 0)))
            posinf_mask = ((zpos_mask & (x > 0)) | (zneg_mask & (x < 0)))
        if (nan_mask.any() or neginf_mask.any() or posinf_mask.any()):
            result = result.astype('float64', copy=False)
            result[nan_mask] = np.nan
            result[posinf_mask] = np.inf
            result[neginf_mask] = (- np.inf)
    return result

def dispatch_fill_zeros(op, left, right, result):
    '\n    Call fill_zeros with the appropriate fill value depending on the operation,\n    with special logic for divmod and rdivmod.\n\n    Parameters\n    ----------\n    op : function (operator.add, operator.div, ...)\n    left : object (np.ndarray for non-reversed ops)\n    right : object (np.ndarray for reversed ops)\n    result : ndarray\n\n    Returns\n    -------\n    result : np.ndarray\n\n    Notes\n    -----\n    For divmod and rdivmod, the `result` parameter and returned `result`\n    is a 2-tuple of ndarray objects.\n    '
    if (op is divmod):
        result = (mask_zero_div_zero(left, right, result[0]), fill_zeros(result[1], left, right))
    elif (op is rdivmod):
        result = (mask_zero_div_zero(right, left, result[0]), fill_zeros(result[1], right, left))
    elif (op is operator.floordiv):
        result = mask_zero_div_zero(left, right, result)
    elif (op is rfloordiv):
        result = mask_zero_div_zero(right, left, result)
    elif (op is operator.mod):
        result = fill_zeros(result, left, right)
    elif (op is rmod):
        result = fill_zeros(result, right, left)
    return result
