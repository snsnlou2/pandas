
'\nEA-compatible analogue to to np.putmask\n'
from typing import Any
import warnings
import numpy as np
from pandas._libs import lib
from pandas._typing import ArrayLike
from pandas.core.dtypes.cast import convert_scalar_for_putitemlike, maybe_promote
from pandas.core.dtypes.common import is_float_dtype, is_integer_dtype, is_list_like
from pandas.core.dtypes.missing import isna_compat

def putmask_inplace(values, mask, value):
    '\n    ExtensionArray-compatible implementation of np.putmask.  The main\n    difference is we do not handle repeating or truncating like numpy.\n\n    Parameters\n    ----------\n    mask : np.ndarray[bool]\n        We assume _extract_bool_array has already been called.\n    value : Any\n    '
    if (lib.is_scalar(value) and isinstance(values, np.ndarray)):
        value = convert_scalar_for_putitemlike(value, values.dtype)
    if ((not isinstance(values, np.ndarray)) or ((values.dtype == object) and (not lib.is_scalar(value)))):
        if (is_list_like(value) and (len(value) == len(values))):
            values[mask] = value[mask]
        else:
            values[mask] = value
    else:
        np.putmask(values, mask, value)

def putmask_smart(values, mask, new):
    '\n    Return a new ndarray, try to preserve dtype if possible.\n\n    Parameters\n    ----------\n    values : np.ndarray\n        `values`, updated in-place.\n    mask : np.ndarray[bool]\n        Applies to both sides (array like).\n    new : `new values` either scalar or an array like aligned with `values`\n\n    Returns\n    -------\n    values : ndarray with updated values\n        this *may* be a copy of the original\n\n    See Also\n    --------\n    ndarray.putmask\n    '
    if (not is_list_like(new)):
        new = np.repeat(new, len(mask))
    try:
        nn = new[mask]
    except TypeError:
        pass
    else:
        if (not isna_compat(values, nn[0])):
            pass
        elif (not (is_float_dtype(nn.dtype) or is_integer_dtype(nn.dtype))):
            pass
        elif (not (is_float_dtype(values.dtype) or is_integer_dtype(values.dtype))):
            pass
        else:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore', np.ComplexWarning)
                nn_at = nn.astype(values.dtype)
            comp = (nn == nn_at)
            if (is_list_like(comp) and comp.all()):
                nv = values.copy()
                nv[mask] = nn_at
                return nv
    new = np.asarray(new)
    if (values.dtype.kind == new.dtype.kind):
        return _putmask_preserve(values, new, mask)
    (dtype, _) = maybe_promote(new.dtype)
    values = values.astype(dtype)
    return _putmask_preserve(values, new, mask)

def _putmask_preserve(new_values, new, mask):
    try:
        new_values[mask] = new[mask]
    except (IndexError, ValueError):
        new_values[mask] = new
    return new_values

def putmask_without_repeat(values, mask, new):
    '\n    np.putmask will truncate or repeat if `new` is a listlike with\n    len(new) != len(values).  We require an exact match.\n\n    Parameters\n    ----------\n    values : np.ndarray\n    mask : np.ndarray[bool]\n    new : Any\n    '
    if (getattr(new, 'ndim', 0) >= 1):
        new = new.astype(values.dtype, copy=False)
    nlocs = mask.sum()
    if ((nlocs > 0) and is_list_like(new) and (getattr(new, 'ndim', 1) == 1)):
        if (nlocs == len(new)):
            np.place(values, mask, new)
        elif ((mask.shape[(- 1)] == len(new)) or (len(new) == 1)):
            np.putmask(values, mask, new)
        else:
            raise ValueError('cannot assign mismatch length to masked array')
    else:
        np.putmask(values, mask, new)
