
'\nFunctions for arithmetic and comparison operations on NumPy arrays and\nExtensionArrays.\n'
from datetime import timedelta
from functools import partial
import operator
from typing import Any
import warnings
import numpy as np
from pandas._libs import Timedelta, Timestamp, lib, ops as libops
from pandas._typing import ArrayLike, Shape
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike, find_common_type, maybe_upcast_putmask
from pandas.core.dtypes.common import ensure_object, is_bool_dtype, is_integer_dtype, is_list_like, is_numeric_v_string_like, is_object_dtype, is_scalar
from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna, notna
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.ops import missing
from pandas.core.ops.dispatch import should_extension_dispatch
from pandas.core.ops.invalid import invalid_comparison
from pandas.core.ops.roperator import rpow

def comp_method_OBJECT_ARRAY(op, x, y):
    if isinstance(y, list):
        y = construct_1d_object_array_from_listlike(y)
    if isinstance(y, (np.ndarray, ABCSeries, ABCIndex)):
        if (not is_object_dtype(y.dtype)):
            y = y.astype(np.object_)
        if isinstance(y, (ABCSeries, ABCIndex)):
            y = y._values
        if (x.shape != y.shape):
            raise ValueError('Shapes must match', x.shape, y.shape)
        result = libops.vec_compare(x.ravel(), y.ravel(), op)
    else:
        result = libops.scalar_compare(x.ravel(), y, op)
    return result.reshape(x.shape)

def _masked_arith_op(x, y, op):
    '\n    If the given arithmetic operation fails, attempt it again on\n    only the non-null elements of the input array(s).\n\n    Parameters\n    ----------\n    x : np.ndarray\n    y : np.ndarray, Series, Index\n    op : binary operator\n    '
    xrav = x.ravel()
    assert isinstance(x, np.ndarray), type(x)
    if isinstance(y, np.ndarray):
        dtype = find_common_type([x.dtype, y.dtype])
        result = np.empty(x.size, dtype=dtype)
        if (len(x) != len(y)):
            raise ValueError(x.shape, y.shape)
        else:
            ymask = notna(y)
        yrav = y.ravel()
        mask = (notna(xrav) & ymask.ravel())
        if mask.any():
            with np.errstate(all='ignore'):
                result[mask] = op(xrav[mask], yrav[mask])
    else:
        if (not is_scalar(y)):
            raise TypeError(f'Cannot broadcast np.ndarray with operand of type {type(y)}')
        result = np.empty(x.size, dtype=x.dtype)
        mask = notna(xrav)
        if (op is pow):
            mask = np.where((x == 1), False, mask)
        elif (op is rpow):
            mask = np.where((y == 1), False, mask)
        if mask.any():
            with np.errstate(all='ignore'):
                result[mask] = op(xrav[mask], y)
    result = maybe_upcast_putmask(result, (~ mask))
    result = result.reshape(x.shape)
    return result

def _na_arithmetic_op(left, right, op, is_cmp=False):
    '\n    Return the result of evaluating op on the passed in values.\n\n    If native types are not compatible, try coercion to object dtype.\n\n    Parameters\n    ----------\n    left : np.ndarray\n    right : np.ndarray or scalar\n    is_cmp : bool, default False\n        If this a comparison operation.\n\n    Returns\n    -------\n    array-like\n\n    Raises\n    ------\n    TypeError : invalid operation\n    '
    import pandas.core.computation.expressions as expressions
    try:
        result = expressions.evaluate(op, left, right)
    except TypeError:
        if is_cmp:
            raise
        result = _masked_arith_op(left, right, op)
    if (is_cmp and (is_scalar(result) or (result is NotImplemented))):
        return invalid_comparison(left, right, op)
    return missing.dispatch_fill_zeros(op, left, right, result)

def arithmetic_op(left, right, op):
    '\n    Evaluate an arithmetic operation `+`, `-`, `*`, `/`, `//`, `%`, `**`, ...\n\n    Parameters\n    ----------\n    left : np.ndarray or ExtensionArray\n    right : object\n        Cannot be a DataFrame or Index.  Series is *not* excluded.\n    op : {operator.add, operator.sub, ...}\n        Or one of the reversed variants from roperator.\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n        Or a 2-tuple of these in the case of divmod or rdivmod.\n    '
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = ensure_wrapped_if_datetimelike(right)
    rvalues = _maybe_upcast_for_op(rvalues, lvalues.shape)
    if (should_extension_dispatch(lvalues, rvalues) or isinstance(rvalues, Timedelta)):
        res_values = op(lvalues, rvalues)
    else:
        with np.errstate(all='ignore'):
            res_values = _na_arithmetic_op(lvalues, rvalues, op)
    return res_values

def comparison_op(left, right, op):
    '\n    Evaluate a comparison operation `=`, `!=`, `>=`, `>`, `<=`, or `<`.\n\n    Parameters\n    ----------\n    left : np.ndarray or ExtensionArray\n    right : object\n        Cannot be a DataFrame, Series, or Index.\n    op : {operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le}\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n    '
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = right
    rvalues = lib.item_from_zerodim(rvalues)
    if isinstance(rvalues, list):
        rvalues = np.asarray(rvalues)
    if isinstance(rvalues, (np.ndarray, ABCExtensionArray)):
        if (len(lvalues) != len(rvalues)):
            raise ValueError('Lengths must match to compare', lvalues.shape, rvalues.shape)
    if should_extension_dispatch(lvalues, rvalues):
        res_values = op(lvalues, rvalues)
    elif (is_scalar(rvalues) and isna(rvalues)):
        if (op is operator.ne):
            res_values = np.ones(lvalues.shape, dtype=bool)
        else:
            res_values = np.zeros(lvalues.shape, dtype=bool)
    elif is_numeric_v_string_like(lvalues, rvalues):
        return invalid_comparison(lvalues, rvalues, op)
    elif is_object_dtype(lvalues.dtype):
        res_values = comp_method_OBJECT_ARRAY(op, lvalues, rvalues)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with np.errstate(all='ignore'):
                res_values = _na_arithmetic_op(lvalues, rvalues, op, is_cmp=True)
    return res_values

def na_logical_op(x, y, op):
    try:
        result = op(x, y)
    except TypeError:
        if isinstance(y, np.ndarray):
            assert (not (is_bool_dtype(x.dtype) and is_bool_dtype(y.dtype)))
            x = ensure_object(x)
            y = ensure_object(y)
            result = libops.vec_binop(x.ravel(), y.ravel(), op)
        else:
            assert lib.is_scalar(y)
            if (not isna(y)):
                y = bool(y)
            try:
                result = libops.scalar_binop(x, y, op)
            except (TypeError, ValueError, AttributeError, OverflowError, NotImplementedError) as err:
                typ = type(y).__name__
                raise TypeError(f"Cannot perform '{op.__name__}' with a dtyped [{x.dtype}] array and scalar of type [{typ}]") from err
    return result.reshape(x.shape)

def logical_op(left, right, op):
    '\n    Evaluate a logical operation `|`, `&`, or `^`.\n\n    Parameters\n    ----------\n    left : np.ndarray or ExtensionArray\n    right : object\n        Cannot be a DataFrame, Series, or Index.\n    op : {operator.and_, operator.or_, operator.xor}\n        Or one of the reversed variants from roperator.\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n    '
    fill_int = (lambda x: x)

    def fill_bool(x, left=None):
        if (x.dtype.kind in ['c', 'f', 'O']):
            mask = isna(x)
            if mask.any():
                x = x.astype(object)
                x[mask] = False
        if ((left is None) or is_bool_dtype(left.dtype)):
            x = x.astype(bool)
        return x
    is_self_int_dtype = is_integer_dtype(left.dtype)
    right = lib.item_from_zerodim(right)
    if (is_list_like(right) and (not hasattr(right, 'dtype'))):
        right = construct_1d_object_array_from_listlike(right)
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = right
    if should_extension_dispatch(lvalues, rvalues):
        res_values = op(lvalues, rvalues)
    else:
        if isinstance(rvalues, np.ndarray):
            is_other_int_dtype = is_integer_dtype(rvalues.dtype)
            rvalues = (rvalues if is_other_int_dtype else fill_bool(rvalues, lvalues))
        else:
            is_other_int_dtype = lib.is_integer(rvalues)
        filler = (fill_int if (is_self_int_dtype and is_other_int_dtype) else fill_bool)
        res_values = na_logical_op(lvalues, rvalues, op)
        res_values = filler(res_values)
    return res_values

def get_array_op(op):
    '\n    Return a binary array operation corresponding to the given operator op.\n\n    Parameters\n    ----------\n    op : function\n        Binary operator from operator or roperator module.\n\n    Returns\n    -------\n    functools.partial\n    '
    if isinstance(op, partial):
        return op
    op_name = op.__name__.strip('_').lstrip('r')
    if (op_name == 'arith_op'):
        return op
    if (op_name in {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}):
        return partial(comparison_op, op=op)
    elif (op_name in {'and', 'or', 'xor', 'rand', 'ror', 'rxor'}):
        return partial(logical_op, op=op)
    elif (op_name in {'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow'}):
        return partial(arithmetic_op, op=op)
    else:
        raise NotImplementedError(op_name)

def _maybe_upcast_for_op(obj, shape):
    '\n    Cast non-pandas objects to pandas types to unify behavior of arithmetic\n    and comparison operations.\n\n    Parameters\n    ----------\n    obj: object\n    shape : tuple[int]\n\n    Returns\n    -------\n    out : object\n\n    Notes\n    -----\n    Be careful to call this *after* determining the `name` attribute to be\n    attached to the result of the arithmetic operation.\n    '
    from pandas.core.arrays import DatetimeArray, TimedeltaArray
    if (type(obj) is timedelta):
        return Timedelta(obj)
    elif isinstance(obj, np.datetime64):
        if isna(obj):
            obj = obj.astype('datetime64[ns]')
            right = np.broadcast_to(obj, shape)
            return DatetimeArray(right)
        return Timestamp(obj)
    elif isinstance(obj, np.timedelta64):
        if isna(obj):
            obj = obj.astype('timedelta64[ns]')
            right = np.broadcast_to(obj, shape)
            return TimedeltaArray(right)
        return Timedelta(obj)
    elif (isinstance(obj, np.ndarray) and (obj.dtype.kind == 'm')):
        return TimedeltaArray._from_sequence(obj)
    return obj
