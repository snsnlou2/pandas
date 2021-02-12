
import numbers
from typing import Dict, List, Optional, Tuple, Type
import warnings
import numpy as np
from pandas._libs import iNaT, lib, missing as libmissing
from pandas._typing import ArrayLike, Dtype, DtypeObj
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.common import is_bool_dtype, is_datetime64_dtype, is_float, is_float_dtype, is_integer_dtype, is_list_like, is_object_dtype, pandas_dtype
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from pandas.core.ops import invalid_comparison
from pandas.core.tools.numeric import to_numeric
from .masked import BaseMaskedArray, BaseMaskedDtype
from .numeric import NumericArray, NumericDtype

class _IntegerDtype(NumericDtype):
    '\n    An ExtensionDtype to hold a single size & kind of integer dtype.\n\n    These specific implementations are subclasses of the non-public\n    _IntegerDtype. For example we have Int8Dtype to represent signed int 8s.\n\n    The attributes name & type are set when these subclasses are created.\n    '

    def __repr__(self):
        sign = ('U' if self.is_unsigned_integer else '')
        return f'{sign}Int{(8 * self.itemsize)}Dtype()'

    @cache_readonly
    def is_signed_integer(self):
        return (self.kind == 'i')

    @cache_readonly
    def is_unsigned_integer(self):
        return (self.kind == 'u')

    @property
    def _is_numeric(self):
        return True

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        return IntegerArray

    def _get_common_dtype(self, dtypes):
        if (not all(((isinstance(t, BaseMaskedDtype) or (isinstance(t, np.dtype) and (np.issubdtype(t, np.number) or np.issubdtype(t, np.bool_)))) for t in dtypes))):
            return None
        np_dtype = np.find_common_type([(t.numpy_dtype if isinstance(t, BaseMaskedDtype) else t) for t in dtypes], [])
        if np.issubdtype(np_dtype, np.integer):
            return INT_STR_TO_DTYPE[str(np_dtype)]
        elif np.issubdtype(np_dtype, np.floating):
            from pandas.core.arrays.floating import FLOAT_STR_TO_DTYPE
            return FLOAT_STR_TO_DTYPE[str(np_dtype)]
        return None

def safe_cast(values, dtype, copy):
    '\n    Safely cast the values to the dtype if they\n    are equivalent, meaning floats must be equivalent to the\n    ints.\n\n    '
    try:
        return values.astype(dtype, casting='safe', copy=copy)
    except TypeError as err:
        casted = values.astype(dtype, copy=copy)
        if (casted == values).all():
            return casted
        raise TypeError(f'cannot safely cast non-equivalent {values.dtype} to {np.dtype(dtype)}') from err

def coerce_to_array(values, dtype, mask=None, copy=False):
    '\n    Coerce the input values array to numpy arrays with a mask\n\n    Parameters\n    ----------\n    values : 1D list-like\n    dtype : integer dtype\n    mask : bool 1D array, optional\n    copy : bool, default False\n        if True, copy the input\n\n    Returns\n    -------\n    tuple of (values, mask)\n    '
    if ((dtype is None) and hasattr(values, 'dtype')):
        if is_integer_dtype(values.dtype):
            dtype = values.dtype
    if (dtype is not None):
        if (isinstance(dtype, str) and (dtype.startswith('Int') or dtype.startswith('UInt'))):
            dtype = dtype.lower()
        if (not issubclass(type(dtype), _IntegerDtype)):
            try:
                dtype = INT_STR_TO_DTYPE[str(np.dtype(dtype))]
            except KeyError as err:
                raise ValueError(f'invalid dtype specified {dtype}') from err
    if isinstance(values, IntegerArray):
        (values, mask) = (values._data, values._mask)
        if (dtype is not None):
            values = values.astype(dtype.numpy_dtype, copy=False)
        if copy:
            values = values.copy()
            mask = mask.copy()
        return (values, mask)
    values = np.array(values, copy=copy)
    if is_object_dtype(values):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if (inferred_type == 'empty'):
            values = np.empty(len(values))
            values.fill(np.nan)
        elif (inferred_type not in ['floating', 'integer', 'mixed-integer', 'integer-na', 'mixed-integer-float']):
            raise TypeError(f'{values.dtype} cannot be converted to an IntegerDtype')
    elif (is_bool_dtype(values) and is_integer_dtype(dtype)):
        values = np.array(values, dtype=int, copy=copy)
    elif (not (is_integer_dtype(values) or is_float_dtype(values))):
        raise TypeError(f'{values.dtype} cannot be converted to an IntegerDtype')
    if (mask is None):
        mask = isna(values)
    else:
        assert (len(mask) == len(values))
    if (not (values.ndim == 1)):
        raise TypeError('values must be a 1D list-like')
    if (not (mask.ndim == 1)):
        raise TypeError('mask must be a 1D list-like')
    if (dtype is None):
        dtype = np.dtype('int64')
    else:
        dtype = dtype.type
    if mask.any():
        values = values.copy()
        values[mask] = 1
        values = safe_cast(values, dtype, copy=False)
    else:
        values = safe_cast(values, dtype, copy=False)
    return (values, mask)

class IntegerArray(NumericArray):
    "\n    Array of integer (optional missing) values.\n\n    .. versionadded:: 0.24.0\n\n    .. versionchanged:: 1.0.0\n\n       Now uses :attr:`pandas.NA` as the missing value rather\n       than :attr:`numpy.nan`.\n\n    .. warning::\n\n       IntegerArray is currently experimental, and its API or internal\n       implementation may change without warning.\n\n    We represent an IntegerArray with 2 numpy arrays:\n\n    - data: contains a numpy integer array of the appropriate dtype\n    - mask: a boolean array holding a mask on the data, True is missing\n\n    To construct an IntegerArray from generic array-like input, use\n    :func:`pandas.array` with one of the integer dtypes (see examples).\n\n    See :ref:`integer_na` for more.\n\n    Parameters\n    ----------\n    values : numpy.ndarray\n        A 1-d integer-dtype array.\n    mask : numpy.ndarray\n        A 1-d boolean-dtype array indicating missing values.\n    copy : bool, default False\n        Whether to copy the `values` and `mask`.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    Returns\n    -------\n    IntegerArray\n\n    Examples\n    --------\n    Create an IntegerArray with :func:`pandas.array`.\n\n    >>> int_array = pd.array([1, None, 3], dtype=pd.Int32Dtype())\n    >>> int_array\n    <IntegerArray>\n    [1, <NA>, 3]\n    Length: 3, dtype: Int32\n\n    String aliases for the dtypes are also available. They are capitalized.\n\n    >>> pd.array([1, None, 3], dtype='Int32')\n    <IntegerArray>\n    [1, <NA>, 3]\n    Length: 3, dtype: Int32\n\n    >>> pd.array([1, None, 3], dtype='UInt16')\n    <IntegerArray>\n    [1, <NA>, 3]\n    Length: 3, dtype: UInt16\n    "
    _internal_fill_value = 1

    @cache_readonly
    def dtype(self):
        return INT_STR_TO_DTYPE[str(self._data.dtype)]

    def __init__(self, values, mask, copy=False):
        if (not (isinstance(values, np.ndarray) and (values.dtype.kind in ['i', 'u']))):
            raise TypeError("values should be integer numpy array. Use the 'pd.array' function instead")
        super().__init__(values, mask, copy=copy)

    def __neg__(self):
        return type(self)((- self._data), self._mask)

    def __pos__(self):
        return self

    def __abs__(self):
        return type(self)(np.abs(self._data), self._mask)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        (values, mask) = coerce_to_array(scalars, dtype=dtype, copy=copy)
        return IntegerArray(values, mask)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        scalars = to_numeric(strings, errors='raise')
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if (method == 'reduce'):
            raise NotImplementedError("The 'reduce' method is not supported.")
        out = kwargs.get('out', ())
        for x in (inputs + out):
            if (not isinstance(x, (self._HANDLED_TYPES + (IntegerArray,)))):
                return NotImplemented
        result = ops.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if (result is not NotImplemented):
            return result
        mask = np.zeros(len(self), dtype=bool)
        inputs2 = []
        for x in inputs:
            if isinstance(x, IntegerArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x):
            if is_integer_dtype(x.dtype):
                m = mask.copy()
                return IntegerArray(x, m)
            else:
                x[mask] = np.nan
            return x
        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if isinstance(result, tuple):
            return tuple((reconstruct(x) for x in result))
        else:
            return reconstruct(result)

    def _coerce_to_array(self, value):
        return coerce_to_array(value, dtype=self.dtype)

    def astype(self, dtype, copy=True):
        "\n        Cast to a NumPy array or ExtensionArray with 'dtype'.\n\n        Parameters\n        ----------\n        dtype : str or dtype\n            Typecode or data-type to which the array is cast.\n        copy : bool, default True\n            Whether to copy the data, even if not necessary. If False,\n            a copy is made only if the old dtype does not match the\n            new dtype.\n\n        Returns\n        -------\n        ndarray or ExtensionArray\n            NumPy ndarray, BooleanArray or IntegerArray with 'dtype' for its dtype.\n\n        Raises\n        ------\n        TypeError\n            if incompatible type with an IntegerDtype, equivalent of same_kind\n            casting\n        "
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        if is_float_dtype(dtype):
            na_value = np.nan
        elif is_datetime64_dtype(dtype):
            na_value = np.datetime64('NaT')
        else:
            na_value = lib.no_default
        return self.to_numpy(dtype=dtype, na_value=na_value, copy=False)

    def _values_for_argsort(self):
        '\n        Return values for sorting.\n\n        Returns\n        -------\n        ndarray\n            The transformed values should maintain the ordering between values\n            within the array.\n\n        See Also\n        --------\n        ExtensionArray.argsort : Return the indices that would sort this array.\n        '
        data = self._data.copy()
        if self._mask.any():
            data[self._mask] = (data.min() - 1)
        return data

    def _cmp_method(self, other, op):
        from pandas.core.arrays import BooleanArray
        mask = None
        if isinstance(other, BaseMaskedArray):
            (other, mask) = (other._data, other._mask)
        elif is_list_like(other):
            other = np.asarray(other)
            if (other.ndim > 1):
                raise NotImplementedError('can only perform ops with 1-d structures')
            if (len(self) != len(other)):
                raise ValueError('Lengths must match to compare')
        if (other is libmissing.NA):
            result = np.zeros(self._data.shape, dtype='bool')
            mask = np.ones(self._data.shape, dtype='bool')
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'elementwise', FutureWarning)
                with np.errstate(all='ignore'):
                    method = getattr(self._data, f'__{op.__name__}__')
                    result = method(other)
                if (result is NotImplemented):
                    result = invalid_comparison(self._data, other, op)
        if (mask is None):
            mask = self._mask.copy()
        else:
            mask = (self._mask | mask)
        return BooleanArray(result, mask)

    def sum(self, *, skipna=True, min_count=0, **kwargs):
        nv.validate_sum((), kwargs)
        return super()._reduce('sum', skipna=skipna, min_count=min_count)

    def prod(self, *, skipna=True, min_count=0, **kwargs):
        nv.validate_prod((), kwargs)
        return super()._reduce('prod', skipna=skipna, min_count=min_count)

    def min(self, *, skipna=True, **kwargs):
        nv.validate_min((), kwargs)
        return super()._reduce('min', skipna=skipna)

    def max(self, *, skipna=True, **kwargs):
        nv.validate_max((), kwargs)
        return super()._reduce('max', skipna=skipna)

    def _maybe_mask_result(self, result, mask, other, op_name):
        '\n        Parameters\n        ----------\n        result : array-like\n        mask : array-like bool\n        other : scalar or array-like\n        op_name : str\n        '
        if ((is_float_dtype(other) or is_float(other)) or (op_name in ['rtruediv', 'truediv'])):
            from pandas.core.arrays import FloatingArray
            return FloatingArray(result, mask, copy=False)
        if (result.dtype == 'timedelta64[ns]'):
            from pandas.core.arrays import TimedeltaArray
            result[mask] = iNaT
            return TimedeltaArray._simple_new(result)
        return type(self)(result, mask, copy=False)
_dtype_docstring = '\nAn ExtensionDtype for {dtype} integer data.\n\n.. versionchanged:: 1.0.0\n\n   Now uses :attr:`pandas.NA` as its missing value,\n   rather than :attr:`numpy.nan`.\n\nAttributes\n----------\nNone\n\nMethods\n-------\nNone\n'

@register_extension_dtype
class Int8Dtype(_IntegerDtype):
    type = np.int8
    name = 'Int8'
    __doc__ = _dtype_docstring.format(dtype='int8')

@register_extension_dtype
class Int16Dtype(_IntegerDtype):
    type = np.int16
    name = 'Int16'
    __doc__ = _dtype_docstring.format(dtype='int16')

@register_extension_dtype
class Int32Dtype(_IntegerDtype):
    type = np.int32
    name = 'Int32'
    __doc__ = _dtype_docstring.format(dtype='int32')

@register_extension_dtype
class Int64Dtype(_IntegerDtype):
    type = np.int64
    name = 'Int64'
    __doc__ = _dtype_docstring.format(dtype='int64')

@register_extension_dtype
class UInt8Dtype(_IntegerDtype):
    type = np.uint8
    name = 'UInt8'
    __doc__ = _dtype_docstring.format(dtype='uint8')

@register_extension_dtype
class UInt16Dtype(_IntegerDtype):
    type = np.uint16
    name = 'UInt16'
    __doc__ = _dtype_docstring.format(dtype='uint16')

@register_extension_dtype
class UInt32Dtype(_IntegerDtype):
    type = np.uint32
    name = 'UInt32'
    __doc__ = _dtype_docstring.format(dtype='uint32')

@register_extension_dtype
class UInt64Dtype(_IntegerDtype):
    type = np.uint64
    name = 'UInt64'
    __doc__ = _dtype_docstring.format(dtype='uint64')
INT_STR_TO_DTYPE = {'int8': Int8Dtype(), 'int16': Int16Dtype(), 'int32': Int32Dtype(), 'int64': Int64Dtype(), 'uint8': UInt8Dtype(), 'uint16': UInt16Dtype(), 'uint32': UInt32Dtype(), 'uint64': UInt64Dtype()}
