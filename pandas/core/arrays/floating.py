
import numbers
from typing import List, Optional, Tuple, Type
import warnings
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas._typing import ArrayLike, DtypeObj
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import astype_nansafe
from pandas.core.dtypes.common import is_bool_dtype, is_datetime64_dtype, is_float_dtype, is_integer_dtype, is_list_like, is_object_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from pandas.core.ops import invalid_comparison
from pandas.core.tools.numeric import to_numeric
from .numeric import NumericArray, NumericDtype

class FloatingDtype(NumericDtype):
    '\n    An ExtensionDtype to hold a single size of floating dtype.\n\n    These specific implementations are subclasses of the non-public\n    FloatingDtype. For example we have Float32Dtype to represent float32.\n\n    The attributes name & type are set when these subclasses are created.\n    '

    def __repr__(self):
        return f'{self.name}Dtype()'

    @property
    def _is_numeric(self):
        return True

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        return FloatingArray

    def _get_common_dtype(self, dtypes):
        if (not all((isinstance(t, FloatingDtype) for t in dtypes))):
            return None
        np_dtype = np.find_common_type([t.numpy_dtype for t in dtypes], [])
        if np.issubdtype(np_dtype, np.floating):
            return FLOAT_STR_TO_DTYPE[str(np_dtype)]
        return None

def coerce_to_array(values, dtype=None, mask=None, copy=False):
    '\n    Coerce the input values array to numpy arrays with a mask.\n\n    Parameters\n    ----------\n    values : 1D list-like\n    dtype : float dtype\n    mask : bool 1D array, optional\n    copy : bool, default False\n        if True, copy the input\n\n    Returns\n    -------\n    tuple of (values, mask)\n    '
    if ((dtype is None) and hasattr(values, 'dtype')):
        if is_float_dtype(values.dtype):
            dtype = values.dtype
    if (dtype is not None):
        if (isinstance(dtype, str) and dtype.startswith('Float')):
            dtype = dtype.lower()
        if (not issubclass(type(dtype), FloatingDtype)):
            try:
                dtype = FLOAT_STR_TO_DTYPE[str(np.dtype(dtype))]
            except KeyError as err:
                raise ValueError(f'invalid dtype specified {dtype}') from err
    if isinstance(values, FloatingArray):
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
            raise TypeError(f'{values.dtype} cannot be converted to a FloatingDtype')
    elif (is_bool_dtype(values) and is_float_dtype(dtype)):
        values = np.array(values, dtype=float, copy=copy)
    elif (not (is_integer_dtype(values) or is_float_dtype(values))):
        raise TypeError(f'{values.dtype} cannot be converted to a FloatingDtype')
    if (mask is None):
        mask = isna(values)
    else:
        assert (len(mask) == len(values))
    if (not (values.ndim == 1)):
        raise TypeError('values must be a 1D list-like')
    if (not (mask.ndim == 1)):
        raise TypeError('mask must be a 1D list-like')
    if (dtype is None):
        dtype = np.dtype('float64')
    else:
        dtype = dtype.type
    if mask.any():
        values = values.copy()
        values[mask] = np.nan
        values = values.astype(dtype, copy=False)
    else:
        values = values.astype(dtype, copy=False)
    return (values, mask)

class FloatingArray(NumericArray):
    '\n    Array of floating (optional missing) values.\n\n    .. versionadded:: 1.2.0\n\n    .. warning::\n\n       FloatingArray is currently experimental, and its API or internal\n       implementation may change without warning. Especially the behaviour\n       regarding NaN (distinct from NA missing values) is subject to change.\n\n    We represent a FloatingArray with 2 numpy arrays:\n\n    - data: contains a numpy float array of the appropriate dtype\n    - mask: a boolean array holding a mask on the data, True is missing\n\n    To construct an FloatingArray from generic array-like input, use\n    :func:`pandas.array` with one of the float dtypes (see examples).\n\n    See :ref:`integer_na` for more.\n\n    Parameters\n    ----------\n    values : numpy.ndarray\n        A 1-d float-dtype array.\n    mask : numpy.ndarray\n        A 1-d boolean-dtype array indicating missing values.\n    copy : bool, default False\n        Whether to copy the `values` and `mask`.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    Returns\n    -------\n    FloatingArray\n\n    Examples\n    --------\n    Create an FloatingArray with :func:`pandas.array`:\n\n    >>> pd.array([0.1, None, 0.3], dtype=pd.Float32Dtype())\n    <FloatingArray>\n    [0.1, <NA>, 0.3]\n    Length: 3, dtype: Float32\n\n    String aliases for the dtypes are also available. They are capitalized.\n\n    >>> pd.array([0.1, None, 0.3], dtype="Float32")\n    <FloatingArray>\n    [0.1, <NA>, 0.3]\n    Length: 3, dtype: Float32\n    '
    _internal_fill_value = 0.0

    @cache_readonly
    def dtype(self):
        return FLOAT_STR_TO_DTYPE[str(self._data.dtype)]

    def __init__(self, values, mask, copy=False):
        if (not (isinstance(values, np.ndarray) and (values.dtype.kind == 'f'))):
            raise TypeError("values should be floating numpy array. Use the 'pd.array' function instead")
        super().__init__(values, mask, copy=copy)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        (values, mask) = coerce_to_array(scalars, dtype=dtype, copy=copy)
        return FloatingArray(values, mask)

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
            if (not isinstance(x, (self._HANDLED_TYPES + (FloatingArray,)))):
                return NotImplemented
        result = ops.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if (result is not NotImplemented):
            return result
        mask = np.zeros(len(self), dtype=bool)
        inputs2 = []
        for x in inputs:
            if isinstance(x, FloatingArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x):
            if is_float_dtype(x.dtype):
                m = mask.copy()
                return FloatingArray(x, m)
            else:
                x[mask] = np.nan
            return x
        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if isinstance(result, tuple):
            tuple((reconstruct(x) for x in result))
        else:
            return reconstruct(result)

    def _coerce_to_array(self, value):
        return coerce_to_array(value, dtype=self.dtype)

    def astype(self, dtype, copy=True):
        "\n        Cast to a NumPy array or ExtensionArray with 'dtype'.\n\n        Parameters\n        ----------\n        dtype : str or dtype\n            Typecode or data-type to which the array is cast.\n        copy : bool, default True\n            Whether to copy the data, even if not necessary. If False,\n            a copy is made only if the old dtype does not match the\n            new dtype.\n\n        Returns\n        -------\n        ndarray or ExtensionArray\n            NumPy ndarray, or BooleanArray, IntegerArray or FloatingArray with\n            'dtype' for its dtype.\n\n        Raises\n        ------\n        TypeError\n            if incompatible type with an FloatingDtype, equivalent of same_kind\n            casting\n        "
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        if is_float_dtype(dtype):
            kwargs = {'na_value': np.nan}
        elif is_datetime64_dtype(dtype):
            kwargs = {'na_value': np.datetime64('NaT')}
        else:
            kwargs = {}
        data = self.to_numpy(dtype=dtype, **kwargs)
        return astype_nansafe(data, dtype, copy=False)

    def _values_for_argsort(self):
        return self._data

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray, IntegerArray
        mask = None
        if isinstance(other, (BooleanArray, IntegerArray, FloatingArray)):
            (other, mask) = (other._data, other._mask)
        elif is_list_like(other):
            other = np.asarray(other)
            if (other.ndim > 1):
                raise NotImplementedError('can only perform ops with 1-d structures')
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
        return type(self)(result, mask, copy=False)
_dtype_docstring = '\nAn ExtensionDtype for {dtype} data.\n\nThis dtype uses ``pd.NA`` as missing value indicator.\n\nAttributes\n----------\nNone\n\nMethods\n-------\nNone\n'

@register_extension_dtype
class Float32Dtype(FloatingDtype):
    type = np.float32
    name = 'Float32'
    __doc__ = _dtype_docstring.format(dtype='float32')

@register_extension_dtype
class Float64Dtype(FloatingDtype):
    type = np.float64
    name = 'Float64'
    __doc__ = _dtype_docstring.format(dtype='float64')
FLOAT_STR_TO_DTYPE = {'float32': Float32Dtype(), 'float64': Float64Dtype()}
