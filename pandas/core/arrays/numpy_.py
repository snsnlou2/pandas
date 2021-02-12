
import numbers
from typing import Tuple, Type, Union
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas._libs import lib
from pandas._typing import Scalar
from pandas.compat.numpy import function as nv
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.core import nanops, ops
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.strings.object_array import ObjectStringArrayMixin

class PandasDtype(ExtensionDtype):
    '\n    A Pandas ExtensionDtype for NumPy dtypes.\n\n    .. versionadded:: 0.24.0\n\n    This is mostly for internal compatibility, and is not especially\n    useful on its own.\n\n    Parameters\n    ----------\n    dtype : object\n        Object to be converted to a NumPy data type object.\n\n    See Also\n    --------\n    numpy.dtype\n    '
    _metadata = ('_dtype',)

    def __init__(self, dtype):
        self._dtype = np.dtype(dtype)

    def __repr__(self):
        return f'PandasDtype({repr(self.name)})'

    @property
    def numpy_dtype(self):
        '\n        The NumPy dtype this PandasDtype wraps.\n        '
        return self._dtype

    @property
    def name(self):
        '\n        A bit-width name for this data-type.\n        '
        return self._dtype.name

    @property
    def type(self):
        '\n        The type object used to instantiate a scalar of this NumPy data-type.\n        '
        return self._dtype.type

    @property
    def _is_numeric(self):
        return (self.kind in set('biufc'))

    @property
    def _is_boolean(self):
        return (self.kind == 'b')

    @classmethod
    def construct_from_string(cls, string):
        try:
            dtype = np.dtype(string)
        except TypeError as err:
            if (not isinstance(string, str)):
                msg = f"'construct_from_string' expects a string, got {type(string)}"
            else:
                msg = f"Cannot construct a 'PandasDtype' from '{string}'"
            raise TypeError(msg) from err
        return cls(dtype)

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        return PandasArray

    @property
    def kind(self):
        "\n        A character code (one of 'biufcmMOSUV') identifying the general kind of data.\n        "
        return self._dtype.kind

    @property
    def itemsize(self):
        '\n        The element size of this data-type object.\n        '
        return self._dtype.itemsize

class PandasArray(OpsMixin, NDArrayBackedExtensionArray, NDArrayOperatorsMixin, ObjectStringArrayMixin):
    '\n    A pandas ExtensionArray for NumPy data.\n\n    .. versionadded:: 0.24.0\n\n    This is mostly for internal compatibility, and is not especially\n    useful on its own.\n\n    Parameters\n    ----------\n    values : ndarray\n        The NumPy ndarray to wrap. Must be 1-dimensional.\n    copy : bool, default False\n        Whether to copy `values`.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n    '
    _typ = 'npy_extension'
    __array_priority__ = 1000

    def __init__(self, values, copy=False):
        if isinstance(values, type(self)):
            values = values._ndarray
        if (not isinstance(values, np.ndarray)):
            raise ValueError(f"'values' must be a NumPy array, not {type(values).__name__}")
        if (values.ndim == 0):
            raise ValueError('PandasArray must be 1-dimensional.')
        if copy:
            values = values.copy()
        self._ndarray = values
        self._dtype = PandasDtype(values.dtype)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if isinstance(dtype, PandasDtype):
            dtype = dtype._dtype
        result = np.asarray(scalars, dtype=dtype)
        if (copy and (result is scalars)):
            result = result.copy()
        return cls(result)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def _from_backing_data(self, arr):
        return type(self)(arr)

    @property
    def dtype(self):
        return self._dtype

    def __array__(self, dtype=None):
        return np.asarray(self._ndarray, dtype=dtype)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in (inputs + out):
            if (not isinstance(x, (self._HANDLED_TYPES + (PandasArray,)))):
                return NotImplemented
        if (ufunc not in [np.logical_or, np.bitwise_or, np.bitwise_xor]):
            result = ops.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
            if (result is not NotImplemented):
                return result
        inputs = tuple(((x._ndarray if isinstance(x, PandasArray) else x) for x in inputs))
        if out:
            kwargs['out'] = tuple(((x._ndarray if isinstance(x, PandasArray) else x) for x in out))
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if ((type(result) is tuple) and len(result)):
            if (not lib.is_scalar(result[0])):
                return tuple((type(self)(x) for x in result))
            else:
                return result
        elif (method == 'at'):
            return None
        else:
            if (not lib.is_scalar(result)):
                result = type(self)(result)
            return result

    def isna(self):
        return isna(self._ndarray)

    def _validate_fill_value(self, fill_value):
        if (fill_value is None):
            fill_value = self.dtype.na_value
        return fill_value

    def _values_for_factorize(self):
        return (self._ndarray, (- 1))

    def any(self, *, axis=None, out=None, keepdims=False, skipna=True):
        nv.validate_any((), {'out': out, 'keepdims': keepdims})
        result = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def all(self, *, axis=None, out=None, keepdims=False, skipna=True):
        nv.validate_all((), {'out': out, 'keepdims': keepdims})
        result = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def min(self, *, axis=None, skipna=True, **kwargs):
        nv.validate_min((), kwargs)
        result = nanops.nanmin(values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def max(self, *, axis=None, skipna=True, **kwargs):
        nv.validate_max((), kwargs)
        result = nanops.nanmax(values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def sum(self, *, axis=None, skipna=True, min_count=0, **kwargs):
        nv.validate_sum((), kwargs)
        result = nanops.nansum(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def prod(self, *, axis=None, skipna=True, min_count=0, **kwargs):
        nv.validate_prod((), kwargs)
        result = nanops.nanprod(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def mean(self, *, axis=None, dtype=None, out=None, keepdims=False, skipna=True):
        nv.validate_mean((), {'dtype': dtype, 'out': out, 'keepdims': keepdims})
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def median(self, *, axis=None, out=None, overwrite_input=False, keepdims=False, skipna=True):
        nv.validate_median((), {'out': out, 'overwrite_input': overwrite_input, 'keepdims': keepdims})
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def std(self, *, axis=None, dtype=None, out=None, ddof=1, keepdims=False, skipna=True):
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='std')
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def var(self, *, axis=None, dtype=None, out=None, ddof=1, keepdims=False, skipna=True):
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='var')
        result = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def sem(self, *, axis=None, dtype=None, out=None, ddof=1, keepdims=False, skipna=True):
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='sem')
        result = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def kurt(self, *, axis=None, dtype=None, out=None, keepdims=False, skipna=True):
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='kurt')
        result = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def skew(self, *, axis=None, dtype=None, out=None, keepdims=False, skipna=True):
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='skew')
        result = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        result = np.asarray(self._ndarray, dtype=dtype)
        if ((copy or (na_value is not lib.no_default)) and (result is self._ndarray)):
            result = result.copy()
        if (na_value is not lib.no_default):
            result[self.isna()] = na_value
        return result

    def __invert__(self):
        return type(self)((~ self._ndarray))

    def _cmp_method(self, other, op):
        if isinstance(other, PandasArray):
            other = other._ndarray
        pd_op = ops.get_array_op(op)
        result = pd_op(self._ndarray, other)
        if ((op is divmod) or (op is ops.rdivmod)):
            (a, b) = result
            if isinstance(a, np.ndarray):
                return (self._wrap_ndarray_result(a), self._wrap_ndarray_result(b))
            return (a, b)
        if isinstance(result, np.ndarray):
            return self._wrap_ndarray_result(result)
        return result
    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result):
        if (result.dtype == 'timedelta64[ns]'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._simple_new(result)
        return type(self)(result)
    _str_na_value = np.nan
