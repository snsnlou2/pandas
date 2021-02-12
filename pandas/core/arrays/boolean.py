
import numbers
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
import warnings
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas._typing import ArrayLike, Dtype
from pandas.compat.numpy import function as nv
from pandas.core.dtypes.common import is_bool_dtype, is_float, is_float_dtype, is_integer_dtype, is_list_like, is_numeric_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from .masked import BaseMaskedArray, BaseMaskedDtype
if TYPE_CHECKING:
    import pyarrow

@register_extension_dtype
class BooleanDtype(BaseMaskedDtype):
    '\n    Extension dtype for boolean data.\n\n    .. versionadded:: 1.0.0\n\n    .. warning::\n\n       BooleanDtype is considered experimental. The implementation and\n       parts of the API may change without warning.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    Examples\n    --------\n    >>> pd.BooleanDtype()\n    BooleanDtype\n    '
    name = 'boolean'

    @property
    def type(self):
        return np.bool_

    @property
    def kind(self):
        return 'b'

    @property
    def numpy_dtype(self):
        return np.dtype('bool')

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        return BooleanArray

    def __repr__(self):
        return 'BooleanDtype'

    @property
    def _is_boolean(self):
        return True

    @property
    def _is_numeric(self):
        return True

    def __from_arrow__(self, array):
        '\n        Construct BooleanArray from pyarrow Array/ChunkedArray.\n        '
        import pyarrow
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            bool_arr = BooleanArray._from_sequence(np.array(arr))
            results.append(bool_arr)
        return BooleanArray._concat_same_type(results)

def coerce_to_array(values, mask=None, copy=False):
    '\n    Coerce the input values array to numpy arrays with a mask.\n\n    Parameters\n    ----------\n    values : 1D list-like\n    mask : bool 1D array, optional\n    copy : bool, default False\n        if True, copy the input\n\n    Returns\n    -------\n    tuple of (values, mask)\n    '
    if isinstance(values, BooleanArray):
        if (mask is not None):
            raise ValueError('cannot pass mask for BooleanArray input')
        (values, mask) = (values._data, values._mask)
        if copy:
            values = values.copy()
            mask = mask.copy()
        return (values, mask)
    mask_values = None
    if (isinstance(values, np.ndarray) and (values.dtype == np.bool_)):
        if copy:
            values = values.copy()
    elif (isinstance(values, np.ndarray) and is_numeric_dtype(values.dtype)):
        mask_values = isna(values)
        values_bool = np.zeros(len(values), dtype=bool)
        values_bool[(~ mask_values)] = values[(~ mask_values)].astype(bool)
        if (not np.all((values_bool[(~ mask_values)].astype(values.dtype) == values[(~ mask_values)]))):
            raise TypeError('Need to pass bool-like values')
        values = values_bool
    else:
        values_object = np.asarray(values, dtype=object)
        inferred_dtype = lib.infer_dtype(values_object, skipna=True)
        integer_like = ('floating', 'integer', 'mixed-integer-float')
        if (inferred_dtype not in (('boolean', 'empty') + integer_like)):
            raise TypeError('Need to pass bool-like values')
        mask_values = isna(values_object)
        values = np.zeros(len(values), dtype=bool)
        values[(~ mask_values)] = values_object[(~ mask_values)].astype(bool)
        if ((inferred_dtype in integer_like) and (not np.all((values[(~ mask_values)].astype(float) == values_object[(~ mask_values)].astype(float))))):
            raise TypeError('Need to pass bool-like values')
    if ((mask is None) and (mask_values is None)):
        mask = np.zeros(len(values), dtype=bool)
    elif (mask is None):
        mask = mask_values
    elif (isinstance(mask, np.ndarray) and (mask.dtype == np.bool_)):
        if (mask_values is not None):
            mask = (mask | mask_values)
        elif copy:
            mask = mask.copy()
    else:
        mask = np.array(mask, dtype=bool)
        if (mask_values is not None):
            mask = (mask | mask_values)
    if (values.ndim != 1):
        raise ValueError('values must be a 1D list-like')
    if (mask.ndim != 1):
        raise ValueError('mask must be a 1D list-like')
    return (values, mask)

class BooleanArray(BaseMaskedArray):
    '\n    Array of boolean (True/False) data with missing values.\n\n    This is a pandas Extension array for boolean data, under the hood\n    represented by 2 numpy arrays: a boolean array with the data and\n    a boolean array with the mask (True indicating missing).\n\n    BooleanArray implements Kleene logic (sometimes called three-value\n    logic) for logical operations. See :ref:`boolean.kleene` for more.\n\n    To construct an BooleanArray from generic array-like input, use\n    :func:`pandas.array` specifying ``dtype="boolean"`` (see examples\n    below).\n\n    .. versionadded:: 1.0.0\n\n    .. warning::\n\n       BooleanArray is considered experimental. The implementation and\n       parts of the API may change without warning.\n\n    Parameters\n    ----------\n    values : numpy.ndarray\n        A 1-d boolean-dtype array with the data.\n    mask : numpy.ndarray\n        A 1-d boolean-dtype array indicating missing values (True\n        indicates missing).\n    copy : bool, default False\n        Whether to copy the `values` and `mask` arrays.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    Returns\n    -------\n    BooleanArray\n\n    Examples\n    --------\n    Create an BooleanArray with :func:`pandas.array`:\n\n    >>> pd.array([True, False, None], dtype="boolean")\n    <BooleanArray>\n    [True, False, <NA>]\n    Length: 3, dtype: boolean\n    '
    _internal_fill_value = False

    def __init__(self, values, mask, copy=False):
        if (not (isinstance(values, np.ndarray) and (values.dtype == np.bool_))):
            raise TypeError("values should be boolean numpy array. Use the 'pd.array' function instead")
        self._dtype = BooleanDtype()
        super().__init__(values, mask, copy=copy)

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if dtype:
            assert (dtype == 'boolean')
        (values, mask) = coerce_to_array(scalars, copy=copy)
        return BooleanArray(values, mask)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):

        def map_string(s):
            if isna(s):
                return s
            elif (s in ['True', 'TRUE', 'true', '1', '1.0']):
                return True
            elif (s in ['False', 'FALSE', 'false', '0', '0.0']):
                return False
            else:
                raise ValueError(f'{s} cannot be cast to bool')
        scalars = [map_string(x) for x in strings]
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)
    _HANDLED_TYPES = (np.ndarray, numbers.Number, bool, np.bool_)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if (method == 'reduce'):
            raise NotImplementedError("The 'reduce' method is not supported.")
        out = kwargs.get('out', ())
        for x in (inputs + out):
            if (not isinstance(x, (self._HANDLED_TYPES + (BooleanArray,)))):
                return NotImplemented
        result = ops.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if (result is not NotImplemented):
            return result
        mask = np.zeros(len(self), dtype=bool)
        inputs2 = []
        for x in inputs:
            if isinstance(x, BooleanArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x):
            if is_bool_dtype(x.dtype):
                m = mask.copy()
                return BooleanArray(x, m)
            else:
                x[mask] = np.nan
            return x
        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if isinstance(result, tuple):
            tuple((reconstruct(x) for x in result))
        else:
            return reconstruct(result)

    def _coerce_to_array(self, value):
        return coerce_to_array(value)

    def astype(self, dtype, copy=True):
        "\n        Cast to a NumPy array or ExtensionArray with 'dtype'.\n\n        Parameters\n        ----------\n        dtype : str or dtype\n            Typecode or data-type to which the array is cast.\n        copy : bool, default True\n            Whether to copy the data, even if not necessary. If False,\n            a copy is made only if the old dtype does not match the\n            new dtype.\n\n        Returns\n        -------\n        ndarray or ExtensionArray\n            NumPy ndarray, BooleanArray or IntegerArray with 'dtype' for its dtype.\n\n        Raises\n        ------\n        TypeError\n            if incompatible type with an BooleanDtype, equivalent of same_kind\n            casting\n        "
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy)
        if is_bool_dtype(dtype):
            if self._hasna:
                raise ValueError('cannot convert float NaN to bool')
            else:
                return self._data.astype(dtype, copy=copy)
        if (is_integer_dtype(dtype) and self._hasna):
            raise ValueError('cannot convert NA to integer')
        na_value = self._na_value
        if is_float_dtype(dtype):
            na_value = np.nan
        return self.to_numpy(dtype=dtype, na_value=na_value, copy=False)

    def _values_for_argsort(self):
        '\n        Return values for sorting.\n\n        Returns\n        -------\n        ndarray\n            The transformed values should maintain the ordering between values\n            within the array.\n\n        See Also\n        --------\n        ExtensionArray.argsort : Return the indices that would sort this array.\n        '
        data = self._data.copy()
        data[self._mask] = (- 1)
        return data

    def any(self, *, skipna=True, **kwargs):
        '\n        Return whether any element is True.\n\n        Returns False unless there is at least one element that is True.\n        By default, NAs are skipped. If ``skipna=False`` is specified and\n        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`\n        is used as for logical operations.\n\n        Parameters\n        ----------\n        skipna : bool, default True\n            Exclude NA values. If the entire array is NA and `skipna` is\n            True, then the result will be False, as for an empty array.\n            If `skipna` is False, the result will still be True if there is\n            at least one element that is True, otherwise NA will be returned\n            if there are NA\'s present.\n        **kwargs : any, default None\n            Additional keywords have no effect but might be accepted for\n            compatibility with NumPy.\n\n        Returns\n        -------\n        bool or :attr:`pandas.NA`\n\n        See Also\n        --------\n        numpy.any : Numpy version of this method.\n        BooleanArray.all : Return whether all elements are True.\n\n        Examples\n        --------\n        The result indicates whether any element is True (and by default\n        skips NAs):\n\n        >>> pd.array([True, False, True]).any()\n        True\n        >>> pd.array([True, False, pd.NA]).any()\n        True\n        >>> pd.array([False, False, pd.NA]).any()\n        False\n        >>> pd.array([], dtype="boolean").any()\n        False\n        >>> pd.array([pd.NA], dtype="boolean").any()\n        False\n\n        With ``skipna=False``, the result can be NA if this is logically\n        required (whether ``pd.NA`` is True or False influences the result):\n\n        >>> pd.array([True, False, pd.NA]).any(skipna=False)\n        True\n        >>> pd.array([False, False, pd.NA]).any(skipna=False)\n        <NA>\n        '
        kwargs.pop('axis', None)
        nv.validate_any((), kwargs)
        values = self._data.copy()
        np.putmask(values, self._mask, False)
        result = values.any()
        if skipna:
            return result
        elif (result or (len(self) == 0) or (not self._mask.any())):
            return result
        else:
            return self.dtype.na_value

    def all(self, *, skipna=True, **kwargs):
        '\n        Return whether all elements are True.\n\n        Returns True unless there is at least one element that is False.\n        By default, NAs are skipped. If ``skipna=False`` is specified and\n        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`\n        is used as for logical operations.\n\n        Parameters\n        ----------\n        skipna : bool, default True\n            Exclude NA values. If the entire array is NA and `skipna` is\n            True, then the result will be True, as for an empty array.\n            If `skipna` is False, the result will still be False if there is\n            at least one element that is False, otherwise NA will be returned\n            if there are NA\'s present.\n        **kwargs : any, default None\n            Additional keywords have no effect but might be accepted for\n            compatibility with NumPy.\n\n        Returns\n        -------\n        bool or :attr:`pandas.NA`\n\n        See Also\n        --------\n        numpy.all : Numpy version of this method.\n        BooleanArray.any : Return whether any element is True.\n\n        Examples\n        --------\n        The result indicates whether any element is True (and by default\n        skips NAs):\n\n        >>> pd.array([True, True, pd.NA]).all()\n        True\n        >>> pd.array([True, False, pd.NA]).all()\n        False\n        >>> pd.array([], dtype="boolean").all()\n        True\n        >>> pd.array([pd.NA], dtype="boolean").all()\n        True\n\n        With ``skipna=False``, the result can be NA if this is logically\n        required (whether ``pd.NA`` is True or False influences the result):\n\n        >>> pd.array([True, True, pd.NA]).all(skipna=False)\n        <NA>\n        >>> pd.array([True, False, pd.NA]).all(skipna=False)\n        False\n        '
        kwargs.pop('axis', None)
        nv.validate_all((), kwargs)
        values = self._data.copy()
        np.putmask(values, self._mask, True)
        result = values.all()
        if skipna:
            return result
        elif ((not result) or (len(self) == 0) or (not self._mask.any())):
            return result
        else:
            return self.dtype.na_value

    def _logical_method(self, other, op):
        assert (op.__name__ in {'or_', 'ror_', 'and_', 'rand_', 'xor', 'rxor'})
        other_is_booleanarray = isinstance(other, BooleanArray)
        other_is_scalar = lib.is_scalar(other)
        mask = None
        if other_is_booleanarray:
            (other, mask) = (other._data, other._mask)
        elif is_list_like(other):
            other = np.asarray(other, dtype='bool')
            if (other.ndim > 1):
                raise NotImplementedError('can only perform ops with 1-d structures')
            (other, mask) = coerce_to_array(other, copy=False)
        elif isinstance(other, np.bool_):
            other = other.item()
        if (other_is_scalar and (other is not libmissing.NA) and (not lib.is_bool(other))):
            raise TypeError(f"'other' should be pandas.NA or a bool. Got {type(other).__name__} instead.")
        if ((not other_is_scalar) and (len(self) != len(other))):
            raise ValueError('Lengths must match to compare')
        if (op.__name__ in {'or_', 'ror_'}):
            (result, mask) = ops.kleene_or(self._data, other, self._mask, mask)
        elif (op.__name__ in {'and_', 'rand_'}):
            (result, mask) = ops.kleene_and(self._data, other, self._mask, mask)
        elif (op.__name__ in {'xor', 'rxor'}):
            (result, mask) = ops.kleene_xor(self._data, other, self._mask, mask)
        return BooleanArray(result, mask)

    def _cmp_method(self, other, op):
        from pandas.arrays import FloatingArray, IntegerArray
        if isinstance(other, (IntegerArray, FloatingArray)):
            return NotImplemented
        mask = None
        if isinstance(other, BooleanArray):
            (other, mask) = (other._data, other._mask)
        elif is_list_like(other):
            other = np.asarray(other)
            if (other.ndim > 1):
                raise NotImplementedError('can only perform ops with 1-d structures')
            if (len(self) != len(other)):
                raise ValueError('Lengths must match to compare')
        if (other is libmissing.NA):
            result = np.zeros_like(self._data)
            mask = np.ones_like(self._data)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'elementwise', FutureWarning)
                with np.errstate(all='ignore'):
                    result = op(self._data, other)
            if (mask is None):
                mask = self._mask.copy()
            else:
                mask = (self._mask | mask)
        return BooleanArray(result, mask, copy=False)

    def _arith_method(self, other, op):
        mask = None
        op_name = op.__name__
        if isinstance(other, BooleanArray):
            (other, mask) = (other._data, other._mask)
        elif is_list_like(other):
            other = np.asarray(other)
            if (other.ndim > 1):
                raise NotImplementedError('can only perform ops with 1-d structures')
            if (len(self) != len(other)):
                raise ValueError('Lengths must match')
        if (mask is None):
            mask = self._mask
            if (other is libmissing.NA):
                mask |= True
        else:
            mask = (self._mask | mask)
        if (other is libmissing.NA):
            if (op_name in {'floordiv', 'rfloordiv', 'mod', 'rmod', 'pow', 'rpow'}):
                dtype = 'int8'
            else:
                dtype = 'bool'
            result = np.zeros(len(self._data), dtype=dtype)
        else:
            if ((op_name in {'pow', 'rpow'}) and isinstance(other, np.bool_)):
                other = bool(other)
            with np.errstate(all='ignore'):
                result = op(self._data, other)
        if (op_name == 'divmod'):
            (div, mod) = result
            return (self._maybe_mask_result(div, mask, other, 'floordiv'), self._maybe_mask_result(mod, mask, other, 'mod'))
        return self._maybe_mask_result(result, mask, other, op_name)

    def _reduce(self, name, *, skipna=True, **kwargs):
        if (name in {'any', 'all'}):
            return getattr(self, name)(skipna=skipna, **kwargs)
        return super()._reduce(name, skipna=skipna, **kwargs)

    def _maybe_mask_result(self, result, mask, other, op_name):
        '\n        Parameters\n        ----------\n        result : array-like\n        mask : array-like bool\n        other : scalar or array-like\n        op_name : str\n        '
        if ((is_float_dtype(other) or is_float(other)) or (op_name in ['rtruediv', 'truediv'])):
            from pandas.core.arrays import FloatingArray
            return FloatingArray(result, mask, copy=False)
        elif is_bool_dtype(result):
            return BooleanArray(result, mask, copy=False)
        elif is_integer_dtype(result):
            from pandas.core.arrays import IntegerArray
            return IntegerArray(result, mask, copy=False)
        else:
            result[mask] = np.nan
            return result
