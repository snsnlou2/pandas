
from typing import TYPE_CHECKING, Type, Union
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas._typing import Scalar
from pandas.compat.numpy import function as nv
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_dtype_equal, is_integer_dtype, is_object_dtype, is_string_dtype, pandas_dtype
from pandas.core import ops
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays import FloatingArray, IntegerArray, PandasArray
from pandas.core.arrays.floating import FloatingDtype
from pandas.core.arrays.integer import _IntegerDtype
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna
if TYPE_CHECKING:
    import pyarrow

@register_extension_dtype
class StringDtype(ExtensionDtype):
    '\n    Extension dtype for string data.\n\n    .. versionadded:: 1.0.0\n\n    .. warning::\n\n       StringDtype is considered experimental. The implementation and\n       parts of the API may change without warning.\n\n       In particular, StringDtype.na_value may change to no longer be\n       ``numpy.nan``.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    Examples\n    --------\n    >>> pd.StringDtype()\n    StringDtype\n    '
    name = 'string'
    na_value = libmissing.NA

    @property
    def type(self):
        return str

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        return StringArray

    def __repr__(self):
        return 'StringDtype'

    def __from_arrow__(self, array):
        '\n        Construct StringArray from pyarrow Array/ChunkedArray.\n        '
        import pyarrow
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            str_arr = StringArray._from_sequence(np.array(arr))
            results.append(str_arr)
        return StringArray._concat_same_type(results)

class StringArray(PandasArray):
    '\n    Extension array for string data.\n\n    .. versionadded:: 1.0.0\n\n    .. warning::\n\n       StringArray is considered experimental. The implementation and\n       parts of the API may change without warning.\n\n    Parameters\n    ----------\n    values : array-like\n        The array of data.\n\n        .. warning::\n\n           Currently, this expects an object-dtype ndarray\n           where the elements are Python strings or :attr:`pandas.NA`.\n           This may change without warning in the future. Use\n           :meth:`pandas.array` with ``dtype="string"`` for a stable way of\n           creating a `StringArray` from any sequence.\n\n    copy : bool, default False\n        Whether to copy the array of data.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    See Also\n    --------\n    array\n        The recommended function for creating a StringArray.\n    Series.str\n        The string methods are available on Series backed by\n        a StringArray.\n\n    Notes\n    -----\n    StringArray returns a BooleanArray for comparison methods.\n\n    Examples\n    --------\n    >>> pd.array([\'This is\', \'some text\', None, \'data.\'], dtype="string")\n    <StringArray>\n    [\'This is\', \'some text\', <NA>, \'data.\']\n    Length: 4, dtype: string\n\n    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``\n    will convert the values to strings.\n\n    >>> pd.array([\'1\', 1], dtype="object")\n    <PandasArray>\n    [\'1\', 1]\n    Length: 2, dtype: object\n    >>> pd.array([\'1\', 1], dtype="string")\n    <StringArray>\n    [\'1\', \'1\']\n    Length: 2, dtype: string\n\n    However, instantiating StringArrays directly with non-strings will raise an error.\n\n    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:\n\n    >>> pd.array(["a", None, "c"], dtype="string") == "a"\n    <BooleanArray>\n    [True, <NA>, False]\n    Length: 3, dtype: boolean\n    '
    _typ = 'extension'

    def __init__(self, values, copy=False):
        values = extract_array(values)
        super().__init__(values, copy=copy)
        self._dtype = StringDtype()
        if (not isinstance(values, type(self))):
            self._validate()

    def _validate(self):
        'Validate that we only store NA or strings.'
        if (len(self._ndarray) and (not lib.is_string_array(self._ndarray, skipna=True))):
            raise ValueError('StringArray requires a sequence of strings or pandas.NA')
        if (self._ndarray.dtype != 'object'):
            raise ValueError(f"StringArray requires a sequence of strings or pandas.NA. Got '{self._ndarray.dtype}' dtype instead.")

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if dtype:
            assert (dtype == 'string')
        from pandas.core.arrays.masked import BaseMaskedArray
        if isinstance(scalars, BaseMaskedArray):
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            result[na_values] = StringDtype.na_value
        else:
            result = lib.ensure_string_array(scalars, na_value=StringDtype.na_value, copy=copy)
        new_string_array = object.__new__(cls)
        new_string_array._dtype = StringDtype()
        new_string_array._ndarray = result
        return new_string_array

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    def __arrow_array__(self, type=None):
        '\n        Convert myself into a pyarrow Array.\n        '
        import pyarrow as pa
        if (type is None):
            type = pa.string()
        values = self._ndarray.copy()
        values[self.isna()] = None
        return pa.array(values, type=type, from_pandas=True)

    def _values_for_factorize(self):
        arr = self._ndarray.copy()
        mask = self.isna()
        arr[mask] = (- 1)
        return (arr, (- 1))

    def __setitem__(self, key, value):
        value = extract_array(value, extract_numpy=True)
        if isinstance(value, type(self)):
            value = value._ndarray
        key = check_array_indexer(self, key)
        scalar_key = lib.is_scalar(key)
        scalar_value = lib.is_scalar(value)
        if (scalar_key and (not scalar_value)):
            raise ValueError('setting an array element with a sequence.')
        if scalar_value:
            if isna(value):
                value = StringDtype.na_value
            elif (not isinstance(value, str)):
                raise ValueError(f"Cannot set non-string value '{value}' into a StringArray.")
        else:
            if (not is_array_like(value)):
                value = np.asarray(value, dtype=object)
            if (len(value) and (not lib.is_string_array(value, skipna=True))):
                raise ValueError('Must provide strings.')
        super().__setitem__(key, value)

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self.dtype):
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, _IntegerDtype):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype.numpy_dtype)
            return IntegerArray(values, mask, copy=False)
        elif isinstance(dtype, FloatingDtype):
            arr = self.copy()
            mask = self.isna()
            arr[mask] = '0'
            values = arr.astype(dtype.numpy_dtype)
            return FloatingArray(values, mask, copy=False)
        elif np.issubdtype(dtype, np.floating):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype)
            values[mask] = np.nan
            return values
        return super().astype(dtype, copy)

    def _reduce(self, name, *, skipna=True, **kwargs):
        if (name in ['min', 'max']):
            return getattr(self, name)(skipna=skipna)
        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    def min(self, axis=None, skipna=True, **kwargs):
        nv.validate_min((), kwargs)
        result = masked_reductions.min(values=self.to_numpy(), mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def max(self, axis=None, skipna=True, **kwargs):
        nv.validate_max((), kwargs)
        result = masked_reductions.max(values=self.to_numpy(), mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def value_counts(self, dropna=False):
        from pandas import value_counts
        return value_counts(self._ndarray, dropna=dropna).astype('Int64')

    def memory_usage(self, deep=False):
        result = self._ndarray.nbytes
        if deep:
            return (result + lib.memory_usage_of_objects(self._ndarray))
        return result

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray
        if isinstance(other, StringArray):
            other = other._ndarray
        mask = (isna(self) | isna(other))
        valid = (~ mask)
        if (not lib.is_scalar(other)):
            if (len(other) != len(self)):
                raise ValueError(f'Lengths of operands do not match: {len(self)} != {len(other)}')
            other = np.asarray(other)
            other = other[valid]
        if (op.__name__ in ops.ARITHMETIC_BINOPS):
            result = np.empty_like(self._ndarray, dtype='object')
            result[mask] = StringDtype.na_value
            result[valid] = op(self._ndarray[valid], other)
            return StringArray(result)
        else:
            result = np.zeros(len(self._ndarray), dtype='bool')
            result[valid] = op(self._ndarray[valid], other)
            return BooleanArray(result, mask)
    _arith_method = _cmp_method
    _str_na_value = StringDtype.na_value

    def _str_map(self, f, na_value=None, dtype=None):
        from pandas.arrays import BooleanArray, IntegerArray, StringArray
        from pandas.core.arrays.string_ import StringDtype
        if (dtype is None):
            dtype = StringDtype()
        if (na_value is None):
            na_value = self.dtype.na_value
        mask = isna(self)
        arr = np.asarray(self)
        if (is_integer_dtype(dtype) or is_bool_dtype(dtype)):
            constructor: Union[(Type[IntegerArray], Type[BooleanArray])]
            if is_integer_dtype(dtype):
                constructor = IntegerArray
            else:
                constructor = BooleanArray
            na_value_is_na = isna(na_value)
            if na_value_is_na:
                na_value = 1
            result = lib.map_infer_mask(arr, f, mask.view('uint8'), convert=False, na_value=na_value, dtype=np.dtype(dtype))
            if (not na_value_is_na):
                mask[:] = False
            return constructor(result, mask)
        elif (is_string_dtype(dtype) and (not is_object_dtype(dtype))):
            result = lib.map_infer_mask(arr, f, mask.view('uint8'), convert=False, na_value=na_value)
            return StringArray(result)
        else:
            return lib.map_infer_mask(arr, f, mask.view('uint8'))
