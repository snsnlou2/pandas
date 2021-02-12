
'\nData structure for 1-dimensional cross-sectional and time series data\n'
from io import StringIO
from shutil import get_terminal_size
from textwrap import dedent
from typing import IO, TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Type, Union
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib, properties, reshape, tslibs
from pandas._libs.lib import no_default
from pandas._typing import AggFuncType, ArrayLike, Axis, DtypeObj, FrameOrSeriesUnion, IndexKeyFunc, Label, StorageOptions, ValueKeyFunc
from pandas.compat.numpy import function as nv
from pandas.errors import InvalidIndexError
from pandas.util._decorators import Appender, Substitution, doc
from pandas.util._validators import validate_bool_kwarg, validate_percentile
from pandas.core.dtypes.cast import convert_dtypes, maybe_cast_to_extension_array, validate_numeric_casting
from pandas.core.dtypes.common import ensure_platform_int, is_bool, is_categorical_dtype, is_dict_like, is_extension_array_dtype, is_integer, is_iterator, is_list_like, is_object_dtype, is_scalar, validate_all_hashable
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna, remove_na_arraylike
from pandas.core import algorithms, base, generic, missing, nanops, ops
from pandas.core.accessor import CachedAccessor
from pandas.core.aggregation import aggregate, transform
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
import pandas.core.common as com
from pandas.core.construction import array as pd_array, create_series_with_explicit_dtype, extract_array, is_empty_data, sanitize_array
from pandas.core.generic import NDFrame
from pandas.core.indexers import deprecate_ndim_indexing, unpack_1tuple
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import CategoricalIndex, Float64Index, Index, MultiIndex, ensure_index
import pandas.core.indexes.base as ibase
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.indexing import check_bool_indexer
from pandas.core.internals import SingleBlockManager
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import ensure_key_mapped, nargsort
from pandas.core.strings import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
import pandas.plotting
if TYPE_CHECKING:
    from pandas._typing import TimedeltaConvertibleTypes, TimestampConvertibleTypes
    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy
    from pandas.core.resample import Resampler
__all__ = ['Series']
_shared_doc_kwargs = {'axes': 'index', 'klass': 'Series', 'axes_single_arg': "{0 or 'index'}", 'axis': "axis : {0 or 'index'}\n        Parameter needed for compatibility with DataFrame.", 'inplace': 'inplace : boolean, default False\n        If True, performs operation inplace and returns None.', 'unique': 'np.ndarray', 'duplicated': 'Series', 'optional_by': '', 'optional_mapper': '', 'optional_labels': '', 'optional_axis': '', 'replace_iloc': '\n    This differs from updating with ``.loc`` or ``.iloc``, which require\n    you to specify a location to update with some value.'}

def _coerce_method(converter):
    '\n    Install the scalar coercion methods.\n    '

    def wrapper(self):
        if (len(self) == 1):
            return converter(self.iloc[0])
        raise TypeError(f'cannot convert the series to {converter}')
    wrapper.__name__ = f'__{converter.__name__}__'
    return wrapper

class Series(base.IndexOpsMixin, generic.NDFrame):
    '\n    One-dimensional ndarray with axis labels (including time series).\n\n    Labels need not be unique but must be a hashable type. The object\n    supports both integer- and label-based indexing and provides a host of\n    methods for performing operations involving the index. Statistical\n    methods from ndarray have been overridden to automatically exclude\n    missing data (currently represented as NaN).\n\n    Operations between Series (+, -, /, *, **) align values based on their\n    associated index values-- they need not be the same length. The result\n    index will be the sorted union of the two indexes.\n\n    Parameters\n    ----------\n    data : array-like, Iterable, dict, or scalar value\n        Contains data stored in Series. If data is a dict, argument order is\n        maintained.\n    index : array-like or Index (1d)\n        Values must be hashable and have the same length as `data`.\n        Non-unique index values are allowed. Will default to\n        RangeIndex (0, 1, 2, ..., n) if not provided. If data is dict-like\n        and index is None, then the values in the index are used to\n        reindex the Series after it is created using the keys in the data.\n    dtype : str, numpy.dtype, or ExtensionDtype, optional\n        Data type for the output Series. If not specified, this will be\n        inferred from `data`.\n        See the :ref:`user guide <basics.dtypes>` for more usages.\n    name : str, optional\n        The name to give to the Series.\n    copy : bool, default False\n        Copy input data.\n    '
    _typ = 'series'
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)
    _metadata = ['name']
    _internal_names_set = ({'index'} | generic.NDFrame._internal_names_set)
    _accessors = {'dt', 'cat', 'str', 'sparse'}
    _hidden_attrs = ((base.IndexOpsMixin._hidden_attrs | generic.NDFrame._hidden_attrs) | frozenset(['compress', 'ptp']))
    hasnans = property(base.IndexOpsMixin.hasnans.func, doc=base.IndexOpsMixin.hasnans.__doc__)
    __hash__ = generic.NDFrame.__hash__

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        if (isinstance(data, SingleBlockManager) and (index is None) and (dtype is None) and (copy is False)):
            NDFrame.__init__(self, data)
            self.name = name
            return
        if fastpath:
            if (not isinstance(data, SingleBlockManager)):
                data = SingleBlockManager.from_array(data, index)
            if copy:
                data = data.copy()
            if (index is None):
                index = data.index
        else:
            name = ibase.maybe_extract_name(name, data, type(self))
            if (is_empty_data(data) and (dtype is None)):
                warnings.warn("The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.", DeprecationWarning, stacklevel=2)
            if (index is not None):
                index = ensure_index(index)
            if (data is None):
                data = {}
            if (dtype is not None):
                dtype = self._validate_dtype(dtype)
            if isinstance(data, MultiIndex):
                raise NotImplementedError('initializing a Series from a MultiIndex is not supported')
            elif isinstance(data, Index):
                if (dtype is not None):
                    data = data.astype(dtype)
                else:
                    data = data._values.copy()
                copy = False
            elif isinstance(data, np.ndarray):
                if len(data.dtype):
                    raise ValueError('Cannot construct a Series from an ndarray with compound dtype.  Use DataFrame instead.')
            elif isinstance(data, Series):
                if (index is None):
                    index = data.index
                else:
                    data = data.reindex(index, copy=copy)
                    copy = False
                data = data._mgr
            elif is_dict_like(data):
                (data, index) = self._init_dict(data, index, dtype)
                dtype = None
                copy = False
            elif isinstance(data, SingleBlockManager):
                if (index is None):
                    index = data.index
                elif ((not data.index.equals(index)) or copy):
                    raise AssertionError('Cannot pass both SingleBlockManager `data` argument and a different `index` argument. `copy` must be False.')
            elif is_extension_array_dtype(data):
                pass
            elif isinstance(data, (set, frozenset)):
                raise TypeError(f"'{type(data).__name__}' type is unordered")
            else:
                data = com.maybe_iterable_to_list(data)
            if (index is None):
                if (not is_list_like(data)):
                    data = [data]
                index = ibase.default_index(len(data))
            elif is_list_like(data):
                try:
                    if (len(index) != len(data)):
                        raise ValueError(f'Length of passed values is {len(data)}, index implies {len(index)}.')
                except TypeError:
                    pass
            if isinstance(data, SingleBlockManager):
                if (dtype is not None):
                    data = data.astype(dtype=dtype, errors='ignore', copy=copy)
                elif copy:
                    data = data.copy()
            else:
                data = sanitize_array(data, index, dtype, copy, raise_cast_failure=True)
                data = SingleBlockManager.from_array(data, index)
        generic.NDFrame.__init__(self, data)
        self.name = name
        self._set_axis(0, index, fastpath=True)

    def _init_dict(self, data, index=None, dtype=None):
        '\n        Derive the "_mgr" and "index" attributes of a new Series from a\n        dictionary input.\n\n        Parameters\n        ----------\n        data : dict or dict-like\n            Data used to populate the new Series.\n        index : Index or index-like, default None\n            Index for the new Series: if None, use dict keys.\n        dtype : dtype, default None\n            The dtype for the new Series: if None, infer from data.\n\n        Returns\n        -------\n        _data : BlockManager for the new Series\n        index : index for the new Series\n        '
        if data:
            keys = tuple(data.keys())
            values = list(data.values())
        elif (index is not None):
            values = na_value_for_dtype(dtype)
            keys = index
        else:
            (keys, values) = ((), [])
        s = create_series_with_explicit_dtype(values, index=keys, dtype=dtype, dtype_if_empty=np.float64)
        if (data and (index is not None)):
            s = s.reindex(index, copy=False)
        return (s._mgr, s.index)

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        from pandas.core.frame import DataFrame
        return DataFrame

    @property
    def _can_hold_na(self):
        return self._mgr._can_hold_na
    _index = None

    def _set_axis(self, axis, labels, fastpath=False):
        '\n        Override generic, we want to set the _typ here.\n\n        This is called from the cython code when we set the `index` attribute\n        directly, e.g. `series.index = [1, 2, 3]`.\n        '
        if (not fastpath):
            labels = ensure_index(labels)
        if labels._is_all_dates:
            deep_labels = labels
            if isinstance(labels, CategoricalIndex):
                deep_labels = labels.categories
            if (not isinstance(deep_labels, (DatetimeIndex, PeriodIndex, TimedeltaIndex))):
                try:
                    labels = DatetimeIndex(labels)
                    if fastpath:
                        self._mgr.set_axis(axis, labels)
                except (tslibs.OutOfBoundsDatetime, ValueError):
                    pass
        object.__setattr__(self, '_index', labels)
        if (not fastpath):
            self._mgr.set_axis(axis, labels)

    @property
    def dtype(self):
        '\n        Return the dtype object of the underlying data.\n        '
        return self._mgr.dtype

    @property
    def dtypes(self):
        '\n        Return the dtype object of the underlying data.\n        '
        return self.dtype

    @property
    def name(self):
        '\n        Return the name of the Series.\n\n        The name of a Series becomes its index or column name if it is used\n        to form a DataFrame. It is also used whenever displaying the Series\n        using the interpreter.\n\n        Returns\n        -------\n        label (hashable object)\n            The name of the Series, also the column name if part of a DataFrame.\n\n        See Also\n        --------\n        Series.rename : Sets the Series name when given a scalar input.\n        Index.name : Corresponding Index property.\n\n        Examples\n        --------\n        The Series name can be set initially when calling the constructor.\n\n        >>> s = pd.Series([1, 2, 3], dtype=np.int64, name=\'Numbers\')\n        >>> s\n        0    1\n        1    2\n        2    3\n        Name: Numbers, dtype: int64\n        >>> s.name = "Integers"\n        >>> s\n        0    1\n        1    2\n        2    3\n        Name: Integers, dtype: int64\n\n        The name of a Series within a DataFrame is its column name.\n\n        >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]],\n        ...                   columns=["Odd Numbers", "Even Numbers"])\n        >>> df\n           Odd Numbers  Even Numbers\n        0            1             2\n        1            3             4\n        2            5             6\n        >>> df["Even Numbers"].name\n        \'Even Numbers\'\n        '
        return self._name

    @name.setter
    def name(self, value):
        validate_all_hashable(value, error_name=f'{type(self).__name__}.name')
        object.__setattr__(self, '_name', value)

    @property
    def values(self):
        "\n        Return Series as ndarray or ndarray-like depending on the dtype.\n\n        .. warning::\n\n           We recommend using :attr:`Series.array` or\n           :meth:`Series.to_numpy`, depending on whether you need\n           a reference to the underlying data or a NumPy array.\n\n        Returns\n        -------\n        numpy.ndarray or ndarray-like\n\n        See Also\n        --------\n        Series.array : Reference to the underlying data.\n        Series.to_numpy : A NumPy array representing the underlying data.\n\n        Examples\n        --------\n        >>> pd.Series([1, 2, 3]).values\n        array([1, 2, 3])\n\n        >>> pd.Series(list('aabc')).values\n        array(['a', 'a', 'b', 'c'], dtype=object)\n\n        >>> pd.Series(list('aabc')).astype('category').values\n        ['a', 'a', 'b', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n\n        Timezone aware datetime data is converted to UTC:\n\n        >>> pd.Series(pd.date_range('20130101', periods=3,\n        ...                         tz='US/Eastern')).values\n        array(['2013-01-01T05:00:00.000000000',\n               '2013-01-02T05:00:00.000000000',\n               '2013-01-03T05:00:00.000000000'], dtype='datetime64[ns]')\n        "
        return self._mgr.external_values()

    @property
    def _values(self):
        '\n        Return the internal repr of this data (defined by Block.interval_values).\n        This are the values as stored in the Block (ndarray or ExtensionArray\n        depending on the Block class), with datetime64[ns] and timedelta64[ns]\n        wrapped in ExtensionArrays to match Index._values behavior.\n\n        Differs from the public ``.values`` for certain data types, because of\n        historical backwards compatibility of the public attribute (e.g. period\n        returns object ndarray and datetimetz a datetime64[ns] ndarray for\n        ``.values`` while it returns an ExtensionArray for ``._values`` in those\n        cases).\n\n        Differs from ``.array`` in that this still returns the numpy array if\n        the Block is backed by a numpy array (except for datetime64 and\n        timedelta64 dtypes), while ``.array`` ensures to always return an\n        ExtensionArray.\n\n        Overview:\n\n        dtype       | values        | _values       | array         |\n        ----------- | ------------- | ------------- | ------------- |\n        Numeric     | ndarray       | ndarray       | PandasArray   |\n        Category    | Categorical   | Categorical   | Categorical   |\n        dt64[ns]    | ndarray[M8ns] | DatetimeArray | DatetimeArray |\n        dt64[ns tz] | ndarray[M8ns] | DatetimeArray | DatetimeArray |\n        td64[ns]    | ndarray[m8ns] | TimedeltaArray| ndarray[m8ns] |\n        Period      | ndarray[obj]  | PeriodArray   | PeriodArray   |\n        Nullable    | EA            | EA            | EA            |\n\n        '
        return self._mgr.internal_values()

    @Appender(base.IndexOpsMixin.array.__doc__)
    @property
    def array(self):
        return self._mgr._block.array_values()

    def ravel(self, order='C'):
        '\n        Return the flattened underlying data as an ndarray.\n\n        Returns\n        -------\n        numpy.ndarray or ndarray-like\n            Flattened data of the Series.\n\n        See Also\n        --------\n        numpy.ndarray.ravel : Return a flattened array.\n        '
        return self._values.ravel(order=order)

    def __len__(self):
        '\n        Return the length of the Series.\n        '
        return len(self._mgr)

    def view(self, dtype=None):
        "\n        Create a new view of the Series.\n\n        This function will return a new Series with a view of the same\n        underlying values in memory, optionally reinterpreted with a new data\n        type. The new data type must preserve the same size in bytes as to not\n        cause index misalignment.\n\n        Parameters\n        ----------\n        dtype : data type\n            Data type object or one of their string representations.\n\n        Returns\n        -------\n        Series\n            A new Series object as a view of the same data in memory.\n\n        See Also\n        --------\n        numpy.ndarray.view : Equivalent numpy function to create a new view of\n            the same data in memory.\n\n        Notes\n        -----\n        Series are instantiated with ``dtype=float64`` by default. While\n        ``numpy.ndarray.view()`` will return a view with the same data type as\n        the original array, ``Series.view()`` (without specified dtype)\n        will try using ``float64`` and may fail if the original data type size\n        in bytes is not the same.\n\n        Examples\n        --------\n        >>> s = pd.Series([-2, -1, 0, 1, 2], dtype='int8')\n        >>> s\n        0   -2\n        1   -1\n        2    0\n        3    1\n        4    2\n        dtype: int8\n\n        The 8 bit signed integer representation of `-1` is `0b11111111`, but\n        the same bytes represent 255 if read as an 8 bit unsigned integer:\n\n        >>> us = s.view('uint8')\n        >>> us\n        0    254\n        1    255\n        2      0\n        3      1\n        4      2\n        dtype: uint8\n\n        The views share the same underlying values:\n\n        >>> us[0] = 128\n        >>> s\n        0   -128\n        1     -1\n        2      0\n        3      1\n        4      2\n        dtype: int8\n        "
        return self._constructor(self._values.view(dtype), index=self.index).__finalize__(self, method='view')
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)

    def __array__(self, dtype=None):
        '\n        Return the values as a NumPy array.\n\n        Users should not call this directly. Rather, it is invoked by\n        :func:`numpy.array` and :func:`numpy.asarray`.\n\n        Parameters\n        ----------\n        dtype : str or numpy.dtype, optional\n            The dtype to use for the resulting NumPy array. By default,\n            the dtype is inferred from the data.\n\n        Returns\n        -------\n        numpy.ndarray\n            The values in the series converted to a :class:`numpy.ndarray`\n            with the specified `dtype`.\n\n        See Also\n        --------\n        array : Create a new array from data.\n        Series.array : Zero-copy view to the array backing the Series.\n        Series.to_numpy : Series method for similar behavior.\n\n        Examples\n        --------\n        >>> ser = pd.Series([1, 2, 3])\n        >>> np.asarray(ser)\n        array([1, 2, 3])\n\n        For timezone-aware data, the timezones may be retained with\n        ``dtype=\'object\'``\n\n        >>> tzser = pd.Series(pd.date_range(\'2000\', periods=2, tz="CET"))\n        >>> np.asarray(tzser, dtype="object")\n        array([Timestamp(\'2000-01-01 00:00:00+0100\', tz=\'CET\', freq=\'D\'),\n               Timestamp(\'2000-01-02 00:00:00+0100\', tz=\'CET\', freq=\'D\')],\n              dtype=object)\n\n        Or the values may be localized to UTC and the tzinfo discarded with\n        ``dtype=\'datetime64[ns]\'``\n\n        >>> np.asarray(tzser, dtype="datetime64[ns]")  # doctest: +ELLIPSIS\n        array([\'1999-12-31T23:00:00.000000000\', ...],\n              dtype=\'datetime64[ns]\')\n        '
        return np.asarray(self.array, dtype)
    __float__ = _coerce_method(float)
    __long__ = _coerce_method(int)
    __int__ = _coerce_method(int)

    @property
    def axes(self):
        '\n        Return a list of the row axis labels.\n        '
        return [self.index]

    @Appender(generic.NDFrame.take.__doc__)
    def take(self, indices, axis=0, is_copy=None, **kwargs):
        if (is_copy is not None):
            warnings.warn("is_copy is deprecated and will be removed in a future version. 'take' always returns a copy, so there is no need to specify this.", FutureWarning, stacklevel=2)
        nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)
        new_index = self.index.take(indices)
        new_values = self._values.take(indices)
        result = self._constructor(new_values, index=new_index, fastpath=True)
        return result.__finalize__(self, method='take')

    def _take_with_is_copy(self, indices, axis=0):
        '\n        Internal version of the `take` method that sets the `_is_copy`\n        attribute to keep track of the parent dataframe (using in indexing\n        for the SettingWithCopyWarning). For Series this does the same\n        as the public take (it never sets `_is_copy`).\n\n        See the docstring of `take` for full explanation of the parameters.\n        '
        return self.take(indices=indices, axis=axis)

    def _ixs(self, i, axis=0):
        '\n        Return the i-th value or values in the Series by location.\n\n        Parameters\n        ----------\n        i : int\n\n        Returns\n        -------\n        scalar (int) or Series (slice, sequence)\n        '
        return self._values[i]

    def _slice(self, slobj, axis=0):
        return self._get_values(slobj)

    def __getitem__(self, key):
        key = com.apply_if_callable(key, self)
        if (key is Ellipsis):
            return self
        key_is_scalar = is_scalar(key)
        if isinstance(key, (list, tuple)):
            key = unpack_1tuple(key)
        if (is_integer(key) and self.index._should_fallback_to_positional()):
            return self._values[key]
        elif key_is_scalar:
            return self._get_value(key)
        if is_hashable(key):
            try:
                result = self._get_value(key)
                return result
            except (KeyError, TypeError):
                if (isinstance(key, tuple) and isinstance(self.index, MultiIndex)):
                    return self._get_values_tuple(key)
        if is_iterator(key):
            key = list(key)
        if com.is_bool_indexer(key):
            key = check_bool_indexer(self.index, key)
            key = np.asarray(key, dtype=bool)
            return self._get_values(key)
        return self._get_with(key)

    def _get_with(self, key):
        if isinstance(key, slice):
            slobj = self.index._convert_slice_indexer(key, kind='getitem')
            return self._slice(slobj)
        elif isinstance(key, ABCDataFrame):
            raise TypeError('Indexing a Series with DataFrame is not supported, use the appropriate DataFrame column')
        elif isinstance(key, tuple):
            return self._get_values_tuple(key)
        elif (not is_list_like(key)):
            return self.loc[key]
        if (not isinstance(key, (list, np.ndarray, ExtensionArray, Series, Index))):
            key = list(key)
        if isinstance(key, Index):
            key_type = key.inferred_type
        else:
            key_type = lib.infer_dtype(key, skipna=False)
        if (key_type == 'integer'):
            if (not self.index._should_fallback_to_positional()):
                return self.loc[key]
            else:
                return self.iloc[key]
        return self.loc[key]

    def _get_values_tuple(self, key):
        if com.any_none(*key):
            result = self._get_values(key)
            deprecate_ndim_indexing(result, stacklevel=5)
            return result
        if (not isinstance(self.index, MultiIndex)):
            raise KeyError('key of type tuple not found and not a MultiIndex')
        (indexer, new_index) = self.index.get_loc_level(key)
        return self._constructor(self._values[indexer], index=new_index).__finalize__(self)

    def _get_values(self, indexer):
        try:
            return self._constructor(self._mgr.get_slice(indexer)).__finalize__(self)
        except ValueError:
            return np.asarray(self._values[indexer])

    def _get_value(self, label, takeable=False):
        '\n        Quickly retrieve single value at passed index label.\n\n        Parameters\n        ----------\n        label : object\n        takeable : interpret the index as indexers, default False\n\n        Returns\n        -------\n        scalar value\n        '
        if takeable:
            return self._values[label]
        loc = self.index.get_loc(label)
        return self.index._get_values_for_loc(self, loc, label)

    def __setitem__(self, key, value):
        key = com.apply_if_callable(key, self)
        cacher_needs_updating = self._check_is_chained_assignment_possible()
        if (key is Ellipsis):
            key = slice(None)
        try:
            self._set_with_engine(key, value)
        except (KeyError, ValueError):
            values = self._values
            if (is_integer(key) and (not (self.index.inferred_type == 'integer'))):
                values[key] = value
            else:
                self.loc[key] = value
        except TypeError as err:
            if (isinstance(key, tuple) and (not isinstance(self.index, MultiIndex))):
                raise KeyError('key of type tuple not found and not a MultiIndex') from err
            if com.is_bool_indexer(key):
                key = check_bool_indexer(self.index, key)
                key = np.asarray(key, dtype=bool)
                try:
                    self._where((~ key), value, inplace=True)
                except InvalidIndexError:
                    self.iloc[key] = value
                return
            else:
                self._set_with(key, value)
        if cacher_needs_updating:
            self._maybe_update_cacher()

    def _set_with_engine(self, key, value):
        loc = self.index._engine.get_loc(key)
        validate_numeric_casting(self.dtype, value)
        self._values[loc] = value

    def _set_with(self, key, value):
        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind='getitem')
            return self._set_values(indexer, value)
        else:
            assert (not isinstance(key, tuple))
            if is_scalar(key):
                key = [key]
            if isinstance(key, Index):
                key_type = key.inferred_type
                key = key._values
            else:
                key_type = lib.infer_dtype(key, skipna=False)
            if (key_type == 'integer'):
                if (not self.index._should_fallback_to_positional()):
                    self._set_labels(key, value)
                else:
                    self._set_values(key, value)
            else:
                self.loc[key] = value

    def _set_labels(self, key, value):
        key = com.asarray_tuplesafe(key)
        indexer: np.ndarray = self.index.get_indexer(key)
        mask = (indexer == (- 1))
        if mask.any():
            raise KeyError(f'{key[mask]} not in index')
        self._set_values(indexer, value)

    def _set_values(self, key, value):
        if isinstance(key, Series):
            key = key._values
        self._mgr = self._mgr.setitem(indexer=key, value=value)
        self._maybe_update_cacher()

    def _set_value(self, label, value, takeable=False):
        '\n        Quickly set single value at passed label.\n\n        If label is not contained, a new object is created with the label\n        placed at the end of the result index.\n\n        Parameters\n        ----------\n        label : object\n            Partial indexing with MultiIndex not allowed.\n        value : object\n            Scalar value.\n        takeable : interpret the index as indexers, default False\n        '
        try:
            if takeable:
                self._values[label] = value
            else:
                loc = self.index.get_loc(label)
                validate_numeric_casting(self.dtype, value)
                self._values[loc] = value
        except KeyError:
            self.loc[label] = value

    @property
    def _is_mixed_type(self):
        return False

    def repeat(self, repeats, axis=None):
        "\n        Repeat elements of a Series.\n\n        Returns a new Series where each element of the current Series\n        is repeated consecutively a given number of times.\n\n        Parameters\n        ----------\n        repeats : int or array of ints\n            The number of repetitions for each element. This should be a\n            non-negative integer. Repeating 0 times will return an empty\n            Series.\n        axis : None\n            Must be ``None``. Has no effect but is accepted for compatibility\n            with numpy.\n\n        Returns\n        -------\n        Series\n            Newly created Series with repeated elements.\n\n        See Also\n        --------\n        Index.repeat : Equivalent function for Index.\n        numpy.repeat : Similar method for :class:`numpy.ndarray`.\n\n        Examples\n        --------\n        >>> s = pd.Series(['a', 'b', 'c'])\n        >>> s\n        0    a\n        1    b\n        2    c\n        dtype: object\n        >>> s.repeat(2)\n        0    a\n        0    a\n        1    b\n        1    b\n        2    c\n        2    c\n        dtype: object\n        >>> s.repeat([1, 2, 3])\n        0    a\n        1    b\n        1    b\n        2    c\n        2    c\n        2    c\n        dtype: object\n        "
        nv.validate_repeat((), {'axis': axis})
        new_index = self.index.repeat(repeats)
        new_values = self._values.repeat(repeats)
        return self._constructor(new_values, index=new_index).__finalize__(self, method='repeat')

    def reset_index(self, level=None, drop=False, name=None, inplace=False):
        "\n        Generate a new DataFrame or Series with the index reset.\n\n        This is useful when the index needs to be treated as a column, or\n        when the index is meaningless and needs to be reset to the default\n        before another operation.\n\n        Parameters\n        ----------\n        level : int, str, tuple, or list, default optional\n            For a Series with a MultiIndex, only remove the specified levels\n            from the index. Removes all levels by default.\n        drop : bool, default False\n            Just reset the index, without inserting it as a column in\n            the new DataFrame.\n        name : object, optional\n            The name to use for the column containing the original Series\n            values. Uses ``self.name`` by default. This argument is ignored\n            when `drop` is True.\n        inplace : bool, default False\n            Modify the Series in place (do not create a new object).\n\n        Returns\n        -------\n        Series or DataFrame or None\n            When `drop` is False (the default), a DataFrame is returned.\n            The newly created columns will come first in the DataFrame,\n            followed by the original Series values.\n            When `drop` is True, a `Series` is returned.\n            In either case, if ``inplace=True``, no value is returned.\n\n        See Also\n        --------\n        DataFrame.reset_index: Analogous function for DataFrame.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3, 4], name='foo',\n        ...               index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))\n\n        Generate a DataFrame with default index.\n\n        >>> s.reset_index()\n          idx  foo\n        0   a    1\n        1   b    2\n        2   c    3\n        3   d    4\n\n        To specify the name of the new column use `name`.\n\n        >>> s.reset_index(name='values')\n          idx  values\n        0   a       1\n        1   b       2\n        2   c       3\n        3   d       4\n\n        To generate a new Series with the default set `drop` to True.\n\n        >>> s.reset_index(drop=True)\n        0    1\n        1    2\n        2    3\n        3    4\n        Name: foo, dtype: int64\n\n        To update the Series in place, without generating a new one\n        set `inplace` to True. Note that it also requires ``drop=True``.\n\n        >>> s.reset_index(inplace=True, drop=True)\n        >>> s\n        0    1\n        1    2\n        2    3\n        3    4\n        Name: foo, dtype: int64\n\n        The `level` parameter is interesting for Series with a multi-level\n        index.\n\n        >>> arrays = [np.array(['bar', 'bar', 'baz', 'baz']),\n        ...           np.array(['one', 'two', 'one', 'two'])]\n        >>> s2 = pd.Series(\n        ...     range(4), name='foo',\n        ...     index=pd.MultiIndex.from_arrays(arrays,\n        ...                                     names=['a', 'b']))\n\n        To remove a specific level from the Index, use `level`.\n\n        >>> s2.reset_index(level='a')\n               a  foo\n        b\n        one  bar    0\n        two  bar    1\n        one  baz    2\n        two  baz    3\n\n        If `level` is not set, all levels are removed from the Index.\n\n        >>> s2.reset_index()\n             a    b  foo\n        0  bar  one    0\n        1  bar  two    1\n        2  baz  one    2\n        3  baz  two    3\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if drop:
            new_index = ibase.default_index(len(self))
            if (level is not None):
                if (not isinstance(level, (tuple, list))):
                    level = [level]
                level = [self.index._get_level_number(lev) for lev in level]
                if (len(level) < self.index.nlevels):
                    new_index = self.index.droplevel(level)
            if inplace:
                self.index = new_index
                self.name = (name or self.name)
            else:
                return self._constructor(self._values.copy(), index=new_index).__finalize__(self, method='reset_index')
        elif inplace:
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')
        else:
            df = self.to_frame(name)
            return df.reset_index(level=level, drop=drop)

    def __repr__(self):
        '\n        Return a string representation for a particular Series.\n        '
        buf = StringIO('')
        (width, height) = get_terminal_size()
        max_rows = (height if (get_option('display.max_rows') == 0) else get_option('display.max_rows'))
        min_rows = (height if (get_option('display.max_rows') == 0) else get_option('display.min_rows'))
        show_dimensions = get_option('display.show_dimensions')
        self.to_string(buf=buf, name=self.name, dtype=self.dtype, min_rows=min_rows, max_rows=max_rows, length=show_dimensions)
        result = buf.getvalue()
        return result

    def to_string(self, buf=None, na_rep='NaN', float_format=None, header=True, index=True, length=False, dtype=False, name=False, max_rows=None, min_rows=None):
        "\n        Render a string representation of the Series.\n\n        Parameters\n        ----------\n        buf : StringIO-like, optional\n            Buffer to write to.\n        na_rep : str, optional\n            String representation of NaN to use, default 'NaN'.\n        float_format : one-parameter function, optional\n            Formatter function to apply to columns' elements if they are\n            floats, default None.\n        header : bool, default True\n            Add the Series header (index name).\n        index : bool, optional\n            Add index (row) labels, default True.\n        length : bool, default False\n            Add the Series length.\n        dtype : bool, default False\n            Add the Series dtype.\n        name : bool, default False\n            Add the Series name if not None.\n        max_rows : int, optional\n            Maximum number of rows to show before truncating. If None, show\n            all.\n        min_rows : int, optional\n            The number of rows to display in a truncated repr (when number\n            of rows is above `max_rows`).\n\n        Returns\n        -------\n        str or None\n            String representation of Series if ``buf=None``, otherwise None.\n        "
        formatter = fmt.SeriesFormatter(self, name=name, length=length, header=header, index=index, dtype=dtype, na_rep=na_rep, float_format=float_format, min_rows=min_rows, max_rows=max_rows)
        result = formatter.to_string()
        if (not isinstance(result, str)):
            raise AssertionError(f'result must be of type str, type of result is {repr(type(result).__name__)}')
        if (buf is None):
            return result
        else:
            try:
                buf.write(result)
            except AttributeError:
                with open(buf, 'w') as f:
                    f.write(result)

    @doc(klass=_shared_doc_kwargs['klass'], storage_options=generic._shared_docs['storage_options'], examples=dedent('\n            Examples\n            --------\n            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")\n            >>> print(s.to_markdown())\n            |    | animal   |\n            |---:|:---------|\n            |  0 | elk      |\n            |  1 | pig      |\n            |  2 | dog      |\n            |  3 | quetzal  |\n            '))
    def to_markdown(self, buf=None, mode='wt', index=True, storage_options=None, **kwargs):
        '\n        Print {klass} in Markdown-friendly format.\n\n        .. versionadded:: 1.0.0\n\n        Parameters\n        ----------\n        buf : str, Path or StringIO-like, optional, default None\n            Buffer to write to. If None, the output is returned as a string.\n        mode : str, optional\n            Mode in which file is opened, "wt" by default.\n        index : bool, optional, default True\n            Add index (row) labels.\n\n            .. versionadded:: 1.1.0\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n\n        **kwargs\n            These parameters will be passed to `tabulate                 <https://pypi.org/project/tabulate>`_.\n\n        Returns\n        -------\n        str\n            {klass} in Markdown-friendly format.\n\n        Notes\n        -----\n        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.\n\n        Examples\n        --------\n        >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")\n        >>> print(s.to_markdown())\n        |    | animal   |\n        |---:|:---------|\n        |  0 | elk      |\n        |  1 | pig      |\n        |  2 | dog      |\n        |  3 | quetzal  |\n\n        Output markdown with a tabulate option.\n\n        >>> print(s.to_markdown(tablefmt="grid"))\n        +----+----------+\n        |    | animal   |\n        +====+==========+\n        |  0 | elk      |\n        +----+----------+\n        |  1 | pig      |\n        +----+----------+\n        |  2 | dog      |\n        +----+----------+\n        |  3 | quetzal  |\n        +----+----------+\n        '
        return self.to_frame().to_markdown(buf, mode, index, storage_options=storage_options, **kwargs)

    def items(self):
        '\n        Lazily iterate over (index, value) tuples.\n\n        This method returns an iterable tuple (index, value). This is\n        convenient if you want to create a lazy iterator.\n\n        Returns\n        -------\n        iterable\n            Iterable of tuples containing the (index, value) pairs from a\n            Series.\n\n        See Also\n        --------\n        DataFrame.items : Iterate over (column name, Series) pairs.\n        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.\n\n        Examples\n        --------\n        >>> s = pd.Series([\'A\', \'B\', \'C\'])\n        >>> for index, value in s.items():\n        ...     print(f"Index : {index}, Value : {value}")\n        Index : 0, Value : A\n        Index : 1, Value : B\n        Index : 2, Value : C\n        '
        return zip(iter(self.index), iter(self))

    @Appender(items.__doc__)
    def iteritems(self):
        return self.items()

    def keys(self):
        '\n        Return alias for index.\n\n        Returns\n        -------\n        Index\n            Index of the Series.\n        '
        return self.index

    def to_dict(self, into=dict):
        "\n        Convert Series to {label -> value} dict or dict-like object.\n\n        Parameters\n        ----------\n        into : class, default dict\n            The collections.abc.Mapping subclass to use as the return\n            object. Can be the actual class or an empty\n            instance of the mapping type you want.  If you want a\n            collections.defaultdict, you must pass it initialized.\n\n        Returns\n        -------\n        collections.abc.Mapping\n            Key-value representation of Series.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> s.to_dict()\n        {0: 1, 1: 2, 2: 3, 3: 4}\n        >>> from collections import OrderedDict, defaultdict\n        >>> s.to_dict(OrderedDict)\n        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])\n        >>> dd = defaultdict(list)\n        >>> s.to_dict(dd)\n        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})\n        "
        into_c = com.standardize_mapping(into)
        return into_c(self.items())

    def to_frame(self, name=None):
        '\n        Convert Series to DataFrame.\n\n        Parameters\n        ----------\n        name : object, default None\n            The passed name should substitute for the series name (if it has\n            one).\n\n        Returns\n        -------\n        DataFrame\n            DataFrame representation of Series.\n\n        Examples\n        --------\n        >>> s = pd.Series(["a", "b", "c"],\n        ...               name="vals")\n        >>> s.to_frame()\n          vals\n        0    a\n        1    b\n        2    c\n        '
        if (name is None):
            df = self._constructor_expanddim(self)
        else:
            df = self._constructor_expanddim({name: self})
        return df

    def _set_name(self, name, inplace=False):
        '\n        Set the Series name.\n\n        Parameters\n        ----------\n        name : str\n        inplace : bool\n            Whether to modify `self` directly or return a copy.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ser = (self if inplace else self.copy())
        ser.name = name
        return ser

    @Appender('\nExamples\n--------\n>>> ser = pd.Series([390., 350., 30., 20.],\n...                 index=[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'], name="Max Speed")\n>>> ser\nFalcon    390.0\nFalcon    350.0\nParrot     30.0\nParrot     20.0\nName: Max Speed, dtype: float64\n>>> ser.groupby(["a", "b", "a", "b"]).mean()\na    210.0\nb    185.0\nName: Max Speed, dtype: float64\n>>> ser.groupby(level=0).mean()\nFalcon    370.0\nParrot     25.0\nName: Max Speed, dtype: float64\n>>> ser.groupby(ser > 100).mean()\nMax Speed\nFalse     25.0\nTrue     370.0\nName: Max Speed, dtype: float64\n\n**Grouping by Indexes**\n\nWe can groupby different levels of a hierarchical index\nusing the `level` parameter:\n\n>>> arrays = [[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n...           [\'Captive\', \'Wild\', \'Captive\', \'Wild\']]\n>>> index = pd.MultiIndex.from_arrays(arrays, names=(\'Animal\', \'Type\'))\n>>> ser = pd.Series([390., 350., 30., 20.], index=index, name="Max Speed")\n>>> ser\nAnimal  Type\nFalcon  Captive    390.0\n        Wild       350.0\nParrot  Captive     30.0\n        Wild        20.0\nName: Max Speed, dtype: float64\n>>> ser.groupby(level=0).mean()\nAnimal\nFalcon    370.0\nParrot     25.0\nName: Max Speed, dtype: float64\n>>> ser.groupby(level="Type").mean()\nType\nCaptive    210.0\nWild       185.0\nName: Max Speed, dtype: float64\n\nWe can also choose to include `NA` in group keys or not by defining\n`dropna` parameter, the default setting is `True`:\n\n>>> ser = pd.Series([1, 2, 3, 3], index=["a", \'a\', \'b\', np.nan])\n>>> ser.groupby(level=0).sum()\na    3\nb    3\ndtype: int64\n\n>>> ser.groupby(level=0, dropna=False).sum()\na    3\nb    3\nNaN  3\ndtype: int64\n\n>>> arrays = [\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\']\n>>> ser = pd.Series([390., 350., 30., 20.], index=arrays, name="Max Speed")\n>>> ser.groupby(["a", "b", "a", np.nan]).mean()\na    210.0\nb    350.0\nName: Max Speed, dtype: float64\n\n>>> ser.groupby(["a", "b", "a", np.nan], dropna=False).mean()\na    210.0\nb    350.0\nNaN   20.0\nName: Max Speed, dtype: float64\n')
    @Appender((generic._shared_docs['groupby'] % _shared_doc_kwargs))
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=no_default, observed=False, dropna=True):
        from pandas.core.groupby.generic import SeriesGroupBy
        if (squeeze is not no_default):
            warnings.warn('The `squeeze` parameter is deprecated and will be removed in a future version.', FutureWarning, stacklevel=2)
        else:
            squeeze = False
        if ((level is None) and (by is None)):
            raise TypeError("You have to supply one of 'by' and 'level'")
        axis = self._get_axis_number(axis)
        return SeriesGroupBy(obj=self, keys=by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, squeeze=squeeze, observed=observed, dropna=dropna)

    def count(self, level=None):
        '\n        Return number of non-NA/null observations in the Series.\n\n        Parameters\n        ----------\n        level : int or level name, default None\n            If the axis is a MultiIndex (hierarchical), count along a\n            particular level, collapsing into a smaller Series.\n\n        Returns\n        -------\n        int or Series (if level specified)\n            Number of non-null values in the Series.\n\n        See Also\n        --------\n        DataFrame.count : Count non-NA cells for each column or row.\n\n        Examples\n        --------\n        >>> s = pd.Series([0.0, 1.0, np.nan])\n        >>> s.count()\n        2\n        '
        if (level is None):
            return notna(self.array).sum()
        elif (not isinstance(self.index, MultiIndex)):
            raise ValueError('Series.count level is only valid with a MultiIndex')
        index = self.index
        assert isinstance(index, MultiIndex)
        if isinstance(level, str):
            level = index._get_level_number(level)
        lev = index.levels[level]
        level_codes = np.array(index.codes[level], subok=False, copy=True)
        mask = (level_codes == (- 1))
        if mask.any():
            level_codes[mask] = cnt = len(lev)
            lev = lev.insert(cnt, lev._na_value)
        obs = level_codes[notna(self._values)]
        out = np.bincount(obs, minlength=(len(lev) or None))
        return self._constructor(out, index=lev, dtype='int64').__finalize__(self, method='count')

    def mode(self, dropna=True):
        "\n        Return the mode(s) of the Series.\n\n        The mode is the value that appears most often. There can be multiple modes.\n\n        Always returns Series even if only one value is returned.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't consider counts of NaN/NaT.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        Series\n            Modes of the Series in sorted order.\n        "
        return algorithms.mode(self, dropna=dropna)

    def unique(self):
        "\n        Return unique values of Series object.\n\n        Uniques are returned in order of appearance. Hash table-based unique,\n        therefore does NOT sort.\n\n        Returns\n        -------\n        ndarray or ExtensionArray\n            The unique values returned as a NumPy array. See Notes.\n\n        See Also\n        --------\n        unique : Top-level unique method for any 1-d array-like object.\n        Index.unique : Return Index with unique values from an Index object.\n\n        Notes\n        -----\n        Returns the unique values as a NumPy array. In case of an\n        extension-array backed Series, a new\n        :class:`~api.extensions.ExtensionArray` of that type with just\n        the unique values is returned. This includes\n\n            * Categorical\n            * Period\n            * Datetime with Timezone\n            * Interval\n            * Sparse\n            * IntegerNA\n\n        See Examples section.\n\n        Examples\n        --------\n        >>> pd.Series([2, 1, 3, 3], name='A').unique()\n        array([2, 1, 3])\n\n        >>> pd.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()\n        array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')\n\n        >>> pd.Series([pd.Timestamp('2016-01-01', tz='US/Eastern')\n        ...            for _ in range(3)]).unique()\n        <DatetimeArray>\n        ['2016-01-01 00:00:00-05:00']\n        Length: 1, dtype: datetime64[ns, US/Eastern]\n\n        An unordered Categorical will return categories in the order of\n        appearance.\n\n        >>> pd.Series(pd.Categorical(list('baabc'))).unique()\n        ['b', 'a', 'c']\n        Categories (3, object): ['b', 'a', 'c']\n\n        An ordered Categorical preserves the category ordering.\n\n        >>> pd.Series(pd.Categorical(list('baabc'), categories=list('abc'),\n        ...                          ordered=True)).unique()\n        ['b', 'a', 'c']\n        Categories (3, object): ['a' < 'b' < 'c']\n        "
        result = super().unique()
        return result

    def drop_duplicates(self, keep='first', inplace=False):
        "\n        Return Series with duplicate values removed.\n\n        Parameters\n        ----------\n        keep : {'first', 'last', ``False``}, default 'first'\n            Method to handle dropping duplicates:\n\n            - 'first' : Drop duplicates except for the first occurrence.\n            - 'last' : Drop duplicates except for the last occurrence.\n            - ``False`` : Drop all duplicates.\n\n        inplace : bool, default ``False``\n            If ``True``, performs operation inplace and returns None.\n\n        Returns\n        -------\n        Series or None\n            Series with duplicates dropped or None if ``inplace=True``.\n\n        See Also\n        --------\n        Index.drop_duplicates : Equivalent method on Index.\n        DataFrame.drop_duplicates : Equivalent method on DataFrame.\n        Series.duplicated : Related method on Series, indicating duplicate\n            Series values.\n\n        Examples\n        --------\n        Generate a Series with duplicated entries.\n\n        >>> s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'],\n        ...               name='animal')\n        >>> s\n        0      lama\n        1       cow\n        2      lama\n        3    beetle\n        4      lama\n        5     hippo\n        Name: animal, dtype: object\n\n        With the 'keep' parameter, the selection behaviour of duplicated values\n        can be changed. The value 'first' keeps the first occurrence for each\n        set of duplicated entries. The default value of keep is 'first'.\n\n        >>> s.drop_duplicates()\n        0      lama\n        1       cow\n        3    beetle\n        5     hippo\n        Name: animal, dtype: object\n\n        The value 'last' for parameter 'keep' keeps the last occurrence for\n        each set of duplicated entries.\n\n        >>> s.drop_duplicates(keep='last')\n        1       cow\n        3    beetle\n        4      lama\n        5     hippo\n        Name: animal, dtype: object\n\n        The value ``False`` for parameter 'keep' discards all sets of\n        duplicated entries. Setting the value of 'inplace' to ``True`` performs\n        the operation inplace and returns ``None``.\n\n        >>> s.drop_duplicates(keep=False, inplace=True)\n        >>> s\n        1       cow\n        3    beetle\n        5     hippo\n        Name: animal, dtype: object\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        result = super().drop_duplicates(keep=keep)
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep='first'):
        "\n        Indicate duplicate Series values.\n\n        Duplicated values are indicated as ``True`` values in the resulting\n        Series. Either all duplicates, all except the first or all except the\n        last occurrence of duplicates can be indicated.\n\n        Parameters\n        ----------\n        keep : {'first', 'last', False}, default 'first'\n            Method to handle dropping duplicates:\n\n            - 'first' : Mark duplicates as ``True`` except for the first\n              occurrence.\n            - 'last' : Mark duplicates as ``True`` except for the last\n              occurrence.\n            - ``False`` : Mark all duplicates as ``True``.\n\n        Returns\n        -------\n        Series\n            Series indicating whether each value has occurred in the\n            preceding values.\n\n        See Also\n        --------\n        Index.duplicated : Equivalent method on pandas.Index.\n        DataFrame.duplicated : Equivalent method on pandas.DataFrame.\n        Series.drop_duplicates : Remove duplicate values from Series.\n\n        Examples\n        --------\n        By default, for each set of duplicated values, the first occurrence is\n        set on False and all others on True:\n\n        >>> animals = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama'])\n        >>> animals.duplicated()\n        0    False\n        1    False\n        2     True\n        3    False\n        4     True\n        dtype: bool\n\n        which is equivalent to\n\n        >>> animals.duplicated(keep='first')\n        0    False\n        1    False\n        2     True\n        3    False\n        4     True\n        dtype: bool\n\n        By using 'last', the last occurrence of each set of duplicated values\n        is set on False and all others on True:\n\n        >>> animals.duplicated(keep='last')\n        0     True\n        1    False\n        2     True\n        3    False\n        4    False\n        dtype: bool\n\n        By setting keep on ``False``, all duplicates are True:\n\n        >>> animals.duplicated(keep=False)\n        0     True\n        1    False\n        2     True\n        3    False\n        4     True\n        dtype: bool\n        "
        res = base.IndexOpsMixin.duplicated(self, keep=keep)
        result = self._constructor(res, index=self.index)
        return result.__finalize__(self, method='duplicated')

    def idxmin(self, axis=0, skipna=True, *args, **kwargs):
        "\n        Return the row label of the minimum value.\n\n        If multiple values equal the minimum, the first row label with that\n        value is returned.\n\n        Parameters\n        ----------\n        axis : int, default 0\n            For compatibility with DataFrame.idxmin. Redundant for application\n            on Series.\n        skipna : bool, default True\n            Exclude NA/null values. If the entire Series is NA, the result\n            will be NA.\n        *args, **kwargs\n            Additional arguments and keywords have no effect but might be\n            accepted for compatibility with NumPy.\n\n        Returns\n        -------\n        Index\n            Label of the minimum value.\n\n        Raises\n        ------\n        ValueError\n            If the Series is empty.\n\n        See Also\n        --------\n        numpy.argmin : Return indices of the minimum values\n            along the given axis.\n        DataFrame.idxmin : Return index of first occurrence of minimum\n            over requested axis.\n        Series.idxmax : Return index *label* of the first occurrence\n            of maximum of values.\n\n        Notes\n        -----\n        This method is the Series version of ``ndarray.argmin``. This method\n        returns the label of the minimum, while ``ndarray.argmin`` returns\n        the position. To get the position, use ``series.values.argmin()``.\n\n        Examples\n        --------\n        >>> s = pd.Series(data=[1, None, 4, 1],\n        ...               index=['A', 'B', 'C', 'D'])\n        >>> s\n        A    1.0\n        B    NaN\n        C    4.0\n        D    1.0\n        dtype: float64\n\n        >>> s.idxmin()\n        'A'\n\n        If `skipna` is False and there is an NA value in the data,\n        the function returns ``nan``.\n\n        >>> s.idxmin(skipna=False)\n        nan\n        "
        i = self.argmin(None, skipna=skipna)
        if (i == (- 1)):
            return np.nan
        return self.index[i]

    def idxmax(self, axis=0, skipna=True, *args, **kwargs):
        "\n        Return the row label of the maximum value.\n\n        If multiple values equal the maximum, the first row label with that\n        value is returned.\n\n        Parameters\n        ----------\n        axis : int, default 0\n            For compatibility with DataFrame.idxmax. Redundant for application\n            on Series.\n        skipna : bool, default True\n            Exclude NA/null values. If the entire Series is NA, the result\n            will be NA.\n        *args, **kwargs\n            Additional arguments and keywords have no effect but might be\n            accepted for compatibility with NumPy.\n\n        Returns\n        -------\n        Index\n            Label of the maximum value.\n\n        Raises\n        ------\n        ValueError\n            If the Series is empty.\n\n        See Also\n        --------\n        numpy.argmax : Return indices of the maximum values\n            along the given axis.\n        DataFrame.idxmax : Return index of first occurrence of maximum\n            over requested axis.\n        Series.idxmin : Return index *label* of the first occurrence\n            of minimum of values.\n\n        Notes\n        -----\n        This method is the Series version of ``ndarray.argmax``. This method\n        returns the label of the maximum, while ``ndarray.argmax`` returns\n        the position. To get the position, use ``series.values.argmax()``.\n\n        Examples\n        --------\n        >>> s = pd.Series(data=[1, None, 4, 3, 4],\n        ...               index=['A', 'B', 'C', 'D', 'E'])\n        >>> s\n        A    1.0\n        B    NaN\n        C    4.0\n        D    3.0\n        E    4.0\n        dtype: float64\n\n        >>> s.idxmax()\n        'C'\n\n        If `skipna` is False and there is an NA value in the data,\n        the function returns ``nan``.\n\n        >>> s.idxmax(skipna=False)\n        nan\n        "
        i = self.argmax(None, skipna=skipna)
        if (i == (- 1)):
            return np.nan
        return self.index[i]

    def round(self, decimals=0, *args, **kwargs):
        '\n        Round each value in a Series to the given number of decimals.\n\n        Parameters\n        ----------\n        decimals : int, default 0\n            Number of decimal places to round to. If decimals is negative,\n            it specifies the number of positions to the left of the decimal point.\n        *args, **kwargs\n            Additional arguments and keywords have no effect but might be\n            accepted for compatibility with NumPy.\n\n        Returns\n        -------\n        Series\n            Rounded values of the Series.\n\n        See Also\n        --------\n        numpy.around : Round values of an np.array.\n        DataFrame.round : Round values of a DataFrame.\n\n        Examples\n        --------\n        >>> s = pd.Series([0.1, 1.3, 2.7])\n        >>> s.round()\n        0    0.0\n        1    1.0\n        2    3.0\n        dtype: float64\n        '
        nv.validate_round(args, kwargs)
        result = self._values.round(decimals)
        result = self._constructor(result, index=self.index).__finalize__(self, method='round')
        return result

    def quantile(self, q=0.5, interpolation='linear'):
        "\n        Return value at the given quantile.\n\n        Parameters\n        ----------\n        q : float or array-like, default 0.5 (50% quantile)\n            The quantile(s) to compute, which can lie in range: 0 <= q <= 1.\n        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}\n            This optional parameter specifies the interpolation method to use,\n            when the desired quantile lies between two data points `i` and `j`:\n\n                * linear: `i + (j - i) * fraction`, where `fraction` is the\n                  fractional part of the index surrounded by `i` and `j`.\n                * lower: `i`.\n                * higher: `j`.\n                * nearest: `i` or `j` whichever is nearest.\n                * midpoint: (`i` + `j`) / 2.\n\n        Returns\n        -------\n        float or Series\n            If ``q`` is an array, a Series will be returned where the\n            index is ``q`` and the values are the quantiles, otherwise\n            a float will be returned.\n\n        See Also\n        --------\n        core.window.Rolling.quantile : Calculate the rolling quantile.\n        numpy.percentile : Returns the q-th percentile(s) of the array elements.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> s.quantile(.5)\n        2.5\n        >>> s.quantile([.25, .5, .75])\n        0.25    1.75\n        0.50    2.50\n        0.75    3.25\n        dtype: float64\n        "
        validate_percentile(q)
        df = self.to_frame()
        result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
        if (result.ndim == 2):
            result = result.iloc[:, 0]
        if is_list_like(q):
            result.name = self.name
            return self._constructor(result, index=Float64Index(q), name=self.name)
        else:
            return result.iloc[0]

    def corr(self, other, method='pearson', min_periods=None):
        "\n        Compute correlation with `other` Series, excluding missing values.\n\n        Parameters\n        ----------\n        other : Series\n            Series with which to compute the correlation.\n        method : {'pearson', 'kendall', 'spearman'} or callable\n            Method used to compute correlation:\n\n            - pearson : Standard correlation coefficient\n            - kendall : Kendall Tau correlation coefficient\n            - spearman : Spearman rank correlation\n            - callable: Callable with input two 1d ndarrays and returning a float.\n\n            .. versionadded:: 0.24.0\n                Note that the returned matrix from corr will have 1 along the\n                diagonals and will be symmetric regardless of the callable's\n                behavior.\n        min_periods : int, optional\n            Minimum number of observations needed to have a valid result.\n\n        Returns\n        -------\n        float\n            Correlation with other.\n\n        See Also\n        --------\n        DataFrame.corr : Compute pairwise correlation between columns.\n        DataFrame.corrwith : Compute pairwise correlation with another\n            DataFrame or Series.\n\n        Examples\n        --------\n        >>> def histogram_intersection(a, b):\n        ...     v = np.minimum(a, b).sum().round(decimals=1)\n        ...     return v\n        >>> s1 = pd.Series([.2, .0, .6, .2])\n        >>> s2 = pd.Series([.3, .6, .0, .1])\n        >>> s1.corr(s2, method=histogram_intersection)\n        0.3\n        "
        (this, other) = self.align(other, join='inner', copy=False)
        if (len(this) == 0):
            return np.nan
        if ((method in ['pearson', 'spearman', 'kendall']) or callable(method)):
            return nanops.nancorr(this.values, other.values, method=method, min_periods=min_periods)
        raise ValueError(f"method must be either 'pearson', 'spearman', 'kendall', or a callable, '{method}' was supplied")

    def cov(self, other, min_periods=None, ddof=1):
        '\n        Compute covariance with Series, excluding missing values.\n\n        Parameters\n        ----------\n        other : Series\n            Series with which to compute the covariance.\n        min_periods : int, optional\n            Minimum number of observations needed to have a valid result.\n        ddof : int, default 1\n            Delta degrees of freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        float\n            Covariance between Series and other normalized by N-1\n            (unbiased estimator).\n\n        See Also\n        --------\n        DataFrame.cov : Compute pairwise covariance of columns.\n\n        Examples\n        --------\n        >>> s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])\n        >>> s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])\n        >>> s1.cov(s2)\n        -0.01685762652715874\n        '
        (this, other) = self.align(other, join='inner', copy=False)
        if (len(this) == 0):
            return np.nan
        return nanops.nancov(this.values, other.values, min_periods=min_periods, ddof=ddof)

    @doc(klass='Series', extra_params='', other_klass='DataFrame', examples=dedent('\n        Difference with previous row\n\n        >>> s = pd.Series([1, 1, 2, 3, 5, 8])\n        >>> s.diff()\n        0    NaN\n        1    0.0\n        2    1.0\n        3    1.0\n        4    2.0\n        5    3.0\n        dtype: float64\n\n        Difference with 3rd previous row\n\n        >>> s.diff(periods=3)\n        0    NaN\n        1    NaN\n        2    NaN\n        3    2.0\n        4    4.0\n        5    6.0\n        dtype: float64\n\n        Difference with following row\n\n        >>> s.diff(periods=-1)\n        0    0.0\n        1   -1.0\n        2   -1.0\n        3   -2.0\n        4   -3.0\n        5    NaN\n        dtype: float64\n\n        Overflow in input dtype\n\n        >>> s = pd.Series([1, 0], dtype=np.uint8)\n        >>> s.diff()\n        0      NaN\n        1    255.0\n        dtype: float64'))
    def diff(self, periods=1):
        '\n        First discrete difference of element.\n\n        Calculates the difference of a {klass} element compared with another\n        element in the {klass} (default is element in previous row).\n\n        Parameters\n        ----------\n        periods : int, default 1\n            Periods to shift for calculating difference, accepts negative\n            values.\n        {extra_params}\n        Returns\n        -------\n        {klass}\n            First differences of the Series.\n\n        See Also\n        --------\n        {klass}.pct_change: Percent change over given number of periods.\n        {klass}.shift: Shift index by desired number of periods with an\n            optional time freq.\n        {other_klass}.diff: First discrete difference of object.\n\n        Notes\n        -----\n        For boolean dtypes, this uses :meth:`operator.xor` rather than\n        :meth:`operator.sub`.\n        The result is calculated according to current dtype in {klass},\n        however dtype of the result is always float64.\n\n        Examples\n        --------\n        {examples}\n        '
        result = algorithms.diff(self.array, periods)
        return self._constructor(result, index=self.index).__finalize__(self, method='diff')

    def autocorr(self, lag=1):
        "\n        Compute the lag-N autocorrelation.\n\n        This method computes the Pearson correlation between\n        the Series and its shifted self.\n\n        Parameters\n        ----------\n        lag : int, default 1\n            Number of lags to apply before performing autocorrelation.\n\n        Returns\n        -------\n        float\n            The Pearson correlation between self and self.shift(lag).\n\n        See Also\n        --------\n        Series.corr : Compute the correlation between two Series.\n        Series.shift : Shift index by desired number of periods.\n        DataFrame.corr : Compute pairwise correlation of columns.\n        DataFrame.corrwith : Compute pairwise correlation between rows or\n            columns of two DataFrame objects.\n\n        Notes\n        -----\n        If the Pearson correlation is not well defined return 'NaN'.\n\n        Examples\n        --------\n        >>> s = pd.Series([0.25, 0.5, 0.2, -0.05])\n        >>> s.autocorr()  # doctest: +ELLIPSIS\n        0.10355...\n        >>> s.autocorr(lag=2)  # doctest: +ELLIPSIS\n        -0.99999...\n\n        If the Pearson correlation is not well defined, then 'NaN' is returned.\n\n        >>> s = pd.Series([1, 0, 0, 0])\n        >>> s.autocorr()\n        nan\n        "
        return self.corr(self.shift(lag))

    def dot(self, other):
        '\n        Compute the dot product between the Series and the columns of other.\n\n        This method computes the dot product between the Series and another\n        one, or the Series and each columns of a DataFrame, or the Series and\n        each columns of an array.\n\n        It can also be called using `self @ other` in Python >= 3.5.\n\n        Parameters\n        ----------\n        other : Series, DataFrame or array-like\n            The other object to compute the dot product with its columns.\n\n        Returns\n        -------\n        scalar, Series or numpy.ndarray\n            Return the dot product of the Series and other if other is a\n            Series, the Series of the dot product of Series and each rows of\n            other if other is a DataFrame or a numpy.ndarray between the Series\n            and each columns of the numpy array.\n\n        See Also\n        --------\n        DataFrame.dot: Compute the matrix product with the DataFrame.\n        Series.mul: Multiplication of series and other, element-wise.\n\n        Notes\n        -----\n        The Series and other has to share the same index if other is a Series\n        or a DataFrame.\n\n        Examples\n        --------\n        >>> s = pd.Series([0, 1, 2, 3])\n        >>> other = pd.Series([-1, 2, -3, 4])\n        >>> s.dot(other)\n        8\n        >>> s @ other\n        8\n        >>> df = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])\n        >>> s.dot(df)\n        0    24\n        1    14\n        dtype: int64\n        >>> arr = np.array([[0, 1], [-2, 3], [4, -5], [6, 7]])\n        >>> s.dot(arr)\n        array([24, 14])\n        '
        if isinstance(other, (Series, ABCDataFrame)):
            common = self.index.union(other.index)
            if ((len(common) > len(self.index)) or (len(common) > len(other.index))):
                raise ValueError('matrices are not aligned')
            left = self.reindex(index=common, copy=False)
            right = other.reindex(index=common, copy=False)
            lvals = left.values
            rvals = right.values
        else:
            lvals = self.values
            rvals = np.asarray(other)
            if (lvals.shape[0] != rvals.shape[0]):
                raise Exception(f'Dot product shape mismatch, {lvals.shape} vs {rvals.shape}')
        if isinstance(other, ABCDataFrame):
            return self._constructor(np.dot(lvals, rvals), index=other.columns).__finalize__(self, method='dot')
        elif isinstance(other, Series):
            return np.dot(lvals, rvals)
        elif isinstance(rvals, np.ndarray):
            return np.dot(lvals, rvals)
        else:
            raise TypeError(f'unsupported type: {type(other)}')

    def __matmul__(self, other):
        '\n        Matrix multiplication using binary `@` operator in Python>=3.5.\n        '
        return self.dot(other)

    def __rmatmul__(self, other):
        '\n        Matrix multiplication using binary `@` operator in Python>=3.5.\n        '
        return self.dot(np.transpose(other))

    @doc(base.IndexOpsMixin.searchsorted, klass='Series')
    def searchsorted(self, value, side='left', sorter=None):
        return algorithms.searchsorted(self._values, value, side=side, sorter=sorter)

    def append(self, to_append, ignore_index=False, verify_integrity=False):
        '\n        Concatenate two or more Series.\n\n        Parameters\n        ----------\n        to_append : Series or list/tuple of Series\n            Series to append with self.\n        ignore_index : bool, default False\n            If True, the resulting axis will be labeled 0, 1, …, n - 1.\n        verify_integrity : bool, default False\n            If True, raise Exception on creating index with duplicates.\n\n        Returns\n        -------\n        Series\n            Concatenated Series.\n\n        See Also\n        --------\n        concat : General function to concatenate DataFrame or Series objects.\n\n        Notes\n        -----\n        Iteratively appending to a Series can be more computationally intensive\n        than a single concatenate. A better solution is to append values to a\n        list and then concatenate the list with the original Series all at\n        once.\n\n        Examples\n        --------\n        >>> s1 = pd.Series([1, 2, 3])\n        >>> s2 = pd.Series([4, 5, 6])\n        >>> s3 = pd.Series([4, 5, 6], index=[3, 4, 5])\n        >>> s1.append(s2)\n        0    1\n        1    2\n        2    3\n        0    4\n        1    5\n        2    6\n        dtype: int64\n\n        >>> s1.append(s3)\n        0    1\n        1    2\n        2    3\n        3    4\n        4    5\n        5    6\n        dtype: int64\n\n        With `ignore_index` set to True:\n\n        >>> s1.append(s2, ignore_index=True)\n        0    1\n        1    2\n        2    3\n        3    4\n        4    5\n        5    6\n        dtype: int64\n\n        With `verify_integrity` set to True:\n\n        >>> s1.append(s2, verify_integrity=True)\n        Traceback (most recent call last):\n        ...\n        ValueError: Indexes have overlapping values: [0, 1, 2]\n        '
        from pandas.core.reshape.concat import concat
        if isinstance(to_append, (list, tuple)):
            to_concat = [self]
            to_concat.extend(to_append)
        else:
            to_concat = [self, to_append]
        if any((isinstance(x, (ABCDataFrame,)) for x in to_concat[1:])):
            msg = 'to_append should be a Series or list/tuple of Series, got DataFrame'
            raise TypeError(msg)
        return concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity)

    def _binop(self, other, func, level=None, fill_value=None):
        '\n        Perform generic binary operation with optional fill value.\n\n        Parameters\n        ----------\n        other : Series\n        func : binary operator\n        fill_value : float or object\n            Value to substitute for NA/null values. If both Series are NA in a\n            location, the result will be NA regardless of the passed fill value.\n        level : int or level name, default None\n            Broadcast across a level, matching Index values on the\n            passed MultiIndex level.\n\n        Returns\n        -------\n        Series\n        '
        if (not isinstance(other, Series)):
            raise AssertionError('Other operand must be Series')
        this = self
        if (not self.index.equals(other.index)):
            (this, other) = self.align(other, level=level, join='outer', copy=False)
        (this_vals, other_vals) = ops.fill_binop(this.values, other.values, fill_value)
        with np.errstate(all='ignore'):
            result = func(this_vals, other_vals)
        name = ops.get_op_result_name(self, other)
        ret = this._construct_result(result, name)
        return ret

    def _construct_result(self, result, name):
        '\n        Construct an appropriately-labelled Series from the result of an op.\n\n        Parameters\n        ----------\n        result : ndarray or ExtensionArray\n        name : Label\n\n        Returns\n        -------\n        Series\n            In the case of __divmod__ or __rdivmod__, a 2-tuple of Series.\n        '
        if isinstance(result, tuple):
            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)
        out = self._constructor(result, index=self.index)
        out = out.__finalize__(self)
        out.name = name
        return out

    @doc(generic._shared_docs['compare'], '\nReturns\n-------\nSeries or DataFrame\n    If axis is 0 or \'index\' the result will be a Series.\n    The resulting index will be a MultiIndex with \'self\' and \'other\'\n    stacked alternately at the inner level.\n\n    If axis is 1 or \'columns\' the result will be a DataFrame.\n    It will have two columns namely \'self\' and \'other\'.\n\nSee Also\n--------\nDataFrame.compare : Compare with another DataFrame and show differences.\n\nNotes\n-----\nMatching NaNs will not appear as a difference.\n\nExamples\n--------\n>>> s1 = pd.Series(["a", "b", "c", "d", "e"])\n>>> s2 = pd.Series(["a", "a", "c", "b", "e"])\n\nAlign the differences on columns\n\n>>> s1.compare(s2)\n  self other\n1    b     a\n3    d     b\n\nStack the differences on indices\n\n>>> s1.compare(s2, align_axis=0)\n1  self     b\n   other    a\n3  self     d\n   other    b\ndtype: object\n\nKeep all original rows\n\n>>> s1.compare(s2, keep_shape=True)\n  self other\n0  NaN   NaN\n1    b     a\n2  NaN   NaN\n3    d     b\n4  NaN   NaN\n\nKeep all original rows and also all original values\n\n>>> s1.compare(s2, keep_shape=True, keep_equal=True)\n  self other\n0    a     a\n1    b     a\n2    c     c\n3    d     b\n4    e     e\n', klass=_shared_doc_kwargs['klass'])
    def compare(self, other, align_axis=1, keep_shape=False, keep_equal=False):
        return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal)

    def combine(self, other, func, fill_value=None):
        "\n        Combine the Series with a Series or scalar according to `func`.\n\n        Combine the Series and `other` using `func` to perform elementwise\n        selection for combined Series.\n        `fill_value` is assumed when value is missing at some index\n        from one of the two objects being combined.\n\n        Parameters\n        ----------\n        other : Series or scalar\n            The value(s) to be combined with the `Series`.\n        func : function\n            Function that takes two scalars as inputs and returns an element.\n        fill_value : scalar, optional\n            The value to assume when an index is missing from\n            one Series or the other. The default specifies to use the\n            appropriate NaN value for the underlying dtype of the Series.\n\n        Returns\n        -------\n        Series\n            The result of combining the Series with the other object.\n\n        See Also\n        --------\n        Series.combine_first : Combine Series values, choosing the calling\n            Series' values first.\n\n        Examples\n        --------\n        Consider 2 Datasets ``s1`` and ``s2`` containing\n        highest clocked speeds of different birds.\n\n        >>> s1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})\n        >>> s1\n        falcon    330.0\n        eagle     160.0\n        dtype: float64\n        >>> s2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})\n        >>> s2\n        falcon    345.0\n        eagle     200.0\n        duck       30.0\n        dtype: float64\n\n        Now, to combine the two datasets and view the highest speeds\n        of the birds across the two datasets\n\n        >>> s1.combine(s2, max)\n        duck        NaN\n        eagle     200.0\n        falcon    345.0\n        dtype: float64\n\n        In the previous example, the resulting value for duck is missing,\n        because the maximum of a NaN and a float is a NaN.\n        So, in the example, we set ``fill_value=0``,\n        so the maximum value returned will be the value from some dataset.\n\n        >>> s1.combine(s2, max, fill_value=0)\n        duck       30.0\n        eagle     200.0\n        falcon    345.0\n        dtype: float64\n        "
        if (fill_value is None):
            fill_value = na_value_for_dtype(self.dtype, compat=False)
        if isinstance(other, Series):
            new_index = self.index.union(other.index)
            new_name = ops.get_op_result_name(self, other)
            new_values = []
            for idx in new_index:
                lv = self.get(idx, fill_value)
                rv = other.get(idx, fill_value)
                with np.errstate(all='ignore'):
                    new_values.append(func(lv, rv))
        else:
            new_index = self.index
            with np.errstate(all='ignore'):
                new_values = [func(lv, other) for lv in self._values]
            new_name = self.name
        if is_categorical_dtype(self.dtype):
            pass
        elif is_extension_array_dtype(self.dtype):
            new_values = maybe_cast_to_extension_array(type(self._values), new_values)
        return self._constructor(new_values, index=new_index, name=new_name)

    def combine_first(self, other):
        "\n        Combine Series values, choosing the calling Series's values first.\n\n        Parameters\n        ----------\n        other : Series\n            The value(s) to be combined with the `Series`.\n\n        Returns\n        -------\n        Series\n            The result of combining the Series with the other object.\n\n        See Also\n        --------\n        Series.combine : Perform elementwise operation on two Series\n            using a given function.\n\n        Notes\n        -----\n        Result index will be the union of the two indexes.\n\n        Examples\n        --------\n        >>> s1 = pd.Series([1, np.nan])\n        >>> s2 = pd.Series([3, 4])\n        >>> s1.combine_first(s2)\n        0    1.0\n        1    4.0\n        dtype: float64\n        "
        new_index = self.index.union(other.index)
        this = self.reindex(new_index, copy=False)
        other = other.reindex(new_index, copy=False)
        if ((this.dtype.kind == 'M') and (other.dtype.kind != 'M')):
            other = to_datetime(other)
        return this.where(notna(this), other)

    def update(self, other):
        "\n        Modify Series in place using values from passed Series.\n\n        Uses non-NA values from passed Series to make updates. Aligns\n        on index.\n\n        Parameters\n        ----------\n        other : Series, or object coercible into Series\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3])\n        >>> s.update(pd.Series([4, 5, 6]))\n        >>> s\n        0    4\n        1    5\n        2    6\n        dtype: int64\n\n        >>> s = pd.Series(['a', 'b', 'c'])\n        >>> s.update(pd.Series(['d', 'e'], index=[0, 2]))\n        >>> s\n        0    d\n        1    b\n        2    e\n        dtype: object\n\n        >>> s = pd.Series([1, 2, 3])\n        >>> s.update(pd.Series([4, 5, 6, 7, 8]))\n        >>> s\n        0    4\n        1    5\n        2    6\n        dtype: int64\n\n        If ``other`` contains NaNs the corresponding values are not updated\n        in the original Series.\n\n        >>> s = pd.Series([1, 2, 3])\n        >>> s.update(pd.Series([4, np.nan, 6]))\n        >>> s\n        0    4\n        1    2\n        2    6\n        dtype: int64\n\n        ``other`` can also be a non-Series object type\n        that is coercible into a Series\n\n        >>> s = pd.Series([1, 2, 3])\n        >>> s.update([4, np.nan, 6])\n        >>> s\n        0    4\n        1    2\n        2    6\n        dtype: int64\n\n        >>> s = pd.Series([1, 2, 3])\n        >>> s.update({1: 9})\n        >>> s\n        0    1\n        1    9\n        2    3\n        dtype: int64\n        "
        if (not isinstance(other, Series)):
            other = Series(other)
        other = other.reindex_like(self)
        mask = notna(other)
        self._mgr = self._mgr.putmask(mask=mask, new=other)
        self._maybe_update_cacher()

    def sort_values(self, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None):
        "\n        Sort by the values.\n\n        Sort a Series in ascending or descending order by some\n        criterion.\n\n        Parameters\n        ----------\n        axis : {0 or 'index'}, default 0\n            Axis to direct sorting. The value 'index' is accepted for\n            compatibility with DataFrame.sort_values.\n        ascending : bool, default True\n            If True, sort values in ascending order, otherwise descending.\n        inplace : bool, default False\n            If True, perform operation in-place.\n        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'\n            Choice of sorting algorithm. See also :func:`numpy.sort` for more\n            information. 'mergesort' and 'stable' are the only stable  algorithms.\n        na_position : {'first' or 'last'}, default 'last'\n            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at\n            the end.\n        ignore_index : bool, default False\n            If True, the resulting axis will be labeled 0, 1, …, n - 1.\n\n            .. versionadded:: 1.0.0\n\n        key : callable, optional\n            If not None, apply the key function to the series values\n            before sorting. This is similar to the `key` argument in the\n            builtin :meth:`sorted` function, with the notable difference that\n            this `key` function should be *vectorized*. It should expect a\n            ``Series`` and return an array-like.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        Series or None\n            Series ordered by values or None if ``inplace=True``.\n\n        See Also\n        --------\n        Series.sort_index : Sort by the Series indices.\n        DataFrame.sort_values : Sort DataFrame by the values along either axis.\n        DataFrame.sort_index : Sort DataFrame by indices.\n\n        Examples\n        --------\n        >>> s = pd.Series([np.nan, 1, 3, 10, 5])\n        >>> s\n        0     NaN\n        1     1.0\n        2     3.0\n        3     10.0\n        4     5.0\n        dtype: float64\n\n        Sort values ascending order (default behaviour)\n\n        >>> s.sort_values(ascending=True)\n        1     1.0\n        2     3.0\n        4     5.0\n        3    10.0\n        0     NaN\n        dtype: float64\n\n        Sort values descending order\n\n        >>> s.sort_values(ascending=False)\n        3    10.0\n        4     5.0\n        2     3.0\n        1     1.0\n        0     NaN\n        dtype: float64\n\n        Sort values inplace\n\n        >>> s.sort_values(ascending=False, inplace=True)\n        >>> s\n        3    10.0\n        4     5.0\n        2     3.0\n        1     1.0\n        0     NaN\n        dtype: float64\n\n        Sort values putting NAs first\n\n        >>> s.sort_values(na_position='first')\n        0     NaN\n        1     1.0\n        2     3.0\n        4     5.0\n        3    10.0\n        dtype: float64\n\n        Sort a series of strings\n\n        >>> s = pd.Series(['z', 'b', 'd', 'a', 'c'])\n        >>> s\n        0    z\n        1    b\n        2    d\n        3    a\n        4    c\n        dtype: object\n\n        >>> s.sort_values()\n        3    a\n        1    b\n        4    c\n        2    d\n        0    z\n        dtype: object\n\n        Sort using a key function. Your `key` function will be\n        given the ``Series`` of values and should return an array-like.\n\n        >>> s = pd.Series(['a', 'B', 'c', 'D', 'e'])\n        >>> s.sort_values()\n        1    B\n        3    D\n        0    a\n        2    c\n        4    e\n        dtype: object\n        >>> s.sort_values(key=lambda x: x.str.lower())\n        0    a\n        1    B\n        2    c\n        3    D\n        4    e\n        dtype: object\n\n        NumPy ufuncs work well here. For example, we can\n        sort by the ``sin`` of the value\n\n        >>> s = pd.Series([-4, -2, 0, 2, 4])\n        >>> s.sort_values(key=np.sin)\n        1   -2\n        4    4\n        2    0\n        0   -4\n        3    2\n        dtype: int64\n\n        More complicated user-defined functions can be used,\n        as long as they expect a Series and return an array-like\n\n        >>> s.sort_values(key=lambda x: (np.tan(x.cumsum())))\n        0   -4\n        3    2\n        4    4\n        1   -2\n        2    0\n        dtype: int64\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._get_axis_number(axis)
        if (inplace and self._is_cached):
            raise ValueError('This Series is a view of some other array, to sort in-place you must create a copy')
        if is_list_like(ascending):
            if (len(ascending) != 1):
                raise ValueError(f'Length of ascending ({len(ascending)}) must be 1 for Series')
            ascending = ascending[0]
        if (not is_bool(ascending)):
            raise ValueError('ascending must be boolean')
        if (na_position not in ['first', 'last']):
            raise ValueError(f'invalid na_position: {na_position}')
        values_to_sort = (ensure_key_mapped(self, key)._values if key else self._values)
        sorted_index = nargsort(values_to_sort, kind, ascending, na_position)
        result = self._constructor(self._values[sorted_index], index=self.index[sorted_index])
        if ignore_index:
            result.index = ibase.default_index(len(sorted_index))
        if inplace:
            self._update_inplace(result)
        else:
            return result.__finalize__(self, method='sort_values')

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None):
        "\n        Sort Series by index labels.\n\n        Returns a new Series sorted by label if `inplace` argument is\n        ``False``, otherwise updates the original series and returns None.\n\n        Parameters\n        ----------\n        axis : int, default 0\n            Axis to direct sorting. This can only be 0 for Series.\n        level : int, optional\n            If not None, sort on values in specified index level(s).\n        ascending : bool or list of bools, default True\n            Sort ascending vs. descending. When the index is a MultiIndex the\n            sort direction can be controlled for each level individually.\n        inplace : bool, default False\n            If True, perform operation in-place.\n        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'\n            Choice of sorting algorithm. See also :func:`numpy.sort` for more\n            information. 'mergesort' and 'stable' are the only stable algorithms. For\n            DataFrames, this option is only applied when sorting on a single\n            column or label.\n        na_position : {'first', 'last'}, default 'last'\n            If 'first' puts NaNs at the beginning, 'last' puts NaNs at the end.\n            Not implemented for MultiIndex.\n        sort_remaining : bool, default True\n            If True and sorting by level and index is multilevel, sort by other\n            levels too (in order) after sorting by specified level.\n        ignore_index : bool, default False\n            If True, the resulting axis will be labeled 0, 1, …, n - 1.\n\n            .. versionadded:: 1.0.0\n\n        key : callable, optional\n            If not None, apply the key function to the index values\n            before sorting. This is similar to the `key` argument in the\n            builtin :meth:`sorted` function, with the notable difference that\n            this `key` function should be *vectorized*. It should expect an\n            ``Index`` and return an ``Index`` of the same shape.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        Series or None\n            The original Series sorted by the labels or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.sort_index: Sort DataFrame by the index.\n        DataFrame.sort_values: Sort DataFrame by the value.\n        Series.sort_values : Sort Series by the value.\n\n        Examples\n        --------\n        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])\n        >>> s.sort_index()\n        1    c\n        2    b\n        3    a\n        4    d\n        dtype: object\n\n        Sort Descending\n\n        >>> s.sort_index(ascending=False)\n        4    d\n        3    a\n        2    b\n        1    c\n        dtype: object\n\n        Sort Inplace\n\n        >>> s.sort_index(inplace=True)\n        >>> s\n        1    c\n        2    b\n        3    a\n        4    d\n        dtype: object\n\n        By default NaNs are put at the end, but use `na_position` to place\n        them at the beginning\n\n        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, np.nan])\n        >>> s.sort_index(na_position='first')\n        NaN     d\n         1.0    c\n         2.0    b\n         3.0    a\n        dtype: object\n\n        Specify index level to sort\n\n        >>> arrays = [np.array(['qux', 'qux', 'foo', 'foo',\n        ...                     'baz', 'baz', 'bar', 'bar']),\n        ...           np.array(['two', 'one', 'two', 'one',\n        ...                     'two', 'one', 'two', 'one'])]\n        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)\n        >>> s.sort_index(level=1)\n        bar  one    8\n        baz  one    6\n        foo  one    4\n        qux  one    2\n        bar  two    7\n        baz  two    5\n        foo  two    3\n        qux  two    1\n        dtype: int64\n\n        Does not sort by remaining levels when sorting by levels\n\n        >>> s.sort_index(level=1, sort_remaining=False)\n        qux  one    2\n        foo  one    4\n        baz  one    6\n        bar  one    8\n        qux  two    1\n        foo  two    3\n        baz  two    5\n        bar  two    7\n        dtype: int64\n\n        Apply a key function before sorting\n\n        >>> s = pd.Series([1, 2, 3, 4], index=['A', 'b', 'C', 'd'])\n        >>> s.sort_index(key=lambda x : x.str.lower())\n        A    1\n        b    2\n        C    3\n        d    4\n        dtype: int64\n        "
        return super().sort_index(axis, level, ascending, inplace, kind, na_position, sort_remaining, ignore_index, key)

    def argsort(self, axis=0, kind='quicksort', order=None):
        '\n        Return the integer indices that would sort the Series values.\n\n        Override ndarray.argsort. Argsorts the value, omitting NA/null values,\n        and places the result in the same locations as the non-NA values.\n\n        Parameters\n        ----------\n        axis : {0 or "index"}\n            Has no effect but is accepted for compatibility with numpy.\n        kind : {\'mergesort\', \'quicksort\', \'heapsort\', \'stable\'}, default \'quicksort\'\n            Choice of sorting algorithm. See :func:`numpy.sort` for more\n            information. \'mergesort\' and \'stable\' are the only stable algorithms.\n        order : None\n            Has no effect but is accepted for compatibility with numpy.\n\n        Returns\n        -------\n        Series\n            Positions of values within the sort order with -1 indicating\n            nan values.\n\n        See Also\n        --------\n        numpy.ndarray.argsort : Returns the indices that would sort this array.\n        '
        values = self._values
        mask = isna(values)
        if mask.any():
            result = Series((- 1), index=self.index, name=self.name, dtype='int64')
            notmask = (~ mask)
            result[notmask] = np.argsort(values[notmask], kind=kind)
            return self._constructor(result, index=self.index).__finalize__(self, method='argsort')
        else:
            return self._constructor(np.argsort(values, kind=kind), index=self.index, dtype='int64').__finalize__(self, method='argsort')

    def nlargest(self, n=5, keep='first'):
        '\n        Return the largest `n` elements.\n\n        Parameters\n        ----------\n        n : int, default 5\n            Return this many descending sorted values.\n        keep : {\'first\', \'last\', \'all\'}, default \'first\'\n            When there are duplicate values that cannot all fit in a\n            Series of `n` elements:\n\n            - ``first`` : return the first `n` occurrences in order\n                of appearance.\n            - ``last`` : return the last `n` occurrences in reverse\n                order of appearance.\n            - ``all`` : keep all occurrences. This can result in a Series of\n                size larger than `n`.\n\n        Returns\n        -------\n        Series\n            The `n` largest values in the Series, sorted in decreasing order.\n\n        See Also\n        --------\n        Series.nsmallest: Get the `n` smallest elements.\n        Series.sort_values: Sort Series by values.\n        Series.head: Return the first `n` rows.\n\n        Notes\n        -----\n        Faster than ``.sort_values(ascending=False).head(n)`` for small `n`\n        relative to the size of the ``Series`` object.\n\n        Examples\n        --------\n        >>> countries_population = {"Italy": 59000000, "France": 65000000,\n        ...                         "Malta": 434000, "Maldives": 434000,\n        ...                         "Brunei": 434000, "Iceland": 337000,\n        ...                         "Nauru": 11300, "Tuvalu": 11300,\n        ...                         "Anguilla": 11300, "Montserrat": 5200}\n        >>> s = pd.Series(countries_population)\n        >>> s\n        Italy       59000000\n        France      65000000\n        Malta         434000\n        Maldives      434000\n        Brunei        434000\n        Iceland       337000\n        Nauru          11300\n        Tuvalu         11300\n        Anguilla       11300\n        Montserrat      5200\n        dtype: int64\n\n        The `n` largest elements where ``n=5`` by default.\n\n        >>> s.nlargest()\n        France      65000000\n        Italy       59000000\n        Malta         434000\n        Maldives      434000\n        Brunei        434000\n        dtype: int64\n\n        The `n` largest elements where ``n=3``. Default `keep` value is \'first\'\n        so Malta will be kept.\n\n        >>> s.nlargest(3)\n        France    65000000\n        Italy     59000000\n        Malta       434000\n        dtype: int64\n\n        The `n` largest elements where ``n=3`` and keeping the last duplicates.\n        Brunei will be kept since it is the last with value 434000 based on\n        the index order.\n\n        >>> s.nlargest(3, keep=\'last\')\n        France      65000000\n        Italy       59000000\n        Brunei        434000\n        dtype: int64\n\n        The `n` largest elements where ``n=3`` with all duplicates kept. Note\n        that the returned Series has five elements due to the three duplicates.\n\n        >>> s.nlargest(3, keep=\'all\')\n        France      65000000\n        Italy       59000000\n        Malta         434000\n        Maldives      434000\n        Brunei        434000\n        dtype: int64\n        '
        return algorithms.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(self, n=5, keep='first'):
        '\n        Return the smallest `n` elements.\n\n        Parameters\n        ----------\n        n : int, default 5\n            Return this many ascending sorted values.\n        keep : {\'first\', \'last\', \'all\'}, default \'first\'\n            When there are duplicate values that cannot all fit in a\n            Series of `n` elements:\n\n            - ``first`` : return the first `n` occurrences in order\n                of appearance.\n            - ``last`` : return the last `n` occurrences in reverse\n                order of appearance.\n            - ``all`` : keep all occurrences. This can result in a Series of\n                size larger than `n`.\n\n        Returns\n        -------\n        Series\n            The `n` smallest values in the Series, sorted in increasing order.\n\n        See Also\n        --------\n        Series.nlargest: Get the `n` largest elements.\n        Series.sort_values: Sort Series by values.\n        Series.head: Return the first `n` rows.\n\n        Notes\n        -----\n        Faster than ``.sort_values().head(n)`` for small `n` relative to\n        the size of the ``Series`` object.\n\n        Examples\n        --------\n        >>> countries_population = {"Italy": 59000000, "France": 65000000,\n        ...                         "Brunei": 434000, "Malta": 434000,\n        ...                         "Maldives": 434000, "Iceland": 337000,\n        ...                         "Nauru": 11300, "Tuvalu": 11300,\n        ...                         "Anguilla": 11300, "Montserrat": 5200}\n        >>> s = pd.Series(countries_population)\n        >>> s\n        Italy       59000000\n        France      65000000\n        Brunei        434000\n        Malta         434000\n        Maldives      434000\n        Iceland       337000\n        Nauru          11300\n        Tuvalu         11300\n        Anguilla       11300\n        Montserrat      5200\n        dtype: int64\n\n        The `n` smallest elements where ``n=5`` by default.\n\n        >>> s.nsmallest()\n        Montserrat    5200\n        Nauru        11300\n        Tuvalu       11300\n        Anguilla     11300\n        Iceland     337000\n        dtype: int64\n\n        The `n` smallest elements where ``n=3``. Default `keep` value is\n        \'first\' so Nauru and Tuvalu will be kept.\n\n        >>> s.nsmallest(3)\n        Montserrat   5200\n        Nauru       11300\n        Tuvalu      11300\n        dtype: int64\n\n        The `n` smallest elements where ``n=3`` and keeping the last\n        duplicates. Anguilla and Tuvalu will be kept since they are the last\n        with value 11300 based on the index order.\n\n        >>> s.nsmallest(3, keep=\'last\')\n        Montserrat   5200\n        Anguilla    11300\n        Tuvalu      11300\n        dtype: int64\n\n        The `n` smallest elements where ``n=3`` with all duplicates kept. Note\n        that the returned Series has four elements due to the three duplicates.\n\n        >>> s.nsmallest(3, keep=\'all\')\n        Montserrat   5200\n        Nauru       11300\n        Tuvalu      11300\n        Anguilla    11300\n        dtype: int64\n        '
        return algorithms.SelectNSeries(self, n=n, keep=keep).nsmallest()

    def swaplevel(self, i=(- 2), j=(- 1), copy=True):
        '\n        Swap levels i and j in a :class:`MultiIndex`.\n\n        Default is to swap the two innermost levels of the index.\n\n        Parameters\n        ----------\n        i, j : int, str\n            Level of the indices to be swapped. Can pass level name as string.\n        copy : bool, default True\n            Whether to copy underlying data.\n\n        Returns\n        -------\n        Series\n            Series with levels swapped in MultiIndex.\n        '
        assert isinstance(self.index, MultiIndex)
        new_index = self.index.swaplevel(i, j)
        return self._constructor(self._values, index=new_index, copy=copy).__finalize__(self, method='swaplevel')

    def reorder_levels(self, order):
        '\n        Rearrange index levels using input order.\n\n        May not drop or duplicate levels.\n\n        Parameters\n        ----------\n        order : list of int representing new level order\n            Reference level by number or key.\n\n        Returns\n        -------\n        type of caller (new object)\n        '
        if (not isinstance(self.index, MultiIndex)):
            raise Exception('Can only reorder levels on a hierarchical axis.')
        result = self.copy()
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index=False):
        "\n        Transform each element of a list-like to a row.\n\n        .. versionadded:: 0.25.0\n\n        Parameters\n        ----------\n        ignore_index : bool, default False\n            If True, the resulting index will be labeled 0, 1, …, n - 1.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        Series\n            Exploded lists to rows; index will be duplicated for these rows.\n\n        See Also\n        --------\n        Series.str.split : Split string values on specified separator.\n        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex\n            to produce DataFrame.\n        DataFrame.melt : Unpivot a DataFrame from wide format to long format.\n        DataFrame.explode : Explode a DataFrame from list-like\n            columns to long format.\n\n        Notes\n        -----\n        This routine will explode list-likes including lists, tuples, sets,\n        Series, and np.ndarray. The result dtype of the subset rows will\n        be object. Scalars will be returned unchanged, and empty list-likes will\n        result in a np.nan for that row. In addition, the ordering of elements in\n        the output will be non-deterministic when exploding sets.\n\n        Examples\n        --------\n        >>> s = pd.Series([[1, 2, 3], 'foo', [], [3, 4]])\n        >>> s\n        0    [1, 2, 3]\n        1          foo\n        2           []\n        3       [3, 4]\n        dtype: object\n\n        >>> s.explode()\n        0      1\n        0      2\n        0      3\n        1    foo\n        2    NaN\n        3      3\n        3      4\n        dtype: object\n        "
        if ((not len(self)) or (not is_object_dtype(self))):
            return self.copy()
        (values, counts) = reshape.explode(np.asarray(self.array))
        if ignore_index:
            index = ibase.default_index(len(values))
        else:
            index = self.index.repeat(counts)
        result = self._constructor(values, index=index, name=self.name)
        return result

    def unstack(self, level=(- 1), fill_value=None):
        "\n        Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.\n\n        Parameters\n        ----------\n        level : int, str, or list of these, default last level\n            Level(s) to unstack, can pass level name.\n        fill_value : scalar value, default None\n            Value to use when replacing NaN values.\n\n        Returns\n        -------\n        DataFrame\n            Unstacked Series.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3, 4],\n        ...               index=pd.MultiIndex.from_product([['one', 'two'],\n        ...                                                 ['a', 'b']]))\n        >>> s\n        one  a    1\n             b    2\n        two  a    3\n             b    4\n        dtype: int64\n\n        >>> s.unstack(level=-1)\n             a  b\n        one  1  2\n        two  3  4\n\n        >>> s.unstack(level=0)\n           one  two\n        a    1    3\n        b    2    4\n        "
        from pandas.core.reshape.reshape import unstack
        return unstack(self, level, fill_value)

    def map(self, arg, na_action=None):
        "\n        Map values of Series according to input correspondence.\n\n        Used for substituting each value in a Series with another value,\n        that may be derived from a function, a ``dict`` or\n        a :class:`Series`.\n\n        Parameters\n        ----------\n        arg : function, collections.abc.Mapping subclass or Series\n            Mapping correspondence.\n        na_action : {None, 'ignore'}, default None\n            If 'ignore', propagate NaN values, without passing them to the\n            mapping correspondence.\n\n        Returns\n        -------\n        Series\n            Same index as caller.\n\n        See Also\n        --------\n        Series.apply : For applying more complex functions on a Series.\n        DataFrame.apply : Apply a function row-/column-wise.\n        DataFrame.applymap : Apply a function elementwise on a whole DataFrame.\n\n        Notes\n        -----\n        When ``arg`` is a dictionary, values in Series that are not in the\n        dictionary (as keys) are converted to ``NaN``. However, if the\n        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.\n        provides a method for default values), then this default is used\n        rather than ``NaN``.\n\n        Examples\n        --------\n        >>> s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])\n        >>> s\n        0      cat\n        1      dog\n        2      NaN\n        3   rabbit\n        dtype: object\n\n        ``map`` accepts a ``dict`` or a ``Series``. Values that are not found\n        in the ``dict`` are converted to ``NaN``, unless the dict has a default\n        value (e.g. ``defaultdict``):\n\n        >>> s.map({'cat': 'kitten', 'dog': 'puppy'})\n        0   kitten\n        1    puppy\n        2      NaN\n        3      NaN\n        dtype: object\n\n        It also accepts a function:\n\n        >>> s.map('I am a {}'.format)\n        0       I am a cat\n        1       I am a dog\n        2       I am a nan\n        3    I am a rabbit\n        dtype: object\n\n        To avoid applying the function to missing values (and keep them as\n        ``NaN``) ``na_action='ignore'`` can be used:\n\n        >>> s.map('I am a {}'.format, na_action='ignore')\n        0     I am a cat\n        1     I am a dog\n        2            NaN\n        3  I am a rabbit\n        dtype: object\n        "
        new_values = super()._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index).__finalize__(self, method='map')

    def _gotitem(self, key, ndim, subset=None):
        '\n        Sub-classes to define. Return a sliced object.\n\n        Parameters\n        ----------\n        key : string / list of selections\n        ndim : 1,2\n            Requested ndim of result.\n        subset : object, default None\n            Subset to act on.\n        '
        return self
    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    Series.apply : Invoke function on a Series.\n    Series.transform : Transform function producing a Series with like indexes.\n    ')
    _agg_examples_doc = dedent("\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n\n    >>> s.agg('min')\n    1\n\n    >>> s.agg(['min', 'max'])\n    min   1\n    max   4\n    dtype: int64\n    ")

    @doc(generic._shared_docs['aggregate'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'], see_also=_agg_see_also_doc, examples=_agg_examples_doc)
    def aggregate(self, func=None, axis=0, *args, **kwargs):
        self._get_axis_number(axis)
        if (func is None):
            func = dict(kwargs.items())
        (result, how) = aggregate(self, func, *args, **kwargs)
        if (result is None):
            kwargs.pop('_axis', None)
            kwargs.pop('_level', None)
            try:
                result = self.apply(func, *args, **kwargs)
            except (ValueError, AttributeError, TypeError):
                result = func(self, *args, **kwargs)
        return result
    agg = aggregate

    @doc(_shared_docs['transform'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'])
    def transform(self, func, axis=0, *args, **kwargs):
        return transform(self, func, axis, *args, **kwargs)

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        "\n        Invoke function on values of Series.\n\n        Can be ufunc (a NumPy function that applies to the entire Series)\n        or a Python function that only works on single values.\n\n        Parameters\n        ----------\n        func : function\n            Python function or NumPy ufunc to apply.\n        convert_dtype : bool, default True\n            Try to find better dtype for elementwise function results. If\n            False, leave as dtype=object.\n        args : tuple\n            Positional arguments passed to func after the series value.\n        **kwds\n            Additional keyword arguments passed to func.\n\n        Returns\n        -------\n        Series or DataFrame\n            If func returns a Series object the result will be a DataFrame.\n\n        See Also\n        --------\n        Series.map: For element-wise operations.\n        Series.agg: Only perform aggregating type operations.\n        Series.transform: Only perform transforming type operations.\n\n        Examples\n        --------\n        Create a series with typical summer temperatures for each city.\n\n        >>> s = pd.Series([20, 21, 12],\n        ...               index=['London', 'New York', 'Helsinki'])\n        >>> s\n        London      20\n        New York    21\n        Helsinki    12\n        dtype: int64\n\n        Square the values by defining a function and passing it as an\n        argument to ``apply()``.\n\n        >>> def square(x):\n        ...     return x ** 2\n        >>> s.apply(square)\n        London      400\n        New York    441\n        Helsinki    144\n        dtype: int64\n\n        Square the values by passing an anonymous function as an\n        argument to ``apply()``.\n\n        >>> s.apply(lambda x: x ** 2)\n        London      400\n        New York    441\n        Helsinki    144\n        dtype: int64\n\n        Define a custom function that needs additional positional\n        arguments and pass these additional arguments using the\n        ``args`` keyword.\n\n        >>> def subtract_custom_value(x, custom_value):\n        ...     return x - custom_value\n\n        >>> s.apply(subtract_custom_value, args=(5,))\n        London      15\n        New York    16\n        Helsinki     7\n        dtype: int64\n\n        Define a custom function that takes keyword arguments\n        and pass these arguments to ``apply``.\n\n        >>> def add_custom_values(x, **kwargs):\n        ...     for month in kwargs:\n        ...         x += kwargs[month]\n        ...     return x\n\n        >>> s.apply(add_custom_values, june=30, july=20, august=25)\n        London      95\n        New York    96\n        Helsinki    87\n        dtype: int64\n\n        Use a function from the Numpy library.\n\n        >>> s.apply(np.log)\n        London      2.995732\n        New York    3.044522\n        Helsinki    2.484907\n        dtype: float64\n        "
        if (len(self) == 0):
            return self._constructor(dtype=self.dtype, index=self.index).__finalize__(self, method='apply')
        if isinstance(func, (list, dict)):
            return self.aggregate(func, *args, **kwds)
        if isinstance(func, str):
            return self._try_aggregate_string_function(func, *args, **kwds)
        if (kwds or (args and (not isinstance(func, np.ufunc)))):

            def f(x):
                return func(x, *args, **kwds)
        else:
            f = func
        with np.errstate(all='ignore'):
            if isinstance(f, np.ufunc):
                return f(self)
            if (is_extension_array_dtype(self.dtype) and hasattr(self._values, 'map')):
                mapped = self._values.map(f)
            else:
                values = self.astype(object)._values
                mapped = lib.map_infer(values, f, convert=convert_dtype)
        if (len(mapped) and isinstance(mapped[0], Series)):
            return self._constructor_expanddim(pd_array(mapped), index=self.index)
        else:
            return self._constructor(mapped, index=self.index).__finalize__(self, method='apply')

    def _reduce(self, op, name, *, axis=0, skipna=True, numeric_only=None, filter_type=None, **kwds):
        '\n        Perform a reduction operation.\n\n        If we have an ndarray as a value, then simply perform the operation,\n        otherwise delegate to the object.\n        '
        delegate = self._values
        if (axis is not None):
            self._get_axis_number(axis)
        if isinstance(delegate, ExtensionArray):
            return delegate._reduce(name, skipna=skipna, **kwds)
        else:
            if numeric_only:
                raise NotImplementedError(f'Series.{name} does not implement numeric_only.')
            with np.errstate(all='ignore'):
                return op(delegate, skipna=skipna, **kwds)

    def _reindex_indexer(self, new_index, indexer, copy):
        if (indexer is None):
            if copy:
                return self.copy()
            return self
        new_values = algorithms.take_1d(self._values, indexer, allow_fill=True, fill_value=None)
        return self._constructor(new_values, index=new_index)

    def _needs_reindex_multi(self, axes, method, level):
        '\n        Check if we do need a multi reindex; this is for compat with\n        higher dims.\n        '
        return False

    @doc(NDFrame.align, klass=_shared_doc_kwargs['klass'], axes_single_arg=_shared_doc_kwargs['axes_single_arg'])
    def align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None, limit=None, fill_axis=0, broadcast_axis=None):
        return super().align(other, join=join, axis=axis, level=level, copy=copy, fill_value=fill_value, method=method, limit=limit, fill_axis=fill_axis, broadcast_axis=broadcast_axis)

    def rename(self, index=None, *, axis=None, copy=True, inplace=False, level=None, errors='ignore'):
        '\n        Alter Series index labels or name.\n\n        Function / dict values must be unique (1-to-1). Labels not contained in\n        a dict / Series will be left as-is. Extra labels listed don\'t throw an\n        error.\n\n        Alternatively, change ``Series.name`` with a scalar value.\n\n        See the :ref:`user guide <basics.rename>` for more.\n\n        Parameters\n        ----------\n        axis : {0 or "index"}\n            Unused. Accepted for compatibility with DataFrame method only.\n        index : scalar, hashable sequence, dict-like or function, optional\n            Functions or dict-like are transformations to apply to\n            the index.\n            Scalar or hashable sequence-like will alter the ``Series.name``\n            attribute.\n\n        **kwargs\n            Additional keyword arguments passed to the function. Only the\n            "inplace" keyword is used.\n\n        Returns\n        -------\n        Series or None\n            Series with index labels or name altered or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.rename : Corresponding DataFrame method.\n        Series.rename_axis : Set the name of the axis.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3])\n        >>> s\n        0    1\n        1    2\n        2    3\n        dtype: int64\n        >>> s.rename("my_name")  # scalar, changes Series.name\n        0    1\n        1    2\n        2    3\n        Name: my_name, dtype: int64\n        >>> s.rename(lambda x: x ** 2)  # function, changes labels\n        0    1\n        1    2\n        4    3\n        dtype: int64\n        >>> s.rename({1: 3, 2: 5})  # mapping, changes labels\n        0    1\n        3    2\n        5    3\n        dtype: int64\n        '
        if (callable(index) or is_dict_like(index)):
            return super().rename(index, copy=copy, inplace=inplace, level=level, errors=errors)
        else:
            return self._set_name(index, inplace=inplace)

    @Appender("\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3])\n        >>> s\n        0    1\n        1    2\n        2    3\n        dtype: int64\n\n        >>> s.set_axis(['a', 'b', 'c'], axis=0)\n        a    1\n        b    2\n        c    3\n        dtype: int64\n    ")
    @Substitution(**_shared_doc_kwargs, extended_summary_sub='', axis_description_sub='', see_also_sub='')
    @Appender(generic.NDFrame.set_axis.__doc__)
    def set_axis(self, labels, axis=0, inplace=False):
        return super().set_axis(labels, axis=axis, inplace=inplace)

    @doc(NDFrame.reindex, klass=_shared_doc_kwargs['klass'], axes=_shared_doc_kwargs['axes'], optional_labels=_shared_doc_kwargs['optional_labels'], optional_axis=_shared_doc_kwargs['optional_axis'])
    def reindex(self, index=None, **kwargs):
        return super().reindex(index=index, **kwargs)

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        "\n        Return Series with specified index labels removed.\n\n        Remove elements of a Series based on specifying the index labels.\n        When using a multi-index, labels on different levels can be removed\n        by specifying the level.\n\n        Parameters\n        ----------\n        labels : single label or list-like\n            Index labels to drop.\n        axis : 0, default 0\n            Redundant for application on Series.\n        index : single label or list-like\n            Redundant for application on Series, but 'index' can be used instead\n            of 'labels'.\n        columns : single label or list-like\n            No change is made to the Series; use 'index' or 'labels' instead.\n        level : int or level name, optional\n            For MultiIndex, level for which the labels will be removed.\n        inplace : bool, default False\n            If True, do operation inplace and return None.\n        errors : {'ignore', 'raise'}, default 'raise'\n            If 'ignore', suppress error and only existing labels are dropped.\n\n        Returns\n        -------\n        Series or None\n            Series with specified index labels removed or None if ``inplace=True``.\n\n        Raises\n        ------\n        KeyError\n            If none of the labels are found in the index.\n\n        See Also\n        --------\n        Series.reindex : Return only specified index labels of Series.\n        Series.dropna : Return series without null values.\n        Series.drop_duplicates : Return Series with duplicate values removed.\n        DataFrame.drop : Drop specified labels from rows or columns.\n\n        Examples\n        --------\n        >>> s = pd.Series(data=np.arange(3), index=['A', 'B', 'C'])\n        >>> s\n        A  0\n        B  1\n        C  2\n        dtype: int64\n\n        Drop labels B en C\n\n        >>> s.drop(labels=['B', 'C'])\n        A  0\n        dtype: int64\n\n        Drop 2nd level label in MultiIndex Series\n\n        >>> midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],\n        ...                              ['speed', 'weight', 'length']],\n        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],\n        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])\n        >>> s = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],\n        ...               index=midx)\n        >>> s\n        lama    speed      45.0\n                weight    200.0\n                length      1.2\n        cow     speed      30.0\n                weight    250.0\n                length      1.5\n        falcon  speed     320.0\n                weight      1.0\n                length      0.3\n        dtype: float64\n\n        >>> s.drop(labels='weight', level=1)\n        lama    speed      45.0\n                length      1.2\n        cow     speed      30.0\n                length      1.5\n        falcon  speed     320.0\n                length      0.3\n        dtype: float64\n        "
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    @doc(NDFrame.fillna, **_shared_doc_kwargs)
    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        return super().fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)

    def pop(self, item):
        '\n        Return item and drops from series. Raise KeyError if not found.\n\n        Parameters\n        ----------\n        item : label\n            Index of the element that needs to be removed.\n\n        Returns\n        -------\n        Value that is popped from series.\n\n        Examples\n        --------\n        >>> ser = pd.Series([1,2,3])\n\n        >>> ser.pop(0)\n        1\n\n        >>> ser\n        1    2\n        2    3\n        dtype: int64\n        '
        return super().pop(item=item)

    @doc(NDFrame.replace, klass=_shared_doc_kwargs['klass'], inplace=_shared_doc_kwargs['inplace'], replace_iloc=_shared_doc_kwargs['replace_iloc'])
    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        return super().replace(to_replace=to_replace, value=value, inplace=inplace, limit=limit, regex=regex, method=method)

    def _replace_single(self, to_replace, method, inplace, limit):
        '\n        Replaces values in a Series using the fill method specified when no\n        replacement value is given in the replace method\n        '
        orig_dtype = self.dtype
        result = (self if inplace else self.copy())
        fill_f = missing.get_fill_func(method)
        mask = missing.mask_missing(result.values, to_replace)
        values = fill_f(result.values, limit=limit, mask=mask)
        if ((values.dtype == orig_dtype) and inplace):
            return
        result = self._constructor(values, index=self.index, dtype=self.dtype)
        result = result.__finalize__(self)
        if inplace:
            self._update_inplace(result)
            return
        return result

    @doc(NDFrame.shift, klass=_shared_doc_kwargs['klass'])
    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        return super().shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)

    def memory_usage(self, index=True, deep=False):
        '\n        Return the memory usage of the Series.\n\n        The memory usage can optionally include the contribution of\n        the index and of elements of `object` dtype.\n\n        Parameters\n        ----------\n        index : bool, default True\n            Specifies whether to include the memory usage of the Series index.\n        deep : bool, default False\n            If True, introspect the data deeply by interrogating\n            `object` dtypes for system-level memory consumption, and include\n            it in the returned value.\n\n        Returns\n        -------\n        int\n            Bytes of memory consumed.\n\n        See Also\n        --------\n        numpy.ndarray.nbytes : Total bytes consumed by the elements of the\n            array.\n        DataFrame.memory_usage : Bytes consumed by a DataFrame.\n\n        Examples\n        --------\n        >>> s = pd.Series(range(3))\n        >>> s.memory_usage()\n        152\n\n        Not including the index gives the size of the rest of the data, which\n        is necessarily smaller:\n\n        >>> s.memory_usage(index=False)\n        24\n\n        The memory footprint of `object` values is ignored by default:\n\n        >>> s = pd.Series(["a", "b"])\n        >>> s.values\n        array([\'a\', \'b\'], dtype=object)\n        >>> s.memory_usage()\n        144\n        >>> s.memory_usage(deep=True)\n        244\n        '
        v = super().memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values):
        "\n        Whether elements in Series are contained in `values`.\n\n        Return a boolean Series showing whether each element in the Series\n        matches an element in the passed sequence of `values` exactly.\n\n        Parameters\n        ----------\n        values : set or list-like\n            The sequence of values to test. Passing in a single string will\n            raise a ``TypeError``. Instead, turn a single string into a\n            list of one element.\n\n        Returns\n        -------\n        Series\n            Series of booleans indicating if each element is in values.\n\n        Raises\n        ------\n        TypeError\n          * If `values` is a string\n\n        See Also\n        --------\n        DataFrame.isin : Equivalent method on DataFrame.\n\n        Examples\n        --------\n        >>> s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama',\n        ...                'hippo'], name='animal')\n        >>> s.isin(['cow', 'lama'])\n        0     True\n        1     True\n        2     True\n        3    False\n        4     True\n        5    False\n        Name: animal, dtype: bool\n\n        Passing a single string as ``s.isin('lama')`` will raise an error. Use\n        a list of one element instead:\n\n        >>> s.isin(['lama'])\n        0     True\n        1    False\n        2     True\n        3    False\n        4     True\n        5    False\n        Name: animal, dtype: bool\n        "
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index).__finalize__(self, method='isin')

    def between(self, left, right, inclusive=True):
        "\n        Return boolean Series equivalent to left <= series <= right.\n\n        This function returns a boolean vector containing `True` wherever the\n        corresponding Series element is between the boundary values `left` and\n        `right`. NA values are treated as `False`.\n\n        Parameters\n        ----------\n        left : scalar or list-like\n            Left boundary.\n        right : scalar or list-like\n            Right boundary.\n        inclusive : bool, default True\n            Include boundaries.\n\n        Returns\n        -------\n        Series\n            Series representing whether each element is between left and\n            right (inclusive).\n\n        See Also\n        --------\n        Series.gt : Greater than of series and other.\n        Series.lt : Less than of series and other.\n\n        Notes\n        -----\n        This function is equivalent to ``(left <= ser) & (ser <= right)``\n\n        Examples\n        --------\n        >>> s = pd.Series([2, 0, 4, 8, np.nan])\n\n        Boundary values are included by default:\n\n        >>> s.between(1, 4)\n        0     True\n        1    False\n        2     True\n        3    False\n        4    False\n        dtype: bool\n\n        With `inclusive` set to ``False`` boundary values are excluded:\n\n        >>> s.between(1, 4, inclusive=False)\n        0     True\n        1    False\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        `left` and `right` can be any scalar value:\n\n        >>> s = pd.Series(['Alice', 'Bob', 'Carol', 'Eve'])\n        >>> s.between('Anna', 'Daniel')\n        0    False\n        1     True\n        2     True\n        3    False\n        dtype: bool\n        "
        if inclusive:
            lmask = (self >= left)
            rmask = (self <= right)
        else:
            lmask = (self > left)
            rmask = (self < right)
        return (lmask & rmask)

    def _convert_dtypes(self, infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True):
        input_series = self
        if infer_objects:
            input_series = input_series.infer_objects()
            if is_object_dtype(input_series):
                input_series = input_series.copy()
        if (convert_string or convert_integer or convert_boolean or convert_floating):
            inferred_dtype = convert_dtypes(input_series._values, convert_string, convert_integer, convert_boolean, convert_floating)
            try:
                result = input_series.astype(inferred_dtype)
            except TypeError:
                result = input_series.copy()
        else:
            result = input_series.copy()
        return result

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isna(self):
        return generic.NDFrame.isna(self)

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isnull(self):
        return super().isnull()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notna(self):
        return super().notna()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notnull(self):
        return super().notnull()

    def dropna(self, axis=0, inplace=False, how=None):
        "\n        Return a new Series with missing values removed.\n\n        See the :ref:`User Guide <missing_data>` for more on which values are\n        considered missing, and how to work with missing data.\n\n        Parameters\n        ----------\n        axis : {0 or 'index'}, default 0\n            There is only one axis to drop values from.\n        inplace : bool, default False\n            If True, do operation inplace and return None.\n        how : str, optional\n            Not in use. Kept for compatibility.\n\n        Returns\n        -------\n        Series or None\n            Series with NA entries dropped from it or None if ``inplace=True``.\n\n        See Also\n        --------\n        Series.isna: Indicate missing values.\n        Series.notna : Indicate existing (non-missing) values.\n        Series.fillna : Replace missing values.\n        DataFrame.dropna : Drop rows or columns which contain NA values.\n        Index.dropna : Drop missing indices.\n\n        Examples\n        --------\n        >>> ser = pd.Series([1., 2., np.nan])\n        >>> ser\n        0    1.0\n        1    2.0\n        2    NaN\n        dtype: float64\n\n        Drop NA values from a Series.\n\n        >>> ser.dropna()\n        0    1.0\n        1    2.0\n        dtype: float64\n\n        Keep the Series with valid entries in the same variable.\n\n        >>> ser.dropna(inplace=True)\n        >>> ser\n        0    1.0\n        1    2.0\n        dtype: float64\n\n        Empty strings are not considered NA values. ``None`` is considered an\n        NA value.\n\n        >>> ser = pd.Series([np.NaN, 2, pd.NaT, '', None, 'I stay'])\n        >>> ser\n        0       NaN\n        1         2\n        2       NaT\n        3\n        4      None\n        5    I stay\n        dtype: object\n        >>> ser.dropna()\n        1         2\n        3\n        5    I stay\n        dtype: object\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._get_axis_number((axis or 0))
        if self._can_hold_na:
            result = remove_na_arraylike(self)
            if inplace:
                self._update_inplace(result)
            else:
                return result
        elif inplace:
            pass
        else:
            return self.copy()

    @doc(NDFrame.asfreq, **_shared_doc_kwargs)
    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        return super().asfreq(freq=freq, method=method, how=how, normalize=normalize, fill_value=fill_value)

    @doc(NDFrame.resample, **_shared_doc_kwargs)
    def resample(self, rule, axis=0, closed=None, label=None, convention='start', kind=None, loffset=None, base=None, on=None, level=None, origin='start_day', offset=None):
        return super().resample(rule=rule, axis=axis, closed=closed, label=label, convention=convention, kind=kind, loffset=loffset, base=base, on=on, level=level, origin=origin, offset=offset)

    def to_timestamp(self, freq=None, how='start', copy=True):
        "\n        Cast to DatetimeIndex of Timestamps, at *beginning* of period.\n\n        Parameters\n        ----------\n        freq : str, default frequency of PeriodIndex\n            Desired frequency.\n        how : {'s', 'e', 'start', 'end'}\n            Convention for converting period to timestamp; start of period\n            vs. end.\n        copy : bool, default True\n            Whether or not to return a copy.\n\n        Returns\n        -------\n        Series with DatetimeIndex\n        "
        new_values = self._values
        if copy:
            new_values = new_values.copy()
        if (not isinstance(self.index, PeriodIndex)):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_index = self.index.to_timestamp(freq=freq, how=how)
        return self._constructor(new_values, index=new_index).__finalize__(self, method='to_timestamp')

    def to_period(self, freq=None, copy=True):
        '\n        Convert Series from DatetimeIndex to PeriodIndex.\n\n        Parameters\n        ----------\n        freq : str, default None\n            Frequency associated with the PeriodIndex.\n        copy : bool, default True\n            Whether or not to return a copy.\n\n        Returns\n        -------\n        Series\n            Series with index converted to PeriodIndex.\n        '
        new_values = self._values
        if copy:
            new_values = new_values.copy()
        if (not isinstance(self.index, DatetimeIndex)):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_index = self.index.to_period(freq=freq)
        return self._constructor(new_values, index=new_index).__finalize__(self, method='to_period')
    _AXIS_ORDERS = ['index']
    _AXIS_REVERSED = False
    _AXIS_LEN = len(_AXIS_ORDERS)
    _info_axis_number = 0
    _info_axis_name = 'index'
    index = properties.AxisProperty(axis=0, doc='The index (axis labels) of the Series.')
    str = CachedAccessor('str', StringMethods)
    dt = CachedAccessor('dt', CombinedDatetimelikeProperties)
    cat = CachedAccessor('cat', CategoricalAccessor)
    plot = CachedAccessor('plot', pandas.plotting.PlotAccessor)
    sparse = CachedAccessor('sparse', SparseAccessor)
    hist = pandas.plotting.hist_series

    def _cmp_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)
        if (isinstance(other, Series) and (not self._indexed_same(other))):
            raise ValueError('Can only compare identically-labeled Series objects')
        lvalues = extract_array(self, extract_numpy=True)
        rvalues = extract_array(other, extract_numpy=True)
        res_values = ops.comparison_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _logical_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)
        (self, other) = ops.align_method_SERIES(self, other, align_asobject=True)
        lvalues = extract_array(self, extract_numpy=True)
        rvalues = extract_array(other, extract_numpy=True)
        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)
        (self, other) = ops.align_method_SERIES(self, other)
        lvalues = extract_array(self, extract_numpy=True)
        rvalues = extract_array(other, extract_numpy=True)
        result = ops.arithmetic_op(lvalues, rvalues, op)
        return self._construct_result(result, name=res_name)
Series._add_numeric_operations()
ops.add_flex_arithmetic_methods(Series)
