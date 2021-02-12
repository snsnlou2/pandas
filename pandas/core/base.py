
'\nBase and utility classes for pandas objects.\n'
import builtins
import textwrap
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Optional, TypeVar, Union, cast
import numpy as np
import pandas._libs.lib as lib
from pandas._typing import DtypeObj, IndexLabel
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import is_categorical_dtype, is_dict_like, is_extension_array_dtype, is_object_dtype, is_scalar
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna, remove_na_arraylike
from pandas.core import algorithms
from pandas.core.accessor import DirNamesMixin
from pandas.core.algorithms import duplicated, unique1d, value_counts
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import create_series_with_explicit_dtype
import pandas.core.nanops as nanops
if TYPE_CHECKING:
    from pandas import Categorical
_shared_docs = {}
_indexops_doc_kwargs = {'klass': 'IndexOpsMixin', 'inplace': '', 'unique': 'IndexOpsMixin', 'duplicated': 'IndexOpsMixin'}
_T = TypeVar('_T', bound='IndexOpsMixin')

class PandasObject(DirNamesMixin):
    '\n    Baseclass for various pandas objects.\n    '

    @property
    def _constructor(self):
        "\n        Class constructor (for this class it's just `__class__`.\n        "
        return type(self)

    def __repr__(self):
        '\n        Return a string representation for a particular object.\n        '
        return object.__repr__(self)

    def _reset_cache(self, key=None):
        '\n        Reset cached properties. If ``key`` is passed, only clears that key.\n        '
        if (getattr(self, '_cache', None) is None):
            return
        if (key is None):
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __sizeof__(self):
        '\n        Generates the total memory usage for an object that returns\n        either a value or Series of values\n        '
        if hasattr(self, 'memory_usage'):
            mem = self.memory_usage(deep=True)
            return int((mem if is_scalar(mem) else mem.sum()))
        return super().__sizeof__()

class NoNewAttributesMixin():
    '\n    Mixin which prevents adding new attributes.\n\n    Prevents additional attributes via xxx.attribute = "something" after a\n    call to `self.__freeze()`. Mainly used to prevent the user from using\n    wrong attributes on an accessor (`Series.cat/.str/.dt`).\n\n    If you really want to add a new attribute at a later time, you need to use\n    `object.__setattr__(self, key, value)`.\n    '

    def _freeze(self):
        '\n        Prevents setting additional attributes.\n        '
        object.__setattr__(self, '__frozen', True)

    def __setattr__(self, key, value):
        if (getattr(self, '__frozen', False) and (not ((key == '_cache') or (key in type(self).__dict__) or (getattr(self, key, None) is not None)))):
            raise AttributeError(f"You cannot add any new attribute '{key}'")
        object.__setattr__(self, key, value)

class DataError(Exception):
    pass

class SpecificationError(Exception):
    pass

class SelectionMixin():
    '\n    mixin implementing the selection & aggregation interface on a group-like\n    object sub-classes need to define: obj, exclusions\n    '
    _selection = None
    _internal_names = ['_cache', '__setstate__']
    _internal_names_set = set(_internal_names)
    _builtin_table = {builtins.sum: np.sum, builtins.max: np.max, builtins.min: np.min}
    _cython_table = {builtins.sum: 'sum', builtins.max: 'max', builtins.min: 'min', np.all: 'all', np.any: 'any', np.sum: 'sum', np.nansum: 'sum', np.mean: 'mean', np.nanmean: 'mean', np.prod: 'prod', np.nanprod: 'prod', np.std: 'std', np.nanstd: 'std', np.var: 'var', np.nanvar: 'var', np.median: 'median', np.nanmedian: 'median', np.max: 'max', np.nanmax: 'max', np.min: 'min', np.nanmin: 'min', np.cumprod: 'cumprod', np.nancumprod: 'cumprod', np.cumsum: 'cumsum', np.nancumsum: 'cumsum'}

    @property
    def _selection_name(self):
        "\n        Return a name for myself;\n\n        This would ideally be called the 'name' property,\n        but we cannot conflict with the Series.name property which can be set.\n        "
        return self._selection

    @property
    def _selection_list(self):
        if (not isinstance(self._selection, (list, tuple, ABCSeries, ABCIndex, np.ndarray))):
            return [self._selection]
        return self._selection

    @cache_readonly
    def _selected_obj(self):
        if ((self._selection is None) or isinstance(self.obj, ABCSeries)):
            return self.obj
        else:
            return self.obj[self._selection]

    @cache_readonly
    def ndim(self):
        return self._selected_obj.ndim

    @cache_readonly
    def _obj_with_exclusions(self):
        if ((self._selection is not None) and isinstance(self.obj, ABCDataFrame)):
            return self.obj.reindex(columns=self._selection_list)
        if (len(self.exclusions) > 0):
            return self.obj.drop(self.exclusions, axis=1)
        else:
            return self.obj

    def __getitem__(self, key):
        if (self._selection is not None):
            raise IndexError(f'Column(s) {self._selection} already selected')
        if isinstance(key, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            if (len(self.obj.columns.intersection(key)) != len(key)):
                bad_keys = list(set(key).difference(self.obj.columns))
                raise KeyError(f'Columns not found: {str(bad_keys)[1:(- 1)]}')
            return self._gotitem(list(key), ndim=2)
        elif (not getattr(self, 'as_index', False)):
            if (key not in self.obj.columns):
                raise KeyError(f'Column not found: {key}')
            return self._gotitem(key, ndim=2)
        else:
            if (key not in self.obj):
                raise KeyError(f'Column not found: {key}')
            return self._gotitem(key, ndim=1)

    def _gotitem(self, key, ndim, subset=None):
        '\n        sub-classes to define\n        return a sliced object\n\n        Parameters\n        ----------\n        key : str / list of selections\n        ndim : {1, 2}\n            requested ndim of result\n        subset : object, default None\n            subset to act on\n        '
        raise AbstractMethodError(self)

    def aggregate(self, func, *args, **kwargs):
        raise AbstractMethodError(self)
    agg = aggregate

    def _try_aggregate_string_function(self, arg, *args, **kwargs):
        '\n        if arg is a string, then try to operate on it:\n        - try to find a function (or attribute) on ourselves\n        - try to find a numpy function\n        - raise\n        '
        assert isinstance(arg, str)
        f = getattr(self, arg, None)
        if (f is not None):
            if callable(f):
                return f(*args, **kwargs)
            assert (len(args) == 0)
            assert (len([kwarg for kwarg in kwargs if (kwarg not in ['axis'])]) == 0)
            return f
        f = getattr(np, arg, None)
        if (f is not None):
            if hasattr(self, '__array__'):
                return f(self, *args, **kwargs)
        raise AttributeError(f"'{arg}' is not a valid function for '{type(self).__name__}' object")

    def _get_cython_func(self, arg):
        '\n        if we define an internal function for this argument, return it\n        '
        return self._cython_table.get(arg)

    def _is_builtin_func(self, arg):
        '\n        if we define an builtin function for this argument, return it,\n        otherwise return the arg\n        '
        return self._builtin_table.get(arg, arg)

class IndexOpsMixin(OpsMixin):
    '\n    Common ops mixin to support a unified interface / docs for Series / Index\n    '
    __array_priority__ = 1000
    _hidden_attrs = frozenset(['tolist'])

    @property
    def dtype(self):
        raise AbstractMethodError(self)

    @property
    def _values(self):
        raise AbstractMethodError(self)

    def transpose(self, *args, **kwargs):
        '\n        Return the transpose, which is by definition self.\n\n        Returns\n        -------\n        %(klass)s\n        '
        nv.validate_transpose(args, kwargs)
        return self
    T = property(transpose, doc='\n        Return the transpose, which is by definition self.\n        ')

    @property
    def shape(self):
        '\n        Return a tuple of the shape of the underlying data.\n        '
        return self._values.shape

    def __len__(self):
        raise AbstractMethodError(self)

    @property
    def ndim(self):
        '\n        Number of dimensions of the underlying data, by definition 1.\n        '
        return 1

    def item(self):
        '\n        Return the first element of the underlying data as a Python scalar.\n\n        Returns\n        -------\n        scalar\n            The first element of %(klass)s.\n\n        Raises\n        ------\n        ValueError\n            If the data is not length-1.\n        '
        if (len(self) == 1):
            return next(iter(self))
        raise ValueError('can only convert an array of size 1 to a Python scalar')

    @property
    def nbytes(self):
        '\n        Return the number of bytes in the underlying data.\n        '
        return self._values.nbytes

    @property
    def size(self):
        '\n        Return the number of elements in the underlying data.\n        '
        return len(self._values)

    @property
    def array(self):
        "\n        The ExtensionArray of the data backing this Series or Index.\n\n        .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        ExtensionArray\n            An ExtensionArray of the values stored within. For extension\n            types, this is the actual array. For NumPy native types, this\n            is a thin (no copy) wrapper around :class:`numpy.ndarray`.\n\n            ``.array`` differs ``.values`` which may require converting the\n            data to a different form.\n\n        See Also\n        --------\n        Index.to_numpy : Similar method that always returns a NumPy array.\n        Series.to_numpy : Similar method that always returns a NumPy array.\n\n        Notes\n        -----\n        This table lays out the different array types for each extension\n        dtype within pandas.\n\n        ================== =============================\n        dtype              array type\n        ================== =============================\n        category           Categorical\n        period             PeriodArray\n        interval           IntervalArray\n        IntegerNA          IntegerArray\n        string             StringArray\n        boolean            BooleanArray\n        datetime64[ns, tz] DatetimeArray\n        ================== =============================\n\n        For any 3rd-party extension types, the array type will be an\n        ExtensionArray.\n\n        For all remaining dtypes ``.array`` will be a\n        :class:`arrays.NumpyExtensionArray` wrapping the actual ndarray\n        stored within. If you absolutely need a NumPy array (possibly with\n        copying / coercing data), then use :meth:`Series.to_numpy` instead.\n\n        Examples\n        --------\n        For regular NumPy types like int, and float, a PandasArray\n        is returned.\n\n        >>> pd.Series([1, 2, 3]).array\n        <PandasArray>\n        [1, 2, 3]\n        Length: 3, dtype: int64\n\n        For extension types, like Categorical, the actual ExtensionArray\n        is returned\n\n        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))\n        >>> ser.array\n        ['a', 'b', 'a']\n        Categories (2, object): ['a', 'b']\n        "
        raise AbstractMethodError(self)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default, **kwargs):
        '\n        A NumPy ndarray representing the values in this Series or Index.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        dtype : str or numpy.dtype, optional\n            The dtype to pass to :meth:`numpy.asarray`.\n        copy : bool, default False\n            Whether to ensure that the returned value is not a view on\n            another array. Note that ``copy=False`` does not *ensure* that\n            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that\n            a copy is made, even if not strictly necessary.\n        na_value : Any, optional\n            The value to use for missing values. The default value depends\n            on `dtype` and the type of the array.\n\n            .. versionadded:: 1.0.0\n\n        **kwargs\n            Additional keywords passed through to the ``to_numpy`` method\n            of the underlying array (for extension arrays).\n\n            .. versionadded:: 1.0.0\n\n        Returns\n        -------\n        numpy.ndarray\n\n        See Also\n        --------\n        Series.array : Get the actual data stored within.\n        Index.array : Get the actual data stored within.\n        DataFrame.to_numpy : Similar method for DataFrame.\n\n        Notes\n        -----\n        The returned array will be the same up to equality (values equal\n        in `self` will be equal in the returned array; likewise for values\n        that are not equal). When `self` contains an ExtensionArray, the\n        dtype may be different. For example, for a category-dtype Series,\n        ``to_numpy()`` will return a NumPy array and the categorical dtype\n        will be lost.\n\n        For NumPy dtypes, this will be a reference to the actual data stored\n        in this Series or Index (assuming ``copy=False``). Modifying the result\n        in place will modify the data stored in the Series or Index (not that\n        we recommend doing that).\n\n        For extension types, ``to_numpy()`` *may* require copying data and\n        coercing the result to a NumPy type (possibly object), which may be\n        expensive. When you need a no-copy reference to the underlying data,\n        :attr:`Series.array` should be used instead.\n\n        This table lays out the different dtypes and default return types of\n        ``to_numpy()`` for various dtypes within pandas.\n\n        ================== ================================\n        dtype              array type\n        ================== ================================\n        category[T]        ndarray[T] (same dtype as input)\n        period             ndarray[object] (Periods)\n        interval           ndarray[object] (Intervals)\n        IntegerNA          ndarray[object]\n        datetime64[ns]     datetime64[ns]\n        datetime64[ns, tz] ndarray[object] (Timestamps)\n        ================== ================================\n\n        Examples\n        --------\n        >>> ser = pd.Series(pd.Categorical([\'a\', \'b\', \'a\']))\n        >>> ser.to_numpy()\n        array([\'a\', \'b\', \'a\'], dtype=object)\n\n        Specify the `dtype` to control how datetime-aware data is represented.\n        Use ``dtype=object`` to return an ndarray of pandas :class:`Timestamp`\n        objects, each with the correct ``tz``.\n\n        >>> ser = pd.Series(pd.date_range(\'2000\', periods=2, tz="CET"))\n        >>> ser.to_numpy(dtype=object)\n        array([Timestamp(\'2000-01-01 00:00:00+0100\', tz=\'CET\', freq=\'D\'),\n               Timestamp(\'2000-01-02 00:00:00+0100\', tz=\'CET\', freq=\'D\')],\n              dtype=object)\n\n        Or ``dtype=\'datetime64[ns]\'`` to return an ndarray of native\n        datetime64 values. The values are converted to UTC and the timezone\n        info is dropped.\n\n        >>> ser.to_numpy(dtype="datetime64[ns]")\n        ... # doctest: +ELLIPSIS\n        array([\'1999-12-31T23:00:00.000000000\', \'2000-01-01T23:00:00...\'],\n              dtype=\'datetime64[ns]\')\n        '
        if is_extension_array_dtype(self.dtype):
            return self.array.to_numpy(dtype, copy=copy, na_value=na_value, **kwargs)
        elif kwargs:
            bad_keys = list(kwargs.keys())[0]
            raise TypeError(f"to_numpy() got an unexpected keyword argument '{bad_keys}'")
        result = np.asarray(self._values, dtype=dtype)
        if (copy or (na_value is not lib.no_default)):
            result = result.copy()
            if (na_value is not lib.no_default):
                result[self.isna()] = na_value
        return result

    @property
    def empty(self):
        return (not self.size)

    def max(self, axis=None, skipna=True, *args, **kwargs):
        "\n        Return the maximum value of the Index.\n\n        Parameters\n        ----------\n        axis : int, optional\n            For compatibility with NumPy. Only 0 or None are allowed.\n        skipna : bool, default True\n            Exclude NA/null values when showing the result.\n        *args, **kwargs\n            Additional arguments and keywords for compatibility with NumPy.\n\n        Returns\n        -------\n        scalar\n            Maximum value.\n\n        See Also\n        --------\n        Index.min : Return the minimum value in an Index.\n        Series.max : Return the maximum value in a Series.\n        DataFrame.max : Return the maximum values in a DataFrame.\n\n        Examples\n        --------\n        >>> idx = pd.Index([3, 2, 1])\n        >>> idx.max()\n        3\n\n        >>> idx = pd.Index(['c', 'b', 'a'])\n        >>> idx.max()\n        'c'\n\n        For a MultiIndex, the maximum is determined lexicographically.\n\n        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])\n        >>> idx.max()\n        ('b', 2)\n        "
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return nanops.nanmax(self._values, skipna=skipna)

    @doc(op='max', oppose='min', value='largest')
    def argmax(self, axis=None, skipna=True, *args, **kwargs):
        "\n        Return int position of the {value} value in the Series.\n\n        If the {op}imum is achieved in multiple locations,\n        the first row position is returned.\n\n        Parameters\n        ----------\n        axis : {{None}}\n            Dummy argument for consistency with Series.\n        skipna : bool, default True\n            Exclude NA/null values when showing the result.\n        *args, **kwargs\n            Additional arguments and keywords for compatibility with NumPy.\n\n        Returns\n        -------\n        int\n            Row position of the {op}imum value.\n\n        See Also\n        --------\n        Series.arg{op} : Return position of the {op}imum value.\n        Series.arg{oppose} : Return position of the {oppose}imum value.\n        numpy.ndarray.arg{op} : Equivalent method for numpy arrays.\n        Series.idxmax : Return index label of the maximum values.\n        Series.idxmin : Return index label of the minimum values.\n\n        Examples\n        --------\n        Consider dataset containing cereal calories\n\n        >>> s = pd.Series({{'Corn Flakes': 100.0, 'Almond Delight': 110.0,\n        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0}})\n        >>> s\n        Corn Flakes              100.0\n        Almond Delight           110.0\n        Cinnamon Toast Crunch    120.0\n        Cocoa Puff               110.0\n        dtype: float64\n\n        >>> s.argmax()\n        2\n        >>> s.argmin()\n        0\n\n        The maximum cereal calories is the third element and\n        the minimum cereal calories is the first element,\n        since series is zero-indexed.\n        "
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            if ((not skipna) and delegate.isna().any()):
                return (- 1)
            else:
                return delegate.argmax()
        else:
            return nanops.nanargmax(delegate, skipna=skipna)

    def min(self, axis=None, skipna=True, *args, **kwargs):
        "\n        Return the minimum value of the Index.\n\n        Parameters\n        ----------\n        axis : {None}\n            Dummy argument for consistency with Series.\n        skipna : bool, default True\n            Exclude NA/null values when showing the result.\n        *args, **kwargs\n            Additional arguments and keywords for compatibility with NumPy.\n\n        Returns\n        -------\n        scalar\n            Minimum value.\n\n        See Also\n        --------\n        Index.max : Return the maximum value of the object.\n        Series.min : Return the minimum value in a Series.\n        DataFrame.min : Return the minimum values in a DataFrame.\n\n        Examples\n        --------\n        >>> idx = pd.Index([3, 2, 1])\n        >>> idx.min()\n        1\n\n        >>> idx = pd.Index(['c', 'b', 'a'])\n        >>> idx.min()\n        'a'\n\n        For a MultiIndex, the minimum is determined lexicographically.\n\n        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])\n        >>> idx.min()\n        ('a', 1)\n        "
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return nanops.nanmin(self._values, skipna=skipna)

    @doc(argmax, op='min', oppose='max', value='smallest')
    def argmin(self, axis=None, skipna=True, *args, **kwargs):
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmin_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            if ((not skipna) and delegate.isna().any()):
                return (- 1)
            else:
                return delegate.argmin()
        else:
            return nanops.nanargmin(delegate, skipna=skipna)

    def tolist(self):
        '\n        Return a list of the values.\n\n        These are each a scalar type, which is a Python scalar\n        (for str, int, float) or a pandas scalar\n        (for Timestamp/Timedelta/Interval/Period)\n\n        Returns\n        -------\n        list\n\n        See Also\n        --------\n        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep\n            nested list of Python scalars.\n        '
        if (not isinstance(self._values, np.ndarray)):
            return list(self._values)
        return self._values.tolist()
    to_list = tolist

    def __iter__(self):
        '\n        Return an iterator of the values.\n\n        These are each a scalar type, which is a Python scalar\n        (for str, int, float) or a pandas scalar\n        (for Timestamp/Timedelta/Interval/Period)\n\n        Returns\n        -------\n        iterator\n        '
        if (not isinstance(self._values, np.ndarray)):
            return iter(self._values)
        else:
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self):
        '\n        Return if I have any nans; enables various perf speedups.\n        '
        return bool(isna(self).any())

    def isna(self):
        return isna(self._values)

    def _reduce(self, op, name, *, axis=0, skipna=True, numeric_only=None, filter_type=None, **kwds):
        '\n        Perform the reduction type operation if we can.\n        '
        func = getattr(self, name, None)
        if (func is None):
            raise TypeError(f'{type(self).__name__} cannot perform the operation {name}')
        return func(skipna=skipna, **kwds)

    def _map_values(self, mapper, na_action=None):
        "\n        An internal function that maps values using the input\n        correspondence (which can be a dict, Series, or function).\n\n        Parameters\n        ----------\n        mapper : function, dict, or Series\n            The input correspondence object\n        na_action : {None, 'ignore'}\n            If 'ignore', propagate NA values, without passing them to the\n            mapping function\n\n        Returns\n        -------\n        Union[Index, MultiIndex], inferred\n            The output of the mapping function applied to the index.\n            If the function returns a tuple with more than one element\n            a MultiIndex will be returned.\n        "
        if is_dict_like(mapper):
            if (isinstance(mapper, dict) and hasattr(mapper, '__missing__')):
                dict_with_default = mapper
                mapper = (lambda x: dict_with_default[x])
            else:
                mapper = create_series_with_explicit_dtype(mapper, dtype_if_empty=np.float64)
        if isinstance(mapper, ABCSeries):
            if is_categorical_dtype(self.dtype):
                self = cast('Categorical', self)
                return self._values.map(mapper)
            values = self._values
            indexer = mapper.index.get_indexer(values)
            new_values = algorithms.take_1d(mapper._values, indexer)
            return new_values
        if (is_extension_array_dtype(self.dtype) and hasattr(self._values, 'map')):
            values = self._values
            if (na_action is not None):
                raise NotImplementedError
            map_f = (lambda values, f: values.map(f))
        else:
            values = self.astype(object)._values
            if (na_action == 'ignore'):
                map_f = (lambda values, f: lib.map_infer_mask(values, f, isna(values).view(np.uint8)))
            elif (na_action is None):
                map_f = lib.map_infer
            else:
                msg = f"na_action must either be 'ignore' or None, {na_action} was passed"
                raise ValueError(msg)
        new_values = map_f(values, mapper)
        return new_values

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        "\n        Return a Series containing counts of unique values.\n\n        The resulting object will be in descending order so that the\n        first element is the most frequently-occurring element.\n        Excludes NA values by default.\n\n        Parameters\n        ----------\n        normalize : bool, default False\n            If True then the object returned will contain the relative\n            frequencies of the unique values.\n        sort : bool, default True\n            Sort by frequencies.\n        ascending : bool, default False\n            Sort in ascending order.\n        bins : int, optional\n            Rather than count values, group them into half-open bins,\n            a convenience for ``pd.cut``, only works with numeric data.\n        dropna : bool, default True\n            Don't include counts of NaN.\n\n        Returns\n        -------\n        Series\n\n        See Also\n        --------\n        Series.count: Number of non-NA elements in a Series.\n        DataFrame.count: Number of non-NA elements in a DataFrame.\n        DataFrame.value_counts: Equivalent method on DataFrames.\n\n        Examples\n        --------\n        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])\n        >>> index.value_counts()\n        3.0    2\n        2.0    1\n        4.0    1\n        1.0    1\n        dtype: int64\n\n        With `normalize` set to `True`, returns the relative frequency by\n        dividing all values by the sum of values.\n\n        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])\n        >>> s.value_counts(normalize=True)\n        3.0    0.4\n        2.0    0.2\n        4.0    0.2\n        1.0    0.2\n        dtype: float64\n\n        **bins**\n\n        Bins can be useful for going from a continuous variable to a\n        categorical variable; instead of counting unique\n        apparitions of values, divide the index in the specified\n        number of half-open bins.\n\n        >>> s.value_counts(bins=3)\n        (0.996, 2.0]    2\n        (2.0, 3.0]      2\n        (3.0, 4.0]      1\n        dtype: int64\n\n        **dropna**\n\n        With `dropna` set to `False` we can also see NaN index values.\n\n        >>> s.value_counts(dropna=False)\n        3.0    2\n        2.0    1\n        NaN    1\n        4.0    1\n        1.0    1\n        dtype: int64\n        "
        result = value_counts(self, sort=sort, ascending=ascending, normalize=normalize, bins=bins, dropna=dropna)
        return result

    def unique(self):
        values = self._values
        if (not isinstance(values, np.ndarray)):
            result = values.unique()
            if ((self.dtype.kind in ['m', 'M']) and isinstance(self, ABCSeries)):
                if (getattr(self.dtype, 'tz', None) is None):
                    result = np.asarray(result)
        else:
            result = unique1d(values)
        return result

    def nunique(self, dropna=True):
        "\n        Return number of unique elements in the object.\n\n        Excludes NA values by default.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include NaN in the count.\n\n        Returns\n        -------\n        int\n\n        See Also\n        --------\n        DataFrame.nunique: Method nunique for DataFrame.\n        Series.count: Count non-NA/null observations in the Series.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 3, 5, 7, 7])\n        >>> s\n        0    1\n        1    3\n        2    5\n        3    7\n        4    7\n        dtype: int64\n\n        >>> s.nunique()\n        4\n        "
        obj = (remove_na_arraylike(self) if dropna else self)
        return len(obj.unique())

    @property
    def is_unique(self):
        '\n        Return boolean if values in the object are unique.\n\n        Returns\n        -------\n        bool\n        '
        return (self.nunique(dropna=False) == len(self))

    @property
    def is_monotonic(self):
        '\n        Return boolean if values in the object are\n        monotonic_increasing.\n\n        Returns\n        -------\n        bool\n        '
        from pandas import Index
        return Index(self).is_monotonic

    @property
    def is_monotonic_increasing(self):
        '\n        Alias for is_monotonic.\n        '
        return self.is_monotonic

    @property
    def is_monotonic_decreasing(self):
        '\n        Return boolean if values in the object are\n        monotonic_decreasing.\n\n        Returns\n        -------\n        bool\n        '
        from pandas import Index
        return Index(self).is_monotonic_decreasing

    def memory_usage(self, deep=False):
        '\n        Memory usage of the values.\n\n        Parameters\n        ----------\n        deep : bool, default False\n            Introspect the data deeply, interrogate\n            `object` dtypes for system-level memory consumption.\n\n        Returns\n        -------\n        bytes used\n\n        See Also\n        --------\n        numpy.ndarray.nbytes : Total bytes consumed by the elements of the\n            array.\n\n        Notes\n        -----\n        Memory usage does not include memory consumed by elements that\n        are not components of the array if deep=False or if used on PyPy\n        '
        if hasattr(self.array, 'memory_usage'):
            return self.array.memory_usage(deep=deep)
        v = self.array.nbytes
        if (deep and is_object_dtype(self) and (not PYPY)):
            v += lib.memory_usage_of_objects(self._values)
        return v

    @doc(algorithms.factorize, values='', order='', size_hint='', sort=textwrap.dedent('            sort : bool, default False\n                Sort `uniques` and shuffle `codes` to maintain the\n                relationship.\n            '))
    def factorize(self, sort=False, na_sentinel=(- 1)):
        return algorithms.factorize(self, sort=sort, na_sentinel=na_sentinel)
    _shared_docs['searchsorted'] = "\n        Find indices where elements should be inserted to maintain order.\n\n        Find the indices into a sorted {klass} `self` such that, if the\n        corresponding elements in `value` were inserted before the indices,\n        the order of `self` would be preserved.\n\n        .. note::\n\n            The {klass} *must* be monotonically sorted, otherwise\n            wrong locations will likely be returned. Pandas does *not*\n            check this for you.\n\n        Parameters\n        ----------\n        value : array_like\n            Values to insert into `self`.\n        side : {{'left', 'right'}}, optional\n            If 'left', the index of the first suitable location found is given.\n            If 'right', return the last such index.  If there is no suitable\n            index, return either 0 or N (where N is the length of `self`).\n        sorter : 1-D array_like, optional\n            Optional array of integer indices that sort `self` into ascending\n            order. They are typically the result of ``np.argsort``.\n\n        Returns\n        -------\n        int or array of int\n            A scalar or array of insertion points with the\n            same shape as `value`.\n\n            .. versionchanged:: 0.24.0\n                If `value` is a scalar, an int is now always returned.\n                Previously, scalar inputs returned an 1-item array for\n                :class:`Series` and :class:`Categorical`.\n\n        See Also\n        --------\n        sort_values : Sort by the values along either axis.\n        numpy.searchsorted : Similar method from NumPy.\n\n        Notes\n        -----\n        Binary search is used to find the required insertion points.\n\n        Examples\n        --------\n        >>> ser = pd.Series([1, 2, 3])\n        >>> ser\n        0    1\n        1    2\n        2    3\n        dtype: int64\n\n        >>> ser.searchsorted(4)\n        3\n\n        >>> ser.searchsorted([0, 4])\n        array([0, 3])\n\n        >>> ser.searchsorted([1, 3], side='left')\n        array([0, 2])\n\n        >>> ser.searchsorted([1, 3], side='right')\n        array([1, 3])\n\n        >>> ser = pd.Series(pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000']))\n        >>> ser\n        0   2000-03-11\n        1   2000-03-12\n        2   2000-03-13\n        dtype: datetime64[ns]\n\n        >>> ser.searchsorted('3/14/2000')\n        3\n\n        >>> ser = pd.Categorical(\n        ...     ['apple', 'bread', 'bread', 'cheese', 'milk'], ordered=True\n        ... )\n        >>> ser\n        ['apple', 'bread', 'bread', 'cheese', 'milk']\n        Categories (4, object): ['apple' < 'bread' < 'cheese' < 'milk']\n\n        >>> ser.searchsorted('bread')\n        1\n\n        >>> ser.searchsorted(['bread'], side='right')\n        array([3])\n\n        If the values are not monotonically sorted, wrong locations\n        may be returned:\n\n        >>> ser = pd.Series([2, 1, 3])\n        >>> ser\n        0    2\n        1    1\n        2    3\n        dtype: int64\n\n        >>> ser.searchsorted(1)  # doctest: +SKIP\n        0  # wrong result, correct would be 1\n        "

    @doc(_shared_docs['searchsorted'], klass='Index')
    def searchsorted(self, value, side='left', sorter=None):
        return algorithms.searchsorted(self._values, value, side=side, sorter=sorter)

    def drop_duplicates(self, keep='first'):
        duplicated = self.duplicated(keep=keep)
        result = self[np.logical_not(duplicated)]
        return result

    def duplicated(self, keep='first'):
        return duplicated(self._values, keep=keep)
