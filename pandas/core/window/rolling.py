
'\nProvide a generic structure to support window functions,\nsimilar to how we have a Groupby object.\n'
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import warnings
import numpy as np
from pandas._libs.tslibs import BaseOffset, to_offset
import pandas._libs.window.aggregations as window_aggregations
from pandas._typing import ArrayLike, Axis, FrameOrSeries, FrameOrSeriesUnion
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.dtypes.common import ensure_float64, is_bool, is_integer, is_list_like, is_scalar, needs_i8_conversion
from pandas.core.dtypes.generic import ABCDataFrame, ABCDatetimeIndex, ABCPeriodIndex, ABCSeries, ABCTimedeltaIndex
from pandas.core.dtypes.missing import notna
from pandas.core.aggregation import aggregate
from pandas.core.base import DataError, SelectionMixin
from pandas.core.construction import extract_array
from pandas.core.groupby.base import GotItemMixin, ShallowMixin
from pandas.core.indexes.api import Index, MultiIndex
from pandas.core.util.numba_ import NUMBA_FUNC_CACHE, maybe_use_numba
from pandas.core.window.common import _doc_template, _shared_docs, flex_binary_moment, zsqrt
from pandas.core.window.indexers import BaseIndexer, FixedWindowIndexer, GroupbyIndexer, VariableWindowIndexer
from pandas.core.window.numba_ import generate_numba_apply_func, generate_numba_table_func
if TYPE_CHECKING:
    from pandas import DataFrame, Series
    from pandas.core.internals import Block

class BaseWindow(ShallowMixin, SelectionMixin):
    'Provides utilities for performing windowing operations.'
    _attributes = ['window', 'min_periods', 'center', 'win_type', 'axis', 'on', 'closed', 'method']
    exclusions = set()

    def __init__(self, obj, window=None, min_periods=None, center=False, win_type=None, axis=0, on=None, closed=None, method='single', **kwargs):
        self.__dict__.update(kwargs)
        self.obj = obj
        self.on = on
        self.closed = closed
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.win_type = win_type
        self.axis = (obj._get_axis_number(axis) if (axis is not None) else None)
        self.method = method
        self._win_freq_i8 = None
        if (self.on is None):
            if (self.axis == 0):
                self._on = self.obj.index
            else:
                self._on = self.obj.columns
        elif isinstance(self.on, Index):
            self._on = self.on
        elif (isinstance(self.obj, ABCDataFrame) and (self.on in self.obj.columns)):
            self._on = Index(self.obj[self.on])
        else:
            raise ValueError(f'invalid on specified as {self.on}, must be a column (of DataFrame), an Index or None')
        self.validate()

    def validate(self):
        if ((self.center is not None) and (not is_bool(self.center))):
            raise ValueError('center must be a boolean')
        if (self.min_periods is not None):
            if (not is_integer(self.min_periods)):
                raise ValueError('min_periods must be an integer')
            elif (self.min_periods < 0):
                raise ValueError('min_periods must be >= 0')
            elif (is_integer(self.window) and (self.min_periods > self.window)):
                raise ValueError(f'min_periods {self.min_periods} must be <= window {self.window}')
        if ((self.closed is not None) and (self.closed not in ['right', 'both', 'left', 'neither'])):
            raise ValueError("closed must be 'right', 'left', 'both' or 'neither'")
        if (not isinstance(self.obj, (ABCSeries, ABCDataFrame))):
            raise TypeError(f'invalid type: {type(self)}')
        if isinstance(self.window, BaseIndexer):
            get_window_bounds_signature = inspect.signature(self.window.get_window_bounds).parameters.keys()
            expected_signature = inspect.signature(BaseIndexer().get_window_bounds).parameters.keys()
            if (get_window_bounds_signature != expected_signature):
                raise ValueError(f'{type(self.window).__name__} does not implement the correct signature for get_window_bounds')
        if (self.method not in ['table', 'single']):
            raise ValueError("method must be 'table' or 'single")

    def _create_data(self, obj):
        '\n        Split data into blocks & return conformed data.\n        '
        if ((self.on is not None) and (not isinstance(self.on, Index)) and (obj.ndim == 2)):
            obj = obj.reindex(columns=obj.columns.difference([self.on]), copy=False)
        if (self.axis == 1):
            obj = obj.select_dtypes(include=['integer', 'float'], exclude=['timedelta'])
            obj = obj.astype('float64', copy=False)
            obj._mgr = obj._mgr.consolidate()
        return obj

    def _gotitem(self, key, ndim, subset=None):
        '\n        Sub-classes to define. Return a sliced object.\n\n        Parameters\n        ----------\n        key : str / list of selections\n        ndim : 1,2\n            requested ndim of result\n        subset : object, default None\n            subset to act on\n        '
        if (subset is None):
            subset = self.obj
        self = self._shallow_copy(subset)
        self._reset_cache()
        if (subset.ndim == 2):
            if ((is_scalar(key) and (key in subset)) or is_list_like(key)):
                self._selection = key
        return self

    def __getattr__(self, attr):
        if (attr in self._internal_names_set):
            return object.__getattribute__(self, attr)
        if (attr in self.obj):
            return self[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def _dir_additions(self):
        return self.obj._dir_additions()

    def _get_cov_corr_window(self, other=None):
        '\n        Return window length.\n\n        Parameters\n        ----------\n        other :\n            Used in Expanding\n\n        Returns\n        -------\n        window : int\n        '
        return self.window

    def __repr__(self):
        '\n        Provide a nice str repr of our rolling object.\n        '
        attrs_list = (f'{attr_name}={getattr(self, attr_name)}' for attr_name in self._attributes if (getattr(self, attr_name, None) is not None))
        attrs = ','.join(attrs_list)
        return f'{type(self).__name__} [{attrs}]'

    def __iter__(self):
        obj = self._create_data(self._selected_obj)
        indexer = self._get_window_indexer()
        (start, end) = indexer.get_window_bounds(num_values=len(obj), min_periods=self.min_periods, center=self.center, closed=self.closed)
        assert (len(start) == len(end))
        for (s, e) in zip(start, end):
            result = obj.iloc[slice(s, e)]
            (yield result)

    def _prep_values(self, values=None):
        'Convert input to numpy arrays for Cython routines'
        if (values is None):
            values = extract_array(self._selected_obj, extract_numpy=True)
        if needs_i8_conversion(values.dtype):
            raise NotImplementedError(f'ops for {type(self).__name__} for this dtype {values.dtype} are not implemented')
        else:
            try:
                values = ensure_float64(values)
            except (ValueError, TypeError) as err:
                raise TypeError(f'cannot handle this type -> {values.dtype}') from err
        inf = np.isinf(values)
        if inf.any():
            values = np.where(inf, np.nan, values)
        return values

    def _insert_on_column(self, result, obj):
        from pandas import Series
        if ((self.on is not None) and (not self._on.equals(obj.index))):
            name = self._on.name
            extra_col = Series(self._on, index=self.obj.index, name=name)
            if (name in result.columns):
                result[name] = extra_col
            elif (name in result.index.names):
                pass
            elif (name in self._selected_obj.columns):
                old_cols = self._selected_obj.columns
                new_cols = result.columns
                old_loc = old_cols.get_loc(name)
                overlap = new_cols.intersection(old_cols[:old_loc])
                new_loc = len(overlap)
                result.insert(new_loc, name, extra_col)
            else:
                result[name] = extra_col

    @property
    def _index_array(self):
        if needs_i8_conversion(self._on.dtype):
            return self._on.asi8
        return None

    def _get_window_indexer(self):
        '\n        Return an indexer class that will compute the window start and end bounds\n        '
        if isinstance(self.window, BaseIndexer):
            return self.window
        if (self._win_freq_i8 is not None):
            return VariableWindowIndexer(index_array=self._index_array, window_size=self._win_freq_i8)
        return FixedWindowIndexer(window_size=self.window)

    def _apply_series(self, homogeneous_func, name=None):
        '\n        Series version of _apply_blockwise\n        '
        obj = self._create_data(self._selected_obj)
        try:
            input = (obj.values if (name != 'count') else notna(obj.values).astype(int))
            values = self._prep_values(input)
        except (TypeError, NotImplementedError) as err:
            raise DataError('No numeric types to aggregate') from err
        result = homogeneous_func(values)
        return obj._constructor(result, index=obj.index, name=obj.name)

    def _apply_blockwise(self, homogeneous_func, name=None):
        '\n        Apply the given function to the DataFrame broken down into homogeneous\n        sub-frames.\n        '
        if (self._selected_obj.ndim == 1):
            return self._apply_series(homogeneous_func, name)
        obj = self._create_data(self._selected_obj)
        if (name == 'count'):
            obj = notna(obj).astype(int)
            obj._mgr = obj._mgr.consolidate()
        mgr = obj._mgr

        def hfunc(bvalues: ArrayLike) -> ArrayLike:
            values = self._prep_values(getattr(bvalues, 'T', bvalues))
            res_values = homogeneous_func(values)
            return getattr(res_values, 'T', res_values)
        new_mgr = mgr.apply(hfunc, ignore_failures=True)
        out = obj._constructor(new_mgr)
        if ((out.shape[1] == 0) and (obj.shape[1] > 0)):
            raise DataError('No numeric types to aggregate')
        elif (out.shape[1] == 0):
            return obj.astype('float64')
        self._insert_on_column(out, obj)
        return out

    def _apply_tablewise(self, homogeneous_func, name=None):
        if (self._selected_obj.ndim == 1):
            raise ValueError("method='table' not applicable for Series objects.")
        obj = self._create_data(self._selected_obj)
        values = self._prep_values(obj.to_numpy())
        values = (values.T if (self.axis == 1) else values)
        result = homogeneous_func(values)
        result = (result.T if (self.axis == 1) else result)
        out = obj._constructor(result, index=obj.index, columns=obj.columns)
        if ((out.shape[1] == 0) and (obj.shape[1] > 0)):
            raise DataError('No numeric types to aggregate')
        elif (out.shape[1] == 0):
            return obj.astype('float64')
        self._insert_on_column(out, obj)
        return out

    def _apply(self, func, name=None, numba_cache_key=None, **kwargs):
        '\n        Rolling statistical measure using supplied function.\n\n        Designed to be used with passed-in Cython array-based functions.\n\n        Parameters\n        ----------\n        func : callable function to apply\n        name : str,\n        numba_cache_key : tuple\n            caching key to be used to store a compiled numba func\n        **kwargs\n            additional arguments for rolling function and window function\n\n        Returns\n        -------\n        y : type of input\n        '
        window_indexer = self._get_window_indexer()
        min_periods = (self.min_periods if (self.min_periods is not None) else window_indexer.window_size)

        def homogeneous_func(values: np.ndarray):
            if (values.size == 0):
                return values.copy()

            def calc(x):
                (start, end) = window_indexer.get_window_bounds(num_values=len(x), min_periods=min_periods, center=self.center, closed=self.closed)
                return func(x, start, end, min_periods)
            with np.errstate(all='ignore'):
                if ((values.ndim > 1) and (self.method == 'single')):
                    result = np.apply_along_axis(calc, self.axis, values)
                else:
                    result = calc(values)
            if (numba_cache_key is not None):
                NUMBA_FUNC_CACHE[numba_cache_key] = func
            return result
        if (self.method == 'single'):
            return self._apply_blockwise(homogeneous_func, name)
        else:
            return self._apply_tablewise(homogeneous_func, name)

    def aggregate(self, func, *args, **kwargs):
        (result, how) = aggregate(self, func, *args, **kwargs)
        if (result is None):
            return self.apply(func, raw=False, args=args, kwargs=kwargs)
        return result
    agg = aggregate
    _shared_docs['sum'] = dedent('\n    Calculate %(name)s sum of given DataFrame or Series.\n\n    Parameters\n    ----------\n    *args, **kwargs\n        For compatibility with other %(name)s methods. Has no effect\n        on the computed value.\n\n    Returns\n    -------\n    Series or DataFrame\n        Same type as the input, with the same index, containing the\n        %(name)s sum.\n\n    See Also\n    --------\n    pandas.Series.sum : Reducing sum for Series.\n    pandas.DataFrame.sum : Reducing sum for DataFrame.\n\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4, 5])\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    4    5\n    dtype: int64\n\n    >>> s.rolling(3).sum()\n    0     NaN\n    1     NaN\n    2     6.0\n    3     9.0\n    4    12.0\n    dtype: float64\n\n    >>> s.expanding(3).sum()\n    0     NaN\n    1     NaN\n    2     6.0\n    3    10.0\n    4    15.0\n    dtype: float64\n\n    >>> s.rolling(3, center=True).sum()\n    0     NaN\n    1     6.0\n    2     9.0\n    3    12.0\n    4     NaN\n    dtype: float64\n\n    For DataFrame, each %(name)s sum is computed column-wise.\n\n    >>> df = pd.DataFrame({"A": s, "B": s ** 2})\n    >>> df\n       A   B\n    0  1   1\n    1  2   4\n    2  3   9\n    3  4  16\n    4  5  25\n\n    >>> df.rolling(3).sum()\n          A     B\n    0   NaN   NaN\n    1   NaN   NaN\n    2   6.0  14.0\n    3   9.0  29.0\n    4  12.0  50.0\n    ')
    _shared_docs['mean'] = dedent('\n    Calculate the %(name)s mean of the values.\n\n    Parameters\n    ----------\n    *args\n        Under Review.\n    **kwargs\n        Under Review.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned object type is determined by the caller of the %(name)s\n        calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.Series.mean : Equivalent method for Series.\n    pandas.DataFrame.mean : Equivalent method for DataFrame.\n\n    Examples\n    --------\n    The below examples will show rolling mean calculations with window sizes of\n    two and three, respectively.\n\n    >>> s = pd.Series([1, 2, 3, 4])\n    >>> s.rolling(2).mean()\n    0    NaN\n    1    1.5\n    2    2.5\n    3    3.5\n    dtype: float64\n\n    >>> s.rolling(3).mean()\n    0    NaN\n    1    NaN\n    2    2.0\n    3    3.0\n    dtype: float64\n    ')
    _shared_docs['var'] = dedent('\n    Calculate unbiased %(name)s variance.\n    %(versionadded)s\n    Normalized by N-1 by default. This can be changed using the `ddof`\n    argument.\n\n    Parameters\n    ----------\n    ddof : int, default 1\n        Delta Degrees of Freedom.  The divisor used in calculations\n        is ``N - ddof``, where ``N`` represents the number of elements.\n    *args, **kwargs\n        For NumPy compatibility. No additional arguments are used.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returns the same object type as the caller of the %(name)s calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.Series.var : Equivalent method for Series.\n    pandas.DataFrame.var : Equivalent method for DataFrame.\n    numpy.var : Equivalent method for Numpy array.\n\n    Notes\n    -----\n    The default `ddof` of 1 used in :meth:`Series.var` is different than the\n    default `ddof` of 0 in :func:`numpy.var`.\n\n    A minimum of 1 period is required for the rolling calculation.\n\n    Examples\n    --------\n    >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n    >>> s.rolling(3).var()\n    0         NaN\n    1         NaN\n    2    0.333333\n    3    1.000000\n    4    1.000000\n    5    1.333333\n    6    0.000000\n    dtype: float64\n\n    >>> s.expanding(3).var()\n    0         NaN\n    1         NaN\n    2    0.333333\n    3    0.916667\n    4    0.800000\n    5    0.700000\n    6    0.619048\n    dtype: float64\n    ')
    _shared_docs['std'] = dedent('\n    Calculate %(name)s standard deviation.\n    %(versionadded)s\n    Normalized by N-1 by default. This can be changed using the `ddof`\n    argument.\n\n    Parameters\n    ----------\n    ddof : int, default 1\n        Delta Degrees of Freedom.  The divisor used in calculations\n        is ``N - ddof``, where ``N`` represents the number of elements.\n    *args, **kwargs\n        For NumPy compatibility. No additional arguments are used.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returns the same object type as the caller of the %(name)s calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.Series.std : Equivalent method for Series.\n    pandas.DataFrame.std : Equivalent method for DataFrame.\n    numpy.std : Equivalent method for Numpy array.\n\n    Notes\n    -----\n    The default `ddof` of 1 used in Series.std is different than the default\n    `ddof` of 0 in numpy.std.\n\n    A minimum of one period is required for the rolling calculation.\n\n    Examples\n    --------\n    >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n    >>> s.rolling(3).std()\n    0         NaN\n    1         NaN\n    2    0.577350\n    3    1.000000\n    4    1.000000\n    5    1.154701\n    6    0.000000\n    dtype: float64\n\n    >>> s.expanding(3).std()\n    0         NaN\n    1         NaN\n    2    0.577350\n    3    0.957427\n    4    0.894427\n    5    0.836660\n    6    0.786796\n    dtype: float64\n    ')

def dispatch(name, *args, **kwargs):
    '\n    Dispatch to groupby apply.\n    '

    def outer(self, *args, **kwargs):

        def f(x):
            x = self._shallow_copy(x, groupby=self._groupby)
            return getattr(x, name)(*args, **kwargs)
        return self._groupby.apply(f)
    outer.__name__ = name
    return outer

class BaseWindowGroupby(GotItemMixin, BaseWindow):
    '\n    Provide the groupby windowing facilities.\n    '

    def __init__(self, obj, *args, **kwargs):
        kwargs.pop('parent', None)
        groupby = kwargs.pop('groupby', None)
        if (groupby is None):
            (groupby, obj) = (obj, obj._selected_obj)
        self._groupby = groupby
        self._groupby.mutated = True
        self._groupby.grouper.mutated = True
        super().__init__(obj, *args, **kwargs)
    corr = dispatch('corr', other=None, pairwise=None)
    cov = dispatch('cov', other=None, pairwise=None)

    def _apply(self, func, name=None, numba_cache_key=None, **kwargs):
        result = super()._apply(func, name, numba_cache_key, **kwargs)
        grouped_object_index = self.obj.index
        grouped_index_name = [*grouped_object_index.names]
        groupby_keys = [grouping.name for grouping in self._groupby.grouper._groupings]
        result_index_names = (groupby_keys + grouped_index_name)
        drop_columns = [key for key in groupby_keys if ((key not in self.obj.index.names) or (key is None))]
        if (len(drop_columns) != len(groupby_keys)):
            result = result.drop(columns=drop_columns, errors='ignore')
        codes = self._groupby.grouper.codes
        levels = self._groupby.grouper.levels
        group_indices = self._groupby.grouper.indices.values()
        if group_indices:
            indexer = np.concatenate(list(group_indices))
        else:
            indexer = np.array([], dtype=np.intp)
        codes = [c.take(indexer) for c in codes]
        if (grouped_object_index is not None):
            idx = grouped_object_index.take(indexer)
            if (not isinstance(idx, MultiIndex)):
                idx = MultiIndex.from_arrays([idx])
            codes.extend(list(idx.codes))
            levels.extend(list(idx.levels))
        result_index = MultiIndex(levels, codes, names=result_index_names, verify_integrity=False)
        result.index = result_index
        return result

    def _create_data(self, obj):
        '\n        Split data into blocks & return conformed data.\n        '
        if (not obj.empty):
            groupby_order = np.concatenate(list(self._groupby.grouper.indices.values())).astype(np.int64)
            obj = obj.take(groupby_order)
        return super()._create_data(obj)

    def _gotitem(self, key, ndim, subset=None):
        if (self.on is not None):
            self.obj = self.obj.set_index(self._on)
        return super()._gotitem(key, ndim, subset=subset)

    def _validate_monotonic(self):
        '\n        Validate that "on" is monotonic; already validated at a higher level.\n        '
        pass

class Window(BaseWindow):
    "\n    Provide rolling window calculations.\n\n    Parameters\n    ----------\n    window : int, offset, or BaseIndexer subclass\n        Size of the moving window. This is the number of observations used for\n        calculating the statistic. Each window will be a fixed size.\n\n        If its an offset then this will be the time period of each window. Each\n        window will be a variable sized based on the observations included in\n        the time-period. This is only valid for datetimelike indexes.\n\n        If a BaseIndexer subclass is passed, calculates the window boundaries\n        based on the defined ``get_window_bounds`` method. Additional rolling\n        keyword arguments, namely `min_periods`, `center`, and\n        `closed` will be passed to `get_window_bounds`.\n    min_periods : int, default None\n        Minimum number of observations in window required to have a value\n        (otherwise result is NA). For a window that is specified by an offset,\n        `min_periods` will default to 1. Otherwise, `min_periods` will default\n        to the size of the window.\n    center : bool, default False\n        Set the labels at the center of the window.\n    win_type : str, default None\n        Provide a window type. If ``None``, all points are evenly weighted.\n        See the notes below for further information.\n    on : str, optional\n        For a DataFrame, a datetime-like column or MultiIndex level on which\n        to calculate the rolling window, rather than the DataFrame's index.\n        Provided integer column is ignored and excluded from result since\n        an integer index is not used to calculate the rolling window.\n    axis : int or str, default 0\n    closed : str, default None\n        Make the interval closed on the 'right', 'left', 'both' or\n        'neither' endpoints. Defaults to 'right'.\n\n        .. versionchanged:: 1.2.0\n\n            The closed parameter with fixed windows is now supported.\n    method : str {'single', 'table'}, default 'single'\n        Execute the rolling operation per single column or row (``'single'``)\n        or over the entire object (``'table'``).\n\n        This argument is only implemented when specifying ``engine='numba'``\n        in the method call.\n\n        .. versionadded:: 1.3.0\n\n    Returns\n    -------\n    a Window or Rolling sub-classed for the particular operation\n\n    See Also\n    --------\n    expanding : Provides expanding transformations.\n    ewm : Provides exponential weighted functions.\n\n    Notes\n    -----\n    By default, the result is set to the right edge of the window. This can be\n    changed to the center of the window by setting ``center=True``.\n\n    To learn more about the offsets & frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n\n    If ``win_type=None``, all points are evenly weighted; otherwise, ``win_type``\n    can accept a string of any `scipy.signal window function\n    <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.\n\n    Certain Scipy window types require additional parameters to be passed\n    in the aggregation function. The additional parameters must match\n    the keywords specified in the Scipy window type method signature.\n    Please see the third example below on how to add the additional parameters.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})\n    >>> df\n         B\n    0  0.0\n    1  1.0\n    2  2.0\n    3  NaN\n    4  4.0\n\n    Rolling sum with a window length of 2, using the 'triang'\n    window type.\n\n    >>> df.rolling(2, win_type='triang').sum()\n         B\n    0  NaN\n    1  0.5\n    2  1.5\n    3  NaN\n    4  NaN\n\n    Rolling sum with a window length of 2, using the 'gaussian'\n    window type (note how we need to specify std).\n\n    >>> df.rolling(2, win_type='gaussian').sum(std=3)\n              B\n    0       NaN\n    1  0.986207\n    2  2.958621\n    3       NaN\n    4       NaN\n\n    Rolling sum with a window length of 2, min_periods defaults\n    to the window length.\n\n    >>> df.rolling(2).sum()\n         B\n    0  NaN\n    1  1.0\n    2  3.0\n    3  NaN\n    4  NaN\n\n    Same as above, but explicitly set the min_periods\n\n    >>> df.rolling(2, min_periods=1).sum()\n         B\n    0  0.0\n    1  1.0\n    2  3.0\n    3  2.0\n    4  4.0\n\n    Same as above, but with forward-looking windows\n\n    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)\n    >>> df.rolling(window=indexer, min_periods=1).sum()\n         B\n    0  1.0\n    1  3.0\n    2  2.0\n    3  4.0\n    4  4.0\n\n    A ragged (meaning not-a-regular frequency), time-indexed DataFrame\n\n    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]},\n    ...                   index = [pd.Timestamp('20130101 09:00:00'),\n    ...                            pd.Timestamp('20130101 09:00:02'),\n    ...                            pd.Timestamp('20130101 09:00:03'),\n    ...                            pd.Timestamp('20130101 09:00:05'),\n    ...                            pd.Timestamp('20130101 09:00:06')])\n\n    >>> df\n                           B\n    2013-01-01 09:00:00  0.0\n    2013-01-01 09:00:02  1.0\n    2013-01-01 09:00:03  2.0\n    2013-01-01 09:00:05  NaN\n    2013-01-01 09:00:06  4.0\n\n    Contrasting to an integer rolling window, this will roll a variable\n    length window corresponding to the time period.\n    The default for min_periods is 1.\n\n    >>> df.rolling('2s').sum()\n                           B\n    2013-01-01 09:00:00  0.0\n    2013-01-01 09:00:02  1.0\n    2013-01-01 09:00:03  3.0\n    2013-01-01 09:00:05  NaN\n    2013-01-01 09:00:06  4.0\n    "

    def validate(self):
        super().validate()
        if (not isinstance(self.win_type, str)):
            raise ValueError(f'Invalid win_type {self.win_type}')
        signal = import_optional_dependency('scipy.signal', extra='Scipy is required to generate window weight.')
        self._scipy_weight_generator = getattr(signal, self.win_type, None)
        if (self._scipy_weight_generator is None):
            raise ValueError(f'Invalid win_type {self.win_type}')
        if isinstance(self.window, BaseIndexer):
            raise NotImplementedError('BaseIndexer subclasses not implemented with win_types.')
        elif ((not is_integer(self.window)) or (self.window < 0)):
            raise ValueError('window must be an integer 0 or greater')
        if (self.method != 'single'):
            raise NotImplementedError("'single' is the only supported method type.")

    def _center_window(self, result, offset):
        '\n        Center the result in the window for weighted rolling aggregations.\n        '
        if (self.axis > (result.ndim - 1)):
            raise ValueError('Requested axis is larger then no. of argument dimensions')
        if (offset > 0):
            lead_indexer = ([slice(None)] * result.ndim)
            lead_indexer[self.axis] = slice(offset, None)
            result = np.copy(result[tuple(lead_indexer)])
        return result

    def _apply(self, func, name=None, numba_cache_key=None, **kwargs):
        '\n        Rolling with weights statistical measure using supplied function.\n\n        Designed to be used with passed-in Cython array-based functions.\n\n        Parameters\n        ----------\n        func : callable function to apply\n        name : str,\n        use_numba_cache : tuple\n            unused\n        **kwargs\n            additional arguments for scipy windows if necessary\n\n        Returns\n        -------\n        y : type of input\n        '
        window = self._scipy_weight_generator(self.window, **kwargs)
        offset = (((len(window) - 1) // 2) if self.center else 0)

        def homogeneous_func(values: np.ndarray):
            if (values.size == 0):
                return values.copy()

            def calc(x):
                additional_nans = np.array(([np.nan] * offset))
                x = np.concatenate((x, additional_nans))
                return func(x, window, (self.min_periods or len(window)))
            with np.errstate(all='ignore'):
                if (values.ndim > 1):
                    result = np.apply_along_axis(calc, self.axis, values)
                else:
                    result = np.asarray(calc(values))
            if self.center:
                result = self._center_window(result, offset)
            return result
        return self._apply_blockwise(homogeneous_func, name)
    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    pandas.DataFrame.aggregate : Similar DataFrame method.\n    pandas.Series.aggregate : Similar Series method.\n    ')
    _agg_examples_doc = dedent('\n    Examples\n    --------\n    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n    >>> df\n       A  B  C\n    0  1  4  7\n    1  2  5  8\n    2  3  6  9\n\n    >>> df.rolling(2, win_type="boxcar").agg("mean")\n         A    B    C\n    0  NaN  NaN  NaN\n    1  1.5  4.5  7.5\n    2  2.5  5.5  8.5\n    ')

    @doc(_shared_docs['aggregate'], see_also=_agg_see_also_doc, examples=_agg_examples_doc, klass='Series/DataFrame', axis='')
    def aggregate(self, func, *args, **kwargs):
        (result, how) = aggregate(self, func, *args, **kwargs)
        if (result is None):
            result = func(self)
        return result
    agg = aggregate

    @Substitution(name='window')
    @Appender(_shared_docs['sum'])
    def sum(self, *args, **kwargs):
        nv.validate_window_func('sum', args, kwargs)
        window_func = window_aggregations.roll_weighted_sum
        return self._apply(window_func, name='sum', **kwargs)

    @Substitution(name='window')
    @Appender(_shared_docs['mean'])
    def mean(self, *args, **kwargs):
        nv.validate_window_func('mean', args, kwargs)
        window_func = window_aggregations.roll_weighted_mean
        return self._apply(window_func, name='mean', **kwargs)

    @Substitution(name='window', versionadded='\n.. versionadded:: 1.0.0\n')
    @Appender(_shared_docs['var'])
    def var(self, ddof=1, *args, **kwargs):
        nv.validate_window_func('var', args, kwargs)
        window_func = partial(window_aggregations.roll_weighted_var, ddof=ddof)
        kwargs.pop('name', None)
        return self._apply(window_func, name='var', **kwargs)

    @Substitution(name='window', versionadded='\n.. versionadded:: 1.0.0\n')
    @Appender(_shared_docs['std'])
    def std(self, ddof=1, *args, **kwargs):
        nv.validate_window_func('std', args, kwargs)
        return zsqrt(self.var(ddof=ddof, name='std', **kwargs))

class RollingAndExpandingMixin(BaseWindow):
    _shared_docs['count'] = dedent('\n    The %(name)s count of any non-NaN observations inside the window.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned object type is determined by the caller of the %(name)s\n        calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.DataFrame.count : Count of the full DataFrame.\n\n    Examples\n    --------\n    >>> s = pd.Series([2, 3, np.nan, 10])\n    >>> s.rolling(2).count()\n    0    1.0\n    1    2.0\n    2    1.0\n    3    1.0\n    dtype: float64\n    >>> s.rolling(3).count()\n    0    1.0\n    1    2.0\n    2    2.0\n    3    2.0\n    dtype: float64\n    >>> s.rolling(4).count()\n    0    1.0\n    1    2.0\n    2    2.0\n    3    3.0\n    dtype: float64\n    ')

    def count(self):
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name='count')
    _shared_docs['apply'] = dedent("\n    Apply an arbitrary function to each %(name)s window.\n\n    Parameters\n    ----------\n    func : function\n        Must produce a single value from an ndarray input if ``raw=True``\n        or a single value from a Series if ``raw=False``. Can also accept a\n        Numba JIT function with ``engine='numba'`` specified.\n\n        .. versionchanged:: 1.0.0\n\n    raw : bool, default None\n        * ``False`` : passes each row or column as a Series to the\n          function.\n        * ``True`` : the passed function will receive ndarray\n          objects instead.\n          If you are just applying a NumPy reduction function this will\n          achieve much better performance.\n    engine : str, default None\n        * ``'cython'`` : Runs rolling apply through C-extensions from cython.\n        * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.\n          Only available when ``raw`` is set to ``True``.\n        * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``\n\n          .. versionadded:: 1.0.0\n\n    engine_kwargs : dict, default None\n        * For ``'cython'`` engine, there are no accepted ``engine_kwargs``\n        * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``\n          and ``parallel`` dictionary keys. The values must either be ``True`` or\n          ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is\n          ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be\n          applied to both the ``func`` and the ``apply`` rolling aggregation.\n\n          .. versionadded:: 1.0.0\n\n    args : tuple, default None\n        Positional arguments to be passed into func.\n    kwargs : dict, default None\n        Keyword arguments to be passed into func.\n\n    Returns\n    -------\n    Series or DataFrame\n        Return type is determined by the caller.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrame data.\n    pandas.Series.apply : Similar method for Series.\n    pandas.DataFrame.apply : Similar method for DataFrame.\n\n    Notes\n    -----\n    See :ref:`window.numba_engine` for extended documentation and performance\n    considerations for the Numba engine.\n    ")

    def apply(self, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None):
        if (args is None):
            args = ()
        if (kwargs is None):
            kwargs = {}
        if (not is_bool(raw)):
            raise ValueError('raw parameter must be `True` or `False`')
        numba_cache_key = None
        if maybe_use_numba(engine):
            if (raw is False):
                raise ValueError('raw must be `True` when using the numba engine')
            caller_name = type(self).__name__
            if (self.method == 'single'):
                apply_func = generate_numba_apply_func(args, kwargs, func, engine_kwargs, caller_name)
                numba_cache_key = (func, f'{caller_name}_apply_single')
            else:
                apply_func = generate_numba_table_func(args, kwargs, func, engine_kwargs, f'{caller_name}_apply')
                numba_cache_key = (func, f'{caller_name}_apply_table')
        elif (engine in ('cython', None)):
            if (engine_kwargs is not None):
                raise ValueError('cython engine does not accept engine_kwargs')
            apply_func = self._generate_cython_apply_func(args, kwargs, raw, func)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")
        return self._apply(apply_func, numba_cache_key=numba_cache_key)

    def _generate_cython_apply_func(self, args, kwargs, raw, function):
        from pandas import Series
        window_func = partial(window_aggregations.roll_apply, args=args, kwargs=kwargs, raw=raw, function=function)

        def apply_func(values, begin, end, min_periods, raw=raw):
            if (not raw):
                values = Series(values, index=self.obj.index)
            return window_func(values, begin, end, min_periods)
        return apply_func

    def sum(self, *args, **kwargs):
        nv.validate_window_func('sum', args, kwargs)
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name='sum', **kwargs)
    _shared_docs['max'] = dedent('\n    Calculate the %(name)s maximum.\n\n    Parameters\n    ----------\n    *args, **kwargs\n        Arguments and keyword arguments to be passed into func.\n    ')

    def max(self, *args, **kwargs):
        nv.validate_window_func('max', args, kwargs)
        window_func = window_aggregations.roll_max
        return self._apply(window_func, name='max', **kwargs)
    _shared_docs['min'] = dedent('\n    Calculate the %(name)s minimum.\n\n    Parameters\n    ----------\n    **kwargs\n        Under Review.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned object type is determined by the caller of the %(name)s\n        calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with a Series.\n    pandas.DataFrame.%(name)s : Calling object with a DataFrame.\n    pandas.Series.min : Similar method for Series.\n    pandas.DataFrame.min : Similar method for DataFrame.\n\n    Examples\n    --------\n    Performing a rolling minimum with a window size of 3.\n\n    >>> s = pd.Series([4, 3, 5, 2, 6])\n    >>> s.rolling(3).min()\n    0    NaN\n    1    NaN\n    2    3.0\n    3    2.0\n    4    2.0\n    dtype: float64\n    ')

    def min(self, *args, **kwargs):
        nv.validate_window_func('min', args, kwargs)
        window_func = window_aggregations.roll_min
        return self._apply(window_func, name='min', **kwargs)

    def mean(self, *args, **kwargs):
        nv.validate_window_func('mean', args, kwargs)
        window_func = window_aggregations.roll_mean
        return self._apply(window_func, name='mean', **kwargs)
    _shared_docs['median'] = dedent('\n    Calculate the %(name)s median.\n\n    Parameters\n    ----------\n    **kwargs\n        For compatibility with other %(name)s methods. Has no effect\n        on the computed median.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned type is the same as the original object.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.Series.median : Equivalent method for Series.\n    pandas.DataFrame.median : Equivalent method for DataFrame.\n\n    Examples\n    --------\n    Compute the rolling median of a series with a window size of 3.\n\n    >>> s = pd.Series([0, 1, 2, 3, 4])\n    >>> s.rolling(3).median()\n    0    NaN\n    1    NaN\n    2    1.0\n    3    2.0\n    4    3.0\n    dtype: float64\n    ')

    def median(self, **kwargs):
        window_func = window_aggregations.roll_median_c
        return self._apply(window_func, name='median', **kwargs)

    def std(self, ddof=1, *args, **kwargs):
        nv.validate_window_func('std', args, kwargs)
        window_func = window_aggregations.roll_var

        def zsqrt_func(values, begin, end, min_periods):
            return zsqrt(window_func(values, begin, end, min_periods, ddof=ddof))
        return self._apply(zsqrt_func, name='std', **kwargs)

    def var(self, ddof=1, *args, **kwargs):
        nv.validate_window_func('var', args, kwargs)
        window_func = partial(window_aggregations.roll_var, ddof=ddof)
        return self._apply(window_func, name='var', **kwargs)
    _shared_docs['skew'] = '\n    Unbiased %(name)s skewness.\n\n    Parameters\n    ----------\n    **kwargs\n        Keyword arguments to be passed into func.\n    '

    def skew(self, **kwargs):
        window_func = window_aggregations.roll_skew
        return self._apply(window_func, name='skew', **kwargs)
    _shared_docs['kurt'] = dedent("\n    Calculate unbiased %(name)s kurtosis.\n\n    This function uses Fisher's definition of kurtosis without bias.\n\n    Parameters\n    ----------\n    **kwargs\n        Under Review.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned object type is determined by the caller of the %(name)s\n        calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.Series.kurt : Equivalent method for Series.\n    pandas.DataFrame.kurt : Equivalent method for DataFrame.\n    scipy.stats.skew : Third moment of a probability density.\n    scipy.stats.kurtosis : Reference SciPy method.\n\n    Notes\n    -----\n    A minimum of 4 periods is required for the %(name)s calculation.\n    ")

    def sem(self, ddof=1, *args, **kwargs):
        return (self.std(*args, **kwargs) / (self.count() - ddof).pow(0.5))
    _shared_docs['sem'] = dedent('\n    Compute %(name)s standard error of mean.\n\n    Parameters\n    ----------\n\n    ddof : int, default 1\n        Delta Degrees of Freedom.  The divisor used in calculations\n        is ``N - ddof``, where ``N`` represents the number of elements.\n\n    *args, **kwargs\n        For NumPy compatibility. No additional arguments are used.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned object type is determined by the caller of the %(name)s\n        calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.Series.sem : Equivalent method for Series.\n    pandas.DataFrame.sem : Equivalent method for DataFrame.\n\n    Notes\n    -----\n    A minimum of one period is required for the rolling calculation.\n\n    Examples\n    --------\n    >>> s = pd.Series([0, 1, 2, 3])\n    >>> s.rolling(2, min_periods=1).sem()\n    0         NaN\n    1    0.707107\n    2    0.707107\n    3    0.707107\n    dtype: float64\n\n    >>> s.expanding().sem()\n    0         NaN\n    1    0.707107\n    2    0.707107\n    3    0.745356\n    dtype: float64\n    ')

    def kurt(self, **kwargs):
        window_func = window_aggregations.roll_kurt
        return self._apply(window_func, name='kurt', **kwargs)
    _shared_docs['quantile'] = dedent("\n    Calculate the %(name)s quantile.\n\n    Parameters\n    ----------\n    quantile : float\n        Quantile to compute. 0 <= quantile <= 1.\n    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}\n        This optional parameter specifies the interpolation method to use,\n        when the desired quantile lies between two data points `i` and `j`:\n\n            * linear: `i + (j - i) * fraction`, where `fraction` is the\n              fractional part of the index surrounded by `i` and `j`.\n            * lower: `i`.\n            * higher: `j`.\n            * nearest: `i` or `j` whichever is nearest.\n            * midpoint: (`i` + `j`) / 2.\n    **kwargs\n        For compatibility with other %(name)s methods. Has no effect on\n        the result.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned object type is determined by the caller of the %(name)s\n        calculation.\n\n    See Also\n    --------\n    pandas.Series.quantile : Computes value at the given quantile over all data\n        in Series.\n    pandas.DataFrame.quantile : Computes values at the given quantile over\n        requested axis in DataFrame.\n\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n    >>> s.rolling(2).quantile(.4, interpolation='lower')\n    0    NaN\n    1    1.0\n    2    2.0\n    3    3.0\n    dtype: float64\n\n    >>> s.rolling(2).quantile(.4, interpolation='midpoint')\n    0    NaN\n    1    1.5\n    2    2.5\n    3    3.5\n    dtype: float64\n    ")

    def quantile(self, quantile, interpolation='linear', **kwargs):
        if (quantile == 1.0):
            window_func = window_aggregations.roll_max
        elif (quantile == 0.0):
            window_func = window_aggregations.roll_min
        else:
            window_func = partial(window_aggregations.roll_quantile, quantile=quantile, interpolation=interpolation)
        return self._apply(window_func, name='quantile', **kwargs)
    _shared_docs['cov'] = '\n        Calculate the %(name)s sample covariance.\n\n        Parameters\n        ----------\n        other : Series, DataFrame, or ndarray, optional\n            If not supplied then will default to self and produce pairwise\n            output.\n        pairwise : bool, default None\n            If False then only matching columns between self and other will be\n            used and the output will be a DataFrame.\n            If True then all pairwise combinations will be calculated and the\n            output will be a MultiIndexed DataFrame in the case of DataFrame\n            inputs. In the case of missing elements, only complete pairwise\n            observations will be used.\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n        **kwargs\n            Keyword arguments to be passed into func.\n    '

    def cov(self, other=None, pairwise=None, ddof=1, **kwargs):
        if (other is None):
            other = self._selected_obj
            pairwise = (True if (pairwise is None) else pairwise)
        other = self._shallow_copy(other)
        window = self._get_cov_corr_window(other)

        def _get_cov(X, Y):
            X = X.astype('float64')
            Y = Y.astype('float64')
            mean = (lambda x: x.rolling(window, self.min_periods, center=self.center).mean(**kwargs))
            count = (X + Y).rolling(window=window, min_periods=0, center=self.center).count(**kwargs)
            bias_adj = (count / (count - ddof))
            return ((mean((X * Y)) - (mean(X) * mean(Y))) * bias_adj)
        return flex_binary_moment(self._selected_obj, other._selected_obj, _get_cov, pairwise=bool(pairwise))
    _shared_docs['corr'] = dedent('\n    Calculate %(name)s correlation.\n\n    Parameters\n    ----------\n    other : Series, DataFrame, or ndarray, optional\n        If not supplied then will default to self.\n    pairwise : bool, default None\n        Calculate pairwise combinations of columns within a\n        DataFrame. If `other` is not specified, defaults to `True`,\n        otherwise defaults to `False`.\n        Not relevant for :class:`~pandas.Series`.\n    **kwargs\n        Unused.\n\n    Returns\n    -------\n    Series or DataFrame\n        Returned object type is determined by the caller of the\n        %(name)s calculation.\n\n    See Also\n    --------\n    pandas.Series.%(name)s : Calling object with Series data.\n    pandas.DataFrame.%(name)s : Calling object with DataFrames.\n    pandas.Series.corr : Equivalent method for Series.\n    pandas.DataFrame.corr : Equivalent method for DataFrame.\n    cov : Similar method to calculate covariance.\n    numpy.corrcoef : NumPy Pearson\'s correlation calculation.\n\n    Notes\n    -----\n    This function uses Pearson\'s definition of correlation\n    (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).\n\n    When `other` is not specified, the output will be self correlation (e.g.\n    all 1\'s), except for :class:`~pandas.DataFrame` inputs with `pairwise`\n    set to `True`.\n\n    Function will return ``NaN`` for correlations of equal valued sequences;\n    this is the result of a 0/0 division error.\n\n    When `pairwise` is set to `False`, only matching columns between `self` and\n    `other` will be used.\n\n    When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame\n    with the original index on the first level, and the `other` DataFrame\n    columns on the second level.\n\n    In the case of missing elements, only complete pairwise observations\n    will be used.\n\n    Examples\n    --------\n    The below example shows a rolling calculation with a window size of\n    four matching the equivalent function call using :meth:`numpy.corrcoef`.\n\n    >>> v1 = [3, 3, 3, 5, 8]\n    >>> v2 = [3, 4, 4, 4, 8]\n    >>> # numpy returns a 2X2 array, the correlation coefficient\n    >>> # is the number at entry [0][1]\n    >>> print(f"{np.corrcoef(v1[:-1], v2[:-1])[0][1]:.6f}")\n    0.333333\n    >>> print(f"{np.corrcoef(v1[1:], v2[1:])[0][1]:.6f}")\n    0.916949\n    >>> s1 = pd.Series(v1)\n    >>> s2 = pd.Series(v2)\n    >>> s1.rolling(4).corr(s2)\n    0         NaN\n    1         NaN\n    2         NaN\n    3    0.333333\n    4    0.916949\n    dtype: float64\n\n    The below example shows a similar rolling calculation on a\n    DataFrame using the pairwise option.\n\n    >>> matrix = np.array([[51., 35.], [49., 30.], [47., 32.],    [46., 31.], [50., 36.]])\n    >>> print(np.corrcoef(matrix[:-1,0], matrix[:-1,1]).round(7))\n    [[1.         0.6263001]\n     [0.6263001  1.       ]]\n    >>> print(np.corrcoef(matrix[1:,0], matrix[1:,1]).round(7))\n    [[1.         0.5553681]\n     [0.5553681  1.        ]]\n    >>> df = pd.DataFrame(matrix, columns=[\'X\',\'Y\'])\n    >>> df\n          X     Y\n    0  51.0  35.0\n    1  49.0  30.0\n    2  47.0  32.0\n    3  46.0  31.0\n    4  50.0  36.0\n    >>> df.rolling(4).corr(pairwise=True)\n                X         Y\n    0 X       NaN       NaN\n      Y       NaN       NaN\n    1 X       NaN       NaN\n      Y       NaN       NaN\n    2 X       NaN       NaN\n      Y       NaN       NaN\n    3 X  1.000000  0.626300\n      Y  0.626300  1.000000\n    4 X  1.000000  0.555368\n      Y  0.555368  1.000000\n    ')

    def corr(self, other=None, pairwise=None, **kwargs):
        if (other is None):
            other = self._selected_obj
            pairwise = (True if (pairwise is None) else pairwise)
        other = self._shallow_copy(other)
        window = self._get_cov_corr_window(other)

        def _get_corr(a, b):
            a = a.rolling(window=window, min_periods=self.min_periods, center=self.center)
            b = b.rolling(window=window, min_periods=self.min_periods, center=self.center)
            return (a.cov(b, **kwargs) / ((a.var(**kwargs) * b.var(**kwargs)) ** 0.5))
        return flex_binary_moment(self._selected_obj, other._selected_obj, _get_corr, pairwise=bool(pairwise))

class Rolling(RollingAndExpandingMixin):

    def validate(self):
        super().validate()
        if ((self.obj.empty or isinstance(self._on, (ABCDatetimeIndex, ABCTimedeltaIndex, ABCPeriodIndex))) and isinstance(self.window, (str, BaseOffset, timedelta))):
            self._validate_monotonic()
            if self.center:
                raise NotImplementedError('center is not implemented for datetimelike and offset based windows')
            try:
                freq = to_offset(self.window)
            except (TypeError, ValueError) as err:
                raise ValueError(f'passed window {self.window} is not compatible with a datetimelike index') from err
            if isinstance(self._on, ABCPeriodIndex):
                self._win_freq_i8 = (freq.nanos / (self._on.freq.nanos / self._on.freq.n))
            else:
                self._win_freq_i8 = freq.nanos
            if (self.min_periods is None):
                self.min_periods = 1
        elif isinstance(self.window, BaseIndexer):
            return
        elif ((not is_integer(self.window)) or (self.window < 0)):
            raise ValueError('window must be an integer 0 or greater')

    def _validate_monotonic(self):
        '\n        Validate monotonic (increasing or decreasing).\n        '
        if (not (self._on.is_monotonic_increasing or self._on.is_monotonic_decreasing)):
            self._raise_monotonic_error()

    def _raise_monotonic_error(self):
        formatted = self.on
        if (self.on is None):
            formatted = 'index'
        raise ValueError(f'{formatted} must be monotonic')
    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    pandas.Series.rolling : Calling object with Series data.\n    pandas.DataFrame.rolling : Calling object with DataFrame data.\n    ')
    _agg_examples_doc = dedent('\n    Examples\n    --------\n    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n    >>> df\n       A  B  C\n    0  1  4  7\n    1  2  5  8\n    2  3  6  9\n\n    >>> df.rolling(2).sum()\n         A     B     C\n    0  NaN   NaN   NaN\n    1  3.0   9.0  15.0\n    2  5.0  11.0  17.0\n\n    >>> df.rolling(2).agg({"A": "sum", "B": "min"})\n         A    B\n    0  NaN  NaN\n    1  3.0  4.0\n    2  5.0  5.0\n    ')

    @doc(_shared_docs['aggregate'], see_also=_agg_see_also_doc, examples=_agg_examples_doc, klass='Series/Dataframe', axis='')
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)
    agg = aggregate

    @Substitution(name='rolling')
    @Appender(_shared_docs['count'])
    def count(self):
        if (self.min_periods is None):
            warnings.warn('min_periods=None will default to the size of window consistent with other methods in a future version. Specify min_periods=0 instead.', FutureWarning)
            self.min_periods = 0
        return super().count()

    @Substitution(name='rolling')
    @Appender(_shared_docs['apply'])
    def apply(self, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None):
        return super().apply(func, raw=raw, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    @Substitution(name='rolling')
    @Appender(_shared_docs['sum'])
    def sum(self, *args, **kwargs):
        nv.validate_rolling_func('sum', args, kwargs)
        return super().sum(*args, **kwargs)

    @Substitution(name='rolling', func_name='max')
    @Appender(_doc_template)
    @Appender(_shared_docs['max'])
    def max(self, *args, **kwargs):
        nv.validate_rolling_func('max', args, kwargs)
        return super().max(*args, **kwargs)

    @Substitution(name='rolling')
    @Appender(_shared_docs['min'])
    def min(self, *args, **kwargs):
        nv.validate_rolling_func('min', args, kwargs)
        return super().min(*args, **kwargs)

    @Substitution(name='rolling')
    @Appender(_shared_docs['mean'])
    def mean(self, *args, **kwargs):
        nv.validate_rolling_func('mean', args, kwargs)
        return super().mean(*args, **kwargs)

    @Substitution(name='rolling')
    @Appender(_shared_docs['median'])
    def median(self, **kwargs):
        return super().median(**kwargs)

    @Substitution(name='rolling', versionadded='')
    @Appender(_shared_docs['std'])
    def std(self, ddof=1, *args, **kwargs):
        nv.validate_rolling_func('std', args, kwargs)
        return super().std(ddof=ddof, **kwargs)

    @Substitution(name='rolling', versionadded='')
    @Appender(_shared_docs['var'])
    def var(self, ddof=1, *args, **kwargs):
        nv.validate_rolling_func('var', args, kwargs)
        return super().var(ddof=ddof, **kwargs)

    @Substitution(name='rolling', func_name='skew')
    @Appender(_doc_template)
    @Appender(_shared_docs['skew'])
    def skew(self, **kwargs):
        return super().skew(**kwargs)

    @Substitution(name='rolling')
    @Appender(_shared_docs['sem'])
    def sem(self, ddof=1, *args, **kwargs):
        return (self.std(*args, **kwargs) / (self.count() - ddof).pow(0.5))
    _agg_doc = dedent('\n    Examples\n    --------\n\n    The example below will show a rolling calculation with a window size of\n    four matching the equivalent function call using `scipy.stats`.\n\n    >>> arr = [1, 2, 3, 4, 999]\n    >>> import scipy.stats\n    >>> print(f"{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}")\n    -1.200000\n    >>> print(f"{scipy.stats.kurtosis(arr[1:], bias=False):.6f}")\n    3.999946\n    >>> s = pd.Series(arr)\n    >>> s.rolling(4).kurt()\n    0         NaN\n    1         NaN\n    2         NaN\n    3   -1.200000\n    4    3.999946\n    dtype: float64\n    ')

    @Appender(_agg_doc)
    @Substitution(name='rolling')
    @Appender(_shared_docs['kurt'])
    def kurt(self, **kwargs):
        return super().kurt(**kwargs)

    @Substitution(name='rolling')
    @Appender(_shared_docs['quantile'])
    def quantile(self, quantile, interpolation='linear', **kwargs):
        return super().quantile(quantile=quantile, interpolation=interpolation, **kwargs)

    @Substitution(name='rolling', func_name='cov')
    @Appender(_doc_template)
    @Appender(_shared_docs['cov'])
    def cov(self, other=None, pairwise=None, ddof=1, **kwargs):
        return super().cov(other=other, pairwise=pairwise, ddof=ddof, **kwargs)

    @Substitution(name='rolling')
    @Appender(_shared_docs['corr'])
    def corr(self, other=None, pairwise=None, **kwargs):
        return super().corr(other=other, pairwise=pairwise, **kwargs)
Rolling.__doc__ = Window.__doc__

class RollingGroupby(BaseWindowGroupby, Rolling):
    '\n    Provide a rolling groupby implementation.\n    '

    @property
    def _constructor(self):
        return Rolling

    def _get_window_indexer(self):
        '\n        Return an indexer class that will compute the window start and end bounds\n\n        Returns\n        -------\n        GroupbyIndexer\n        '
        rolling_indexer: Type[BaseIndexer]
        indexer_kwargs: Optional[Dict[(str, Any)]] = None
        index_array = self._index_array
        window = self.window
        if isinstance(self.window, BaseIndexer):
            rolling_indexer = type(self.window)
            indexer_kwargs = self.window.__dict__
            assert isinstance(indexer_kwargs, dict)
            indexer_kwargs.pop('index_array', None)
            window = 0
        elif (self._win_freq_i8 is not None):
            rolling_indexer = VariableWindowIndexer
            window = self._win_freq_i8
        else:
            rolling_indexer = FixedWindowIndexer
            index_array = None
        window_indexer = GroupbyIndexer(index_array=index_array, window_size=window, groupby_indicies=self._groupby.indices, window_indexer=rolling_indexer, indexer_kwargs=indexer_kwargs)
        return window_indexer

    def _validate_monotonic(self):
        '\n        Validate that on is monotonic;\n        in this case we have to check only for nans, because\n        monotonicity was already validated at a higher level.\n        '
        if self._on.hasnans:
            self._raise_monotonic_error()
