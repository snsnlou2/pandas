
'\nProvide the groupby split-apply-combine paradigm. Define the GroupBy\nclass providing the base-class of operations.\n\nThe SeriesGroupBy and DataFrameGroupBy sub-class\n(defined in pandas.core.groupby.generic)\nexpose these user-facing objects to provide specific functionality.\n'
from contextlib import contextmanager
import datetime
from functools import partial, wraps
import inspect
from textwrap import dedent
import types
from typing import Callable, Dict, FrozenSet, Generic, Hashable, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union
import numpy as np
from pandas._config.config import option_context
from pandas._libs import Timestamp, lib
import pandas._libs.groupby as libgroupby
from pandas._typing import F, FrameOrSeries, FrameOrSeriesUnion, IndexLabel, Label, Scalar, final
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution, cache_readonly, doc
from pandas.core.dtypes.cast import maybe_downcast_numeric
from pandas.core.dtypes.common import ensure_float, is_bool_dtype, is_datetime64_dtype, is_extension_array_dtype, is_integer_dtype, is_numeric_dtype, is_object_dtype, is_scalar, is_timedelta64_dtype
from pandas.core.dtypes.missing import isna, notna
from pandas.core import nanops
import pandas.core.algorithms as algorithms
from pandas.core.arrays import Categorical, DatetimeArray
from pandas.core.base import DataError, PandasObject, SelectionMixin
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import base, numba_, ops
from pandas.core.indexes.api import CategoricalIndex, Index, MultiIndex
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import NUMBA_FUNC_CACHE
_common_see_also = '\n        See Also\n        --------\n        Series.%(name)s : Apply a function %(name)s to a Series.\n        DataFrame.%(name)s : Apply a function %(name)s\n            to each row or column of a DataFrame.\n'
_apply_docs = {'template': '\n    Apply function `func` group-wise and combine the results together.\n\n    The function passed to `apply` must take a {input} as its first\n    argument and return a DataFrame, Series or scalar. `apply` will\n    then take care of combining the results back together into a single\n    dataframe or series. `apply` is therefore a highly flexible\n    grouping method.\n\n    While `apply` is a very flexible method, its downside is that\n    using it can be quite a bit slower than using more specific methods\n    like `agg` or `transform`. Pandas offers a wide range of method that will\n    be much faster than using `apply` for their specific purposes, so try to\n    use them before reaching for `apply`.\n\n    Parameters\n    ----------\n    func : callable\n        A callable that takes a {input} as its first argument, and\n        returns a dataframe, a series or a scalar. In addition the\n        callable may take positional and keyword arguments.\n    args, kwargs : tuple and dict\n        Optional positional and keyword arguments to pass to `func`.\n\n    Returns\n    -------\n    applied : Series or DataFrame\n\n    See Also\n    --------\n    pipe : Apply function to the full GroupBy object instead of to each\n        group.\n    aggregate : Apply aggregate function to the GroupBy object.\n    transform : Apply function column-by-column to the GroupBy object.\n    Series.apply : Apply a function to a Series.\n    DataFrame.apply : Apply a function to each row or column of a DataFrame.\n    ', 'dataframe_examples': "\n    >>> df = pd.DataFrame({'A': 'a a b'.split(),\n                           'B': [1,2,3],\n                           'C': [4,6, 5]})\n    >>> g = df.groupby('A')\n\n    Notice that ``g`` has two groups, ``a`` and ``b``.\n    Calling `apply` in various ways, we can get different grouping results:\n\n    Example 1: below the function passed to `apply` takes a DataFrame as\n    its argument and returns a DataFrame. `apply` combines the result for\n    each group together into a new DataFrame:\n\n    >>> g[['B', 'C']].apply(lambda x: x / x.sum())\n              B    C\n    0  0.333333  0.4\n    1  0.666667  0.6\n    2  1.000000  1.0\n\n    Example 2: The function passed to `apply` takes a DataFrame as\n    its argument and returns a Series.  `apply` combines the result for\n    each group together into a new DataFrame:\n\n    >>> g[['B', 'C']].apply(lambda x: x.max() - x.min())\n       B  C\n    A\n    a  1  2\n    b  0  0\n\n    Example 3: The function passed to `apply` takes a DataFrame as\n    its argument and returns a scalar. `apply` combines the result for\n    each group together into a Series, including setting the index as\n    appropriate:\n\n    >>> g.apply(lambda x: x.C.max() - x.B.min())\n    A\n    a    5\n    b    2\n    dtype: int64\n    ", 'series_examples': "\n    >>> s = pd.Series([0, 1, 2], index='a a b'.split())\n    >>> g = s.groupby(s.index)\n\n    From ``s`` above we can see that ``g`` has two groups, ``a`` and ``b``.\n    Calling `apply` in various ways, we can get different grouping results:\n\n    Example 1: The function passed to `apply` takes a Series as\n    its argument and returns a Series.  `apply` combines the result for\n    each group together into a new Series:\n\n    >>> g.apply(lambda x:  x*2 if x.name == 'b' else x/2)\n    0    0.0\n    1    0.5\n    2    4.0\n    dtype: float64\n\n    Example 2: The function passed to `apply` takes a Series as\n    its argument and returns a scalar. `apply` combines the result for\n    each group together into a Series, including setting the index as\n    appropriate:\n\n    >>> g.apply(lambda x: x.max() - x.min())\n    a    1\n    b    0\n    dtype: int64\n\n    Notes\n    -----\n    In the current implementation `apply` calls `func` twice on the\n    first group to decide whether it can take a fast or slow code\n    path. This can lead to unexpected behavior if `func` has\n    side-effects, as they will take effect twice for the first\n    group.\n\n    Examples\n    --------\n    {examples}\n    "}
_groupby_agg_method_template = '\nCompute {fname} of group values.\n\nParameters\n----------\nnumeric_only : bool, default {no}\n    Include only float, int, boolean columns. If None, will attempt to use\n    everything, then use only numeric data.\nmin_count : int, default {mc}\n    The required number of valid values to perform the operation. If fewer\n    than ``min_count`` non-NA values are present the result will be NA.\n\nReturns\n-------\nSeries or DataFrame\n    Computed {fname} of values within each group.\n'
_pipe_template = "\nApply a function `func` with arguments to this %(klass)s object and return\nthe function's result.\n\nUse `.pipe` when you want to improve readability by chaining together\nfunctions that expect Series, DataFrames, GroupBy or Resampler objects.\nInstead of writing\n\n>>> h(g(f(df.groupby('group')), arg1=a), arg2=b, arg3=c)  # doctest: +SKIP\n\nYou can write\n\n>>> (df.groupby('group')\n...    .pipe(f)\n...    .pipe(g, arg1=a)\n...    .pipe(h, arg2=b, arg3=c))  # doctest: +SKIP\n\nwhich is much more readable.\n\nParameters\n----------\nfunc : callable or tuple of (callable, str)\n    Function to apply to this %(klass)s object or, alternatively,\n    a `(callable, data_keyword)` tuple where `data_keyword` is a\n    string indicating the keyword of `callable` that expects the\n    %(klass)s object.\nargs : iterable, optional\n       Positional arguments passed into `func`.\nkwargs : dict, optional\n         A dictionary of keyword arguments passed into `func`.\n\nReturns\n-------\nobject : the return type of `func`.\n\nSee Also\n--------\nSeries.pipe : Apply a function with arguments to a series.\nDataFrame.pipe: Apply a function with arguments to a dataframe.\napply : Apply function to each group instead of to the\n    full %(klass)s object.\n\nNotes\n-----\nSee more `here\n<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_\n\nExamples\n--------\n%(examples)s\n"
_transform_template = '\nCall function producing a like-indexed %(klass)s on each group and\nreturn a %(klass)s having the same indexes as the original object\nfilled with the transformed values\n\nParameters\n----------\nf : function\n    Function to apply to each group.\n\n    Can also accept a Numba JIT function with\n    ``engine=\'numba\'`` specified.\n\n    If the ``\'numba\'`` engine is chosen, the function must be\n    a user defined function with ``values`` and ``index`` as the\n    first and second arguments respectively in the function signature.\n    Each group\'s index will be passed to the user defined function\n    and optionally available for use.\n\n    .. versionchanged:: 1.1.0\n*args\n    Positional arguments to pass to func.\nengine : str, default None\n    * ``\'cython\'`` : Runs the function through C-extensions from cython.\n    * ``\'numba\'`` : Runs the function through JIT compiled code from numba.\n    * ``None`` : Defaults to ``\'cython\'`` or globally setting ``compute.use_numba``\n\n    .. versionadded:: 1.1.0\nengine_kwargs : dict, default None\n    * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``\n    * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``\n      and ``parallel`` dictionary keys. The values must either be ``True`` or\n      ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is\n      ``{\'nopython\': True, \'nogil\': False, \'parallel\': False}`` and will be\n      applied to the function\n\n    .. versionadded:: 1.1.0\n**kwargs\n    Keyword arguments to be passed into func.\n\nReturns\n-------\n%(klass)s\n\nSee Also\n--------\n%(klass)s.groupby.apply : Apply function func group-wise\n    and combine the results together.\n%(klass)s.groupby.aggregate : Aggregate using one or more\n    operations over the specified axis.\n%(klass)s.transform : Transforms the Series on each group\n    based on the given function.\n\nNotes\n-----\nEach group is endowed the attribute \'name\' in case you need to know\nwhich group you are working on.\n\nThe current implementation imposes three requirements on f:\n\n* f must return a value that either has the same shape as the input\n  subframe or can be broadcast to the shape of the input subframe.\n  For example, if `f` returns a scalar it will be broadcast to have the\n  same shape as the input subframe.\n* if this is a DataFrame, f must support application column-by-column\n  in the subframe. If f also supports application to the entire subframe,\n  then a fast path is used starting from the second chunk.\n* f must not mutate groups. Mutation is not supported and may\n  produce unexpected results.\n\nWhen using ``engine=\'numba\'``, there will be no "fall back" behavior internally.\nThe group data and group index will be passed as numpy arrays to the JITed\nuser defined function, and no alternative execution attempts will be tried.\n\nExamples\n--------\n\n>>> df = pd.DataFrame({\'A\' : [\'foo\', \'bar\', \'foo\', \'bar\',\n...                           \'foo\', \'bar\'],\n...                    \'B\' : [\'one\', \'one\', \'two\', \'three\',\n...                           \'two\', \'two\'],\n...                    \'C\' : [1, 5, 5, 2, 5, 5],\n...                    \'D\' : [2.0, 5., 8., 1., 2., 9.]})\n>>> grouped = df.groupby(\'A\')\n>>> grouped.transform(lambda x: (x - x.mean()) / x.std())\n          C         D\n0 -1.154701 -0.577350\n1  0.577350  0.000000\n2  0.577350  1.154701\n3 -1.154701 -1.000000\n4  0.577350 -0.577350\n5  0.577350  1.000000\n\nBroadcast result of the transformation\n\n>>> grouped.transform(lambda x: x.max() - x.min())\n   C    D\n0  4  6.0\n1  3  8.0\n2  4  6.0\n3  3  8.0\n4  4  6.0\n5  3  8.0\n'
_agg_template = '\nAggregate using one or more operations over the specified axis.\n\nParameters\n----------\nfunc : function, str, list or dict\n    Function to use for aggregating the data. If a function, must either\n    work when passed a {klass} or when passed to {klass}.apply.\n\n    Accepted combinations are:\n\n    - function\n    - string function name\n    - list of functions and/or function names, e.g. ``[np.sum, \'mean\']``\n    - dict of axis labels -> functions, function names or list of such.\n\n    Can also accept a Numba JIT function with\n    ``engine=\'numba\'`` specified. Only passing a single function is supported\n    with this engine.\n\n    If the ``\'numba\'`` engine is chosen, the function must be\n    a user defined function with ``values`` and ``index`` as the\n    first and second arguments respectively in the function signature.\n    Each group\'s index will be passed to the user defined function\n    and optionally available for use.\n\n    .. versionchanged:: 1.1.0\n*args\n    Positional arguments to pass to func.\nengine : str, default None\n    * ``\'cython\'`` : Runs the function through C-extensions from cython.\n    * ``\'numba\'`` : Runs the function through JIT compiled code from numba.\n    * ``None`` : Defaults to ``\'cython\'`` or globally setting ``compute.use_numba``\n\n    .. versionadded:: 1.1.0\nengine_kwargs : dict, default None\n    * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``\n    * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``\n      and ``parallel`` dictionary keys. The values must either be ``True`` or\n      ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is\n      ``{{\'nopython\': True, \'nogil\': False, \'parallel\': False}}`` and will be\n      applied to the function\n\n    .. versionadded:: 1.1.0\n**kwargs\n    Keyword arguments to be passed into func.\n\nReturns\n-------\n{klass}\n\nSee Also\n--------\n{klass}.groupby.apply : Apply function func group-wise\n    and combine the results together.\n{klass}.groupby.transform : Aggregate using one or more\n    operations over the specified axis.\n{klass}.aggregate : Transforms the Series on each group\n    based on the given function.\n\nNotes\n-----\nWhen using ``engine=\'numba\'``, there will be no "fall back" behavior internally.\nThe group data and group index will be passed as numpy arrays to the JITed\nuser defined function, and no alternative execution attempts will be tried.\n{examples}\n'

@final
class GroupByPlot(PandasObject):
    '\n    Class implementing the .plot attribute for groupby objects.\n    '

    def __init__(self, groupby):
        self._groupby = groupby

    def __call__(self, *args, **kwargs):

        def f(self):
            return self.plot(*args, **kwargs)
        f.__name__ = 'plot'
        return self._groupby.apply(f)

    def __getattr__(self, name):

        def attr(*args, **kwargs):

            def f(self):
                return getattr(self.plot, name)(*args, **kwargs)
            return self._groupby.apply(f)
        return attr

@contextmanager
def group_selection_context(groupby):
    '\n    Set / reset the group_selection_context.\n    '
    groupby._set_group_selection()
    try:
        (yield groupby)
    finally:
        groupby._reset_group_selection()
_KeysArgType = Union[(Hashable, List[Hashable], Callable[([Hashable], Hashable)], List[Callable[([Hashable], Hashable)]], Mapping[(Hashable, Hashable)])]

class BaseGroupBy(PandasObject, SelectionMixin, Generic[FrameOrSeries]):
    _group_selection = None
    _apply_allowlist = frozenset()
    _hidden_attrs = (PandasObject._hidden_attrs | {'as_index', 'axis', 'dropna', 'exclusions', 'grouper', 'group_keys', 'keys', 'level', 'mutated', 'obj', 'observed', 'sort', 'squeeze'})

    def __init__(self, obj, keys=None, axis=0, level=None, grouper=None, exclusions=None, selection=None, as_index=True, sort=True, group_keys=True, squeeze=False, observed=False, mutated=False, dropna=True):
        self._selection = selection
        assert isinstance(obj, NDFrame), type(obj)
        self.level = level
        if (not as_index):
            if (not isinstance(obj, DataFrame)):
                raise TypeError('as_index=False only valid with DataFrame')
            if (axis != 0):
                raise ValueError('as_index=False only valid for axis=0')
        self.as_index = as_index
        self.keys = keys
        self.sort = sort
        self.group_keys = group_keys
        self.squeeze = squeeze
        self.observed = observed
        self.mutated = mutated
        self.dropna = dropna
        if (grouper is None):
            from pandas.core.groupby.grouper import get_grouper
            (grouper, exclusions, obj) = get_grouper(obj, keys, axis=axis, level=level, sort=sort, observed=observed, mutated=self.mutated, dropna=self.dropna)
        self.obj = obj
        self.axis = obj._get_axis_number(axis)
        self.grouper = grouper
        self.exclusions = (exclusions or set())

    @final
    def __len__(self):
        return len(self.groups)

    @final
    def __repr__(self):
        return object.__repr__(self)

    def _assure_grouper(self):
        '\n        We create the grouper on instantiation sub-classes may have a\n        different policy.\n        '
        pass

    @final
    @property
    def groups(self):
        '\n        Dict {group name -> group labels}.\n        '
        self._assure_grouper()
        return self.grouper.groups

    @final
    @property
    def ngroups(self):
        self._assure_grouper()
        return self.grouper.ngroups

    @final
    @property
    def indices(self):
        '\n        Dict {group name -> group indices}.\n        '
        self._assure_grouper()
        return self.grouper.indices

    @final
    def _get_indices(self, names):
        '\n        Safe get multiple indices, translate keys for\n        datelike to underlying repr.\n        '

        def get_converter(s):
            if isinstance(s, datetime.datetime):
                return (lambda key: Timestamp(key))
            elif isinstance(s, np.datetime64):
                return (lambda key: Timestamp(key).asm8)
            else:
                return (lambda key: key)
        if (len(names) == 0):
            return []
        if (len(self.indices) > 0):
            index_sample = next(iter(self.indices))
        else:
            index_sample = None
        name_sample = names[0]
        if isinstance(index_sample, tuple):
            if (not isinstance(name_sample, tuple)):
                msg = 'must supply a tuple to get_group with multiple grouping keys'
                raise ValueError(msg)
            if (not (len(name_sample) == len(index_sample))):
                try:
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    msg = 'must supply a same-length tuple to get_group with multiple grouping keys'
                    raise ValueError(msg) from err
            converters = [get_converter(s) for s in index_sample]
            names = (tuple((f(n) for (f, n) in zip(converters, name))) for name in names)
        else:
            converter = get_converter(index_sample)
            names = (converter(name) for name in names)
        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name):
        '\n        Safe get index, translate keys for datelike to underlying repr.\n        '
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self):
        if ((self._selection is None) or isinstance(self.obj, Series)):
            if (self._group_selection is not None):
                return self.obj[self._group_selection]
            return self.obj
        else:
            return self.obj[self._selection]

    @final
    def _reset_group_selection(self):
        '\n        Clear group based selection.\n\n        Used for methods needing to return info on each group regardless of\n        whether a group selection was previously set.\n        '
        if (self._group_selection is not None):
            self._group_selection = None
            self._reset_cache('_selected_obj')

    @final
    def _set_group_selection(self):
        '\n        Create group based selection.\n\n        Used when selection is not passed directly but instead via a grouper.\n\n        NOTE: this should be paired with a call to _reset_group_selection\n        '
        grp = self.grouper
        if (not (self.as_index and (getattr(grp, 'groupings', None) is not None) and (self.obj.ndim > 1) and (self._group_selection is None))):
            return
        groupers = [g.name for g in grp.groupings if ((g.level is None) and g.in_axis)]
        if len(groupers):
            ax = self.obj._info_axis
            self._group_selection = ax.difference(Index(groupers), sort=False).tolist()
            self._reset_cache('_selected_obj')

    @final
    def _set_result_index_ordered(self, result):
        if self.grouper.is_monotonic:
            result.set_axis(self.obj._get_axis(self.axis), axis=self.axis, inplace=True)
            return result
        original_positions = Index(np.concatenate(self._get_indices(self.grouper.result_index)))
        result.set_axis(original_positions, axis=self.axis, inplace=True)
        result = result.sort_index(axis=self.axis)
        dropped_rows = (len(result.index) < len(self.obj.index))
        if dropped_rows:
            sorted_indexer = result.index
            result.index = self._selected_obj.index[sorted_indexer]
        else:
            result.set_axis(self.obj._get_axis(self.axis), axis=self.axis, inplace=True)
        return result

    @final
    def _dir_additions(self):
        return (self.obj._dir_additions() | self._apply_allowlist)

    def __getattr__(self, attr):
        if (attr in self._internal_names_set):
            return object.__getattribute__(self, attr)
        if (attr in self.obj):
            return self[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @Substitution(klass='GroupBy', examples=dedent("        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})\n        >>> df\n           A  B\n        0  a  1\n        1  b  2\n        2  a  3\n        3  b  4\n\n        To get the difference between each groups maximum and minimum value in one\n        pass, you can do\n\n        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())\n           B\n        A\n        a  2\n        b  2"))
    @Appender(_pipe_template)
    def pipe(self, func, *args, **kwargs):
        return com.pipe(self, func, *args, **kwargs)
    plot = property(GroupByPlot)

    @final
    def _make_wrapper(self, name):
        assert (name in self._apply_allowlist)
        with group_selection_context(self):
            f = getattr(self._obj_with_exclusions, name)
            if (not isinstance(f, types.MethodType)):
                return self.apply((lambda self: getattr(self, name)))
        f = getattr(type(self._obj_with_exclusions), name)
        sig = inspect.signature(f)

        def wrapper(*args, **kwargs):
            if ('axis' in sig.parameters):
                if (kwargs.get('axis', None) is None):
                    kwargs['axis'] = self.axis

            def curried(x):
                return f(x, *args, **kwargs)
            curried.__name__ = name
            if (name in base.plotting_methods):
                return self.apply(curried)
            return self._python_apply_general(curried, self._obj_with_exclusions)
        wrapper.__name__ = name
        return wrapper

    @final
    def get_group(self, name, obj=None):
        '\n        Construct DataFrame from group with provided name.\n\n        Parameters\n        ----------\n        name : object\n            The name of the group to get as a DataFrame.\n        obj : DataFrame, default None\n            The DataFrame to take the DataFrame out of.  If\n            it is None, the object groupby was called on will\n            be used.\n\n        Returns\n        -------\n        group : same type as obj\n        '
        if (obj is None):
            obj = self._selected_obj
        inds = self._get_index(name)
        if (not len(inds)):
            raise KeyError(name)
        return obj._take_with_is_copy(inds, axis=self.axis)

    def __iter__(self):
        '\n        Groupby iterator.\n\n        Returns\n        -------\n        Generator yielding sequence of (name, subsetted object)\n        for each group\n        '
        return self.grouper.get_iterator(self.obj, axis=self.axis)

    @Appender(_apply_docs['template'].format(input='dataframe', examples=_apply_docs['dataframe_examples']))
    def apply(self, func, *args, **kwargs):
        func = self._is_builtin_func(func)
        if (args or kwargs):
            if callable(func):

                @wraps(func)
                def f(g):
                    with np.errstate(all='ignore'):
                        return func(g, *args, **kwargs)
            elif hasattr(nanops, ('nan' + func)):
                f = getattr(nanops, ('nan' + func))
            else:
                raise ValueError('func must be a callable if args or kwargs are supplied')
        else:
            f = func
        with option_context('mode.chained_assignment', None):
            try:
                result = self._python_apply_general(f, self._selected_obj)
            except TypeError:
                with group_selection_context(self):
                    return self._python_apply_general(f, self._selected_obj)
        return result

    @final
    def _python_apply_general(self, f, data):
        '\n        Apply function f in python space\n\n        Parameters\n        ----------\n        f : callable\n            Function to apply\n        data : Series or DataFrame\n            Data to apply f to\n\n        Returns\n        -------\n        Series or DataFrame\n            data after applying f\n        '
        (keys, values, mutated) = self.grouper.apply(f, data, self.axis)
        return self._wrap_applied_output(keys, values, not_indexed_same=(mutated or self.mutated))

    def _iterate_slices(self):
        raise AbstractMethodError(self)

    def transform(self, func, *args, **kwargs):
        raise AbstractMethodError(self)

    @final
    def _cumcount_array(self, ascending=True):
        '\n        Parameters\n        ----------\n        ascending : bool, default True\n            If False, number in reverse, from length of group - 1 to 0.\n\n        Notes\n        -----\n        this is currently implementing sort=False\n        (though the default is sort=True) for groupby in general\n        '
        (ids, _, ngroups) = self.grouper.group_info
        sorter = get_group_index_sorter(ids, ngroups)
        (ids, count) = (ids[sorter], len(ids))
        if (count == 0):
            return np.empty(0, dtype=np.int64)
        run = np.r_[(True, (ids[:(- 1)] != ids[1:]))]
        rep = np.diff(np.r_[(np.nonzero(run)[0], count)])
        out = (~ run).cumsum()
        if ascending:
            out -= np.repeat(out[run], rep)
        else:
            out = (np.repeat(out[np.r_[(run[1:], True)]], rep) - out)
        rev = np.empty(count, dtype=np.intp)
        rev[sorter] = np.arange(count, dtype=np.intp)
        return out[rev].astype(np.int64, copy=False)

    @final
    def _cython_transform(self, how, numeric_only=True, axis=0, **kwargs):
        output: Dict[(base.OutputKey, np.ndarray)] = {}
        for (idx, obj) in enumerate(self._iterate_slices()):
            name = obj.name
            is_numeric = is_numeric_dtype(obj.dtype)
            if (numeric_only and (not is_numeric)):
                continue
            try:
                result = self.grouper._cython_operation('transform', obj._values, how, axis, **kwargs)
            except NotImplementedError:
                continue
            key = base.OutputKey(label=name, position=idx)
            output[key] = result
        if (not output):
            raise DataError('No numeric types to aggregate')
        return self._wrap_transformed_output(output)

    def _wrap_aggregated_output(self, output, index):
        raise AbstractMethodError(self)

    def _wrap_transformed_output(self, output):
        raise AbstractMethodError(self)

    def _wrap_applied_output(self, keys, values, not_indexed_same=False):
        raise AbstractMethodError(self)

    @final
    def _agg_general(self, numeric_only=True, min_count=(- 1), *, alias, npfunc):
        with group_selection_context(self):
            result = None
            try:
                result = self._cython_agg_general(how=alias, alt=npfunc, numeric_only=numeric_only, min_count=min_count)
            except DataError:
                pass
            except NotImplementedError as err:
                if (('function is not implemented for this dtype' in str(err)) or ('category dtype not supported' in str(err))):
                    pass
                else:
                    raise
            if (result is None):
                result = self.aggregate((lambda x: npfunc(x, axis=self.axis)))
            return result.__finalize__(self.obj, method='groupby')

    def _cython_agg_general(self, how, alt=None, numeric_only=True, min_count=(- 1)):
        output: Dict[(base.OutputKey, Union[(np.ndarray, DatetimeArray)])] = {}
        idx = 0
        for obj in self._iterate_slices():
            name = obj.name
            is_numeric = is_numeric_dtype(obj.dtype)
            if (numeric_only and (not is_numeric)):
                continue
            result = self.grouper._cython_operation('aggregate', obj._values, how, axis=0, min_count=min_count)
            if (how == 'ohlc'):
                agg_names = ['open', 'high', 'low', 'close']
                assert (len(agg_names) == result.shape[1])
                for (result_column, result_name) in zip(result.T, agg_names):
                    key = base.OutputKey(label=result_name, position=idx)
                    output[key] = result_column
                    idx += 1
            else:
                assert (result.ndim == 1)
                key = base.OutputKey(label=name, position=idx)
                output[key] = result
                idx += 1
        if (not output):
            raise DataError('No numeric types to aggregate')
        return self._wrap_aggregated_output(output, index=self.grouper.result_index)

    @final
    def _transform_with_numba(self, data, func, *args, engine_kwargs=None, **kwargs):
        '\n        Perform groupby transform routine with the numba engine.\n\n        This routine mimics the data splitting routine of the DataSplitter class\n        to generate the indices of each group in the sorted data and then passes the\n        data and indices into a Numba jitted function.\n        '
        if (not callable(func)):
            raise NotImplementedError('Numba engine can only be used with a single function.')
        group_keys = self.grouper._get_group_keys()
        (labels, _, n_groups) = self.grouper.group_info
        sorted_index = get_group_index_sorter(labels, n_groups)
        sorted_labels = algorithms.take_nd(labels, sorted_index, allow_fill=False)
        sorted_data = data.take(sorted_index, axis=self.axis).to_numpy()
        (starts, ends) = lib.generate_slices(sorted_labels, n_groups)
        numba_transform_func = numba_.generate_numba_transform_func(tuple(args), kwargs, func, engine_kwargs)
        result = numba_transform_func(sorted_data, sorted_index, starts, ends, len(group_keys), len(data.columns))
        cache_key = (func, 'groupby_transform')
        if (cache_key not in NUMBA_FUNC_CACHE):
            NUMBA_FUNC_CACHE[cache_key] = numba_transform_func
        return result.take(np.argsort(sorted_index), axis=0)

    @final
    def _aggregate_with_numba(self, data, func, *args, engine_kwargs=None, **kwargs):
        '\n        Perform groupby aggregation routine with the numba engine.\n\n        This routine mimics the data splitting routine of the DataSplitter class\n        to generate the indices of each group in the sorted data and then passes the\n        data and indices into a Numba jitted function.\n        '
        if (not callable(func)):
            raise NotImplementedError('Numba engine can only be used with a single function.')
        group_keys = self.grouper._get_group_keys()
        (labels, _, n_groups) = self.grouper.group_info
        sorted_index = get_group_index_sorter(labels, n_groups)
        sorted_labels = algorithms.take_nd(labels, sorted_index, allow_fill=False)
        sorted_data = data.take(sorted_index, axis=self.axis).to_numpy()
        (starts, ends) = lib.generate_slices(sorted_labels, n_groups)
        numba_agg_func = numba_.generate_numba_agg_func(tuple(args), kwargs, func, engine_kwargs)
        result = numba_agg_func(sorted_data, sorted_index, starts, ends, len(group_keys), len(data.columns))
        cache_key = (func, 'groupby_agg')
        if (cache_key not in NUMBA_FUNC_CACHE):
            NUMBA_FUNC_CACHE[cache_key] = numba_agg_func
        if (self.grouper.nkeys > 1):
            index = MultiIndex.from_tuples(group_keys, names=self.grouper.names)
        else:
            index = Index(group_keys, name=self.grouper.names[0])
        return (result, index)

    @final
    def _python_agg_general(self, func, *args, **kwargs):
        func = self._is_builtin_func(func)
        f = (lambda x: func(x, *args, **kwargs))
        output: Dict[(base.OutputKey, np.ndarray)] = {}
        for (idx, obj) in enumerate(self._iterate_slices()):
            name = obj.name
            if (self.grouper.ngroups == 0):
                continue
            try:
                (result, counts) = self.grouper.agg_series(obj, f)
            except TypeError:
                continue
            assert (result is not None)
            key = base.OutputKey(label=name, position=idx)
            if is_numeric_dtype(obj.dtype):
                result = maybe_downcast_numeric(result, obj.dtype)
            if self.grouper._filter_empty_groups:
                mask = (counts.ravel() > 0)
                values = result
                if is_numeric_dtype(values.dtype):
                    values = ensure_float(values)
                    result = maybe_downcast_numeric(values[mask], result.dtype)
            output[key] = result
        if (not output):
            return self._python_apply_general(f, self._selected_obj)
        return self._wrap_aggregated_output(output, index=self.grouper.result_index)

    @final
    def _concat_objects(self, keys, values, not_indexed_same=False):
        from pandas.core.reshape.concat import concat

        def reset_identity(values):
            for v in com.not_none(*values):
                ax = v._get_axis(self.axis)
                ax._reset_identity()
            return values
        if (not not_indexed_same):
            result = concat(values, axis=self.axis)
            ax = self.filter((lambda x: True)).axes[self.axis]
            if (ax.has_duplicates and (not result.axes[self.axis].equals(ax))):
                (indexer, _) = result.index.get_indexer_non_unique(ax._values)
                indexer = algorithms.unique1d(indexer)
                result = result.take(indexer, axis=self.axis)
            else:
                result = result.reindex(ax, axis=self.axis, copy=False)
        elif self.group_keys:
            values = reset_identity(values)
            if self.as_index:
                group_keys = keys
                group_levels = self.grouper.levels
                group_names = self.grouper.names
                result = concat(values, axis=self.axis, keys=group_keys, levels=group_levels, names=group_names, sort=False)
            else:
                keys = list(range(len(values)))
                result = concat(values, axis=self.axis, keys=keys)
        else:
            values = reset_identity(values)
            result = concat(values, axis=self.axis)
        if (isinstance(result, Series) and (self._selection_name is not None)):
            result.name = self._selection_name
        return result

    @final
    def _apply_filter(self, indices, dropna):
        if (len(indices) == 0):
            indices = np.array([], dtype='int64')
        else:
            indices = np.sort(np.concatenate(indices))
        if dropna:
            filtered = self._selected_obj.take(indices, axis=self.axis)
        else:
            mask = np.empty(len(self._selected_obj.index), dtype=bool)
            mask.fill(False)
            mask[indices.astype(int)] = True
            mask = np.tile(mask, (list(self._selected_obj.shape[1:]) + [1])).T
            filtered = self._selected_obj.where(mask)
        return filtered
OutputFrameOrSeries = TypeVar('OutputFrameOrSeries', bound=NDFrame)

class GroupBy(BaseGroupBy[FrameOrSeries]):
    '\n    Class for grouping and aggregating relational data.\n\n    See aggregate, transform, and apply functions on this object.\n\n    It\'s easiest to use obj.groupby(...) to use GroupBy, but you can also do:\n\n    ::\n\n        grouped = groupby(obj, ...)\n\n    Parameters\n    ----------\n    obj : pandas object\n    axis : int, default 0\n    level : int, default None\n        Level of MultiIndex\n    groupings : list of Grouping objects\n        Most users should ignore this\n    exclusions : array-like, optional\n        List of columns to exclude\n    name : str\n        Most users should ignore this\n\n    Returns\n    -------\n    **Attributes**\n    groups : dict\n        {group name -> group labels}\n    len(grouped) : int\n        Number of groups\n\n    Notes\n    -----\n    After grouping, see aggregate, apply, and transform functions. Here are\n    some other brief notes about usage. When grouping by multiple groups, the\n    result index will be a MultiIndex (hierarchical) by default.\n\n    Iteration produces (key, group) tuples, i.e. chunking the data by group. So\n    you can write code like:\n\n    ::\n\n        grouped = obj.groupby(keys, axis=axis)\n        for key, group in grouped:\n            # do something with the data\n\n    Function calls on GroupBy, if not specially implemented, "dispatch" to the\n    grouped data. So if you group a DataFrame and wish to invoke the std()\n    method on each group, you can simply do:\n\n    ::\n\n        df.groupby(mapper).std()\n\n    rather than\n\n    ::\n\n        df.groupby(mapper).aggregate(np.std)\n\n    You can pass arguments to these "wrapped" functions, too.\n\n    See the online documentation for full exposition on these topics and much\n    more\n    '

    @final
    @property
    def _obj_1d_constructor(self):
        if isinstance(self.obj, DataFrame):
            return self.obj._constructor_sliced
        assert isinstance(self.obj, Series)
        return self.obj._constructor

    @final
    def _bool_agg(self, val_test, skipna):
        '\n        Shared func to call any / all Cython GroupBy implementations.\n        '

        def objs_to_bool(vals: np.ndarray) -> Tuple[(np.ndarray, Type)]:
            if is_object_dtype(vals):
                vals = np.array([bool(x) for x in vals])
            else:
                vals = vals.astype(bool)
            return (vals.view(np.uint8), bool)

        def result_to_bool(result: np.ndarray, inference: Type) -> np.ndarray:
            return result.astype(inference, copy=False)
        return self._get_cythonized_result('group_any_all', aggregate=True, numeric_only=False, cython_dtype=np.dtype(np.uint8), needs_values=True, needs_mask=True, pre_processing=objs_to_bool, post_processing=result_to_bool, val_test=val_test, skipna=skipna)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def any(self, skipna=True):
        '\n        Return True if any value in the group is truthful, else False.\n\n        Parameters\n        ----------\n        skipna : bool, default True\n            Flag to ignore nan values during truth testing.\n\n        Returns\n        -------\n        Series or DataFrame\n            DataFrame or Series of boolean values, where a value is True if any element\n            is True within its respective group, False otherwise.\n        '
        return self._bool_agg('any', skipna)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def all(self, skipna=True):
        '\n        Return True if all values in the group are truthful, else False.\n\n        Parameters\n        ----------\n        skipna : bool, default True\n            Flag to ignore nan values during truth testing.\n\n        Returns\n        -------\n        Series or DataFrame\n            DataFrame or Series of boolean values, where a value is True if all elements\n            are True within its respective group, False otherwise.\n        '
        return self._bool_agg('all', skipna)

    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def count(self):
        '\n        Compute count of group, excluding missing values.\n\n        Returns\n        -------\n        Series or DataFrame\n            Count of values within each group.\n        '
        raise NotImplementedError

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def mean(self, numeric_only=True):
        "\n        Compute mean of groups, excluding missing values.\n\n        Parameters\n        ----------\n        numeric_only : bool, default True\n            Include only float, int, boolean columns. If None, will attempt to use\n            everything, then use only numeric data.\n\n        Returns\n        -------\n        pandas.Series or pandas.DataFrame\n        %(see_also)s\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],\n        ...                    'B': [np.nan, 2, 3, 4, 5],\n        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])\n\n        Groupby one column and return the mean of the remaining columns in\n        each group.\n\n        >>> df.groupby('A').mean()\n             B         C\n        A\n        1  3.0  1.333333\n        2  4.0  1.500000\n\n        Groupby two columns and return the mean of the remaining column.\n\n        >>> df.groupby(['A', 'B']).mean()\n               C\n        A B\n        1 2.0  2\n          4.0  1\n        2 3.0  1\n          5.0  2\n\n        Groupby one column and return the mean of only particular column in\n        the group.\n\n        >>> df.groupby('A')['B'].mean()\n        A\n        1    3.0\n        2    4.0\n        Name: B, dtype: float64\n        "
        return self._cython_agg_general('mean', alt=(lambda x, axis: Series(x).mean(numeric_only=numeric_only)), numeric_only=numeric_only)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def median(self, numeric_only=True):
        '\n        Compute median of groups, excluding missing values.\n\n        For multiple groupings, the result index will be a MultiIndex\n\n        Parameters\n        ----------\n        numeric_only : bool, default True\n            Include only float, int, boolean columns. If None, will attempt to use\n            everything, then use only numeric data.\n\n        Returns\n        -------\n        Series or DataFrame\n            Median of values within each group.\n        '
        return self._cython_agg_general('median', alt=(lambda x, axis: Series(x).median(axis=axis, numeric_only=numeric_only)), numeric_only=numeric_only)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def std(self, ddof=1):
        '\n        Compute standard deviation of groups, excluding missing values.\n\n        For multiple groupings, the result index will be a MultiIndex.\n\n        Parameters\n        ----------\n        ddof : int, default 1\n            Degrees of freedom.\n\n        Returns\n        -------\n        Series or DataFrame\n            Standard deviation of values within each group.\n        '
        return self._get_cythonized_result('group_var_float64', aggregate=True, needs_counts=True, needs_values=True, needs_2d=True, cython_dtype=np.dtype(np.float64), post_processing=(lambda vals, inference: np.sqrt(vals)), ddof=ddof)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def var(self, ddof=1):
        '\n        Compute variance of groups, excluding missing values.\n\n        For multiple groupings, the result index will be a MultiIndex.\n\n        Parameters\n        ----------\n        ddof : int, default 1\n            Degrees of freedom.\n\n        Returns\n        -------\n        Series or DataFrame\n            Variance of values within each group.\n        '
        if (ddof == 1):
            return self._cython_agg_general('var', alt=(lambda x, axis: Series(x).var(ddof=ddof)))
        else:
            func = (lambda x: x.var(ddof=ddof))
            with group_selection_context(self):
                return self._python_agg_general(func)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def sem(self, ddof=1):
        '\n        Compute standard error of the mean of groups, excluding missing values.\n\n        For multiple groupings, the result index will be a MultiIndex.\n\n        Parameters\n        ----------\n        ddof : int, default 1\n            Degrees of freedom.\n\n        Returns\n        -------\n        Series or DataFrame\n            Standard error of the mean of values within each group.\n        '
        result = self.std(ddof=ddof)
        if (result.ndim == 1):
            result /= np.sqrt(self.count())
        else:
            cols = result.columns.difference(self.exclusions).unique()
            counts = self.count()
            result_ilocs = result.columns.get_indexer_for(cols)
            count_ilocs = counts.columns.get_indexer_for(cols)
            result.iloc[:, result_ilocs] /= np.sqrt(counts.iloc[:, count_ilocs])
        return result

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def size(self):
        '\n        Compute group sizes.\n\n        Returns\n        -------\n        DataFrame or Series\n            Number of rows in each group as a Series if as_index is True\n            or a DataFrame if as_index is False.\n        '
        result = self.grouper.size()
        if issubclass(self.obj._constructor, Series):
            result = self._obj_1d_constructor(result, name=self.obj.name)
        else:
            result = self._obj_1d_constructor(result)
        if (not self.as_index):
            result = result.rename('size').reset_index()
        return self._reindex_output(result, fill_value=0)

    @final
    @doc(_groupby_agg_method_template, fname='sum', no=True, mc=0)
    def sum(self, numeric_only=True, min_count=0):
        with com.temp_setattr(self, 'observed', True):
            result = self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='add', npfunc=np.sum)
        return self._reindex_output(result, fill_value=0)

    @final
    @doc(_groupby_agg_method_template, fname='prod', no=True, mc=0)
    def prod(self, numeric_only=True, min_count=0):
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='prod', npfunc=np.prod)

    @final
    @doc(_groupby_agg_method_template, fname='min', no=False, mc=(- 1))
    def min(self, numeric_only=False, min_count=(- 1)):
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='min', npfunc=np.min)

    @final
    @doc(_groupby_agg_method_template, fname='max', no=False, mc=(- 1))
    def max(self, numeric_only=False, min_count=(- 1)):
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='max', npfunc=np.max)

    @final
    @doc(_groupby_agg_method_template, fname='first', no=False, mc=(- 1))
    def first(self, numeric_only=False, min_count=(- 1)):

        def first_compat(obj: FrameOrSeries, axis: int=0):

            def first(x: Series):
                "Helper function for first item that isn't NA."
                arr = x.array[notna(x.array)]
                if (not len(arr)):
                    return np.nan
                return arr[0]
            if isinstance(obj, DataFrame):
                return obj.apply(first, axis=axis)
            elif isinstance(obj, Series):
                return first(obj)
            else:
                raise TypeError(type(obj))
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='first', npfunc=first_compat)

    @final
    @doc(_groupby_agg_method_template, fname='last', no=False, mc=(- 1))
    def last(self, numeric_only=False, min_count=(- 1)):

        def last_compat(obj: FrameOrSeries, axis: int=0):

            def last(x: Series):
                "Helper function for last item that isn't NA."
                arr = x.array[notna(x.array)]
                if (not len(arr)):
                    return np.nan
                return arr[(- 1)]
            if isinstance(obj, DataFrame):
                return obj.apply(last, axis=axis)
            elif isinstance(obj, Series):
                return last(obj)
            else:
                raise TypeError(type(obj))
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='last', npfunc=last_compat)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def ohlc(self):
        '\n        Compute open, high, low and close values of a group, excluding missing values.\n\n        For multiple groupings, the result index will be a MultiIndex\n\n        Returns\n        -------\n        DataFrame\n            Open, high, low and close values within each group.\n        '
        return self._apply_to_column_groupbys((lambda x: x._cython_agg_general('ohlc')))

    @final
    @doc(DataFrame.describe)
    def describe(self, **kwargs):
        with group_selection_context(self):
            result = self.apply((lambda x: x.describe(**kwargs)))
            if (self.axis == 1):
                return result.T
            return result.unstack()

    @final
    def resample(self, rule, *args, **kwargs):
        '\n        Provide resampling when using a TimeGrouper.\n\n        Given a grouper, the function resamples it according to a string\n        "string" -> "frequency".\n\n        See the :ref:`frequency aliases <timeseries.offset_aliases>`\n        documentation for more details.\n\n        Parameters\n        ----------\n        rule : str or DateOffset\n            The offset string or object representing target grouper conversion.\n        *args, **kwargs\n            Possible arguments are `how`, `fill_method`, `limit`, `kind` and\n            `on`, and other arguments of `TimeGrouper`.\n\n        Returns\n        -------\n        Grouper\n            Return a new grouper with our resampler appended.\n\n        See Also\n        --------\n        Grouper : Specify a frequency to resample with when\n            grouping by a key.\n        DatetimeIndex.resample : Frequency conversion and resampling of\n            time series.\n\n        Examples\n        --------\n        >>> idx = pd.date_range(\'1/1/2000\', periods=4, freq=\'T\')\n        >>> df = pd.DataFrame(data=4 * [range(2)],\n        ...                   index=idx,\n        ...                   columns=[\'a\', \'b\'])\n        >>> df.iloc[2, 0] = 5\n        >>> df\n                            a  b\n        2000-01-01 00:00:00  0  1\n        2000-01-01 00:01:00  0  1\n        2000-01-01 00:02:00  5  1\n        2000-01-01 00:03:00  0  1\n\n        Downsample the DataFrame into 3 minute bins and sum the values of\n        the timestamps falling into a bin.\n\n        >>> df.groupby(\'a\').resample(\'3T\').sum()\n                                 a  b\n        a\n        0   2000-01-01 00:00:00  0  2\n            2000-01-01 00:03:00  0  1\n        5   2000-01-01 00:00:00  5  1\n\n        Upsample the series into 30 second bins.\n\n        >>> df.groupby(\'a\').resample(\'30S\').sum()\n                            a  b\n        a\n        0   2000-01-01 00:00:00  0  1\n            2000-01-01 00:00:30  0  0\n            2000-01-01 00:01:00  0  1\n            2000-01-01 00:01:30  0  0\n            2000-01-01 00:02:00  0  0\n            2000-01-01 00:02:30  0  0\n            2000-01-01 00:03:00  0  1\n        5   2000-01-01 00:02:00  5  1\n\n        Resample by month. Values are assigned to the month of the period.\n\n        >>> df.groupby(\'a\').resample(\'M\').sum()\n                    a  b\n        a\n        0   2000-01-31  0  3\n        5   2000-01-31  5  1\n\n        Downsample the series into 3 minute bins as above, but close the right\n        side of the bin interval.\n\n        >>> df.groupby(\'a\').resample(\'3T\', closed=\'right\').sum()\n                                 a  b\n        a\n        0   1999-12-31 23:57:00  0  1\n            2000-01-01 00:00:00  0  2\n        5   2000-01-01 00:00:00  5  1\n\n        Downsample the series into 3 minute bins and close the right side of\n        the bin interval, but label each bin using the right edge instead of\n        the left.\n\n        >>> df.groupby(\'a\').resample(\'3T\', closed=\'right\', label=\'right\').sum()\n                                 a  b\n        a\n        0   2000-01-01 00:00:00  0  1\n            2000-01-01 00:03:00  0  2\n        5   2000-01-01 00:03:00  5  1\n        '
        from pandas.core.resample import get_resampler_for_grouping
        return get_resampler_for_grouping(self, rule, *args, **kwargs)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def rolling(self, *args, **kwargs):
        '\n        Return a rolling grouper, providing rolling functionality per group.\n        '
        from pandas.core.window import RollingGroupby
        return RollingGroupby(self, *args, **kwargs)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def expanding(self, *args, **kwargs):
        '\n        Return an expanding grouper, providing expanding\n        functionality per group.\n        '
        from pandas.core.window import ExpandingGroupby
        return ExpandingGroupby(self, *args, **kwargs)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def ewm(self, *args, **kwargs):
        '\n        Return an ewm grouper, providing ewm functionality per group.\n        '
        from pandas.core.window import ExponentialMovingWindowGroupby
        return ExponentialMovingWindowGroupby(self, *args, **kwargs)

    @final
    def _fill(self, direction, limit=None):
        "\n        Shared function for `pad` and `backfill` to call Cython method.\n\n        Parameters\n        ----------\n        direction : {'ffill', 'bfill'}\n            Direction passed to underlying Cython function. `bfill` will cause\n            values to be filled backwards. `ffill` and any other values will\n            default to a forward fill\n        limit : int, default None\n            Maximum number of consecutive values to fill. If `None`, this\n            method will convert to -1 prior to passing to Cython\n\n        Returns\n        -------\n        `Series` or `DataFrame` with filled values\n\n        See Also\n        --------\n        pad : Returns Series with minimum number of char in object.\n        backfill : Backward fill the missing values in the dataset.\n        "
        if (limit is None):
            limit = (- 1)
        return self._get_cythonized_result('group_fillna_indexer', numeric_only=False, needs_mask=True, cython_dtype=np.dtype(np.int64), result_is_index=True, direction=direction, limit=limit, dropna=self.dropna)

    @final
    @Substitution(name='groupby')
    def pad(self, limit=None):
        '\n        Forward fill the values.\n\n        Parameters\n        ----------\n        limit : int, optional\n            Limit of how many values to fill.\n\n        Returns\n        -------\n        Series or DataFrame\n            Object with missing values filled.\n\n        See Also\n        --------\n        Series.pad: Returns Series with minimum number of char in object.\n        DataFrame.pad: Object with missing values filled or None if inplace=True.\n        Series.fillna: Fill NaN values of a Series.\n        DataFrame.fillna: Fill NaN values of a DataFrame.\n        '
        return self._fill('ffill', limit=limit)
    ffill = pad

    @final
    @Substitution(name='groupby')
    def backfill(self, limit=None):
        '\n        Backward fill the values.\n\n        Parameters\n        ----------\n        limit : int, optional\n            Limit of how many values to fill.\n\n        Returns\n        -------\n        Series or DataFrame\n            Object with missing values filled.\n\n        See Also\n        --------\n        Series.backfill :  Backward fill the missing values in the dataset.\n        DataFrame.backfill:  Backward fill the missing values in the dataset.\n        Series.fillna: Fill NaN values of a Series.\n        DataFrame.fillna: Fill NaN values of a DataFrame.\n        '
        return self._fill('bfill', limit=limit)
    bfill = backfill

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def nth(self, n, dropna=None):
        "\n        Take the nth row from each group if n is an int, or a subset of rows\n        if n is a list of ints.\n\n        If dropna, will take the nth non-null row, dropna is either\n        'all' or 'any'; this is equivalent to calling dropna(how=dropna)\n        before the groupby.\n\n        Parameters\n        ----------\n        n : int or list of ints\n            A single nth value for the row or a list of nth values.\n        dropna : None or str, optional\n            Apply the specified dropna operation before counting which row is\n            the nth row. Needs to be None, 'any' or 'all'.\n\n        Returns\n        -------\n        Series or DataFrame\n            N-th value within each group.\n        %(see_also)s\n        Examples\n        --------\n\n        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],\n        ...                    'B': [np.nan, 2, 3, 4, 5]}, columns=['A', 'B'])\n        >>> g = df.groupby('A')\n        >>> g.nth(0)\n             B\n        A\n        1  NaN\n        2  3.0\n        >>> g.nth(1)\n             B\n        A\n        1  2.0\n        2  5.0\n        >>> g.nth(-1)\n             B\n        A\n        1  4.0\n        2  5.0\n        >>> g.nth([0, 1])\n             B\n        A\n        1  NaN\n        1  2.0\n        2  3.0\n        2  5.0\n\n        Specifying `dropna` allows count ignoring ``NaN``\n\n        >>> g.nth(0, dropna='any')\n             B\n        A\n        1  2.0\n        2  3.0\n\n        NaNs denote group exhausted when using dropna\n\n        >>> g.nth(3, dropna='any')\n            B\n        A\n        1 NaN\n        2 NaN\n\n        Specifying `as_index=False` in `groupby` keeps the original index.\n\n        >>> df.groupby('A', as_index=False).nth(1)\n           A    B\n        1  1  2.0\n        4  2  5.0\n        "
        valid_containers = (set, list, tuple)
        if (not isinstance(n, (valid_containers, int))):
            raise TypeError('n needs to be an int or a list/set/tuple of ints')
        if (not dropna):
            if isinstance(n, int):
                nth_values = [n]
            elif isinstance(n, valid_containers):
                nth_values = list(set(n))
            nth_array = np.array(nth_values, dtype=np.intp)
            with group_selection_context(self):
                mask_left = np.in1d(self._cumcount_array(), nth_array)
                mask_right = np.in1d((self._cumcount_array(ascending=False) + 1), (- nth_array))
                mask = (mask_left | mask_right)
                (ids, _, _) = self.grouper.group_info
                mask = (mask & (ids != (- 1)))
                out = self._selected_obj[mask]
                if (not self.as_index):
                    return out
                result_index = self.grouper.result_index
                out.index = result_index[ids[mask]]
                if ((not self.observed) and isinstance(result_index, CategoricalIndex)):
                    out = out.reindex(result_index)
                out = self._reindex_output(out)
                return (out.sort_index() if self.sort else out)
        if isinstance(n, valid_containers):
            raise ValueError('dropna option with a list of nth values is not supported')
        if (dropna not in ['any', 'all']):
            raise ValueError(f"For a DataFrame groupby, dropna must be either None, 'any' or 'all', (was passed {dropna}).")
        max_len = (n if (n >= 0) else ((- 1) - n))
        dropped = self.obj.dropna(how=dropna, axis=self.axis)
        if ((self.keys is None) and (self.level is None)):
            axis = self.grouper.axis
            grouper = axis[axis.isin(dropped.index)]
        else:
            from pandas.core.groupby.grouper import get_grouper
            (grouper, _, _) = get_grouper(dropped, key=self.keys, axis=self.axis, level=self.level, sort=self.sort, mutated=self.mutated)
        grb = dropped.groupby(grouper, as_index=self.as_index, sort=self.sort)
        (sizes, result) = (grb.size(), grb.nth(n))
        mask = (sizes < max_len)._values
        if (len(result) and mask.any()):
            result.loc[mask] = np.nan
        if ((len(self.obj) == len(dropped)) or (len(result) == len(self.grouper.result_index))):
            result.index = self.grouper.result_index
        else:
            result = result.reindex(self.grouper.result_index)
        return result

    @final
    def quantile(self, q=0.5, interpolation='linear'):
        "\n        Return group values at the given quantile, a la numpy.percentile.\n\n        Parameters\n        ----------\n        q : float or array-like, default 0.5 (50% quantile)\n            Value(s) between 0 and 1 providing the quantile(s) to compute.\n        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}\n            Method to use when the desired quantile falls between two points.\n\n        Returns\n        -------\n        Series or DataFrame\n            Return type determined by caller of GroupBy object.\n\n        See Also\n        --------\n        Series.quantile : Similar method for Series.\n        DataFrame.quantile : Similar method for DataFrame.\n        numpy.percentile : NumPy method to compute qth percentile.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([\n        ...     ['a', 1], ['a', 2], ['a', 3],\n        ...     ['b', 1], ['b', 3], ['b', 5]\n        ... ], columns=['key', 'val'])\n        >>> df.groupby('key').quantile()\n            val\n        key\n        a    2.0\n        b    3.0\n        "
        from pandas import concat

        def pre_processor(vals: np.ndarray) -> Tuple[(np.ndarray, Optional[Type])]:
            if is_object_dtype(vals):
                raise TypeError("'quantile' cannot be performed against 'object' dtypes!")
            inference = None
            if is_integer_dtype(vals.dtype):
                if is_extension_array_dtype(vals.dtype):
                    vals = vals.to_numpy(dtype=float, na_value=np.nan)
                inference = np.int64
            elif (is_bool_dtype(vals.dtype) and is_extension_array_dtype(vals.dtype)):
                vals = vals.to_numpy(dtype=float, na_value=np.nan)
            elif is_datetime64_dtype(vals.dtype):
                inference = 'datetime64[ns]'
                vals = np.asarray(vals).astype(float)
            elif is_timedelta64_dtype(vals.dtype):
                inference = 'timedelta64[ns]'
                vals = np.asarray(vals).astype(float)
            return (vals, inference)

        def post_processor(vals: np.ndarray, inference: Optional[Type]) -> np.ndarray:
            if inference:
                if (not (is_integer_dtype(inference) and (interpolation in {'linear', 'midpoint'}))):
                    vals = vals.astype(inference)
            return vals
        if is_scalar(q):
            return self._get_cythonized_result('group_quantile', aggregate=True, numeric_only=False, needs_values=True, needs_mask=True, cython_dtype=np.dtype(np.float64), pre_processing=pre_processor, post_processing=post_processor, q=q, interpolation=interpolation)
        else:
            results = [self._get_cythonized_result('group_quantile', aggregate=True, needs_values=True, needs_mask=True, cython_dtype=np.dtype(np.float64), pre_processing=pre_processor, post_processing=post_processor, q=qi, interpolation=interpolation) for qi in q]
            result = concat(results, axis=self.axis, keys=q)
            order = (list(range(1, result.axes[self.axis].nlevels)) + [0])
            index_names = np.array(result.axes[self.axis].names)
            result.axes[self.axis].names = np.arange(len(index_names))
            if isinstance(result, Series):
                result = result.reorder_levels(order)
            else:
                result = result.reorder_levels(order, axis=self.axis)
            result.axes[self.axis].names = index_names[order]
            indices = np.arange(result.shape[self.axis]).reshape([len(q), self.ngroups]).T.flatten()
            return result.take(indices, axis=self.axis)

    @final
    @Substitution(name='groupby')
    def ngroup(self, ascending=True):
        '\n        Number each group from 0 to the number of groups - 1.\n\n        This is the enumerative complement of cumcount.  Note that the\n        numbers given to the groups match the order in which the groups\n        would be seen when iterating over the groupby object, not the\n        order they are first observed.\n\n        Parameters\n        ----------\n        ascending : bool, default True\n            If False, number in reverse, from number of group - 1 to 0.\n\n        Returns\n        -------\n        Series\n            Unique numbers for each group.\n\n        See Also\n        --------\n        .cumcount : Number the rows in each group.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": list("aaabba")})\n        >>> df\n           A\n        0  a\n        1  a\n        2  a\n        3  b\n        4  b\n        5  a\n        >>> df.groupby(\'A\').ngroup()\n        0    0\n        1    0\n        2    0\n        3    1\n        4    1\n        5    0\n        dtype: int64\n        >>> df.groupby(\'A\').ngroup(ascending=False)\n        0    1\n        1    1\n        2    1\n        3    0\n        4    0\n        5    1\n        dtype: int64\n        >>> df.groupby(["A", [1,1,2,3,2,1]]).ngroup()\n        0    0\n        1    0\n        2    1\n        3    3\n        4    2\n        5    0\n        dtype: int64\n        '
        with group_selection_context(self):
            index = self._selected_obj.index
            result = self._obj_1d_constructor(self.grouper.group_info[0], index)
            if (not ascending):
                result = ((self.ngroups - 1) - result)
            return result

    @final
    @Substitution(name='groupby')
    def cumcount(self, ascending=True):
        "\n        Number each item in each group from 0 to the length of that group - 1.\n\n        Essentially this is equivalent to\n\n        .. code-block:: python\n\n            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))\n\n        Parameters\n        ----------\n        ascending : bool, default True\n            If False, number in reverse, from length of group - 1 to 0.\n\n        Returns\n        -------\n        Series\n            Sequence number of each element within each group.\n\n        See Also\n        --------\n        .ngroup : Number the groups themselves.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],\n        ...                   columns=['A'])\n        >>> df\n           A\n        0  a\n        1  a\n        2  a\n        3  b\n        4  b\n        5  a\n        >>> df.groupby('A').cumcount()\n        0    0\n        1    1\n        2    2\n        3    0\n        4    1\n        5    3\n        dtype: int64\n        >>> df.groupby('A').cumcount(ascending=False)\n        0    3\n        1    2\n        2    1\n        3    1\n        4    0\n        5    0\n        dtype: int64\n        "
        with group_selection_context(self):
            index = self._selected_obj._get_axis(self.axis)
            cumcounts = self._cumcount_array(ascending=ascending)
            return self._obj_1d_constructor(cumcounts, index)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def rank(self, method='average', ascending=True, na_option='keep', pct=False, axis=0):
        "\n        Provide the rank of values within each group.\n\n        Parameters\n        ----------\n        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'\n            * average: average rank of group.\n            * min: lowest rank in group.\n            * max: highest rank in group.\n            * first: ranks assigned in order they appear in the array.\n            * dense: like 'min', but rank always increases by 1 between groups.\n        ascending : bool, default True\n            False for ranks by high (1) to low (N).\n        na_option : {'keep', 'top', 'bottom'}, default 'keep'\n            * keep: leave NA values where they are.\n            * top: smallest rank if ascending.\n            * bottom: smallest rank if descending.\n        pct : bool, default False\n            Compute percentage rank of data within each group.\n        axis : int, default 0\n            The axis of the object over which to compute the rank.\n\n        Returns\n        -------\n        DataFrame with ranking of values within each group\n        "
        if (na_option not in {'keep', 'top', 'bottom'}):
            msg = "na_option must be one of 'keep', 'top', or 'bottom'"
            raise ValueError(msg)
        return self._cython_transform('rank', numeric_only=False, ties_method=method, ascending=ascending, na_option=na_option, pct=pct, axis=axis)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def cumprod(self, axis=0, *args, **kwargs):
        '\n        Cumulative product for each group.\n\n        Returns\n        -------\n        Series or DataFrame\n        '
        nv.validate_groupby_func('cumprod', args, kwargs, ['numeric_only', 'skipna'])
        if (axis != 0):
            return self.apply((lambda x: x.cumprod(axis=axis, **kwargs)))
        return self._cython_transform('cumprod', **kwargs)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def cumsum(self, axis=0, *args, **kwargs):
        '\n        Cumulative sum for each group.\n\n        Returns\n        -------\n        Series or DataFrame\n        '
        nv.validate_groupby_func('cumsum', args, kwargs, ['numeric_only', 'skipna'])
        if (axis != 0):
            return self.apply((lambda x: x.cumsum(axis=axis, **kwargs)))
        return self._cython_transform('cumsum', **kwargs)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def cummin(self, axis=0, **kwargs):
        '\n        Cumulative min for each group.\n\n        Returns\n        -------\n        Series or DataFrame\n        '
        if (axis != 0):
            return self.apply((lambda x: np.minimum.accumulate(x, axis)))
        return self._cython_transform('cummin', numeric_only=False)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def cummax(self, axis=0, **kwargs):
        '\n        Cumulative max for each group.\n\n        Returns\n        -------\n        Series or DataFrame\n        '
        if (axis != 0):
            return self.apply((lambda x: np.maximum.accumulate(x, axis)))
        return self._cython_transform('cummax', numeric_only=False)

    @final
    def _get_cythonized_result(self, how, cython_dtype, aggregate=False, numeric_only=True, needs_counts=False, needs_values=False, needs_2d=False, min_count=None, needs_mask=False, needs_ngroups=False, result_is_index=False, pre_processing=None, post_processing=None, **kwargs):
        '\n        Get result for Cythonized functions.\n\n        Parameters\n        ----------\n        how : str, Cythonized function name to be called\n        cython_dtype : np.dtype\n            Type of the array that will be modified by the Cython call.\n        aggregate : bool, default False\n            Whether the result should be aggregated to match the number of\n            groups\n        numeric_only : bool, default True\n            Whether only numeric datatypes should be computed\n        needs_counts : bool, default False\n            Whether the counts should be a part of the Cython call\n        needs_values : bool, default False\n            Whether the values should be a part of the Cython call\n            signature\n        needs_2d : bool, default False\n            Whether the values and result of the Cython call signature\n            are 2-dimensional.\n        min_count : int, default None\n            When not None, min_count for the Cython call\n        needs_mask : bool, default False\n            Whether boolean mask needs to be part of the Cython call\n            signature\n        needs_ngroups : bool, default False\n            Whether number of groups is part of the Cython call signature\n        result_is_index : bool, default False\n            Whether the result of the Cython operation is an index of\n            values to be retrieved, instead of the actual values themselves\n        pre_processing : function, default None\n            Function to be applied to `values` prior to passing to Cython.\n            Function should return a tuple where the first element is the\n            values to be passed to Cython and the second element is an optional\n            type which the values should be converted to after being returned\n            by the Cython operation. This function is also responsible for\n            raising a TypeError if the values have an invalid type. Raises\n            if `needs_values` is False.\n        post_processing : function, default None\n            Function to be applied to result of Cython function. Should accept\n            an array of values as the first argument and type inferences as its\n            second argument, i.e. the signature should be\n            (ndarray, Type).\n        **kwargs : dict\n            Extra arguments to be passed back to Cython funcs\n\n        Returns\n        -------\n        `Series` or `DataFrame`  with filled values\n        '
        if (result_is_index and aggregate):
            raise ValueError("'result_is_index' and 'aggregate' cannot both be True!")
        if (post_processing and (not callable(post_processing))):
            raise ValueError("'post_processing' must be a callable!")
        if pre_processing:
            if (not callable(pre_processing)):
                raise ValueError("'pre_processing' must be a callable!")
            if (not needs_values):
                raise ValueError("Cannot use 'pre_processing' without specifying 'needs_values'!")
        grouper = self.grouper
        (labels, _, ngroups) = grouper.group_info
        output: Dict[(base.OutputKey, np.ndarray)] = {}
        base_func = getattr(libgroupby, how)
        error_msg = ''
        for (idx, obj) in enumerate(self._iterate_slices()):
            name = obj.name
            values = obj._values
            if (numeric_only and (not is_numeric_dtype(values))):
                continue
            if aggregate:
                result_sz = ngroups
            else:
                result_sz = len(values)
            result = np.zeros(result_sz, dtype=cython_dtype)
            if needs_2d:
                result = result.reshape(((- 1), 1))
            func = partial(base_func, result)
            inferences = None
            if needs_counts:
                counts = np.zeros(self.ngroups, dtype=np.int64)
                func = partial(func, counts)
            if needs_values:
                vals = values
                if pre_processing:
                    try:
                        (vals, inferences) = pre_processing(vals)
                    except TypeError as e:
                        error_msg = str(e)
                        continue
                vals = vals.astype(cython_dtype, copy=False)
                if needs_2d:
                    vals = vals.reshape(((- 1), 1))
                func = partial(func, vals)
            func = partial(func, labels)
            if (min_count is not None):
                func = partial(func, min_count)
            if needs_mask:
                mask = isna(values).view(np.uint8)
                func = partial(func, mask)
            if needs_ngroups:
                func = partial(func, ngroups)
            func(**kwargs)
            if needs_2d:
                result = result.reshape((- 1))
            if result_is_index:
                result = algorithms.take_nd(values, result)
            if post_processing:
                result = post_processing(result, inferences)
            key = base.OutputKey(label=name, position=idx)
            output[key] = result
        if ((not output) and (error_msg != '')):
            raise TypeError(error_msg)
        if aggregate:
            return self._wrap_aggregated_output(output, index=self.grouper.result_index)
        else:
            return self._wrap_transformed_output(output)

    @final
    @Substitution(name='groupby')
    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        '\n        Shift each group by periods observations.\n\n        If freq is passed, the index will be increased using the periods and the freq.\n\n        Parameters\n        ----------\n        periods : int, default 1\n            Number of periods to shift.\n        freq : str, optional\n            Frequency string.\n        axis : axis to shift, default 0\n            Shift direction.\n        fill_value : optional\n            The scalar value to use for newly introduced missing values.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        Series or DataFrame\n            Object shifted within each group.\n\n        See Also\n        --------\n        Index.shift : Shift values of Index.\n        tshift : Shift the time index, using the index’s frequency\n            if available.\n        '
        if ((freq is not None) or (axis != 0) or (not isna(fill_value))):
            return self.apply((lambda x: x.shift(periods, freq, axis, fill_value)))
        return self._get_cythonized_result('group_shift_indexer', numeric_only=False, cython_dtype=np.dtype(np.int64), needs_ngroups=True, result_is_index=True, periods=periods)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def pct_change(self, periods=1, fill_method='pad', limit=None, freq=None, axis=0):
        '\n        Calculate pct_change of each value to previous entry in group.\n\n        Returns\n        -------\n        Series or DataFrame\n            Percentage changes within each group.\n        '
        if ((freq is not None) or (axis != 0)):
            return self.apply((lambda x: x.pct_change(periods=periods, fill_method=fill_method, limit=limit, freq=freq, axis=axis)))
        if (fill_method is None):
            fill_method = 'pad'
            limit = 0
        filled = getattr(self, fill_method)(limit=limit)
        fill_grp = filled.groupby(self.grouper.codes, axis=self.axis)
        shifted = fill_grp.shift(periods=periods, freq=freq, axis=self.axis)
        return ((filled / shifted) - 1)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def head(self, n=5):
        "\n        Return first n rows of each group.\n\n        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows\n        from the original DataFrame with original index and order preserved\n        (``as_index`` flag is ignored).\n\n        Does not work for negative values of `n`.\n\n        Returns\n        -------\n        Series or DataFrame\n        %(see_also)s\n        Examples\n        --------\n\n        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]],\n        ...                   columns=['A', 'B'])\n        >>> df.groupby('A').head(1)\n           A  B\n        0  1  2\n        2  5  6\n        >>> df.groupby('A').head(-1)\n        Empty DataFrame\n        Columns: [A, B]\n        Index: []\n        "
        self._reset_group_selection()
        mask = (self._cumcount_array() < n)
        if (self.axis == 0):
            return self._selected_obj[mask]
        else:
            return self._selected_obj.iloc[:, mask]

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def tail(self, n=5):
        "\n        Return last n rows of each group.\n\n        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows\n        from the original DataFrame with original index and order preserved\n        (``as_index`` flag is ignored).\n\n        Does not work for negative values of `n`.\n\n        Returns\n        -------\n        Series or DataFrame\n        %(see_also)s\n        Examples\n        --------\n\n        >>> df = pd.DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2]],\n        ...                   columns=['A', 'B'])\n        >>> df.groupby('A').tail(1)\n           A  B\n        1  a  2\n        3  b  2\n        >>> df.groupby('A').tail(-1)\n        Empty DataFrame\n        Columns: [A, B]\n        Index: []\n        "
        self._reset_group_selection()
        mask = (self._cumcount_array(ascending=False) < n)
        if (self.axis == 0):
            return self._selected_obj[mask]
        else:
            return self._selected_obj.iloc[:, mask]

    @final
    def _reindex_output(self, output, fill_value=np.NaN):
        '\n        If we have categorical groupers, then we might want to make sure that\n        we have a fully re-indexed output to the levels. This means expanding\n        the output space to accommodate all values in the cartesian product of\n        our groups, regardless of whether they were observed in the data or\n        not. This will expand the output space if there are missing groups.\n\n        The method returns early without modifying the input if the number of\n        groupings is less than 2, self.observed == True or none of the groupers\n        are categorical.\n\n        Parameters\n        ----------\n        output : Series or DataFrame\n            Object resulting from grouping and applying an operation.\n        fill_value : scalar, default np.NaN\n            Value to use for unobserved categories if self.observed is False.\n\n        Returns\n        -------\n        Series or DataFrame\n            Object (potentially) re-indexed to include all possible groups.\n        '
        groupings = self.grouper.groupings
        if (groupings is None):
            return output
        elif (len(groupings) == 1):
            return output
        elif self.observed:
            return output
        elif (not any((isinstance(ping.grouper, (Categorical, CategoricalIndex)) for ping in groupings))):
            return output
        levels_list = [ping.group_index for ping in groupings]
        (index, _) = MultiIndex.from_product(levels_list, names=self.grouper.names).sortlevel()
        if self.as_index:
            d = {self.obj._get_axis_name(self.axis): index, 'copy': False, 'fill_value': fill_value}
            return output.reindex(**d)
        in_axis_grps = ((i, ping.name) for (i, ping) in enumerate(groupings) if ping.in_axis)
        (g_nums, g_names) = zip(*in_axis_grps)
        output = output.drop(labels=list(g_names), axis=1)
        output = output.set_index(self.grouper.result_index).reindex(index, copy=False, fill_value=fill_value)
        output = output.reset_index(level=g_nums)
        return output.reset_index(drop=True)

    @final
    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None):
        '\n        Return a random sample of items from each group.\n\n        You can use `random_state` for reproducibility.\n\n        .. versionadded:: 1.1.0\n\n        Parameters\n        ----------\n        n : int, optional\n            Number of items to return for each group. Cannot be used with\n            `frac` and must be no larger than the smallest group unless\n            `replace` is True. Default is one if `frac` is None.\n        frac : float, optional\n            Fraction of items to return. Cannot be used with `n`.\n        replace : bool, default False\n            Allow or disallow sampling of the same row more than once.\n        weights : list-like, optional\n            Default None results in equal probability weighting.\n            If passed a list-like then values must have the same length as\n            the underlying DataFrame or Series object and will be used as\n            sampling probabilities after normalization within each group.\n            Values must be non-negative with at least one positive element\n            within each group.\n        random_state : int, array-like, BitGenerator, np.random.RandomState, optional\n            If int, array-like, or BitGenerator (NumPy>=1.17), seed for\n            random number generator\n            If np.random.RandomState, use as numpy RandomState object.\n\n        Returns\n        -------\n        Series or DataFrame\n            A new object of same type as caller containing items randomly\n            sampled within each group from the caller object.\n\n        See Also\n        --------\n        DataFrame.sample: Generate random samples from a DataFrame object.\n        numpy.random.choice: Generate a random sample from a given 1-D numpy\n            array.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(\n        ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}\n        ... )\n        >>> df\n               a  b\n        0    red  0\n        1    red  1\n        2   blue  2\n        3   blue  3\n        4  black  4\n        5  black  5\n\n        Select one row at random for each distinct value in column a. The\n        `random_state` argument can be used to guarantee reproducibility:\n\n        >>> df.groupby("a").sample(n=1, random_state=1)\n               a  b\n        4  black  4\n        2   blue  2\n        1    red  1\n\n        Set `frac` to sample fixed proportions rather than counts:\n\n        >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2)\n        5    5\n        2    2\n        0    0\n        Name: b, dtype: int64\n\n        Control sample probabilities within groups by setting weights:\n\n        >>> df.groupby("a").sample(\n        ...     n=1,\n        ...     weights=[1, 1, 1, 0, 0, 1],\n        ...     random_state=1,\n        ... )\n               a  b\n        5  black  5\n        2   blue  2\n        0    red  0\n        '
        from pandas.core.reshape.concat import concat
        if (weights is not None):
            weights = Series(weights, index=self._selected_obj.index)
            ws = [weights[idx] for idx in self.indices.values()]
        else:
            ws = ([None] * self.ngroups)
        if (random_state is not None):
            random_state = com.random_state(random_state)
        samples = [obj.sample(n=n, frac=frac, replace=replace, weights=w, random_state=random_state) for ((_, obj), w) in zip(self, ws)]
        return concat(samples, axis=self.axis)

@doc(GroupBy)
def get_groupby(obj, by=None, axis=0, level=None, grouper=None, exclusions=None, selection=None, as_index=True, sort=True, group_keys=True, squeeze=False, observed=False, mutated=False, dropna=True):
    klass: Type[GroupBy]
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy
        klass = SeriesGroupBy
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy
        klass = DataFrameGroupBy
    else:
        raise TypeError(f'invalid type: {obj}')
    return klass(obj=obj, keys=by, axis=axis, level=level, grouper=grouper, exclusions=exclusions, selection=selection, as_index=as_index, sort=sort, group_keys=group_keys, squeeze=squeeze, observed=observed, mutated=mutated, dropna=dropna)
