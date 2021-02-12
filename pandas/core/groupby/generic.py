
'\nDefine the SeriesGroupBy and DataFrameGroupBy\nclasses that hold the groupby interfaces (and some implementations).\n\nThese are user facing as the result of the ``df.groupby(...)`` operations,\nwhich here returns a DataFrameGroupBy object.\n'
from collections import abc, namedtuple
import copy
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar, Union, cast
import warnings
import numpy as np
from pandas._libs import lib, reduction as libreduction
from pandas._typing import ArrayLike, FrameOrSeries, FrameOrSeriesUnion, Label
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.dtypes.cast import find_common_type, maybe_cast_result_dtype, maybe_downcast_numeric
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int, is_bool, is_categorical_dtype, is_integer_dtype, is_interval_dtype, is_numeric_dtype, is_scalar, needs_i8_conversion
from pandas.core.dtypes.missing import isna, notna
from pandas.core import algorithms, nanops
from pandas.core.aggregation import agg_list_like, aggregate, maybe_mangle_lambdas, reconstruct_func, validate_func_kwargs
from pandas.core.arrays import Categorical, ExtensionArray
from pandas.core.base import DataError, SpecificationError
import pandas.core.common as com
from pandas.core.construction import create_series_with_explicit_dtype
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import base
from pandas.core.groupby.groupby import GroupBy, _agg_template, _apply_docs, _transform_template, get_groupby, group_selection_context
from pandas.core.indexes.api import Index, MultiIndex, all_indexes_same
import pandas.core.indexes.base as ibase
from pandas.core.internals import BlockManager
from pandas.core.series import Series
from pandas.core.util.numba_ import maybe_use_numba
from pandas.plotting import boxplot_frame_groupby
if TYPE_CHECKING:
    from pandas.core.internals import Block
NamedAgg = namedtuple('NamedAgg', ['column', 'aggfunc'])
AggScalar = Union[(str, Callable[(..., Any)])]
ScalarResult = TypeVar('ScalarResult')

def generate_property(name, klass):
    '\n    Create a property for a GroupBy subclass to dispatch to DataFrame/Series.\n\n    Parameters\n    ----------\n    name : str\n    klass : {DataFrame, Series}\n\n    Returns\n    -------\n    property\n    '

    def prop(self):
        return self._make_wrapper(name)
    parent_method = getattr(klass, name)
    prop.__doc__ = (parent_method.__doc__ or '')
    prop.__name__ = name
    return property(prop)

def pin_allowlisted_properties(klass, allowlist):
    "\n    Create GroupBy member defs for DataFrame/Series names in a allowlist.\n\n    Parameters\n    ----------\n    klass : DataFrame or Series class\n        class where members are defined.\n    allowlist : frozenset[str]\n        Set of names of klass methods to be constructed\n\n    Returns\n    -------\n    class decorator\n\n    Notes\n    -----\n    Since we don't want to override methods explicitly defined in the\n    base class, any such name is skipped.\n    "

    def pinner(cls):
        for name in allowlist:
            if hasattr(cls, name):
                continue
            prop = generate_property(name, klass)
            setattr(cls, name, prop)
        return cls
    return pinner

@pin_allowlisted_properties(Series, base.series_apply_allowlist)
class SeriesGroupBy(GroupBy[Series]):
    _apply_allowlist = base.series_apply_allowlist

    def _iterate_slices(self):
        (yield self._selected_obj)

    @property
    def _selection_name(self):
        '\n        since we are a series, we by definition only have\n        a single name, but may be the result of a selection or\n        the name of our object\n        '
        if (self._selection is None):
            return self.obj.name
        else:
            return self._selection
    _agg_examples_doc = dedent("\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).min()\n    1    1\n    2    3\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).agg('min')\n    1    1\n    2    3\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])\n       min  max\n    1    1    2\n    2    3    4\n\n    The output column names can be controlled by passing\n    the desired column names and aggregations as keyword arguments.\n\n    >>> s.groupby([1, 1, 2, 2]).agg(\n    ...     minimum='min',\n    ...     maximum='max',\n    ... )\n       minimum  maximum\n    1        1        2\n    2        3        4")

    @Appender(_apply_docs['template'].format(input='series', examples=_apply_docs['series_examples']))
    def apply(self, func, *args, **kwargs):
        return super().apply(func, *args, **kwargs)

    @doc(_agg_template, examples=_agg_examples_doc, klass='Series')
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        if maybe_use_numba(engine):
            with group_selection_context(self):
                data = self._selected_obj
            (result, index) = self._aggregate_with_numba(data.to_frame(), func, *args, engine_kwargs=engine_kwargs, **kwargs)
            return self.obj._constructor(result.ravel(), index=index, name=data.name)
        relabeling = (func is None)
        columns = None
        if relabeling:
            (columns, func) = validate_func_kwargs(kwargs)
            kwargs = {}
        if isinstance(func, str):
            return getattr(self, func)(*args, **kwargs)
        elif isinstance(func, abc.Iterable):
            func = maybe_mangle_lambdas(func)
            ret = self._aggregate_multiple_funcs(func)
            if relabeling:
                ret.columns = columns
        else:
            cyfunc = self._get_cython_func(func)
            if (cyfunc and (not args) and (not kwargs)):
                return getattr(self, cyfunc)()
            if (self.grouper.nkeys > 1):
                return self._python_agg_general(func, *args, **kwargs)
            try:
                return self._python_agg_general(func, *args, **kwargs)
            except (ValueError, KeyError):
                result = self._aggregate_named(func, *args, **kwargs)
            index = Index(sorted(result), name=self.grouper.names[0])
            ret = create_series_with_explicit_dtype(result, index=index, dtype_if_empty=object)
        if (not self.as_index):
            print('Warning, ignoring as_index=True')
        if isinstance(ret, dict):
            from pandas import concat
            ret = concat(ret.values(), axis=1, keys=[key.label for key in ret.keys()])
        return ret
    agg = aggregate

    def _aggregate_multiple_funcs(self, arg):
        if isinstance(arg, dict):
            if isinstance(self._selected_obj, Series):
                raise SpecificationError('nested renamer is not supported')
            columns = list(arg.keys())
            arg = arg.items()
        elif any((isinstance(x, (tuple, list)) for x in arg)):
            arg = [((x, x) if (not isinstance(x, (tuple, list))) else x) for x in arg]
            columns = next(zip(*arg))
        else:
            columns = []
            for f in arg:
                columns.append((com.get_callable_name(f) or f))
            arg = zip(columns, arg)
        results: Dict[(base.OutputKey, FrameOrSeriesUnion)] = {}
        for (idx, (name, func)) in enumerate(arg):
            obj = self
            if (name in self._selected_obj):
                obj = copy.copy(obj)
                obj._reset_cache()
                obj._selection = name
            results[base.OutputKey(label=name, position=idx)] = obj.aggregate(func)
        if any((isinstance(x, DataFrame) for x in results.values())):
            return results
        output = self._wrap_aggregated_output(results, index=None)
        return self.obj._constructor_expanddim(output, columns=columns)

    def _wrap_series_output(self, output, index):
        '\n        Wraps the output of a SeriesGroupBy operation into the expected result.\n\n        Parameters\n        ----------\n        output : Mapping[base.OutputKey, Union[Series, np.ndarray]]\n            Data to wrap.\n        index : pd.Index or None\n            Index to apply to the output.\n\n        Returns\n        -------\n        Series or DataFrame\n\n        Notes\n        -----\n        In the vast majority of cases output and columns will only contain one\n        element. The exception is operations that expand dimensions, like ohlc.\n        '
        indexed_output = {key.position: val for (key, val) in output.items()}
        columns = Index((key.label for key in output))
        result: FrameOrSeriesUnion
        if (len(output) > 1):
            result = self.obj._constructor_expanddim(indexed_output, index=index)
            result.columns = columns
        elif (not columns.empty):
            result = self.obj._constructor(indexed_output[0], index=index, name=columns[0])
        else:
            result = self.obj._constructor_expanddim()
        return result

    def _wrap_aggregated_output(self, output, index):
        '\n        Wraps the output of a SeriesGroupBy aggregation into the expected result.\n\n        Parameters\n        ----------\n        output : Mapping[base.OutputKey, Union[Series, np.ndarray]]\n            Data to wrap.\n\n        Returns\n        -------\n        Series or DataFrame\n\n        Notes\n        -----\n        In the vast majority of cases output will only contain one element.\n        The exception is operations that expand dimensions, like ohlc.\n        '
        result = self._wrap_series_output(output=output, index=index)
        return self._reindex_output(result)

    def _wrap_transformed_output(self, output):
        '\n        Wraps the output of a SeriesGroupBy aggregation into the expected result.\n\n        Parameters\n        ----------\n        output : dict[base.OutputKey, Union[Series, np.ndarray]]\n            Dict with a sole key of 0 and a value of the result values.\n\n        Returns\n        -------\n        Series\n\n        Notes\n        -----\n        output should always contain one element. It is specified as a dict\n        for consistency with DataFrame methods and _wrap_aggregated_output.\n        '
        assert (len(output) == 1)
        result = self._wrap_series_output(output=output, index=self.obj.index)
        assert isinstance(result, Series)
        return result

    def _wrap_applied_output(self, keys, values, not_indexed_same=False):
        '\n        Wrap the output of SeriesGroupBy.apply into the expected result.\n\n        Parameters\n        ----------\n        keys : Index\n            Keys of groups that Series was grouped by.\n        values : Optional[List[Any]]\n            Applied output for each group.\n        not_indexed_same : bool, default False\n            Whether the applied outputs are not indexed the same as the group axes.\n\n        Returns\n        -------\n        DataFrame or Series\n        '
        if (len(keys) == 0):
            return self.obj._constructor([], name=self._selection_name, index=keys, dtype=np.float64)
        assert (values is not None)

        def _get_index() -> Index:
            if (self.grouper.nkeys > 1):
                index = MultiIndex.from_tuples(keys, names=self.grouper.names)
            else:
                index = Index(keys, name=self.grouper.names[0])
            return index
        if isinstance(values[0], dict):
            index = _get_index()
            result: FrameOrSeriesUnion = self._reindex_output(self.obj._constructor_expanddim(values, index=index))
            result = result.stack(dropna=self.observed)
            result.name = self._selection_name
            return result
        elif isinstance(values[0], (Series, DataFrame)):
            return self._concat_objects(keys, values, not_indexed_same=not_indexed_same)
        else:
            result = self.obj._constructor(data=values, index=_get_index(), name=self._selection_name)
            return self._reindex_output(result)

    def _aggregate_named(self, func, *args, **kwargs):
        result = {}
        initialized = False
        for (name, group) in self:
            group.name = name
            output = func(group, *args, **kwargs)
            output = libreduction.extract_result(output)
            if (not initialized):
                libreduction.check_result_array(output, 0)
                initialized = True
            result[name] = output
        return result

    @Substitution(klass='Series')
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        if maybe_use_numba(engine):
            with group_selection_context(self):
                data = self._selected_obj
            result = self._transform_with_numba(data.to_frame(), func, *args, engine_kwargs=engine_kwargs, **kwargs)
            return self.obj._constructor(result.ravel(), index=data.index, name=data.name)
        func = (self._get_cython_func(func) or func)
        if (not isinstance(func, str)):
            return self._transform_general(func, *args, **kwargs)
        elif (func not in base.transform_kernel_allowlist):
            msg = f"'{func}' is not a valid function name for transform(name)"
            raise ValueError(msg)
        elif ((func in base.cythonized_kernels) or (func in base.transformation_kernels)):
            return getattr(self, func)(*args, **kwargs)
        with com.temp_setattr(self, 'observed', True):
            result = getattr(self, func)(*args, **kwargs)
        return self._transform_fast(result)

    def _transform_general(self, func, *args, **kwargs):
        '\n        Transform with a non-str `func`.\n        '
        klass = type(self._selected_obj)
        results = []
        for (name, group) in self:
            object.__setattr__(group, 'name', name)
            res = func(group, *args, **kwargs)
            if isinstance(res, (DataFrame, Series)):
                res = res._values
            results.append(klass(res, index=group.index))
        if results:
            from pandas.core.reshape.concat import concat
            concatenated = concat(results)
            result = self._set_result_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)
        if is_numeric_dtype(result.dtype):
            common_dtype = find_common_type([self._selected_obj.dtype, result.dtype])
            if (common_dtype is result.dtype):
                result = maybe_downcast_numeric(result, self._selected_obj.dtype)
        result.name = self._selected_obj.name
        return result

    def _transform_fast(self, result):
        '\n        fast version of transform, only applicable to\n        builtin/cythonizable functions\n        '
        (ids, _, ngroup) = self.grouper.group_info
        result = result.reindex(self.grouper.result_index, copy=False)
        out = algorithms.take_1d(result._values, ids)
        return self.obj._constructor(out, index=self.obj.index, name=self.obj.name)

    def filter(self, func, dropna=True, *args, **kwargs):
        "\n        Return a copy of a Series excluding elements from groups that\n        do not satisfy the boolean criterion specified by func.\n\n        Parameters\n        ----------\n        func : function\n            To apply to each group. Should return True or False.\n        dropna : Drop groups that do not pass the filter. True by default;\n            if False, groups that evaluate False are filled with NaNs.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',\n        ...                           'foo', 'bar'],\n        ...                    'B' : [1, 2, 3, 4, 5, 6],\n        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})\n        >>> grouped = df.groupby('A')\n        >>> df.groupby('A').B.filter(lambda x: x.mean() > 3.)\n        1    2\n        3    4\n        5    6\n        Name: B, dtype: int64\n\n        Returns\n        -------\n        filtered : Series\n        "
        if isinstance(func, str):
            wrapper = (lambda x: getattr(x, func)(*args, **kwargs))
        else:
            wrapper = (lambda x: func(x, *args, **kwargs))

        def true_and_notna(x) -> bool:
            b = wrapper(x)
            return (b and notna(b))
        try:
            indices = [self._get_index(name) for (name, group) in self if true_and_notna(group)]
        except (ValueError, TypeError) as err:
            raise TypeError('the filter must return a boolean result') from err
        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna=True):
        '\n        Return number of unique elements in the group.\n\n        Returns\n        -------\n        Series\n            Number of unique values within each group.\n        '
        (ids, _, _) = self.grouper.group_info
        val = self.obj._values
        (codes, _) = algorithms.factorize(val, sort=False)
        sorter = np.lexsort((codes, ids))
        codes = codes[sorter]
        ids = ids[sorter]
        idx = np.r_[(0, (1 + np.nonzero((ids[1:] != ids[:(- 1)]))[0]))]
        inc = np.r_[(1, (codes[1:] != codes[:(- 1)]))]
        mask = (codes == (- 1))
        if dropna:
            inc[idx] = 1
            inc[mask] = 0
        else:
            inc[(mask & np.r_[(False, mask[:(- 1)])])] = 0
            inc[idx] = 1
        out = np.add.reduceat(inc, idx).astype('int64', copy=False)
        if len(ids):
            if (ids[0] == (- 1)):
                res = out[1:]
                idx = idx[np.flatnonzero(idx)]
            else:
                res = out
        else:
            res = out[1:]
        ri = self.grouper.result_index
        if (len(res) != len(ri)):
            (res, out) = (np.zeros(len(ri), dtype=out.dtype), res)
            res[ids[idx]] = out
        result = self.obj._constructor(res, index=ri, name=self._selection_name)
        return self._reindex_output(result, fill_value=0)

    @doc(Series.describe)
    def describe(self, **kwargs):
        result = self.apply((lambda x: x.describe(**kwargs)))
        if (self.axis == 1):
            return result.T
        return result.unstack()

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut
        (ids, _, _) = self.grouper.group_info
        val = self.obj._values

        def apply_series_value_counts():
            return self.apply(Series.value_counts, normalize=normalize, sort=sort, ascending=ascending, bins=bins)
        if (bins is not None):
            if (not np.iterable(bins)):
                return apply_series_value_counts()
        elif is_categorical_dtype(val):
            return apply_series_value_counts()
        mask = (ids != (- 1))
        (ids, val) = (ids[mask], val[mask])
        if (bins is None):
            (lab, lev) = algorithms.factorize(val, sort=True)
            llab = (lambda lab, inc: lab[inc])
        else:
            lab = cut(Series(val), bins, include_lowest=True)
            lev = lab.cat.categories
            lab = lev.take(lab.cat.codes, allow_fill=True, fill_value=lev._na_value)
            llab = (lambda lab, inc: lab[inc]._multiindex.codes[(- 1)])
        if is_interval_dtype(lab.dtype):
            sorter = np.lexsort((lab.left, lab.right, ids))
        else:
            sorter = np.lexsort((lab, ids))
        (ids, lab) = (ids[sorter], lab[sorter])
        idx = np.r_[(0, (1 + np.nonzero((ids[1:] != ids[:(- 1)]))[0]))]
        lchanges = (llab(lab, slice(1, None)) != llab(lab, slice(None, (- 1))))
        inc = np.r_[(True, lchanges)]
        inc[idx] = True
        out = np.diff(np.nonzero(np.r_[(inc, True)])[0])
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))
        codes = self.grouper.reconstructed_codes
        codes = ([rep(level_codes) for level_codes in codes] + [llab(lab, inc)])
        levels = ([ping.group_index for ping in self.grouper.groupings] + [lev])
        names = (self.grouper.names + [self._selection_name])
        if dropna:
            mask = (codes[(- 1)] != (- 1))
            if mask.all():
                dropna = False
            else:
                (out, codes) = (out[mask], [level_codes[mask] for level_codes in codes])
        if normalize:
            out = out.astype('float')
            d = np.diff(np.r_[(idx, len(ids))])
            if dropna:
                m = ids[(lab == (- 1))]
                np.add.at(d, m, (- 1))
                acc = rep(d)[mask]
            else:
                acc = rep(d)
            out /= acc
        if (sort and (bins is None)):
            cat = (ids[inc][mask] if dropna else ids[inc])
            sorter = np.lexsort(((out if ascending else (- out)), cat))
            (out, codes[(- 1)]) = (out[sorter], codes[(- 1)][sorter])
        if (bins is None):
            mi = MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=False)
            if is_integer_dtype(out):
                out = ensure_int64(out)
            return self.obj._constructor(out, index=mi, name=self._selection_name)
        diff = np.zeros(len(out), dtype='bool')
        for level_codes in codes[:(- 1)]:
            diff |= np.r_[(True, (level_codes[1:] != level_codes[:(- 1)]))]
        (ncat, nbin) = (diff.sum(), len(levels[(- 1)]))
        left = [np.repeat(np.arange(ncat), nbin), np.tile(np.arange(nbin), ncat)]
        right = [(diff.cumsum() - 1), codes[(- 1)]]
        (_, idx) = get_join_indexers(left, right, sort=False, how='left')
        out = np.where((idx != (- 1)), out[idx], 0)
        if sort:
            sorter = np.lexsort(((out if ascending else (- out)), left[0]))
            (out, left[(- 1)]) = (out[sorter], left[(- 1)][sorter])

        def build_codes(lev_codes: np.ndarray) -> np.ndarray:
            return np.repeat(lev_codes[diff], nbin)
        codes = [build_codes(lev_codes) for lev_codes in codes[:(- 1)]]
        codes.append(left[(- 1)])
        mi = MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=False)
        if is_integer_dtype(out):
            out = ensure_int64(out)
        return self.obj._constructor(out, index=mi, name=self._selection_name)

    def count(self):
        '\n        Compute count of group, excluding missing values.\n\n        Returns\n        -------\n        Series\n            Count of values within each group.\n        '
        (ids, _, ngroups) = self.grouper.group_info
        val = self.obj._values
        mask = ((ids != (- 1)) & (~ isna(val)))
        ids = ensure_platform_int(ids)
        minlength = (ngroups or 0)
        out = np.bincount(ids[mask], minlength=minlength)
        result = self.obj._constructor(out, index=self.grouper.result_index, name=self._selection_name, dtype='int64')
        return self._reindex_output(result, fill_value=0)

    def _apply_to_column_groupbys(self, func):
        ' return a pass thru '
        return func(self)

    def pct_change(self, periods=1, fill_method='pad', limit=None, freq=None):
        'Calculate pct_change of each value to previous entry in group'
        if freq:
            return self.apply((lambda x: x.pct_change(periods=periods, fill_method=fill_method, limit=limit, freq=freq)))
        if (fill_method is None):
            fill_method = 'pad'
            limit = 0
        filled = getattr(self, fill_method)(limit=limit)
        fill_grp = filled.groupby(self.grouper.codes)
        shifted = fill_grp.shift(periods=periods, freq=freq)
        return ((filled / shifted) - 1)

@pin_allowlisted_properties(DataFrame, base.dataframe_apply_allowlist)
class DataFrameGroupBy(GroupBy[DataFrame]):
    _apply_allowlist = base.dataframe_apply_allowlist
    _agg_examples_doc = dedent('\n    Examples\n    --------\n    >>> df = pd.DataFrame(\n    ...     {\n    ...         "A": [1, 1, 2, 2],\n    ...         "B": [1, 2, 3, 4],\n    ...         "C": [0.362838, 0.227877, 1.267767, -0.562860],\n    ...     }\n    ... )\n\n    >>> df\n       A  B         C\n    0  1  1  0.362838\n    1  1  2  0.227877\n    2  2  3  1.267767\n    3  2  4 -0.562860\n\n    The aggregation is for each column.\n\n    >>> df.groupby(\'A\').agg(\'min\')\n       B         C\n    A\n    1  1  0.227877\n    2  3 -0.562860\n\n    Multiple aggregations\n\n    >>> df.groupby(\'A\').agg([\'min\', \'max\'])\n        B             C\n      min max       min       max\n    A\n    1   1   2  0.227877  0.362838\n    2   3   4 -0.562860  1.267767\n\n    Select a column for aggregation\n\n    >>> df.groupby(\'A\').B.agg([\'min\', \'max\'])\n       min  max\n    A\n    1    1    2\n    2    3    4\n\n    Different aggregations per column\n\n    >>> df.groupby(\'A\').agg({\'B\': [\'min\', \'max\'], \'C\': \'sum\'})\n        B             C\n      min max       sum\n    A\n    1   1   2  0.590715\n    2   3   4  0.704907\n\n    To control the output names with different aggregations per column,\n    pandas supports "named aggregation"\n\n    >>> df.groupby("A").agg(\n    ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),\n    ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum"))\n       b_min     c_sum\n    A\n    1      1  0.590715\n    2      3  0.704907\n\n    - The keywords are the *output* column names\n    - The values are tuples whose first element is the column to select\n      and the second element is the aggregation to apply to that column.\n      Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields\n      ``[\'column\', \'aggfunc\']`` to make it clearer what the arguments are.\n      As usual, the aggregation can be a callable or a string alias.\n\n    See :ref:`groupby.aggregate.named` for more.')

    @doc(_agg_template, examples=_agg_examples_doc, klass='DataFrame')
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        if maybe_use_numba(engine):
            with group_selection_context(self):
                data = self._selected_obj
            (result, index) = self._aggregate_with_numba(data, func, *args, engine_kwargs=engine_kwargs, **kwargs)
            return self.obj._constructor(result, index=index, columns=data.columns)
        (relabeling, func, columns, order) = reconstruct_func(func, **kwargs)
        func = maybe_mangle_lambdas(func)
        (result, how) = aggregate(self, func, *args, **kwargs)
        if (how is None):
            return result
        if (result is None):
            if (self.grouper.nkeys > 1):
                return self._python_agg_general(func, *args, **kwargs)
            elif (args or kwargs):
                result = self._aggregate_frame(func, *args, **kwargs)
            elif (self.axis == 1):
                result = self._aggregate_frame(func)
            else:
                try:
                    result = agg_list_like(self, [func], _axis=self.axis)
                    result.columns = result.columns.rename(([self._selected_obj.columns.name] * result.columns.nlevels)).droplevel((- 1))
                except ValueError as err:
                    if ('no results' not in str(err)):
                        raise
                    result = self._aggregate_frame(func)
                except AttributeError:
                    result = self._aggregate_frame(func)
        if relabeling:
            result = result.iloc[:, order]
            result.columns = columns
        if (not self.as_index):
            self._insert_inaxis_grouper_inplace(result)
            result.index = np.arange(len(result))
        return result._convert(datetime=True)
    agg = aggregate

    def _iterate_slices(self):
        obj = self._selected_obj
        if (self.axis == 1):
            obj = obj.T
        if (isinstance(obj, Series) and (obj.name not in self.exclusions)):
            (yield obj)
        else:
            for (label, values) in obj.items():
                if (label in self.exclusions):
                    continue
                (yield values)

    def _cython_agg_general(self, how, alt=None, numeric_only=True, min_count=(- 1)):
        agg_mgr = self._cython_agg_blocks(how, alt=alt, numeric_only=numeric_only, min_count=min_count)
        return self._wrap_agged_blocks(agg_mgr.blocks, items=agg_mgr.items)

    def _cython_agg_blocks(self, how, alt=None, numeric_only=True, min_count=(- 1)):
        data: BlockManager = self._get_data_to_aggregate()
        if numeric_only:
            data = data.get_numeric_data(copy=False)

        def cast_agg_result(result, values: ArrayLike, how: str) -> ArrayLike:
            assert (not isinstance(result, DataFrame))
            dtype = maybe_cast_result_dtype(values.dtype, how)
            result = maybe_downcast_numeric(result, dtype)
            if (isinstance(values, Categorical) and isinstance(result, np.ndarray)):
                result = type(values)._from_sequence(result.ravel(), dtype=values.dtype)
            elif (isinstance(result, np.ndarray) and (result.ndim == 1)):
                result = result.reshape(1, (- 1))
            return result

        def py_fallback(bvalues: ArrayLike) -> ArrayLike:
            obj: FrameOrSeriesUnion
            if isinstance(bvalues, ExtensionArray):
                obj = Series(bvalues)
            else:
                obj = DataFrame(bvalues.T)
                if (obj.shape[1] == 1):
                    obj = obj.iloc[:, 0]
            sgb = get_groupby(obj, self.grouper, observed=True)
            result = sgb.aggregate((lambda x: alt(x, axis=self.axis)))
            assert isinstance(result, (Series, DataFrame))
            result = result._consolidate()
            assert isinstance(result, (Series, DataFrame))
            assert (len(result._mgr.blocks) == 1)
            result = result._mgr.blocks[0].values
            return result

        def blk_func(bvalues: ArrayLike) -> ArrayLike:
            try:
                result = self.grouper._cython_operation('aggregate', bvalues, how, axis=1, min_count=min_count)
            except NotImplementedError:
                if (alt is None):
                    assert (how == 'ohlc')
                    raise
                result = py_fallback(bvalues)
            return cast_agg_result(result, bvalues, how)
        new_mgr = data.apply(blk_func, ignore_failures=True)
        if (not len(new_mgr)):
            raise DataError('No numeric types to aggregate')
        return new_mgr

    def _aggregate_frame(self, func, *args, **kwargs):
        if (self.grouper.nkeys != 1):
            raise AssertionError('Number of keys must be 1')
        axis = self.axis
        obj = self._obj_with_exclusions
        result: Dict[(Label, Union[(NDFrame, np.ndarray)])] = {}
        if (axis != obj._info_axis_number):
            for (name, data) in self:
                fres = func(data, *args, **kwargs)
                result[name] = fres
        else:
            for name in self.indices:
                data = self.get_group(name, obj=obj)
                fres = func(data, *args, **kwargs)
                result[name] = fres
        return self._wrap_frame_output(result, obj)

    def _aggregate_item_by_item(self, func, *args, **kwargs):
        obj = self._obj_with_exclusions
        result: Dict[(Union[(int, str)], NDFrame)] = {}
        cannot_agg = []
        for item in obj:
            data = obj[item]
            colg = SeriesGroupBy(data, selection=item, grouper=self.grouper)
            try:
                result[item] = colg.aggregate(func, *args, **kwargs)
            except ValueError as err:
                if ('Must produce aggregated value' in str(err)):
                    raise
                cannot_agg.append(item)
                continue
        result_columns = obj.columns
        if cannot_agg:
            result_columns = result_columns.drop(cannot_agg)
        return self.obj._constructor(result, columns=result_columns)

    def _wrap_applied_output(self, keys, values, not_indexed_same=False):
        if (len(keys) == 0):
            return self.obj._constructor(index=keys)
        first_not_none = next(com.not_none(*values), None)
        if (first_not_none is None):
            return self.obj._constructor()
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(keys, values, not_indexed_same=not_indexed_same)
        key_index = (self.grouper.result_index if self.as_index else None)
        if isinstance(first_not_none, (np.ndarray, Index)):
            return self.obj._constructor_sliced(values, index=key_index, name=self._selection_name)
        elif (not isinstance(first_not_none, Series)):
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = DataFrame(values, index=key_index, columns=[self._selection])
                self._insert_inaxis_grouper_inplace(result)
                return result
        else:
            return self._wrap_applied_output_series(keys, values, not_indexed_same, first_not_none, key_index)

    def _wrap_applied_output_series(self, keys, values, not_indexed_same, first_not_none, key_index):
        kwargs = first_not_none._construct_axes_dict()
        backup = create_series_with_explicit_dtype(dtype_if_empty=object, **kwargs)
        values = [(x if (x is not None) else backup) for x in values]
        all_indexed_same = all_indexes_same((x.index for x in values))
        if self.squeeze:
            applied_index = self._selected_obj._get_axis(self.axis)
            singular_series = ((len(values) == 1) and (applied_index.nlevels == 1))
            if singular_series:
                values[0].name = keys[0]
                return self._concat_objects(keys, values, not_indexed_same=not_indexed_same)
            elif all_indexed_same:
                from pandas.core.reshape.concat import concat
                return concat(values)
        if (not all_indexed_same):
            return self._concat_objects(keys, values, not_indexed_same=True)
        stacked_values = np.vstack([np.asarray(v) for v in values])
        if (self.axis == 0):
            index = key_index
            columns = first_not_none.index.copy()
            if (columns.name is None):
                names = {v.name for v in values}
                if (len(names) == 1):
                    columns.name = list(names)[0]
        else:
            index = first_not_none.index
            columns = key_index
            stacked_values = stacked_values.T
        result = self.obj._constructor(stacked_values, index=index, columns=columns)
        so = self._selected_obj
        if ((so.ndim == 2) and so.dtypes.apply(needs_i8_conversion).any()):
            result = result._convert(datetime=True)
        else:
            result = result._convert(datetime=True)
        if (not self.as_index):
            self._insert_inaxis_grouper_inplace(result)
        return self._reindex_output(result)

    def _transform_general(self, func, *args, **kwargs):
        from pandas.core.reshape.concat import concat
        applied = []
        obj = self._obj_with_exclusions
        gen = self.grouper.get_iterator(obj, axis=self.axis)
        (fast_path, slow_path) = self._define_paths(func, *args, **kwargs)
        for (name, group) in gen:
            object.__setattr__(group, 'name', name)
            try:
                (path, res) = self._choose_path(fast_path, slow_path, group)
            except TypeError:
                return self._transform_item_by_item(obj, fast_path)
            except ValueError as err:
                msg = 'transform must return a scalar value for each group'
                raise ValueError(msg) from err
            if isinstance(res, Series):
                if (not np.prod(group.shape)):
                    continue
                elif res.index.is_(obj.index):
                    r = concat(([res] * len(group.columns)), axis=1)
                    r.columns = group.columns
                    r.index = group.index
                else:
                    r = self.obj._constructor(np.concatenate(([res.values] * len(group.index))).reshape(group.shape), columns=group.columns, index=group.index)
                applied.append(r)
            else:
                applied.append(res)
        concat_index = (obj.columns if (self.axis == 0) else obj.index)
        other_axis = (1 if (self.axis == 0) else 0)
        concatenated = concat(applied, axis=self.axis, verify_integrity=False)
        concatenated = concatenated.reindex(concat_index, axis=other_axis, copy=False)
        return self._set_result_index_ordered(concatenated)

    @Substitution(klass='DataFrame')
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        if maybe_use_numba(engine):
            with group_selection_context(self):
                data = self._selected_obj
            result = self._transform_with_numba(data, func, *args, engine_kwargs=engine_kwargs, **kwargs)
            return self.obj._constructor(result, index=data.index, columns=data.columns)
        func = (self._get_cython_func(func) or func)
        if (not isinstance(func, str)):
            return self._transform_general(func, *args, **kwargs)
        elif (func not in base.transform_kernel_allowlist):
            msg = f"'{func}' is not a valid function name for transform(name)"
            raise ValueError(msg)
        elif ((func in base.cythonized_kernels) or (func in base.transformation_kernels)):
            return getattr(self, func)(*args, **kwargs)
        if (func in base.reduction_kernels):
            with com.temp_setattr(self, 'observed', True):
                result = getattr(self, func)(*args, **kwargs)
            if (isinstance(result, DataFrame) and result.columns.equals(self._obj_with_exclusions.columns)):
                return self._transform_fast(result)
        return self._transform_general(func, *args, **kwargs)

    def _transform_fast(self, result):
        '\n        Fast transform path for aggregations\n        '
        obj = self._obj_with_exclusions
        (ids, _, ngroup) = self.grouper.group_info
        result = result.reindex(self.grouper.result_index, copy=False)
        output = [algorithms.take_1d(result.iloc[:, i].values, ids) for (i, _) in enumerate(result.columns)]
        return self.obj._constructor._from_arrays(output, columns=result.columns, index=obj.index)

    def _define_paths(self, func, *args, **kwargs):
        if isinstance(func, str):
            fast_path = (lambda group: getattr(group, func)(*args, **kwargs))
            slow_path = (lambda group: group.apply((lambda x: getattr(x, func)(*args, **kwargs)), axis=self.axis))
        else:
            fast_path = (lambda group: func(group, *args, **kwargs))
            slow_path = (lambda group: group.apply((lambda x: func(x, *args, **kwargs)), axis=self.axis))
        return (fast_path, slow_path)

    def _choose_path(self, fast_path, slow_path, group):
        path = slow_path
        res = slow_path(group)
        try:
            res_fast = fast_path(group)
        except AssertionError:
            raise
        except Exception:
            return (path, res)
        if (not isinstance(res_fast, DataFrame)):
            return (path, res)
        if (not res_fast.columns.equals(group.columns)):
            return (path, res)
        if res_fast.equals(res):
            path = fast_path
        return (path, res)

    def _transform_item_by_item(self, obj, wrapper):
        output = {}
        inds = []
        for (i, col) in enumerate(obj):
            try:
                output[col] = self[col].transform(wrapper)
            except TypeError:
                pass
            else:
                inds.append(i)
        if (not output):
            raise TypeError('Transform function invalid for data types')
        columns = obj.columns
        if (len(output) < len(obj.columns)):
            columns = columns.take(inds)
        return self.obj._constructor(output, index=obj.index, columns=columns)

    def filter(self, func, dropna=True, *args, **kwargs):
        "\n        Return a copy of a DataFrame excluding filtered elements.\n\n        Elements from groups are filtered if they do not satisfy the\n        boolean criterion specified by func.\n\n        Parameters\n        ----------\n        func : function\n            Function to apply to each subframe. Should return True or False.\n        dropna : Drop groups that do not pass the filter. True by default;\n            If False, groups that evaluate False are filled with NaNs.\n\n        Returns\n        -------\n        filtered : DataFrame\n\n        Notes\n        -----\n        Each subframe is endowed the attribute 'name' in case you need to know\n        which group you are working on.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',\n        ...                           'foo', 'bar'],\n        ...                    'B' : [1, 2, 3, 4, 5, 6],\n        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})\n        >>> grouped = df.groupby('A')\n        >>> grouped.filter(lambda x: x['B'].mean() > 3.)\n             A  B    C\n        1  bar  2  5.0\n        3  bar  4  1.0\n        5  bar  6  9.0\n        "
        indices = []
        obj = self._selected_obj
        gen = self.grouper.get_iterator(obj, axis=self.axis)
        for (name, group) in gen:
            object.__setattr__(group, 'name', name)
            res = func(group, *args, **kwargs)
            try:
                res = res.squeeze()
            except AttributeError:
                pass
            if (is_bool(res) or (is_scalar(res) and isna(res))):
                if (res and notna(res)):
                    indices.append(self._get_index(name))
            else:
                raise TypeError(f'filter function returned a {type(res).__name__}, but expected a scalar bool')
        return self._apply_filter(indices, dropna)

    def __getitem__(self, key):
        if (self.axis == 1):
            raise ValueError('Cannot subset columns when using axis=1')
        if (isinstance(key, tuple) and (len(key) > 1)):
            warnings.warn('Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.', FutureWarning, stacklevel=2)
        return super().__getitem__(key)

    def _gotitem(self, key, ndim, subset=None):
        '\n        sub-classes to define\n        return a sliced object\n\n        Parameters\n        ----------\n        key : string / list of selections\n        ndim : {1, 2}\n            requested ndim of result\n        subset : object, default None\n            subset to act on\n        '
        if (ndim == 2):
            if (subset is None):
                subset = self.obj
            return DataFrameGroupBy(subset, self.grouper, axis=self.axis, level=self.level, grouper=self.grouper, exclusions=self.exclusions, selection=key, as_index=self.as_index, sort=self.sort, group_keys=self.group_keys, squeeze=self.squeeze, observed=self.observed, mutated=self.mutated, dropna=self.dropna)
        elif (ndim == 1):
            if (subset is None):
                subset = self.obj[key]
            return SeriesGroupBy(subset, level=self.level, grouper=self.grouper, selection=key, sort=self.sort, group_keys=self.group_keys, squeeze=self.squeeze, observed=self.observed, dropna=self.dropna)
        raise AssertionError('invalid ndim for _gotitem')

    def _wrap_frame_output(self, result, obj):
        result_index = self.grouper.levels[0]
        if (self.axis == 0):
            return self.obj._constructor(result, index=obj.columns, columns=result_index).T
        else:
            return self.obj._constructor(result, index=obj.index, columns=result_index)

    def _get_data_to_aggregate(self):
        obj = self._obj_with_exclusions
        if (self.axis == 1):
            return obj.T._mgr
        else:
            return obj._mgr

    def _insert_inaxis_grouper_inplace(self, result):
        columns = result.columns
        for (name, lev, in_axis) in zip(reversed(self.grouper.names), reversed(self.grouper.get_group_levels()), reversed([grp.in_axis for grp in self.grouper.groupings])):
            if (in_axis and (name not in columns)):
                result.insert(0, name, lev)

    def _wrap_aggregated_output(self, output, index):
        '\n        Wraps the output of DataFrameGroupBy aggregations into the expected result.\n\n        Parameters\n        ----------\n        output : Mapping[base.OutputKey, Union[Series, np.ndarray]]\n           Data to wrap.\n\n        Returns\n        -------\n        DataFrame\n        '
        indexed_output = {key.position: val for (key, val) in output.items()}
        columns = Index([key.label for key in output])
        columns._set_names(self._obj_with_exclusions._get_axis((1 - self.axis)).names)
        result = self.obj._constructor(indexed_output)
        result.columns = columns
        if (not self.as_index):
            self._insert_inaxis_grouper_inplace(result)
            result = result._consolidate()
        else:
            result.index = self.grouper.result_index
        if (self.axis == 1):
            result = result.T
        return self._reindex_output(result)

    def _wrap_transformed_output(self, output):
        '\n        Wraps the output of DataFrameGroupBy transformations into the expected result.\n\n        Parameters\n        ----------\n        output : Mapping[base.OutputKey, Union[Series, np.ndarray]]\n            Data to wrap.\n\n        Returns\n        -------\n        DataFrame\n        '
        indexed_output = {key.position: val for (key, val) in output.items()}
        result = self.obj._constructor(indexed_output)
        if (self.axis == 1):
            result = result.T
            result.columns = self.obj.columns
        else:
            columns = Index((key.label for key in output))
            columns.name = self.obj.columns.name
            result.columns = columns
        result.index = self.obj.index
        return result

    def _wrap_agged_blocks(self, blocks, items):
        if (not self.as_index):
            index = np.arange(blocks[0].values.shape[(- 1)])
            mgr = BlockManager(blocks, axes=[items, index])
            result = self.obj._constructor(mgr)
            self._insert_inaxis_grouper_inplace(result)
            result = result._consolidate()
        else:
            index = self.grouper.result_index
            mgr = BlockManager(blocks, axes=[items, index])
            result = self.obj._constructor(mgr)
        if (self.axis == 1):
            result = result.T
        return self._reindex_output(result)._convert(datetime=True)

    def _iterate_column_groupbys(self):
        for (i, colname) in enumerate(self._selected_obj.columns):
            (yield (colname, SeriesGroupBy(self._selected_obj.iloc[:, i], selection=colname, grouper=self.grouper, exclusions=self.exclusions)))

    def _apply_to_column_groupbys(self, func):
        from pandas.core.reshape.concat import concat
        return concat((func(col_groupby) for (_, col_groupby) in self._iterate_column_groupbys()), keys=self._selected_obj.columns, axis=1)

    def count(self):
        '\n        Compute count of group, excluding missing values.\n\n        Returns\n        -------\n        DataFrame\n            Count of values within each group.\n        '
        data = self._get_data_to_aggregate()
        (ids, _, ngroups) = self.grouper.group_info
        mask = (ids != (- 1))

        def hfunc(bvalues: ArrayLike) -> ArrayLike:
            if (bvalues.ndim == 1):
                masked = (mask & (~ isna(bvalues).reshape(1, (- 1))))
            else:
                masked = (mask & (~ isna(bvalues)))
            counted = lib.count_level_2d(masked, labels=ids, max_bin=ngroups, axis=1)
            return counted
        new_mgr = data.apply(hfunc)
        with com.temp_setattr(self, 'observed', True):
            result = self._wrap_agged_blocks(new_mgr.blocks, items=data.items)
        return self._reindex_output(result, fill_value=0)

    def nunique(self, dropna=True):
        "\n        Return DataFrame with counts of unique elements in each position.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include NaN in the counts.\n\n        Returns\n        -------\n        nunique: DataFrame\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'id': ['spam', 'egg', 'egg', 'spam',\n        ...                           'ham', 'ham'],\n        ...                    'value1': [1, 5, 5, 2, 5, 5],\n        ...                    'value2': list('abbaxy')})\n        >>> df\n             id  value1 value2\n        0  spam       1      a\n        1   egg       5      b\n        2   egg       5      b\n        3  spam       2      a\n        4   ham       5      x\n        5   ham       5      y\n\n        >>> df.groupby('id').nunique()\n              value1  value2\n        id\n        egg        1       1\n        ham        1       2\n        spam       2       1\n\n        Check for rows with the same id but conflicting values:\n\n        >>> df.groupby('id').filter(lambda g: (g.nunique() > 1).any())\n             id  value1 value2\n        0  spam       1      a\n        3  spam       2      a\n        4   ham       5      x\n        5   ham       5      y\n        "
        from pandas.core.reshape.concat import concat
        obj = self._obj_with_exclusions
        axis_number = obj._get_axis_number(self.axis)
        other_axis = int((not axis_number))
        if (axis_number == 0):
            iter_func = obj.items
        else:
            iter_func = obj.iterrows
        results = concat([SeriesGroupBy(content, selection=label, grouper=self.grouper).nunique(dropna) for (label, content) in iter_func()], axis=1)
        results = cast(DataFrame, results)
        if (axis_number == 1):
            results = results.T
        results._get_axis(other_axis).names = obj._get_axis(other_axis).names
        if (not self.as_index):
            results.index = ibase.default_index(len(results))
            self._insert_inaxis_grouper_inplace(results)
        return results

    @Appender(DataFrame.idxmax.__doc__)
    def idxmax(self, axis=0, skipna=True):
        axis = DataFrame._get_axis_number(axis)
        numeric_only = (None if (axis == 0) else False)

        def func(df):
            res = df._reduce(nanops.nanargmax, 'argmax', axis=axis, skipna=skipna, numeric_only=numeric_only)
            indices = res._values
            index = df._get_axis(axis)
            result = [(index[i] if (i >= 0) else np.nan) for i in indices]
            return df._constructor_sliced(result, index=res.index)
        return self._python_apply_general(func, self._obj_with_exclusions)

    @Appender(DataFrame.idxmin.__doc__)
    def idxmin(self, axis=0, skipna=True):
        axis = DataFrame._get_axis_number(axis)
        numeric_only = (None if (axis == 0) else False)

        def func(df):
            res = df._reduce(nanops.nanargmin, 'argmin', axis=axis, skipna=skipna, numeric_only=numeric_only)
            indices = res._values
            index = df._get_axis(axis)
            result = [(index[i] if (i >= 0) else np.nan) for i in indices]
            return df._constructor_sliced(result, index=res.index)
        return self._python_apply_general(func, self._obj_with_exclusions)
    boxplot = boxplot_frame_groupby
