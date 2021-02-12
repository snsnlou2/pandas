
'\nConcat routines.\n'
from collections import abc
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Type, Union, cast, overload
import numpy as np
from pandas._typing import FrameOrSeriesUnion, Label
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import factorize_from_iterable, factorize_from_iterables
import pandas.core.common as com
from pandas.core.indexes.api import Index, MultiIndex, all_indexes_same, ensure_index, get_objs_combined_axis, get_unanimous_names
import pandas.core.indexes.base as ibase
from pandas.core.internals import concatenate_block_managers
if TYPE_CHECKING:
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame

@overload
def concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True):
    ...

@overload
def concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True):
    ...

def concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True):
    '\n    Concatenate pandas objects along a particular axis with optional set logic\n    along the other axes.\n\n    Can also add a layer of hierarchical indexing on the concatenation axis,\n    which may be useful if the labels are the same (or overlapping) on\n    the passed axis number.\n\n    Parameters\n    ----------\n    objs : a sequence or mapping of Series or DataFrame objects\n        If a mapping is passed, the sorted keys will be used as the `keys`\n        argument, unless it is passed, in which case the values will be\n        selected (see below). Any None objects will be dropped silently unless\n        they are all None in which case a ValueError will be raised.\n    axis : {0/\'index\', 1/\'columns\'}, default 0\n        The axis to concatenate along.\n    join : {\'inner\', \'outer\'}, default \'outer\'\n        How to handle indexes on other axis (or axes).\n    ignore_index : bool, default False\n        If True, do not use the index values along the concatenation axis. The\n        resulting axis will be labeled 0, ..., n - 1. This is useful if you are\n        concatenating objects where the concatenation axis does not have\n        meaningful indexing information. Note the index values on the other\n        axes are still respected in the join.\n    keys : sequence, default None\n        If multiple levels passed, should contain tuples. Construct\n        hierarchical index using the passed keys as the outermost level.\n    levels : list of sequences, default None\n        Specific levels (unique values) to use for constructing a\n        MultiIndex. Otherwise they will be inferred from the keys.\n    names : list, default None\n        Names for the levels in the resulting hierarchical index.\n    verify_integrity : bool, default False\n        Check whether the new concatenated axis contains duplicates. This can\n        be very expensive relative to the actual data concatenation.\n    sort : bool, default False\n        Sort non-concatenation axis if it is not already aligned when `join`\n        is \'outer\'.\n        This has no effect when ``join=\'inner\'``, which already preserves\n        the order of the non-concatenation axis.\n\n        .. versionchanged:: 1.0.0\n\n           Changed to not sort by default.\n\n    copy : bool, default True\n        If False, do not copy data unnecessarily.\n\n    Returns\n    -------\n    object, type of objs\n        When concatenating all ``Series`` along the index (axis=0), a\n        ``Series`` is returned. When ``objs`` contains at least one\n        ``DataFrame``, a ``DataFrame`` is returned. When concatenating along\n        the columns (axis=1), a ``DataFrame`` is returned.\n\n    See Also\n    --------\n    Series.append : Concatenate Series.\n    DataFrame.append : Concatenate DataFrames.\n    DataFrame.join : Join DataFrames using indexes.\n    DataFrame.merge : Merge DataFrames by indexes or columns.\n\n    Notes\n    -----\n    The keys, levels, and names arguments are all optional.\n\n    A walkthrough of how this method fits in with other tools for combining\n    pandas objects can be found `here\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>`__.\n\n    Examples\n    --------\n    Combine two ``Series``.\n\n    >>> s1 = pd.Series([\'a\', \'b\'])\n    >>> s2 = pd.Series([\'c\', \'d\'])\n    >>> pd.concat([s1, s2])\n    0    a\n    1    b\n    0    c\n    1    d\n    dtype: object\n\n    Clear the existing index and reset it in the result\n    by setting the ``ignore_index`` option to ``True``.\n\n    >>> pd.concat([s1, s2], ignore_index=True)\n    0    a\n    1    b\n    2    c\n    3    d\n    dtype: object\n\n    Add a hierarchical index at the outermost level of\n    the data with the ``keys`` option.\n\n    >>> pd.concat([s1, s2], keys=[\'s1\', \'s2\'])\n    s1  0    a\n        1    b\n    s2  0    c\n        1    d\n    dtype: object\n\n    Label the index keys you create with the ``names`` option.\n\n    >>> pd.concat([s1, s2], keys=[\'s1\', \'s2\'],\n    ...           names=[\'Series name\', \'Row ID\'])\n    Series name  Row ID\n    s1           0         a\n                 1         b\n    s2           0         c\n                 1         d\n    dtype: object\n\n    Combine two ``DataFrame`` objects with identical columns.\n\n    >>> df1 = pd.DataFrame([[\'a\', 1], [\'b\', 2]],\n    ...                    columns=[\'letter\', \'number\'])\n    >>> df1\n      letter  number\n    0      a       1\n    1      b       2\n    >>> df2 = pd.DataFrame([[\'c\', 3], [\'d\', 4]],\n    ...                    columns=[\'letter\', \'number\'])\n    >>> df2\n      letter  number\n    0      c       3\n    1      d       4\n    >>> pd.concat([df1, df2])\n      letter  number\n    0      a       1\n    1      b       2\n    0      c       3\n    1      d       4\n\n    Combine ``DataFrame`` objects with overlapping columns\n    and return everything. Columns outside the intersection will\n    be filled with ``NaN`` values.\n\n    >>> df3 = pd.DataFrame([[\'c\', 3, \'cat\'], [\'d\', 4, \'dog\']],\n    ...                    columns=[\'letter\', \'number\', \'animal\'])\n    >>> df3\n      letter  number animal\n    0      c       3    cat\n    1      d       4    dog\n    >>> pd.concat([df1, df3], sort=False)\n      letter  number animal\n    0      a       1    NaN\n    1      b       2    NaN\n    0      c       3    cat\n    1      d       4    dog\n\n    Combine ``DataFrame`` objects with overlapping columns\n    and return only those that are shared by passing ``inner`` to\n    the ``join`` keyword argument.\n\n    >>> pd.concat([df1, df3], join="inner")\n      letter  number\n    0      a       1\n    1      b       2\n    0      c       3\n    1      d       4\n\n    Combine ``DataFrame`` objects horizontally along the x axis by\n    passing in ``axis=1``.\n\n    >>> df4 = pd.DataFrame([[\'bird\', \'polly\'], [\'monkey\', \'george\']],\n    ...                    columns=[\'animal\', \'name\'])\n    >>> pd.concat([df1, df4], axis=1)\n      letter  number  animal    name\n    0      a       1    bird   polly\n    1      b       2  monkey  george\n\n    Prevent the result from including duplicate index values with the\n    ``verify_integrity`` option.\n\n    >>> df5 = pd.DataFrame([1], index=[\'a\'])\n    >>> df5\n       0\n    a  1\n    >>> df6 = pd.DataFrame([2], index=[\'a\'])\n    >>> df6\n       0\n    a  2\n    >>> pd.concat([df5, df6], verify_integrity=True)\n    Traceback (most recent call last):\n        ...\n    ValueError: Indexes have overlapping values: [\'a\']\n    '
    op = _Concatenator(objs, axis=axis, ignore_index=ignore_index, join=join, keys=keys, levels=levels, names=names, verify_integrity=verify_integrity, copy=copy, sort=sort)
    return op.get_result()

class _Concatenator():
    '\n    Orchestrates a concatenation operation for BlockManagers\n    '

    def __init__(self, objs, axis=0, join='outer', keys=None, levels=None, names=None, ignore_index=False, verify_integrity=False, copy=True, sort=False):
        if isinstance(objs, (ABCSeries, ABCDataFrame, str)):
            raise TypeError(f'first argument must be an iterable of pandas objects, you passed an object of type "{type(objs).__name__}"')
        if (join == 'outer'):
            self.intersect = False
        elif (join == 'inner'):
            self.intersect = True
        else:
            raise ValueError('Only can inner (intersect) or outer (union) join the other axis')
        if isinstance(objs, abc.Mapping):
            if (keys is None):
                keys = list(objs.keys())
            objs = [objs[k] for k in keys]
        else:
            objs = list(objs)
        if (len(objs) == 0):
            raise ValueError('No objects to concatenate')
        if (keys is None):
            objs = list(com.not_none(*objs))
        else:
            clean_keys = []
            clean_objs = []
            for (k, v) in zip(keys, objs):
                if (v is None):
                    continue
                clean_keys.append(k)
                clean_objs.append(v)
            objs = clean_objs
            name = getattr(keys, 'name', None)
            keys = Index(clean_keys, name=name)
        if (len(objs) == 0):
            raise ValueError('All objects passed were None')
        ndims = set()
        for obj in objs:
            if (not isinstance(obj, (ABCSeries, ABCDataFrame))):
                msg = f"cannot concatenate object of type '{type(obj)}'; only Series and DataFrame objs are valid"
                raise TypeError(msg)
            ndims.add(obj.ndim)
        sample: Optional['NDFrame'] = None
        if (len(ndims) > 1):
            max_ndim = max(ndims)
            for obj in objs:
                if ((obj.ndim == max_ndim) and np.sum(obj.shape)):
                    sample = obj
                    break
        else:
            non_empties = [obj for obj in objs if ((sum(obj.shape) > 0) or isinstance(obj, ABCSeries))]
            if (len(non_empties) and ((keys is None) and (names is None) and (levels is None) and (not self.intersect))):
                objs = non_empties
                sample = objs[0]
        if (sample is None):
            sample = objs[0]
        self.objs = objs
        if isinstance(sample, ABCSeries):
            axis = sample._constructor_expanddim._get_axis_number(axis)
        else:
            axis = sample._get_axis_number(axis)
        self._is_frame = isinstance(sample, ABCDataFrame)
        if self._is_frame:
            axis = sample._get_block_manager_axis(axis)
        self._is_series = isinstance(sample, ABCSeries)
        if (not (0 <= axis <= sample.ndim)):
            raise AssertionError(f'axis must be between 0 and {sample.ndim}, input was {axis}')
        if (len(ndims) > 1):
            current_column = 0
            max_ndim = sample.ndim
            (self.objs, objs) = ([], self.objs)
            for obj in objs:
                ndim = obj.ndim
                if (ndim == max_ndim):
                    pass
                elif (ndim != (max_ndim - 1)):
                    raise ValueError('cannot concatenate unaligned mixed dimensional NDFrame objects')
                else:
                    name = getattr(obj, 'name', None)
                    if (ignore_index or (name is None)):
                        name = current_column
                        current_column += 1
                    if (self._is_frame and (axis == 1)):
                        name = 0
                    sample = cast('FrameOrSeriesUnion', sample)
                    obj = sample._constructor({name: obj})
                self.objs.append(obj)
        self.bm_axis = axis
        self.axis = ((1 - self.bm_axis) if self._is_frame else 0)
        self.keys = keys
        self.names = (names or getattr(keys, 'names', None))
        self.levels = levels
        self.sort = sort
        self.ignore_index = ignore_index
        self.verify_integrity = verify_integrity
        self.copy = copy
        self.new_axes = self._get_new_axes()

    def get_result(self):
        cons: Type[FrameOrSeriesUnion]
        sample: FrameOrSeriesUnion
        if self._is_series:
            sample = cast('Series', self.objs[0])
            if (self.bm_axis == 0):
                name = com.consensus_name_attr(self.objs)
                cons = sample._constructor
                arrs = [ser._values for ser in self.objs]
                res = concat_compat(arrs, axis=0)
                result = cons(res, index=self.new_axes[0], name=name, dtype=res.dtype)
                return result.__finalize__(self, method='concat')
            else:
                data = dict(zip(range(len(self.objs)), self.objs))
                cons = sample._constructor_expanddim
                (index, columns) = self.new_axes
                df = cons(data, index=index)
                df.columns = columns
                return df.__finalize__(self, method='concat')
        else:
            sample = cast('DataFrame', self.objs[0])
            mgrs_indexers = []
            for obj in self.objs:
                indexers = {}
                for (ax, new_labels) in enumerate(self.new_axes):
                    if (ax == self.bm_axis):
                        continue
                    obj_labels = obj.axes[(1 - ax)]
                    if (not new_labels.equals(obj_labels)):
                        indexers[ax] = obj_labels.get_indexer(new_labels)
                mgrs_indexers.append((obj._mgr, indexers))
            new_data = concatenate_block_managers(mgrs_indexers, self.new_axes, concat_axis=self.bm_axis, copy=self.copy)
            if (not self.copy):
                new_data._consolidate_inplace()
            cons = sample._constructor
            return cons(new_data).__finalize__(self, method='concat')

    def _get_result_dim(self):
        if (self._is_series and (self.bm_axis == 1)):
            return 2
        else:
            return self.objs[0].ndim

    def _get_new_axes(self):
        ndim = self._get_result_dim()
        return [(self._get_concat_axis if (i == self.bm_axis) else self._get_comb_axis(i)) for i in range(ndim)]

    def _get_comb_axis(self, i):
        data_axis = self.objs[0]._get_block_manager_axis(i)
        return get_objs_combined_axis(self.objs, axis=data_axis, intersect=self.intersect, sort=self.sort, copy=self.copy)

    @cache_readonly
    def _get_concat_axis(self):
        '\n        Return index to be used along concatenation axis.\n        '
        if self._is_series:
            if (self.bm_axis == 0):
                indexes = [x.index for x in self.objs]
            elif self.ignore_index:
                idx = ibase.default_index(len(self.objs))
                return idx
            elif (self.keys is None):
                names: List[Label] = ([None] * len(self.objs))
                num = 0
                has_names = False
                for (i, x) in enumerate(self.objs):
                    if (not isinstance(x, ABCSeries)):
                        raise TypeError(f"Cannot concatenate type 'Series' with object of type '{type(x).__name__}'")
                    if (x.name is not None):
                        names[i] = x.name
                        has_names = True
                    else:
                        names[i] = num
                        num += 1
                if has_names:
                    return Index(names)
                else:
                    return ibase.default_index(len(self.objs))
            else:
                return ensure_index(self.keys).set_names(self.names)
        else:
            indexes = [x.axes[self.axis] for x in self.objs]
        if self.ignore_index:
            idx = ibase.default_index(sum((len(i) for i in indexes)))
            return idx
        if (self.keys is None):
            concat_axis = _concat_indexes(indexes)
        else:
            concat_axis = _make_concat_multiindex(indexes, self.keys, self.levels, self.names)
        self._maybe_check_integrity(concat_axis)
        return concat_axis

    def _maybe_check_integrity(self, concat_index):
        if self.verify_integrity:
            if (not concat_index.is_unique):
                overlap = concat_index[concat_index.duplicated()].unique()
                raise ValueError(f'Indexes have overlapping values: {overlap}')

def _concat_indexes(indexes):
    return indexes[0].append(indexes[1:])

def _make_concat_multiindex(indexes, keys, levels=None, names=None):
    if (((levels is None) and isinstance(keys[0], tuple)) or ((levels is not None) and (len(levels) > 1))):
        zipped = list(zip(*keys))
        if (names is None):
            names = ([None] * len(zipped))
        if (levels is None):
            (_, levels) = factorize_from_iterables(zipped)
        else:
            levels = [ensure_index(x) for x in levels]
    else:
        zipped = [keys]
        if (names is None):
            names = [None]
        if (levels is None):
            levels = [ensure_index(keys)]
        else:
            levels = [ensure_index(x) for x in levels]
    if (not all_indexes_same(indexes)):
        codes_list = []
        for (hlevel, level) in zip(zipped, levels):
            to_concat = []
            for (key, index) in zip(hlevel, indexes):
                mask = ((isna(level) & isna(key)) | (level == key))
                if (not mask.any()):
                    raise ValueError(f'Key {key} not in level {level}')
                i = np.nonzero(mask)[0][0]
                to_concat.append(np.repeat(i, len(index)))
            codes_list.append(np.concatenate(to_concat))
        concat_index = _concat_indexes(indexes)
        if isinstance(concat_index, MultiIndex):
            levels.extend(concat_index.levels)
            codes_list.extend(concat_index.codes)
        else:
            (codes, categories) = factorize_from_iterable(concat_index)
            levels.append(categories)
            codes_list.append(codes)
        if (len(names) == len(levels)):
            names = list(names)
        else:
            if (not (len({idx.nlevels for idx in indexes}) == 1)):
                raise AssertionError('Cannot concat indices that do not have the same number of levels')
            names = (list(names) + list(get_unanimous_names(*indexes)))
        return MultiIndex(levels=levels, codes=codes_list, names=names, verify_integrity=False)
    new_index = indexes[0]
    n = len(new_index)
    kpieces = len(indexes)
    new_names = list(names)
    new_levels = list(levels)
    new_codes = []
    for (hlevel, level) in zip(zipped, levels):
        hlevel = ensure_index(hlevel)
        mapped = level.get_indexer(hlevel)
        mask = (mapped == (- 1))
        if mask.any():
            raise ValueError(f'Values not found in passed level: {hlevel[mask]!s}')
        new_codes.append(np.repeat(mapped, n))
    if isinstance(new_index, MultiIndex):
        new_levels.extend(new_index.levels)
        new_codes.extend([np.tile(lab, kpieces) for lab in new_index.codes])
    else:
        new_levels.append(new_index)
        new_codes.append(np.tile(np.arange(n), kpieces))
    if (len(new_names) < len(new_levels)):
        new_names.extend(new_index.names)
    return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
