
import itertools
from typing import List, Optional, Union
import numpy as np
import pandas._libs.algos as libalgos
import pandas._libs.reshape as libreshape
from pandas._libs.sparse import IntIndex
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import ensure_platform_int, is_bool_dtype, is_extension_array_dtype, is_integer, is_integer_dtype, is_list_like, is_object_dtype, needs_i8_conversion
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index, MultiIndex
from pandas.core.series import Series
from pandas.core.sorting import compress_group_index, decons_obs_group_ids, get_compressed_ids, get_group_index

class _Unstacker():
    '\n    Helper class to unstack data / pivot with multi-level index\n\n    Parameters\n    ----------\n    index : MultiIndex\n    level : int or str, default last level\n        Level to "unstack". Accepts a name for the level.\n    fill_value : scalar, optional\n        Default value to fill in missing values if subgroups do not have the\n        same set of labels. By default, missing values will be replaced with\n        the default fill value for that data type, NaN for float, NaT for\n        datetimelike, etc. For integer types, by default data will converted to\n        float and missing values will be set to NaN.\n    constructor : object\n        Pandas ``DataFrame`` or subclass used to create unstacked\n        response.  If None, DataFrame will be used.\n\n    Examples\n    --------\n    >>> index = pd.MultiIndex.from_tuples([(\'one\', \'a\'), (\'one\', \'b\'),\n    ...                                    (\'two\', \'a\'), (\'two\', \'b\')])\n    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)\n    >>> s\n    one  a    1\n         b    2\n    two  a    3\n         b    4\n    dtype: int64\n\n    >>> s.unstack(level=-1)\n         a  b\n    one  1  2\n    two  3  4\n\n    >>> s.unstack(level=0)\n       one  two\n    a    1    3\n    b    2    4\n\n    Returns\n    -------\n    unstacked : DataFrame\n    '

    def __init__(self, index, level=(- 1), constructor=None):
        if (constructor is None):
            constructor = DataFrame
        self.constructor = constructor
        self.index = index.remove_unused_levels()
        self.level = self.index._get_level_number(level)
        self.lift = (1 if ((- 1) in self.index.codes[self.level]) else 0)
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)
        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]
        num_rows = np.max([index_level.size for index_level in self.new_index_levels])
        num_columns = self.removed_level.size
        num_cells = np.multiply(num_rows, num_columns, dtype=np.int32)
        if ((num_rows > 0) and (num_columns > 0) and (num_cells <= 0)):
            raise ValueError('Unstacked DataFrame is too big, causing int32 overflow')
        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(self):
        v = self.level
        codes = list(self.index.codes)
        levs = list(self.index.levels)
        to_sort = ((codes[:v] + codes[(v + 1):]) + [codes[v]])
        sizes = [len(x) for x in ((levs[:v] + levs[(v + 1):]) + [levs[v]])]
        (comp_index, obs_ids) = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)
        indexer = libalgos.groupsort_indexer(comp_index, ngroups)[0]
        indexer = ensure_platform_int(indexer)
        return (indexer, to_sort)

    @cache_readonly
    def sorted_labels(self):
        (indexer, to_sort) = self._indexer_and_to_sort
        return [line.take(indexer) for line in to_sort]

    def _make_sorted_values(self, values):
        (indexer, _) = self._indexer_and_to_sort
        sorted_values = algos.take_nd(values, indexer, axis=0)
        return sorted_values

    def _make_selectors(self):
        new_levels = self.new_index_levels
        remaining_labels = self.sorted_labels[:(- 1)]
        level_sizes = [len(x) for x in new_levels]
        (comp_index, obs_ids) = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)
        comp_index = ensure_platform_int(comp_index)
        stride = (self.index.levshape[self.level] + self.lift)
        self.full_shape = (ngroups, stride)
        selector = ((self.sorted_labels[(- 1)] + (stride * comp_index)) + self.lift)
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)
        if (mask.sum() < len(self.index)):
            raise ValueError('Index contains duplicate entries, cannot reshape')
        self.group_index = comp_index
        self.mask = mask
        self.unique_groups = obs_ids
        self.compressor = comp_index.searchsorted(np.arange(ngroups))

    def get_result(self, values, value_columns, fill_value):
        if (values.ndim == 1):
            values = values[:, np.newaxis]
        if ((value_columns is None) and (values.shape[1] != 1)):
            raise ValueError('must pass column labels for multi-column data')
        (values, _) = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index
        return self.constructor(values, index=index, columns=columns)

    def get_new_values(self, values, fill_value=None):
        if (values.ndim == 1):
            values = values[:, np.newaxis]
        sorted_values = self._make_sorted_values(values)
        (length, width) = self.full_shape
        stride = values.shape[1]
        result_width = (width * stride)
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = mask.all()
        if (mask_all and len(values)):
            new_values = sorted_values.reshape(length, width, stride).swapaxes(1, 2).reshape(result_shape)
            new_mask = np.ones(result_shape, dtype=bool)
            return (new_values, new_mask)
        if mask_all:
            dtype = values.dtype
            new_values = np.empty(result_shape, dtype=dtype)
        else:
            (dtype, fill_value) = maybe_promote(values.dtype, fill_value)
            new_values = np.empty(result_shape, dtype=dtype)
            new_values.fill(fill_value)
        new_mask = np.zeros(result_shape, dtype=bool)
        name = np.dtype(dtype).name
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view('i8')
            new_values = new_values.view('i8')
        elif is_bool_dtype(values.dtype):
            sorted_values = sorted_values.astype('object')
            new_values = new_values.astype('object')
        else:
            sorted_values = sorted_values.astype(name, copy=False)
        libreshape.unstack(sorted_values, mask.view('u1'), stride, length, width, new_values, new_mask.view('u1'))
        if needs_i8_conversion(values.dtype):
            new_values = new_values.view(values.dtype)
        return (new_values, new_mask)

    def get_new_columns(self, value_columns):
        if (value_columns is None):
            if (self.lift == 0):
                return self.removed_level._shallow_copy(name=self.removed_name)
            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)
        stride = (len(self.removed_level) + self.lift)
        width = len(value_columns)
        propagator = np.repeat(np.arange(width), stride)
        if isinstance(value_columns, MultiIndex):
            new_levels = (value_columns.levels + (self.removed_level_full,))
            new_names = (value_columns.names + (self.removed_name,))
            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [value_columns, self.removed_level_full]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]
        if (len(self.removed_level_full) != len(self.removed_level)):
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.lift:
                repeater = np.insert(repeater, 0, (- 1))
        else:
            repeater = (np.arange(stride) - self.lift)
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    @cache_readonly
    def new_index(self):
        result_codes = [lab.take(self.compressor) for lab in self.sorted_labels[:(- 1)]]
        if (len(self.new_index_levels) == 1):
            (level, level_codes) = (self.new_index_levels[0], result_codes[0])
            if (level_codes == (- 1)).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])
        return MultiIndex(levels=self.new_index_levels, codes=result_codes, names=self.new_index_names, verify_integrity=False)

def _unstack_multiple(data, clocs, fill_value=None):
    if (len(clocs) == 0):
        return data
    index = data.index
    if (clocs in index.names):
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]
    rlocs = [i for i in range(index.nlevels) if (i not in clocs)]
    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]
    shape = [len(x) for x in clevels]
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)
    (comp_ids, obs_ids) = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)
    if (not rlocs):
        dummy_index = Index(obs_ids, name='__placeholder__')
    else:
        dummy_index = MultiIndex(levels=(rlevels + [obs_ids]), codes=(rcodes + [comp_ids]), names=(rnames + ['__placeholder__']), verify_integrity=False)
    if isinstance(data, Series):
        dummy = data.copy()
        dummy.index = dummy_index
        unstacked = dummy.unstack('__placeholder__', fill_value=fill_value)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            for i in range(len(clocs)):
                val = clocs[i]
                result = result.unstack(val, fill_value=fill_value)
                clocs = [(v if (v < val) else (v - 1)) for v in clocs]
            return result
        dummy = data.copy()
        dummy.index = dummy_index
        unstacked = dummy.unstack('__placeholder__', fill_value=fill_value)
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        assert isinstance(unstcols, MultiIndex)
        new_levels = ([unstcols.levels[0]] + clevels)
        new_names = ([data.columns.name] + cnames)
        new_codes = [unstcols.codes[0]]
        for rec in recons_codes:
            new_codes.append(rec.take(unstcols.codes[(- 1)]))
    new_columns = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns
    return unstacked

def unstack(obj, level, fill_value=None):
    if isinstance(level, (tuple, list)):
        if (len(level) != 1):
            return _unstack_multiple(obj, level, fill_value=fill_value)
        else:
            level = level[0]
    if ((not is_integer(level)) and (not (level == '__placeholder__'))):
        level = obj.index._get_level_number(level)
    if isinstance(obj, DataFrame):
        if isinstance(obj.index, MultiIndex):
            return _unstack_frame(obj, level, fill_value=fill_value)
        else:
            return obj.T.stack(dropna=False)
    elif (not isinstance(obj.index, MultiIndex)):
        raise ValueError(f'index must be a MultiIndex to unstack, {type(obj.index)} was passed')
    else:
        if is_extension_array_dtype(obj.dtype):
            return _unstack_extension_series(obj, level, fill_value)
        unstacker = _Unstacker(obj.index, level=level, constructor=obj._constructor_expanddim)
        return unstacker.get_result(obj.values, value_columns=None, fill_value=fill_value)

def _unstack_frame(obj, level, fill_value=None):
    if (not obj._can_fast_transpose):
        unstacker = _Unstacker(obj.index, level=level)
        mgr = obj._mgr.unstack(unstacker, fill_value=fill_value)
        return obj._constructor(mgr)
    else:
        return _Unstacker(obj.index, level=level, constructor=obj._constructor).get_result(obj._values, value_columns=obj.columns, fill_value=fill_value)

def _unstack_extension_series(series, level, fill_value):
    '\n    Unstack an ExtensionArray-backed Series.\n\n    The ExtensionDtype is preserved.\n\n    Parameters\n    ----------\n    series : Series\n        A Series with an ExtensionArray for values\n    level : Any\n        The level name or number.\n    fill_value : Any\n        The user-level (not physical storage) fill value to use for\n        missing values introduced by the reshape. Passed to\n        ``series.values.take``.\n\n    Returns\n    -------\n    DataFrame\n        Each column of the DataFrame will have the same dtype as\n        the input Series.\n    '
    df = series.to_frame()
    result = df.unstack(level=level, fill_value=fill_value)
    return result.droplevel(level=0, axis=1)

def stack(frame, level=(- 1), dropna=True):
    '\n    Convert DataFrame to Series with multi-level Index. Columns become the\n    second level of the resulting hierarchical index\n\n    Returns\n    -------\n    stacked : Series\n    '

    def factorize(index):
        if index.is_unique:
            return (index, np.arange(len(index)))
        (codes, categories) = factorize_from_iterable(index)
        return (categories, codes)
    (N, K) = frame.shape
    level_num = frame.columns._get_level_number(level)
    if isinstance(frame.columns, MultiIndex):
        return _stack_multi_columns(frame, level_num=level_num, dropna=dropna)
    elif isinstance(frame.index, MultiIndex):
        new_levels = list(frame.index.levels)
        new_codes = [lab.repeat(K) for lab in frame.index.codes]
        (clev, clab) = factorize(frame.columns)
        new_levels.append(clev)
        new_codes.append(np.tile(clab, N).ravel())
        new_names = list(frame.index.names)
        new_names.append(frame.columns.name)
        new_index = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    else:
        (levels, (ilab, clab)) = zip(*map(factorize, (frame.index, frame.columns)))
        codes = (ilab.repeat(K), np.tile(clab, N).ravel())
        new_index = MultiIndex(levels=levels, codes=codes, names=[frame.index.name, frame.columns.name], verify_integrity=False)
    if ((not frame.empty) and frame._is_homogeneous_type):
        dtypes = list(frame.dtypes._values)
        dtype = dtypes[0]
        if is_extension_array_dtype(dtype):
            arr = dtype.construct_array_type()
            new_values = arr._concat_same_type([col._values for (_, col) in frame.items()])
            new_values = _reorder_for_extension_array_stack(new_values, N, K)
        else:
            new_values = frame._values.ravel()
    else:
        new_values = frame._values.ravel()
    if dropna:
        mask = notna(new_values)
        new_values = new_values[mask]
        new_index = new_index[mask]
    return frame._constructor_sliced(new_values, index=new_index)

def stack_multiple(frame, level, dropna=True):
    if all(((lev in frame.columns.names) for lev in level)):
        result = frame
        for lev in level:
            result = stack(result, lev, dropna=dropna)
    elif all((isinstance(lev, int) for lev in level)):
        result = frame
        level = [frame.columns._get_level_number(lev) for lev in level]
        for index in range(len(level)):
            lev = level[index]
            result = stack(result, lev, dropna=dropna)
            updated_level = []
            for other in level:
                if (other > lev):
                    updated_level.append((other - 1))
                else:
                    updated_level.append(other)
            level = updated_level
    else:
        raise ValueError('level should contain all level names or all level numbers, not a mixture of the two.')
    return result

def _stack_multi_columns(frame, level_num=(- 1), dropna=True):

    def _convert_level_number(level_num, columns):
        '\n        Logic for converting the level number to something we can safely pass\n        to swaplevel.\n\n        If `level_num` matches a column name return the name from\n        position `level_num`, otherwise return `level_num`.\n        '
        if (level_num in columns.names):
            return columns.names[level_num]
        return level_num
    this = frame.copy()
    if (level_num != (frame.columns.nlevels - 1)):
        roll_columns = this.columns
        for i in range(level_num, (frame.columns.nlevels - 1)):
            lev1 = _convert_level_number(i, roll_columns)
            lev2 = _convert_level_number((i + 1), roll_columns)
            roll_columns = roll_columns.swaplevel(lev1, lev2)
        this.columns = roll_columns
    if (not this.columns.is_lexsorted()):
        level_to_sort = _convert_level_number(0, this.columns)
        this = this.sort_index(level=level_to_sort, axis=1)
    if (len(frame.columns.levels) > 2):
        tuples = list(zip(*[lev.take(level_codes) for (lev, level_codes) in zip(this.columns.levels[:(- 1)], this.columns.codes[:(- 1)])]))
        unique_groups = [key for (key, _) in itertools.groupby(tuples)]
        new_names = this.columns.names[:(- 1)]
        new_columns = MultiIndex.from_tuples(unique_groups, names=new_names)
    else:
        new_columns = this.columns.levels[0]._shallow_copy(name=this.columns.names[0])
        unique_groups = new_columns
    new_data = {}
    level_vals = this.columns.levels[(- 1)]
    level_codes = sorted(set(this.columns.codes[(- 1)]))
    level_vals_used = level_vals[level_codes]
    levsize = len(level_codes)
    drop_cols = []
    for key in unique_groups:
        try:
            loc = this.columns.get_loc(key)
        except KeyError:
            drop_cols.append(key)
            continue
        if (not isinstance(loc, slice)):
            slice_len = len(loc)
        else:
            slice_len = (loc.stop - loc.start)
        if (slice_len != levsize):
            chunk = this.loc[:, this.columns[loc]]
            chunk.columns = level_vals.take(chunk.columns.codes[(- 1)])
            value_slice = chunk.reindex(columns=level_vals_used).values
        elif (frame._is_homogeneous_type and is_extension_array_dtype(frame.dtypes.iloc[0])):
            dtype = this[this.columns[loc]].dtypes.iloc[0]
            subset = this[this.columns[loc]]
            value_slice = dtype.construct_array_type()._concat_same_type([x._values for (_, x) in subset.items()])
            (N, K) = this.shape
            idx = np.arange((N * K)).reshape(K, N).T.ravel()
            value_slice = value_slice.take(idx)
        elif frame._is_mixed_type:
            value_slice = this[this.columns[loc]].values
        else:
            value_slice = this.values[:, loc]
        if (value_slice.ndim > 1):
            value_slice = value_slice.ravel()
        new_data[key] = value_slice
    if (len(drop_cols) > 0):
        new_columns = new_columns.difference(drop_cols)
    N = len(this)
    if isinstance(this.index, MultiIndex):
        new_levels = list(this.index.levels)
        new_names = list(this.index.names)
        new_codes = [lab.repeat(levsize) for lab in this.index.codes]
    else:
        (old_codes, old_levels) = factorize_from_iterable(this.index)
        new_levels = [old_levels]
        new_codes = [old_codes.repeat(levsize)]
        new_names = [this.index.name]
    new_levels.append(level_vals)
    new_codes.append(np.tile(level_codes, N))
    new_names.append(frame.columns.names[level_num])
    new_index = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    result = frame._constructor(new_data, index=new_index, columns=new_columns)
    if dropna:
        result = result.dropna(axis=0, how='all')
    return result

def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None):
    "\n    Convert categorical variable into dummy/indicator variables.\n\n    Parameters\n    ----------\n    data : array-like, Series, or DataFrame\n        Data of which to get dummy indicators.\n    prefix : str, list of str, or dict of str, default None\n        String to append DataFrame column names.\n        Pass a list with length equal to the number of columns\n        when calling get_dummies on a DataFrame. Alternatively, `prefix`\n        can be a dictionary mapping column names to prefixes.\n    prefix_sep : str, default '_'\n        If appending prefix, separator/delimiter to use. Or pass a\n        list or dictionary as with `prefix`.\n    dummy_na : bool, default False\n        Add a column to indicate NaNs, if False NaNs are ignored.\n    columns : list-like, default None\n        Column names in the DataFrame to be encoded.\n        If `columns` is None then all the columns with\n        `object` or `category` dtype will be converted.\n    sparse : bool, default False\n        Whether the dummy-encoded columns should be backed by\n        a :class:`SparseArray` (True) or a regular NumPy array (False).\n    drop_first : bool, default False\n        Whether to get k-1 dummies out of k categorical levels by removing the\n        first level.\n    dtype : dtype, default np.uint8\n        Data type for new columns. Only a single dtype is allowed.\n\n    Returns\n    -------\n    DataFrame\n        Dummy-coded data.\n\n    See Also\n    --------\n    Series.str.get_dummies : Convert Series to dummy codes.\n\n    Examples\n    --------\n    >>> s = pd.Series(list('abca'))\n\n    >>> pd.get_dummies(s)\n       a  b  c\n    0  1  0  0\n    1  0  1  0\n    2  0  0  1\n    3  1  0  0\n\n    >>> s1 = ['a', 'b', np.nan]\n\n    >>> pd.get_dummies(s1)\n       a  b\n    0  1  0\n    1  0  1\n    2  0  0\n\n    >>> pd.get_dummies(s1, dummy_na=True)\n       a  b  NaN\n    0  1  0    0\n    1  0  1    0\n    2  0  0    1\n\n    >>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],\n    ...                    'C': [1, 2, 3]})\n\n    >>> pd.get_dummies(df, prefix=['col1', 'col2'])\n       C  col1_a  col1_b  col2_a  col2_b  col2_c\n    0  1       1       0       0       1       0\n    1  2       0       1       1       0       0\n    2  3       1       0       0       0       1\n\n    >>> pd.get_dummies(pd.Series(list('abcaa')))\n       a  b  c\n    0  1  0  0\n    1  0  1  0\n    2  0  0  1\n    3  1  0  0\n    4  1  0  0\n\n    >>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)\n       b  c\n    0  0  0\n    1  1  0\n    2  0  1\n    3  0  0\n    4  0  0\n\n    >>> pd.get_dummies(pd.Series(list('abc')), dtype=float)\n         a    b    c\n    0  1.0  0.0  0.0\n    1  0.0  1.0  0.0\n    2  0.0  0.0  1.0\n    "
    from pandas.core.reshape.concat import concat
    dtypes_to_encode = ['object', 'category']
    if isinstance(data, DataFrame):
        if (columns is None):
            data_to_encode = data.select_dtypes(include=dtypes_to_encode)
        elif (not is_list_like(columns)):
            raise TypeError('Input must be a list-like for parameter `columns`')
        else:
            data_to_encode = data[columns]

        def check_len(item, name):
            if is_list_like(item):
                if (not (len(item) == data_to_encode.shape[1])):
                    len_msg = f"Length of '{name}' ({len(item)}) did not match the length of the columns being encoded ({data_to_encode.shape[1]})."
                    raise ValueError(len_msg)
        check_len(prefix, 'prefix')
        check_len(prefix_sep, 'prefix_sep')
        if isinstance(prefix, str):
            prefix = itertools.cycle([prefix])
        if isinstance(prefix, dict):
            prefix = [prefix[col] for col in data_to_encode.columns]
        if (prefix is None):
            prefix = data_to_encode.columns
        if isinstance(prefix_sep, str):
            prefix_sep = itertools.cycle([prefix_sep])
        elif isinstance(prefix_sep, dict):
            prefix_sep = [prefix_sep[col] for col in data_to_encode.columns]
        with_dummies: List[DataFrame]
        if (data_to_encode.shape == data.shape):
            with_dummies = []
        elif (columns is not None):
            with_dummies = [data.drop(columns, axis=1)]
        else:
            with_dummies = [data.select_dtypes(exclude=dtypes_to_encode)]
        for (col, pre, sep) in zip(data_to_encode.items(), prefix, prefix_sep):
            dummy = _get_dummies_1d(col[1], prefix=pre, prefix_sep=sep, dummy_na=dummy_na, sparse=sparse, drop_first=drop_first, dtype=dtype)
            with_dummies.append(dummy)
        result = concat(with_dummies, axis=1)
    else:
        result = _get_dummies_1d(data, prefix, prefix_sep, dummy_na, sparse=sparse, drop_first=drop_first, dtype=dtype)
    return result

def _get_dummies_1d(data, prefix, prefix_sep='_', dummy_na=False, sparse=False, drop_first=False, dtype=None):
    from pandas.core.reshape.concat import concat
    (codes, levels) = factorize_from_iterable(Series(data))
    if (dtype is None):
        dtype = np.uint8
    dtype = np.dtype(dtype)
    if is_object_dtype(dtype):
        raise ValueError('dtype=object is not a valid dtype for get_dummies')

    def get_empty_frame(data) -> DataFrame:
        if isinstance(data, Series):
            index = data.index
        else:
            index = np.arange(len(data))
        return DataFrame(index=index)
    if ((not dummy_na) and (len(levels) == 0)):
        return get_empty_frame(data)
    codes = codes.copy()
    if dummy_na:
        codes[(codes == (- 1))] = len(levels)
        levels = np.append(levels, np.nan)
    if (drop_first and (len(levels) == 1)):
        return get_empty_frame(data)
    number_of_cols = len(levels)
    if (prefix is None):
        dummy_cols = levels
    else:
        dummy_cols = [f'{prefix}{prefix_sep}{level}' for level in levels]
    index: Optional[Index]
    if isinstance(data, Series):
        index = data.index
    else:
        index = None
    if sparse:
        fill_value: Union[(bool, float, int)]
        if is_integer_dtype(dtype):
            fill_value = 0
        elif (dtype == bool):
            fill_value = False
        else:
            fill_value = 0.0
        sparse_series = []
        N = len(data)
        sp_indices: List[List] = [[] for _ in range(len(dummy_cols))]
        mask = (codes != (- 1))
        codes = codes[mask]
        n_idx = np.arange(N)[mask]
        for (ndx, code) in zip(n_idx, codes):
            sp_indices[code].append(ndx)
        if drop_first:
            sp_indices = sp_indices[1:]
            dummy_cols = dummy_cols[1:]
        for (col, ixs) in zip(dummy_cols, sp_indices):
            sarr = SparseArray(np.ones(len(ixs), dtype=dtype), sparse_index=IntIndex(N, ixs), fill_value=fill_value, dtype=dtype)
            sparse_series.append(Series(data=sarr, index=index, name=col))
        out = concat(sparse_series, axis=1, copy=False)
        return out
    else:
        dummy_mat = np.eye(number_of_cols, dtype=dtype).take(codes, axis=0)
        if (not dummy_na):
            dummy_mat[(codes == (- 1))] = 0
        if drop_first:
            dummy_mat = dummy_mat[:, 1:]
            dummy_cols = dummy_cols[1:]
        return DataFrame(dummy_mat, index=index, columns=dummy_cols)

def _reorder_for_extension_array_stack(arr, n_rows, n_columns):
    "\n    Re-orders the values when stacking multiple extension-arrays.\n\n    The indirect stacking method used for EAs requires a followup\n    take to get the order correct.\n\n    Parameters\n    ----------\n    arr : ExtensionArray\n    n_rows, n_columns : int\n        The number of rows and columns in the original DataFrame.\n\n    Returns\n    -------\n    taken : ExtensionArray\n        The original `arr` with elements re-ordered appropriately\n\n    Examples\n    --------\n    >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])\n    >>> _reorder_for_extension_array_stack(arr, 2, 3)\n    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')\n\n    >>> _reorder_for_extension_array_stack(arr, 3, 2)\n    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')\n    "
    idx = np.arange((n_rows * n_columns)).reshape(n_columns, n_rows).T.ravel()
    return arr.take(idx)
