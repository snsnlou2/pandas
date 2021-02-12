
' miscellaneous sorting / groupby utilities '
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from pandas._libs import algos, hashtable, lib
from pandas._libs.hashtable import unique_label_indices
from pandas._typing import IndexKeyFunc
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int, is_extension_array_dtype
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.dtypes.missing import isna
import pandas.core.algorithms as algorithms
from pandas.core.construction import extract_array
if TYPE_CHECKING:
    from pandas import MultiIndex
    from pandas.core.indexes.base import Index
_INT64_MAX = np.iinfo(np.int64).max

def get_indexer_indexer(target, level, ascending, kind, na_position, sort_remaining, key):
    "\n    Helper method that return the indexer according to input parameters for\n    the sort_index method of DataFrame and Series.\n\n    Parameters\n    ----------\n    target : Index\n    level : int or level name or list of ints or list of level names\n    ascending : bool or list of bools, default True\n    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'\n    na_position : {'first', 'last'}, default 'last'\n    sort_remaining : bool, default True\n    key : callable, optional\n\n    Returns\n    -------\n    Optional[ndarray]\n        The indexer for the new index.\n    "
    target = ensure_key_mapped(target, key, levels=level)
    target = target._sort_levels_monotonic()
    if (level is not None):
        (_, indexer) = target.sortlevel(level, ascending=ascending, sort_remaining=sort_remaining)
    elif isinstance(target, ABCMultiIndex):
        indexer = lexsort_indexer(target._get_codes_for_sorting(), orders=ascending, na_position=na_position)
    else:
        if ((ascending and target.is_monotonic_increasing) or ((not ascending) and target.is_monotonic_decreasing)):
            return None
        indexer = nargsort(target, kind=kind, ascending=ascending, na_position=na_position)
    return indexer

def get_group_index(labels, shape, sort, xnull):
    "\n    For the particular label_list, gets the offsets into the hypothetical list\n    representing the totally ordered cartesian product of all possible label\n    combinations, *as long as* this space fits within int64 bounds;\n    otherwise, though group indices identify unique combinations of\n    labels, they cannot be deconstructed.\n    - If `sort`, rank of returned ids preserve lexical ranks of labels.\n      i.e. returned id's can be used to do lexical sort on labels;\n    - If `xnull` nulls (-1 labels) are passed through.\n\n    Parameters\n    ----------\n    labels : sequence of arrays\n        Integers identifying levels at each location\n    shape : sequence of ints\n        Number of unique levels at each location\n    sort : bool\n        If the ranks of returned ids should match lexical ranks of labels\n    xnull : bool\n        If true nulls are excluded. i.e. -1 values in the labels are\n        passed through.\n\n    Returns\n    -------\n    An array of type int64 where two elements are equal if their corresponding\n    labels are equal at all location.\n\n    Notes\n    -----\n    The length of `labels` and `shape` must be identical.\n    "

    def _int64_cut_off(shape) -> int:
        acc = 1
        for (i, mul) in enumerate(shape):
            acc *= int(mul)
            if (not (acc < _INT64_MAX)):
                return i
        return len(shape)

    def maybe_lift(lab, size):
        return (((lab + 1), (size + 1)) if (lab == (- 1)).any() else (lab, size))
    labels = map(ensure_int64, labels)
    if (not xnull):
        (labels, shape) = map(list, zip(*map(maybe_lift, labels, shape)))
    labels = list(labels)
    shape = list(shape)
    while True:
        nlev = _int64_cut_off(shape)
        stride = np.prod(shape[1:nlev], dtype='i8')
        out = (stride * labels[0].astype('i8', subok=False, copy=False))
        for i in range(1, nlev):
            if (shape[i] == 0):
                stride = 0
            else:
                stride //= shape[i]
            out += (labels[i] * stride)
        if xnull:
            mask = (labels[0] == (- 1))
            for lab in labels[1:nlev]:
                mask |= (lab == (- 1))
            out[mask] = (- 1)
        if (nlev == len(shape)):
            break
        (comp_ids, obs_ids) = compress_group_index(out, sort=sort)
        labels = ([comp_ids] + labels[nlev:])
        shape = ([len(obs_ids)] + shape[nlev:])
    return out

def get_compressed_ids(labels, sizes):
    '\n    Group_index is offsets into cartesian product of all possible labels. This\n    space can be huge, so this function compresses it, by computing offsets\n    (comp_ids) into the list of unique labels (obs_group_ids).\n\n    Parameters\n    ----------\n    labels : list of label arrays\n    sizes : list of size of the levels\n\n    Returns\n    -------\n    tuple of (comp_ids, obs_group_ids)\n    '
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)

def is_int64_overflow_possible(shape):
    the_prod = 1
    for x in shape:
        the_prod *= int(x)
    return (the_prod >= _INT64_MAX)

def decons_group_index(comp_labels, shape):
    if is_int64_overflow_possible(shape):
        raise ValueError('cannot deconstruct factorized group indices!')
    label_list = []
    factor = 1
    y = 0
    x = comp_labels
    for i in reversed(range(len(shape))):
        labels = (((x - y) % (factor * shape[i])) // factor)
        np.putmask(labels, (comp_labels < 0), (- 1))
        label_list.append(labels)
        y = (labels * factor)
        factor *= shape[i]
    return label_list[::(- 1)]

def decons_obs_group_ids(comp_ids, obs_ids, shape, labels, xnull):
    '\n    Reconstruct labels from observed group ids.\n\n    Parameters\n    ----------\n    xnull : bool\n        If nulls are excluded; i.e. -1 labels are passed through.\n    '
    if (not xnull):
        lift = np.fromiter(((a == (- 1)).any() for a in labels), dtype='i8')
        shape = (np.asarray(shape, dtype='i8') + lift)
    if (not is_int64_overflow_possible(shape)):
        out = decons_group_index(obs_ids, shape)
        return (out if (xnull or (not lift.any())) else [(x - y) for (x, y) in zip(out, lift)])
    i = unique_label_indices(comp_ids)
    i8copy = (lambda a: a.astype('i8', subok=False, copy=True))
    return [i8copy(lab[i]) for lab in labels]

def indexer_from_factorized(labels, shape, compress=True):
    ids = get_group_index(labels, shape, sort=True, xnull=False)
    if (not compress):
        ngroups = ((ids.size and ids.max()) + 1)
    else:
        (ids, obs) = compress_group_index(ids, sort=True)
        ngroups = len(obs)
    return get_group_index_sorter(ids, ngroups)

def lexsort_indexer(keys, orders=None, na_position='last', key=None):
    '\n    Performs lexical sorting on a set of keys\n\n    Parameters\n    ----------\n    keys : sequence of arrays\n        Sequence of ndarrays to be sorted by the indexer\n    orders : boolean or list of booleans, optional\n        Determines the sorting order for each element in keys. If a list,\n        it must be the same length as keys. This determines whether the\n        corresponding element in keys should be sorted in ascending\n        (True) or descending (False) order. if bool, applied to all\n        elements as above. if None, defaults to True.\n    na_position : {\'first\', \'last\'}, default \'last\'\n        Determines placement of NA elements in the sorted list ("last" or "first")\n    key : Callable, optional\n        Callable key function applied to every element in keys before sorting\n\n        .. versionadded:: 1.0.0\n    '
    from pandas.core.arrays import Categorical
    labels = []
    shape = []
    if isinstance(orders, bool):
        orders = ([orders] * len(keys))
    elif (orders is None):
        orders = ([True] * len(keys))
    keys = [ensure_key_mapped(k, key) for k in keys]
    for (k, order) in zip(keys, orders):
        cat = Categorical(k, ordered=True)
        if (na_position not in ['last', 'first']):
            raise ValueError(f'invalid na_position: {na_position}')
        n = len(cat.categories)
        codes = cat.codes.copy()
        mask = (cat.codes == (- 1))
        if order:
            if (na_position == 'last'):
                codes = np.where(mask, n, codes)
            elif (na_position == 'first'):
                codes += 1
        elif (na_position == 'last'):
            codes = np.where(mask, n, ((n - codes) - 1))
        elif (na_position == 'first'):
            codes = np.where(mask, 0, (n - codes))
        if mask.any():
            n += 1
        shape.append(n)
        labels.append(codes)
    return indexer_from_factorized(labels, shape)

def nargsort(items, kind='quicksort', ascending=True, na_position='last', key=None, mask=None):
    "\n    Intended to be a drop-in replacement for np.argsort which handles NaNs.\n\n    Adds ascending, na_position, and key parameters.\n\n    (GH #6399, #5231, #27237)\n\n    Parameters\n    ----------\n    kind : str, default 'quicksort'\n    ascending : bool, default True\n    na_position : {'first', 'last'}, default 'last'\n    key : Optional[Callable], default None\n    mask : Optional[np.ndarray], default None\n        Passed when called by ExtensionArray.argsort.\n    "
    if (key is not None):
        items = ensure_key_mapped(items, key)
        return nargsort(items, kind=kind, ascending=ascending, na_position=na_position, key=None, mask=mask)
    items = extract_array(items)
    if (mask is None):
        mask = np.asarray(isna(items))
    if is_extension_array_dtype(items):
        return items.argsort(ascending=ascending, kind=kind, na_position=na_position)
    else:
        items = np.asanyarray(items)
    idx = np.arange(len(items))
    non_nans = items[(~ mask)]
    non_nan_idx = idx[(~ mask)]
    nan_idx = np.nonzero(mask)[0]
    if (not ascending):
        non_nans = non_nans[::(- 1)]
        non_nan_idx = non_nan_idx[::(- 1)]
    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
    if (not ascending):
        indexer = indexer[::(- 1)]
    if (na_position == 'last'):
        indexer = np.concatenate([indexer, nan_idx])
    elif (na_position == 'first'):
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError(f'invalid na_position: {na_position}')
    return indexer

def nargminmax(values, method):
    '\n    Implementation of np.argmin/argmax but for ExtensionArray and which\n    handles missing values.\n\n    Parameters\n    ----------\n    values : ExtensionArray\n    method : {"argmax", "argmin"}\n\n    Returns\n    -------\n    int\n    '
    assert (method in {'argmax', 'argmin'})
    func = (np.argmax if (method == 'argmax') else np.argmin)
    mask = np.asarray(isna(values))
    values = values._values_for_argsort()
    idx = np.arange(len(values))
    non_nans = values[(~ mask)]
    non_nan_idx = idx[(~ mask)]
    return non_nan_idx[func(non_nans)]

def _ensure_key_mapped_multiindex(index, key, level=None):
    '\n    Returns a new MultiIndex in which key has been applied\n    to all levels specified in level (or all levels if level\n    is None). Used for key sorting for MultiIndex.\n\n    Parameters\n    ----------\n    index : MultiIndex\n        Index to which to apply the key function on the\n        specified levels.\n    key : Callable\n        Function that takes an Index and returns an Index of\n        the same shape. This key is applied to each level\n        separately. The name of the level can be used to\n        distinguish different levels for application.\n    level : list-like, int or str, default None\n        Level or list of levels to apply the key function to.\n        If None, key function is applied to all levels. Other\n        levels are left unchanged.\n\n    Returns\n    -------\n    labels : MultiIndex\n        Resulting MultiIndex with modified levels.\n    '
    if (level is not None):
        if isinstance(level, (str, int)):
            sort_levels = [level]
        else:
            sort_levels = level
        sort_levels = [index._get_level_number(lev) for lev in sort_levels]
    else:
        sort_levels = list(range(index.nlevels))
    mapped = [(ensure_key_mapped(index._get_level_values(level), key) if (level in sort_levels) else index._get_level_values(level)) for level in range(index.nlevels)]
    labels = type(index).from_arrays(mapped)
    return labels

def ensure_key_mapped(values, key, levels=None):
    '\n    Applies a callable key function to the values function and checks\n    that the resulting value has the same shape. Can be called on Index\n    subclasses, Series, DataFrames, or ndarrays.\n\n    Parameters\n    ----------\n    values : Series, DataFrame, Index subclass, or ndarray\n    key : Optional[Callable], key to be called on the values array\n    levels : Optional[List], if values is a MultiIndex, list of levels to\n    apply the key to.\n    '
    from pandas.core.indexes.api import Index
    if (not key):
        return values
    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)
    result = key(values.copy())
    if (len(result) != len(values)):
        raise ValueError('User-provided `key` function must not change the shape of the array.')
    try:
        if isinstance(values, Index):
            result = Index(result)
        else:
            type_of_values = type(values)
            result = type_of_values(result)
    except TypeError:
        raise TypeError(f'User-provided `key` function returned an invalid type {type(result)}             which could not be converted to {type(values)}.')
    return result

def get_flattened_list(comp_ids, ngroups, levels, labels):
    'Map compressed group id -> key tuple.'
    comp_ids = comp_ids.astype(np.int64, copy=False)
    arrays: DefaultDict[(int, List[int])] = defaultdict(list)
    for (labs, level) in zip(labels, levels):
        table = hashtable.Int64HashTable(ngroups)
        table.map(comp_ids, labs.astype(np.int64, copy=False))
        for i in range(ngroups):
            arrays[i].append(level[table.get_item(i)])
    return [tuple(array) for array in arrays.values()]

def get_indexer_dict(label_list, keys):
    '\n    Returns\n    -------\n    dict:\n        Labels mapped to indexers.\n    '
    shape = [len(x) for x in keys]
    group_index = get_group_index(label_list, shape, sort=True, xnull=True)
    if np.all((group_index == (- 1))):
        return {}
    ngroups = (((group_index.size and group_index.max()) + 1) if is_int64_overflow_possible(shape) else np.prod(shape, dtype='i8'))
    sorter = get_group_index_sorter(group_index, ngroups)
    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)
    return lib.indices_fast(sorter, group_index, keys, sorted_labels)

def get_group_index_sorter(group_index, ngroups):
    "\n    algos.groupsort_indexer implements `counting sort` and it is at least\n    O(ngroups), where\n        ngroups = prod(shape)\n        shape = map(len, keys)\n    that is, linear in the number of combinations (cartesian product) of unique\n    values of groupby keys. This can be huge when doing multi-key groupby.\n    np.argsort(kind='mergesort') is O(count x log(count)) where count is the\n    length of the data-frame;\n    Both algorithms are `stable` sort and that is necessary for correctness of\n    groupby operations. e.g. consider:\n        df.groupby(key)[col].transform('first')\n    "
    count = len(group_index)
    alpha = 0.0
    beta = 1.0
    do_groupsort = ((count > 0) and ((alpha + (beta * ngroups)) < (count * np.log(count))))
    if do_groupsort:
        (sorter, _) = algos.groupsort_indexer(ensure_int64(group_index), ngroups)
        return ensure_platform_int(sorter)
    else:
        return group_index.argsort(kind='mergesort')

def compress_group_index(group_index, sort=True):
    '\n    Group_index is offsets into cartesian product of all possible labels. This\n    space can be huge, so this function compresses it, by computing offsets\n    (comp_ids) into the list of unique labels (obs_group_ids).\n    '
    size_hint = min(len(group_index), hashtable.SIZE_HINT_LIMIT)
    table = hashtable.Int64HashTable(size_hint)
    group_index = ensure_int64(group_index)
    (comp_ids, obs_group_ids) = table.get_labels_groupby(group_index)
    if (sort and (len(obs_group_ids) > 0)):
        (obs_group_ids, comp_ids) = _reorder_by_uniques(obs_group_ids, comp_ids)
    return (ensure_int64(comp_ids), ensure_int64(obs_group_ids))

def _reorder_by_uniques(uniques, labels):
    sorter = uniques.argsort()
    reverse_indexer = np.empty(len(sorter), dtype=np.int64)
    reverse_indexer.put(sorter, np.arange(len(sorter)))
    mask = (labels < 0)
    labels = algorithms.take_nd(reverse_indexer, labels, allow_fill=False)
    np.putmask(labels, mask, (- 1))
    uniques = algorithms.take_nd(uniques, sorter, allow_fill=False)
    return (uniques, labels)
