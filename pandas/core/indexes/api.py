
import textwrap
from typing import List, Set
from pandas._libs import NaT, lib
from pandas.errors import InvalidIndexError
from pandas.core.indexes.base import Index, _new_Index, ensure_index, ensure_index_from_sequences, get_unanimous_names
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.numeric import Float64Index, Int64Index, NumericIndex, UInt64Index
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
_sort_msg = textwrap.dedent("Sorting because non-concatenation axis is not aligned. A future version\nof pandas will change to not sort by default.\n\nTo accept the future behavior, pass 'sort=False'.\n\nTo retain the current behavior and silence the warning, pass 'sort=True'.\n")
__all__ = ['Index', 'MultiIndex', 'NumericIndex', 'Float64Index', 'Int64Index', 'CategoricalIndex', 'IntervalIndex', 'RangeIndex', 'UInt64Index', 'InvalidIndexError', 'TimedeltaIndex', 'PeriodIndex', 'DatetimeIndex', '_new_Index', 'NaT', 'ensure_index', 'ensure_index_from_sequences', 'get_objs_combined_axis', 'union_indexes', 'get_unanimous_names', 'all_indexes_same']

def get_objs_combined_axis(objs, intersect=False, axis=0, sort=True, copy=False):
    '\n    Extract combined index: return intersection or union (depending on the\n    value of "intersect") of indexes on given axis, or None if all objects\n    lack indexes (e.g. they are numpy arrays).\n\n    Parameters\n    ----------\n    objs : list\n        Series or DataFrame objects, may be mix of the two.\n    intersect : bool, default False\n        If True, calculate the intersection between indexes. Otherwise,\n        calculate the union.\n    axis : {0 or \'index\', 1 or \'outer\'}, default 0\n        The axis to extract indexes from.\n    sort : bool, default True\n        Whether the result index should come out sorted or not.\n    copy : bool, default False\n        If True, return a copy of the combined index.\n\n    Returns\n    -------\n    Index\n    '
    obs_idxes = [obj._get_axis(axis) for obj in objs]
    return _get_combined_index(obs_idxes, intersect=intersect, sort=sort, copy=copy)

def _get_distinct_objs(objs):
    '\n    Return a list with distinct elements of "objs" (different ids).\n    Preserves order.\n    '
    ids: Set[int] = set()
    res = []
    for obj in objs:
        if (id(obj) not in ids):
            ids.add(id(obj))
            res.append(obj)
    return res

def _get_combined_index(indexes, intersect=False, sort=False, copy=False):
    '\n    Return the union or intersection of indexes.\n\n    Parameters\n    ----------\n    indexes : list of Index or list objects\n        When intersect=True, do not accept list of lists.\n    intersect : bool, default False\n        If True, calculate the intersection between indexes. Otherwise,\n        calculate the union.\n    sort : bool, default False\n        Whether the result index should come out sorted or not.\n    copy : bool, default False\n        If True, return a copy of the combined index.\n\n    Returns\n    -------\n    Index\n    '
    indexes = _get_distinct_objs(indexes)
    if (len(indexes) == 0):
        index = Index([])
    elif (len(indexes) == 1):
        index = indexes[0]
    elif intersect:
        index = indexes[0]
        for other in indexes[1:]:
            index = index.intersection(other)
    else:
        index = union_indexes(indexes, sort=sort)
        index = ensure_index(index)
    if sort:
        try:
            index = index.sort_values()
        except TypeError:
            pass
    if copy:
        index = index.copy()
    return index

def union_indexes(indexes, sort=True):
    '\n    Return the union of indexes.\n\n    The behavior of sort and names is not consistent.\n\n    Parameters\n    ----------\n    indexes : list of Index or list objects\n    sort : bool, default True\n        Whether the result index should come out sorted or not.\n\n    Returns\n    -------\n    Index\n    '
    if (len(indexes) == 0):
        raise AssertionError('Must have at least 1 Index to union')
    if (len(indexes) == 1):
        result = indexes[0]
        if isinstance(result, list):
            result = Index(sorted(result))
        return result
    (indexes, kind) = _sanitize_and_check(indexes)

    def _unique_indices(inds) -> Index:
        '\n        Convert indexes to lists and concatenate them, removing duplicates.\n\n        The final dtype is inferred.\n\n        Parameters\n        ----------\n        inds : list of Index or list objects\n\n        Returns\n        -------\n        Index\n        '

        def conv(i):
            if isinstance(i, Index):
                i = i.tolist()
            return i
        return Index(lib.fast_unique_multiple_list([conv(i) for i in inds], sort=sort))
    if (kind == 'special'):
        result = indexes[0]
        if hasattr(result, 'union_many'):
            return result.union_many(indexes[1:])
        else:
            for other in indexes[1:]:
                result = result.union(other)
            return result
    elif (kind == 'array'):
        index = indexes[0]
        if (not all((index.equals(other) for other in indexes[1:]))):
            index = _unique_indices(indexes)
        name = get_unanimous_names(*indexes)[0]
        if (name != index.name):
            index = index.rename(name)
        return index
    else:
        return _unique_indices(indexes)

def _sanitize_and_check(indexes):
    "\n    Verify the type of indexes and convert lists to Index.\n\n    Cases:\n\n    - [list, list, ...]: Return ([list, list, ...], 'list')\n    - [list, Index, ...]: Return _sanitize_and_check([Index, Index, ...])\n        Lists are sorted and converted to Index.\n    - [Index, Index, ...]: Return ([Index, Index, ...], TYPE)\n        TYPE = 'special' if at least one special type, 'array' otherwise.\n\n    Parameters\n    ----------\n    indexes : list of Index or list objects\n\n    Returns\n    -------\n    sanitized_indexes : list of Index or list objects\n    type : {'list', 'array', 'special'}\n    "
    kinds = list({type(index) for index in indexes})
    if (list in kinds):
        if (len(kinds) > 1):
            indexes = [(Index(list(x)) if (not isinstance(x, Index)) else x) for x in indexes]
            kinds.remove(list)
        else:
            return (indexes, 'list')
    if ((len(kinds) > 1) or (Index not in kinds)):
        return (indexes, 'special')
    else:
        return (indexes, 'array')

def all_indexes_same(indexes):
    '\n    Determine if all indexes contain the same elements.\n\n    Parameters\n    ----------\n    indexes : iterable of Index objects\n\n    Returns\n    -------\n    bool\n        True if all indexes contain the same elements, False otherwise.\n    '
    itr = iter(indexes)
    first = next(itr)
    return all((first.equals(index) for index in itr))
