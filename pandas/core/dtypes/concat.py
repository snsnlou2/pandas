
'\nUtility functions related to concat.\n'
from typing import Set, cast
import numpy as np
from pandas._typing import ArrayLike, DtypeObj
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_categorical_dtype, is_dtype_equal, is_extension_array_dtype, is_sparse
from pandas.core.dtypes.generic import ABCCategoricalIndex, ABCRangeIndex, ABCSeries
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.sparse import SparseArray
from pandas.core.construction import array, ensure_wrapped_if_datetimelike

def _get_dtype_kinds(arrays):
    '\n    Parameters\n    ----------\n    arrays : list of arrays\n\n    Returns\n    -------\n    set[str]\n        A set of kinds that exist in this list of arrays.\n    '
    typs: Set[str] = set()
    for arr in arrays:
        dtype = arr.dtype
        if (not isinstance(dtype, np.dtype)):
            typ = str(dtype)
        elif isinstance(arr, ABCRangeIndex):
            typ = 'range'
        elif (dtype.kind == 'M'):
            typ = 'datetime'
        elif (dtype.kind == 'm'):
            typ = 'timedelta'
        elif (dtype.kind in ['O', 'b']):
            typ = str(dtype)
        else:
            typ = dtype.kind
        typs.add(typ)
    return typs

def _cast_to_common_type(arr, dtype):
    '\n    Helper function for `arr.astype(common_dtype)` but handling all special\n    cases.\n    '
    if (is_categorical_dtype(arr.dtype) and isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.integer)):
        try:
            return arr.astype(dtype, copy=False)
        except ValueError:
            return arr.astype(object, copy=False)
    if (is_sparse(arr) and (not is_sparse(dtype))):
        arr = cast(SparseArray, arr)
        return arr.to_dense().astype(dtype, copy=False)
    if (isinstance(arr, np.ndarray) and (arr.dtype.kind in ['m', 'M']) and (dtype is np.dtype('object'))):
        arr = ensure_wrapped_if_datetimelike(arr)
    if is_extension_array_dtype(dtype):
        if isinstance(arr, np.ndarray):
            return array(arr, dtype=dtype, copy=False)
    return arr.astype(dtype, copy=False)

def concat_compat(to_concat, axis=0):
    "\n    provide concatenation of an array of arrays each of which is a single\n    'normalized' dtypes (in that for example, if it's object, then it is a\n    non-datetimelike and provide a combined dtype for the resulting array that\n    preserves the overall dtype if possible)\n\n    Parameters\n    ----------\n    to_concat : array of arrays\n    axis : axis to provide concatenation\n\n    Returns\n    -------\n    a single array, preserving the combined dtypes\n    "

    def is_nonempty(x) -> bool:
        if (x.ndim <= axis):
            return True
        return (x.shape[axis] > 0)
    non_empties = [x for x in to_concat if is_nonempty(x)]
    if (non_empties and (axis == 0)):
        to_concat = non_empties
    typs = _get_dtype_kinds(to_concat)
    _contains_datetime = any((typ.startswith('datetime') for typ in typs))
    all_empty = (not len(non_empties))
    single_dtype = (len({x.dtype for x in to_concat}) == 1)
    any_ea = any((is_extension_array_dtype(x.dtype) for x in to_concat))
    if any_ea:
        if (not single_dtype):
            target_dtype = find_common_type([x.dtype for x in to_concat])
            to_concat = [_cast_to_common_type(arr, target_dtype) for arr in to_concat]
        if isinstance(to_concat[0], ExtensionArray):
            cls = type(to_concat[0])
            return cls._concat_same_type(to_concat)
        else:
            return np.concatenate(to_concat)
    elif (_contains_datetime or ('timedelta' in typs)):
        return _concat_datetime(to_concat, axis=axis)
    elif all_empty:
        typs = _get_dtype_kinds(to_concat)
        if (len(typs) != 1):
            if ((not len((typs - {'i', 'u', 'f'}))) or (not len((typs - {'bool', 'i', 'u'})))):
                pass
            else:
                to_concat = [x.astype('object') for x in to_concat]
    return np.concatenate(to_concat, axis=axis)

def union_categoricals(to_union, sort_categories=False, ignore_order=False):
    '\n    Combine list-like of Categorical-like, unioning categories.\n\n    All categories must have the same dtype.\n\n    Parameters\n    ----------\n    to_union : list-like\n        Categorical, CategoricalIndex, or Series with dtype=\'category\'.\n    sort_categories : bool, default False\n        If true, resulting categories will be lexsorted, otherwise\n        they will be ordered as they appear in the data.\n    ignore_order : bool, default False\n        If true, the ordered attribute of the Categoricals will be ignored.\n        Results in an unordered categorical.\n\n    Returns\n    -------\n    Categorical\n\n    Raises\n    ------\n    TypeError\n        - all inputs do not have the same dtype\n        - all inputs do not have the same ordered property\n        - all inputs are ordered and their categories are not identical\n        - sort_categories=True and Categoricals are ordered\n    ValueError\n        Empty list of categoricals passed\n\n    Notes\n    -----\n    To learn more about categories, see `link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#unioning>`__\n\n    Examples\n    --------\n    >>> from pandas.api.types import union_categoricals\n\n    If you want to combine categoricals that do not necessarily have\n    the same categories, `union_categoricals` will combine a list-like\n    of categoricals. The new categories will be the union of the\n    categories being combined.\n\n    >>> a = pd.Categorical(["b", "c"])\n    >>> b = pd.Categorical(["a", "b"])\n    >>> union_categoricals([a, b])\n    [\'b\', \'c\', \'a\', \'b\']\n    Categories (3, object): [\'b\', \'c\', \'a\']\n\n    By default, the resulting categories will be ordered as they appear\n    in the `categories` of the data. If you want the categories to be\n    lexsorted, use `sort_categories=True` argument.\n\n    >>> union_categoricals([a, b], sort_categories=True)\n    [\'b\', \'c\', \'a\', \'b\']\n    Categories (3, object): [\'a\', \'b\', \'c\']\n\n    `union_categoricals` also works with the case of combining two\n    categoricals of the same categories and order information (e.g. what\n    you could also `append` for).\n\n    >>> a = pd.Categorical(["a", "b"], ordered=True)\n    >>> b = pd.Categorical(["a", "b", "a"], ordered=True)\n    >>> union_categoricals([a, b])\n    [\'a\', \'b\', \'a\', \'b\', \'a\']\n    Categories (2, object): [\'a\' < \'b\']\n\n    Raises `TypeError` because the categories are ordered and not identical.\n\n    >>> a = pd.Categorical(["a", "b"], ordered=True)\n    >>> b = pd.Categorical(["a", "b", "c"], ordered=True)\n    >>> union_categoricals([a, b])\n    Traceback (most recent call last):\n        ...\n    TypeError: to union ordered Categoricals, all categories must be the same\n\n    New in version 0.20.0\n\n    Ordered categoricals with different categories or orderings can be\n    combined by using the `ignore_ordered=True` argument.\n\n    >>> a = pd.Categorical(["a", "b", "c"], ordered=True)\n    >>> b = pd.Categorical(["c", "b", "a"], ordered=True)\n    >>> union_categoricals([a, b], ignore_order=True)\n    [\'a\', \'b\', \'c\', \'c\', \'b\', \'a\']\n    Categories (3, object): [\'a\', \'b\', \'c\']\n\n    `union_categoricals` also works with a `CategoricalIndex`, or `Series`\n    containing categorical data, but note that the resulting array will\n    always be a plain `Categorical`\n\n    >>> a = pd.Series(["b", "c"], dtype=\'category\')\n    >>> b = pd.Series(["a", "b"], dtype=\'category\')\n    >>> union_categoricals([a, b])\n    [\'b\', \'c\', \'a\', \'b\']\n    Categories (3, object): [\'b\', \'c\', \'a\']\n    '
    from pandas import Categorical
    from pandas.core.arrays.categorical import recode_for_categories
    if (len(to_union) == 0):
        raise ValueError('No Categoricals to union')

    def _maybe_unwrap(x):
        if isinstance(x, (ABCCategoricalIndex, ABCSeries)):
            return x._values
        elif isinstance(x, Categorical):
            return x
        else:
            raise TypeError('all components to combine must be Categorical')
    to_union = [_maybe_unwrap(x) for x in to_union]
    first = to_union[0]
    if (not all((is_dtype_equal(other.categories.dtype, first.categories.dtype) for other in to_union[1:]))):
        raise TypeError('dtype of categories must be the same')
    ordered = False
    if all((first._categories_match_up_to_permutation(other) for other in to_union[1:])):
        categories = first.categories
        ordered = first.ordered
        all_codes = [first._encode_with_my_categories(x)._codes for x in to_union]
        new_codes = np.concatenate(all_codes)
        if (sort_categories and (not ignore_order) and ordered):
            raise TypeError('Cannot use sort_categories=True with ordered Categoricals')
        if (sort_categories and (not categories.is_monotonic_increasing)):
            categories = categories.sort_values()
            indexer = categories.get_indexer(first.categories)
            from pandas.core.algorithms import take_1d
            new_codes = take_1d(indexer, new_codes, fill_value=(- 1))
    elif (ignore_order or all(((not c.ordered) for c in to_union))):
        cats = first.categories.append([c.categories for c in to_union[1:]])
        categories = cats.unique()
        if sort_categories:
            categories = categories.sort_values()
        new_codes = [recode_for_categories(c.codes, c.categories, categories) for c in to_union]
        new_codes = np.concatenate(new_codes)
    elif all((c.ordered for c in to_union)):
        msg = 'to union ordered Categoricals, all categories must be the same'
        raise TypeError(msg)
    else:
        raise TypeError('Categorical.ordered must be the same')
    if ignore_order:
        ordered = False
    return Categorical(new_codes, categories=categories, ordered=ordered, fastpath=True)

def _concatenate_2d(to_concat, axis):
    if (axis == 1):
        to_concat = [np.atleast_2d(x) for x in to_concat]
    return np.concatenate(to_concat, axis=axis)

def _concat_datetime(to_concat, axis=0):
    '\n    provide concatenation of an datetimelike array of arrays each of which is a\n    single M8[ns], datetime64[ns, tz] or m8[ns] dtype\n\n    Parameters\n    ----------\n    to_concat : array of arrays\n    axis : axis to provide concatenation\n\n    Returns\n    -------\n    a single array, preserving the combined dtypes\n    '
    to_concat = [ensure_wrapped_if_datetimelike(x) for x in to_concat]
    single_dtype = (len({x.dtype for x in to_concat}) == 1)
    if (not single_dtype):
        return _concatenate_2d([x.astype(object) for x in to_concat], axis=axis)
    if (axis == 1):
        to_concat = [(x.reshape(1, (- 1)) if (x.ndim == 1) else x) for x in to_concat]
    result = type(to_concat[0])._concat_same_type(to_concat, axis=axis)
    if ((result.ndim == 2) and is_extension_array_dtype(result.dtype)):
        assert (result.shape[0] == 1)
        result = result[0]
    return result
