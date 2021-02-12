
from functools import wraps
from sys import getsizeof
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import algos as libalgos, index as libindex, lib
from pandas._libs.hashtable import duplicated_int64
from pandas._typing import AnyArrayLike, DtypeObj, Label, Scalar, Shape
from pandas.compat.numpy import function as nv
from pandas.errors import InvalidIndexError, PerformanceWarning, UnsortedIndexError
from pandas.util._decorators import Appender, cache_readonly, doc
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int, is_categorical_dtype, is_hashable, is_integer, is_iterator, is_list_like, is_object_dtype, is_scalar, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCDatetimeIndex, ABCTimedeltaIndex
from pandas.core.dtypes.missing import array_equivalent, isna
import pandas.core.algorithms as algos
from pandas.core.arrays import Categorical
from pandas.core.arrays.categorical import factorize_from_iterables
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs, ensure_index, get_unanimous_names
from pandas.core.indexes.frozen import FrozenList
from pandas.core.indexes.numeric import Int64Index
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import get_group_index, indexer_from_factorized, lexsort_indexer
from pandas.io.formats.printing import format_object_attrs, format_object_summary, pprint_thing
if TYPE_CHECKING:
    from pandas import Series
_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'klass': 'MultiIndex', 'target_klass': 'MultiIndex or list of tuples'})

class MultiIndexUIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    '\n    This class manages a MultiIndex by mapping label combinations to positive\n    integers.\n    '
    _base = libindex.UInt64Engine

    def _codes_to_ints(self, codes):
        '\n        Transform combination(s) of uint64 in one uint64 (each), in a strictly\n        monotonic way (i.e. respecting the lexicographic order of integer\n        combinations): see BaseMultiIndexCodesEngine documentation.\n\n        Parameters\n        ----------\n        codes : 1- or 2-dimensional array of dtype uint64\n            Combinations of integers (one per row)\n\n        Returns\n        -------\n        scalar or 1-dimensional array, of dtype uint64\n            Integer(s) representing one combination (each).\n        '
        codes <<= self.offsets
        if (codes.ndim == 1):
            return np.bitwise_or.reduce(codes)
        return np.bitwise_or.reduce(codes, axis=1)

class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    '\n    This class manages those (extreme) cases in which the number of possible\n    label combinations overflows the 64 bits integers, and uses an ObjectEngine\n    containing Python integers.\n    '
    _base = libindex.ObjectEngine

    def _codes_to_ints(self, codes):
        '\n        Transform combination(s) of uint64 in one Python integer (each), in a\n        strictly monotonic way (i.e. respecting the lexicographic order of\n        integer combinations): see BaseMultiIndexCodesEngine documentation.\n\n        Parameters\n        ----------\n        codes : 1- or 2-dimensional array of dtype uint64\n            Combinations of integers (one per row)\n\n        Returns\n        -------\n        int, or 1-dimensional array of dtype object\n            Integer(s) representing one combination (each).\n        '
        codes = (codes.astype('object') << self.offsets)
        if (codes.ndim == 1):
            return np.bitwise_or.reduce(codes)
        return np.bitwise_or.reduce(codes, axis=1)

def names_compat(meth):
    '\n    A decorator to allow either `name` or `names` keyword but not both.\n\n    This makes it easier to share code with base class.\n    '

    @wraps(meth)
    def new_meth(self_or_cls, *args, **kwargs):
        if (('name' in kwargs) and ('names' in kwargs)):
            raise TypeError('Can only provide one of `names` and `name`')
        elif ('name' in kwargs):
            kwargs['names'] = kwargs.pop('name')
        return meth(self_or_cls, *args, **kwargs)
    return new_meth

class MultiIndex(Index):
    "\n    A multi-level, or hierarchical, index object for pandas objects.\n\n    Parameters\n    ----------\n    levels : sequence of arrays\n        The unique labels for each level.\n    codes : sequence of arrays\n        Integers for each level designating which label at each location.\n\n        .. versionadded:: 0.24.0\n    sortorder : optional int\n        Level of sortedness (must be lexicographically sorted by that\n        level).\n    names : optional sequence of objects\n        Names for each of the index levels. (name is accepted for compat).\n    copy : bool, default False\n        Copy the meta-data.\n    verify_integrity : bool, default True\n        Check that the levels/codes are consistent and valid.\n\n    Attributes\n    ----------\n    names\n    levels\n    codes\n    nlevels\n    levshape\n\n    Methods\n    -------\n    from_arrays\n    from_tuples\n    from_product\n    from_frame\n    set_levels\n    set_codes\n    to_frame\n    to_flat_index\n    is_lexsorted\n    sortlevel\n    droplevel\n    swaplevel\n    reorder_levels\n    remove_unused_levels\n    get_locs\n\n    See Also\n    --------\n    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.\n    MultiIndex.from_product : Create a MultiIndex from the cartesian product\n                              of iterables.\n    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.\n    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.\n    Index : The base pandas Index type.\n\n    Notes\n    -----\n    See the `user guide\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`_\n    for more.\n\n    Examples\n    --------\n    A new ``MultiIndex`` is typically constructed using one of the helper\n    methods :meth:`MultiIndex.from_arrays`, :meth:`MultiIndex.from_product`\n    and :meth:`MultiIndex.from_tuples`. For example (using ``.from_arrays``):\n\n    >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]\n    >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))\n    MultiIndex([(1,  'red'),\n                (1, 'blue'),\n                (2,  'red'),\n                (2, 'blue')],\n               names=['number', 'color'])\n\n    See further examples for how to construct a MultiIndex in the doc strings\n    of the mentioned helper methods.\n    "
    _hidden_attrs = (Index._hidden_attrs | frozenset())
    _typ = 'multiindex'
    _names = FrozenList()
    _levels = FrozenList()
    _codes = FrozenList()
    _comparables = ['names']
    rename = Index.set_names

    def __new__(cls, levels=None, codes=None, sortorder=None, names=None, dtype=None, copy=False, name=None, verify_integrity=True):
        if (name is not None):
            names = name
        if ((levels is None) or (codes is None)):
            raise TypeError('Must pass both levels and codes')
        if (len(levels) != len(codes)):
            raise ValueError('Length of levels and codes must be the same.')
        if (len(levels) == 0):
            raise ValueError('Must pass non-zero number of levels/codes')
        result = object.__new__(MultiIndex)
        result._cache = {}
        result._set_levels(levels, copy=copy, validate=False)
        result._set_codes(codes, copy=copy, validate=False)
        result._names = ([None] * len(levels))
        if (names is not None):
            result._set_names(names)
        if (sortorder is not None):
            result.sortorder = int(sortorder)
        else:
            result.sortorder = sortorder
        if verify_integrity:
            new_codes = result._verify_integrity()
            result._codes = new_codes
        result._reset_identity()
        return result

    def _validate_codes(self, level, code):
        '\n        Reassign code values as -1 if their corresponding levels are NaN.\n\n        Parameters\n        ----------\n        code : list\n            Code to reassign.\n        level : list\n            Level to check for missing values (NaN, NaT, None).\n\n        Returns\n        -------\n        new code where code value = -1 if it corresponds\n        to a level with missing values (NaN, NaT, None).\n        '
        null_mask = isna(level)
        if np.any(null_mask):
            code = np.where(null_mask[code], (- 1), code)
        return code

    def _verify_integrity(self, codes=None, levels=None):
        "\n        Parameters\n        ----------\n        codes : optional list\n            Codes to check for validity. Defaults to current codes.\n        levels : optional list\n            Levels to check for validity. Defaults to current levels.\n\n        Raises\n        ------\n        ValueError\n            If length of levels and codes don't match, if the codes for any\n            level would exceed level bounds, or there are any duplicate levels.\n\n        Returns\n        -------\n        new codes where code value = -1 if it corresponds to a\n        NaN level.\n        "
        codes = (codes or self.codes)
        levels = (levels or self.levels)
        if (len(levels) != len(codes)):
            raise ValueError('Length of levels and codes must match. NOTE: this index is in an inconsistent state.')
        codes_length = len(codes[0])
        for (i, (level, level_codes)) in enumerate(zip(levels, codes)):
            if (len(level_codes) != codes_length):
                raise ValueError(f'Unequal code lengths: {[len(code_) for code_ in codes]}')
            if (len(level_codes) and (level_codes.max() >= len(level))):
                raise ValueError(f'On level {i}, code max ({level_codes.max()}) >= length of level ({len(level)}). NOTE: this index is in an inconsistent state')
            if (len(level_codes) and (level_codes.min() < (- 1))):
                raise ValueError(f'On level {i}, code value ({level_codes.min()}) < -1')
            if (not level.is_unique):
                raise ValueError(f'Level values must be unique: {list(level)} on level {i}')
        if (self.sortorder is not None):
            if (self.sortorder > self._lexsort_depth()):
                raise ValueError(f'Value for sortorder must be inferior or equal to actual lexsort_depth: sortorder {self.sortorder} with lexsort_depth {self._lexsort_depth()}')
        codes = [self._validate_codes(level, code) for (level, code) in zip(levels, codes)]
        new_codes = FrozenList(codes)
        return new_codes

    @classmethod
    def from_arrays(cls, arrays, sortorder=None, names=lib.no_default):
        "\n        Convert arrays to MultiIndex.\n\n        Parameters\n        ----------\n        arrays : list / sequence of array-likes\n            Each array-like gives one level's value for each data point.\n            len(arrays) is the number of levels.\n        sortorder : int or None\n            Level of sortedness (must be lexicographically sorted by that\n            level).\n        names : list / sequence of str, optional\n            Names for the levels in the index.\n\n        Returns\n        -------\n        MultiIndex\n\n        See Also\n        --------\n        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.\n        MultiIndex.from_product : Make a MultiIndex from cartesian product\n                                  of iterables.\n        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.\n\n        Examples\n        --------\n        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]\n        >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))\n        MultiIndex([(1,  'red'),\n                    (1, 'blue'),\n                    (2,  'red'),\n                    (2, 'blue')],\n                   names=['number', 'color'])\n        "
        error_msg = 'Input must be a list / sequence of array-likes.'
        if (not is_list_like(arrays)):
            raise TypeError(error_msg)
        elif is_iterator(arrays):
            arrays = list(arrays)
        for array in arrays:
            if (not is_list_like(array)):
                raise TypeError(error_msg)
        for i in range(1, len(arrays)):
            if (len(arrays[i]) != len(arrays[(i - 1)])):
                raise ValueError('all arrays must be same length')
        (codes, levels) = factorize_from_iterables(arrays)
        if (names is lib.no_default):
            names = [getattr(arr, 'name', None) for arr in arrays]
        return cls(levels=levels, codes=codes, sortorder=sortorder, names=names, verify_integrity=False)

    @classmethod
    @names_compat
    def from_tuples(cls, tuples, sortorder=None, names=None):
        "\n        Convert list of tuples to MultiIndex.\n\n        Parameters\n        ----------\n        tuples : list / sequence of tuple-likes\n            Each tuple is the index of one row/column.\n        sortorder : int or None\n            Level of sortedness (must be lexicographically sorted by that\n            level).\n        names : list / sequence of str, optional\n            Names for the levels in the index.\n\n        Returns\n        -------\n        MultiIndex\n\n        See Also\n        --------\n        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.\n        MultiIndex.from_product : Make a MultiIndex from cartesian product\n                                  of iterables.\n        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.\n\n        Examples\n        --------\n        >>> tuples = [(1, 'red'), (1, 'blue'),\n        ...           (2, 'red'), (2, 'blue')]\n        >>> pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))\n        MultiIndex([(1,  'red'),\n                    (1, 'blue'),\n                    (2,  'red'),\n                    (2, 'blue')],\n                   names=['number', 'color'])\n        "
        if (not is_list_like(tuples)):
            raise TypeError('Input must be a list / sequence of tuple-likes.')
        elif is_iterator(tuples):
            tuples = list(tuples)
        arrays: List[Sequence[Label]]
        if (len(tuples) == 0):
            if (names is None):
                raise TypeError('Cannot infer number of levels from empty list')
            arrays = ([[]] * len(names))
        elif isinstance(tuples, (np.ndarray, Index)):
            if isinstance(tuples, Index):
                tuples = tuples._values
            arrays = list(lib.tuples_to_object_array(tuples).T)
        elif isinstance(tuples, list):
            arrays = list(lib.to_object_array_tuples(tuples).T)
        else:
            arrays = zip(*tuples)
        return cls.from_arrays(arrays, sortorder=sortorder, names=names)

    @classmethod
    def from_product(cls, iterables, sortorder=None, names=lib.no_default):
        "\n        Make a MultiIndex from the cartesian product of multiple iterables.\n\n        Parameters\n        ----------\n        iterables : list / sequence of iterables\n            Each iterable has unique labels for each level of the index.\n        sortorder : int or None\n            Level of sortedness (must be lexicographically sorted by that\n            level).\n        names : list / sequence of str, optional\n            Names for the levels in the index.\n\n            .. versionchanged:: 1.0.0\n\n               If not explicitly provided, names will be inferred from the\n               elements of iterables if an element has a name attribute\n\n        Returns\n        -------\n        MultiIndex\n\n        See Also\n        --------\n        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.\n        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.\n        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.\n\n        Examples\n        --------\n        >>> numbers = [0, 1, 2]\n        >>> colors = ['green', 'purple']\n        >>> pd.MultiIndex.from_product([numbers, colors],\n        ...                            names=['number', 'color'])\n        MultiIndex([(0,  'green'),\n                    (0, 'purple'),\n                    (1,  'green'),\n                    (1, 'purple'),\n                    (2,  'green'),\n                    (2, 'purple')],\n                   names=['number', 'color'])\n        "
        from pandas.core.reshape.util import cartesian_product
        if (not is_list_like(iterables)):
            raise TypeError('Input must be a list / sequence of iterables.')
        elif is_iterator(iterables):
            iterables = list(iterables)
        (codes, levels) = factorize_from_iterables(iterables)
        if (names is lib.no_default):
            names = [getattr(it, 'name', None) for it in iterables]
        codes = cartesian_product(codes)
        return cls(levels, codes, sortorder=sortorder, names=names)

    @classmethod
    def from_frame(cls, df, sortorder=None, names=None):
        "\n        Make a MultiIndex from a DataFrame.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        df : DataFrame\n            DataFrame to be converted to MultiIndex.\n        sortorder : int, optional\n            Level of sortedness (must be lexicographically sorted by that\n            level).\n        names : list-like, optional\n            If no names are provided, use the column names, or tuple of column\n            names if the columns is a MultiIndex. If a sequence, overwrite\n            names with the given sequence.\n\n        Returns\n        -------\n        MultiIndex\n            The MultiIndex representation of the given DataFrame.\n\n        See Also\n        --------\n        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.\n        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.\n        MultiIndex.from_product : Make a MultiIndex from cartesian product\n                                  of iterables.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],\n        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],\n        ...                   columns=['a', 'b'])\n        >>> df\n              a       b\n        0    HI    Temp\n        1    HI  Precip\n        2    NJ    Temp\n        3    NJ  Precip\n\n        >>> pd.MultiIndex.from_frame(df)\n        MultiIndex([('HI',   'Temp'),\n                    ('HI', 'Precip'),\n                    ('NJ',   'Temp'),\n                    ('NJ', 'Precip')],\n                   names=['a', 'b'])\n\n        Using explicit names, instead of the column names\n\n        >>> pd.MultiIndex.from_frame(df, names=['state', 'observation'])\n        MultiIndex([('HI',   'Temp'),\n                    ('HI', 'Precip'),\n                    ('NJ',   'Temp'),\n                    ('NJ', 'Precip')],\n                   names=['state', 'observation'])\n        "
        if (not isinstance(df, ABCDataFrame)):
            raise TypeError('Input must be a DataFrame')
        (column_names, columns) = zip(*df.items())
        names = (column_names if (names is None) else names)
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    @cache_readonly
    def _values(self):
        values = []
        for i in range(self.nlevels):
            vals = self._get_level_values(i)
            if is_categorical_dtype(vals.dtype):
                vals = vals._internal_get_values()
            if (isinstance(vals.dtype, ExtensionDtype) or isinstance(vals, (ABCDatetimeIndex, ABCTimedeltaIndex))):
                vals = vals.astype(object)
            vals = np.array(vals, copy=False)
            values.append(vals)
        arr = lib.fast_zip(values)
        return arr

    @property
    def values(self):
        return self._values

    @property
    def array(self):
        "\n        Raises a ValueError for `MultiIndex` because there's no single\n        array backing a MultiIndex.\n\n        Raises\n        ------\n        ValueError\n        "
        raise ValueError("MultiIndex has no single backing array. Use 'MultiIndex.to_numpy()' to get a NumPy array of tuples.")

    @cache_readonly
    def dtypes(self):
        '\n        Return the dtypes as a Series for the underlying MultiIndex\n        '
        from pandas import Series
        return Series({(f'level_{idx}' if (level.name is None) else level.name): level.dtype for (idx, level) in enumerate(self.levels)})

    @property
    def shape(self):
        '\n        Return a tuple of the shape of the underlying data.\n        '
        return (len(self),)

    def __len__(self):
        return len(self.codes[0])

    @cache_readonly
    def levels(self):
        result = [x._shallow_copy(name=name) for (x, name) in zip(self._levels, self._names)]
        for level in result:
            level._no_setting_name = True
        return FrozenList(result)

    def _set_levels(self, levels, level=None, copy=False, validate=True, verify_integrity=False):
        if validate:
            if (len(levels) == 0):
                raise ValueError('Must set non-zero number of levels.')
            if ((level is None) and (len(levels) != self.nlevels)):
                raise ValueError('Length of levels must match number of levels.')
            if ((level is not None) and (len(levels) != len(level))):
                raise ValueError('Length of levels must match length of level.')
        if (level is None):
            new_levels = FrozenList((ensure_index(lev, copy=copy)._shallow_copy() for lev in levels))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_levels_list = list(self._levels)
            for (lev_num, lev) in zip(level_numbers, levels):
                new_levels_list[lev_num] = ensure_index(lev, copy=copy)._shallow_copy()
            new_levels = FrozenList(new_levels_list)
        if verify_integrity:
            new_codes = self._verify_integrity(levels=new_levels)
            self._codes = new_codes
        names = self.names
        self._levels = new_levels
        if any(names):
            self._set_names(names)
        self._reset_cache()

    def set_levels(self, levels, level=None, inplace=None, verify_integrity=True):
        '\n        Set new levels on MultiIndex. Defaults to returning new index.\n\n        Parameters\n        ----------\n        levels : sequence or list of sequence\n            New level(s) to apply.\n        level : int, level name, or sequence of int/level names (default None)\n            Level(s) to set (None for all levels).\n        inplace : bool\n            If True, mutates in place.\n\n            .. deprecated:: 1.2.0\n        verify_integrity : bool, default True\n            If True, checks that levels and codes are compatible.\n\n        Returns\n        -------\n        new index (of same type and class...etc) or None\n            The same type as the caller or None if ``inplace=True``.\n\n        Examples\n        --------\n        >>> idx = pd.MultiIndex.from_tuples(\n        ...     [\n        ...         (1, "one"),\n        ...         (1, "two"),\n        ...         (2, "one"),\n        ...         (2, "two"),\n        ...         (3, "one"),\n        ...         (3, "two")\n        ...     ],\n        ...     names=["foo", "bar"]\n        ... )\n        >>> idx\n        MultiIndex([(1, \'one\'),\n            (1, \'two\'),\n            (2, \'one\'),\n            (2, \'two\'),\n            (3, \'one\'),\n            (3, \'two\')],\n           names=[\'foo\', \'bar\'])\n\n        >>> idx.set_levels([[\'a\', \'b\', \'c\'], [1, 2]])\n        MultiIndex([(\'a\', 1),\n                    (\'a\', 2),\n                    (\'b\', 1),\n                    (\'b\', 2),\n                    (\'c\', 1),\n                    (\'c\', 2)],\n                   names=[\'foo\', \'bar\'])\n        >>> idx.set_levels([\'a\', \'b\', \'c\'], level=0)\n        MultiIndex([(\'a\', \'one\'),\n                    (\'a\', \'two\'),\n                    (\'b\', \'one\'),\n                    (\'b\', \'two\'),\n                    (\'c\', \'one\'),\n                    (\'c\', \'two\')],\n                   names=[\'foo\', \'bar\'])\n        >>> idx.set_levels([\'a\', \'b\'], level=\'bar\')\n        MultiIndex([(1, \'a\'),\n                    (1, \'b\'),\n                    (2, \'a\'),\n                    (2, \'b\'),\n                    (3, \'a\'),\n                    (3, \'b\')],\n                   names=[\'foo\', \'bar\'])\n\n        If any of the levels passed to ``set_levels()`` exceeds the\n        existing length, all of the values from that argument will\n        be stored in the MultiIndex levels, though the values will\n        be truncated in the MultiIndex output.\n\n        >>> idx.set_levels([[\'a\', \'b\', \'c\'], [1, 2, 3, 4]], level=[0, 1])\n        MultiIndex([(\'a\', 1),\n            (\'a\', 2),\n            (\'b\', 1),\n            (\'b\', 2),\n            (\'c\', 1),\n            (\'c\', 2)],\n           names=[\'foo\', \'bar\'])\n        >>> idx.set_levels([[\'a\', \'b\', \'c\'], [1, 2, 3, 4]], level=[0, 1]).levels\n        FrozenList([[\'a\', \'b\', \'c\'], [1, 2, 3, 4]])\n        '
        if (inplace is not None):
            warnings.warn('inplace is deprecated and will be removed in a future version.', FutureWarning, stacklevel=2)
        else:
            inplace = False
        if (is_list_like(levels) and (not isinstance(levels, Index))):
            levels = list(levels)
        if ((level is not None) and (not is_list_like(level))):
            if (not is_list_like(levels)):
                raise TypeError('Levels must be list-like')
            if is_list_like(levels[0]):
                raise TypeError('Levels must be list-like')
            level = [level]
            levels = [levels]
        elif ((level is None) or is_list_like(level)):
            if ((not is_list_like(levels)) or (not is_list_like(levels[0]))):
                raise TypeError('Levels must be list of lists-like')
        if inplace:
            idx = self
        else:
            idx = self._shallow_copy()
        idx._reset_identity()
        idx._set_levels(levels, level=level, validate=True, verify_integrity=verify_integrity)
        if (not inplace):
            return idx

    @property
    def nlevels(self):
        "\n        Integer number of levels in this MultiIndex.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])\n        >>> mi\n        MultiIndex([('a', 'b', 'c')],\n                   )\n        >>> mi.nlevels\n        3\n        "
        return len(self._levels)

    @property
    def levshape(self):
        "\n        A tuple with the length of each level.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])\n        >>> mi\n        MultiIndex([('a', 'b', 'c')],\n                   )\n        >>> mi.levshape\n        (1, 1, 1)\n        "
        return tuple((len(x) for x in self.levels))

    @property
    def codes(self):
        return self._codes

    def _set_codes(self, codes, level=None, copy=False, validate=True, verify_integrity=False):
        if validate:
            if ((level is None) and (len(codes) != self.nlevels)):
                raise ValueError('Length of codes must match number of levels')
            if ((level is not None) and (len(codes) != len(level))):
                raise ValueError('Length of codes must match length of levels.')
        if (level is None):
            new_codes = FrozenList((_coerce_indexer_frozen(level_codes, lev, copy=copy).view() for (lev, level_codes) in zip(self._levels, codes)))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_codes_list = list(self._codes)
            for (lev_num, level_codes) in zip(level_numbers, codes):
                lev = self.levels[lev_num]
                new_codes_list[lev_num] = _coerce_indexer_frozen(level_codes, lev, copy=copy)
            new_codes = FrozenList(new_codes_list)
        if verify_integrity:
            new_codes = self._verify_integrity(codes=new_codes)
        self._codes = new_codes
        self._reset_cache()

    def set_codes(self, codes, level=None, inplace=None, verify_integrity=True):
        '\n        Set new codes on MultiIndex. Defaults to returning new index.\n\n        .. versionadded:: 0.24.0\n\n           New name for deprecated method `set_labels`.\n\n        Parameters\n        ----------\n        codes : sequence or list of sequence\n            New codes to apply.\n        level : int, level name, or sequence of int/level names (default None)\n            Level(s) to set (None for all levels).\n        inplace : bool\n            If True, mutates in place.\n\n            .. deprecated:: 1.2.0\n        verify_integrity : bool (default True)\n            If True, checks that levels and codes are compatible.\n\n        Returns\n        -------\n        new index (of same type and class...etc) or None\n            The same type as the caller or None if ``inplace=True``.\n\n        Examples\n        --------\n        >>> idx = pd.MultiIndex.from_tuples(\n        ...     [(1, "one"), (1, "two"), (2, "one"), (2, "two")], names=["foo", "bar"]\n        ... )\n        >>> idx\n        MultiIndex([(1, \'one\'),\n            (1, \'two\'),\n            (2, \'one\'),\n            (2, \'two\')],\n           names=[\'foo\', \'bar\'])\n\n        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]])\n        MultiIndex([(2, \'one\'),\n                    (1, \'one\'),\n                    (2, \'two\'),\n                    (1, \'two\')],\n                   names=[\'foo\', \'bar\'])\n        >>> idx.set_codes([1, 0, 1, 0], level=0)\n        MultiIndex([(2, \'one\'),\n                    (1, \'two\'),\n                    (2, \'one\'),\n                    (1, \'two\')],\n                   names=[\'foo\', \'bar\'])\n        >>> idx.set_codes([0, 0, 1, 1], level=\'bar\')\n        MultiIndex([(1, \'one\'),\n                    (1, \'one\'),\n                    (2, \'two\'),\n                    (2, \'two\')],\n                   names=[\'foo\', \'bar\'])\n        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]], level=[0, 1])\n        MultiIndex([(2, \'one\'),\n                    (1, \'one\'),\n                    (2, \'two\'),\n                    (1, \'two\')],\n                   names=[\'foo\', \'bar\'])\n        '
        if (inplace is not None):
            warnings.warn('inplace is deprecated and will be removed in a future version.', FutureWarning, stacklevel=2)
        else:
            inplace = False
        if ((level is not None) and (not is_list_like(level))):
            if (not is_list_like(codes)):
                raise TypeError('Codes must be list-like')
            if is_list_like(codes[0]):
                raise TypeError('Codes must be list-like')
            level = [level]
            codes = [codes]
        elif ((level is None) or is_list_like(level)):
            if ((not is_list_like(codes)) or (not is_list_like(codes[0]))):
                raise TypeError('Codes must be list of lists-like')
        if inplace:
            idx = self
        else:
            idx = self._shallow_copy()
        idx._reset_identity()
        idx._set_codes(codes, level=level, verify_integrity=verify_integrity)
        if (not inplace):
            return idx

    @cache_readonly
    def _engine(self):
        sizes = np.ceil(np.log2([(len(level) + 1) for level in self.levels]))
        lev_bits = np.cumsum(sizes[::(- 1)])[::(- 1)]
        offsets = np.concatenate([lev_bits[1:], [0]]).astype('uint64')
        if (lev_bits[0] > 64):
            return MultiIndexPyIntEngine(self.levels, self.codes, offsets)
        return MultiIndexUIntEngine(self.levels, self.codes, offsets)

    @property
    def _constructor(self):
        return type(self).from_tuples

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values=None, name=lib.no_default):
        names = (name if (name is not lib.no_default) else self.names)
        if (values is not None):
            return type(self).from_tuples(values, sortorder=None, names=names)
        result = type(self)(levels=self.levels, codes=self.codes, sortorder=None, names=names, verify_integrity=False)
        result._cache = self._cache.copy()
        result._cache.pop('levels', None)
        return result

    def copy(self, names=None, dtype=None, levels=None, codes=None, deep=False, name=None):
        '\n        Make a copy of this object. Names, dtype, levels and codes can be\n        passed and will be set on new copy.\n\n        Parameters\n        ----------\n        names : sequence, optional\n        dtype : numpy dtype or pandas type, optional\n\n            .. deprecated:: 1.2.0\n        levels : sequence, optional\n\n            .. deprecated:: 1.2.0\n        codes : sequence, optional\n\n            .. deprecated:: 1.2.0\n        deep : bool, default False\n        name : Label\n            Kept for compatibility with 1-dimensional Index. Should not be used.\n\n        Returns\n        -------\n        MultiIndex\n\n        Notes\n        -----\n        In most cases, there should be no functional difference from using\n        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.\n        This could be potentially expensive on large MultiIndex objects.\n        '
        names = self._validate_names(name=name, names=names, deep=deep)
        if (levels is not None):
            warnings.warn('parameter levels is deprecated and will be removed in a future version. Use the set_levels method instead.', FutureWarning, stacklevel=2)
        if (codes is not None):
            warnings.warn('parameter codes is deprecated and will be removed in a future version. Use the set_codes method instead.', FutureWarning, stacklevel=2)
        if deep:
            from copy import deepcopy
            if (levels is None):
                levels = deepcopy(self.levels)
            if (codes is None):
                codes = deepcopy(self.codes)
        levels = (levels if (levels is not None) else self.levels)
        codes = (codes if (codes is not None) else self.codes)
        new_index = type(self)(levels=levels, codes=codes, sortorder=self.sortorder, names=names, verify_integrity=False)
        new_index._cache = self._cache.copy()
        new_index._cache.pop('levels', None)
        if dtype:
            warnings.warn('parameter dtype is deprecated and will be removed in a future version. Use the astype method instead.', FutureWarning, stacklevel=2)
            new_index = new_index.astype(dtype)
        return new_index

    def __array__(self, dtype=None):
        ' the array interface, return my values '
        return self.values

    def view(self, cls=None):
        ' this is defined as a copy with the same identity '
        result = self.copy()
        result._id = self._id
        return result

    @doc(Index.__contains__)
    def __contains__(self, key):
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError, ValueError):
            return False

    @cache_readonly
    def dtype(self):
        return np.dtype('O')

    def _is_memory_usage_qualified(self):
        ' return a boolean if we need a qualified .info display '

        def f(level):
            return (('mixed' in level) or ('string' in level) or ('unicode' in level))
        return any((f(level) for level in self._inferred_type_levels))

    @doc(Index.memory_usage)
    def memory_usage(self, deep=False):
        return self._nbytes(deep)

    @cache_readonly
    def nbytes(self):
        ' return the number of bytes in the underlying data '
        return self._nbytes(False)

    def _nbytes(self, deep=False):
        '\n        return the number of bytes in the underlying data\n        deeply introspect the level data if deep=True\n\n        include the engine hashtable\n\n        *this is in internal routine*\n\n        '
        objsize = 24
        level_nbytes = sum((i.memory_usage(deep=deep) for i in self.levels))
        label_nbytes = sum((i.nbytes for i in self.codes))
        names_nbytes = sum((getsizeof(i, objsize) for i in self.names))
        result = ((level_nbytes + label_nbytes) + names_nbytes)
        result += self._engine.sizeof(deep=deep)
        return result

    def _formatter_func(self, tup):
        "\n        Formats each item in tup according to its level's formatter function.\n        "
        formatter_funcs = [level._formatter_func for level in self.levels]
        return tuple((func(val) for (func, val) in zip(formatter_funcs, tup)))

    def _format_data(self, name=None):
        '\n        Return the formatted data as a unicode string\n        '
        return format_object_summary(self, self._formatter_func, name=name, line_break_each_value=True)

    def _format_attrs(self):
        '\n        Return a list of tuples of the (attr,formatted_value).\n        '
        return format_object_attrs(self, include_dtype=False)

    def _format_native_types(self, na_rep='nan', **kwargs):
        new_levels = []
        new_codes = []
        for (level, level_codes) in zip(self.levels, self.codes):
            level = level._format_native_types(na_rep=na_rep, **kwargs)
            mask = (level_codes == (- 1))
            if mask.any():
                nan_index = len(level)
                level = np.append(level, na_rep)
                assert (not level_codes.flags.writeable)
                level_codes = level_codes.copy()
                level_codes[mask] = nan_index
            new_levels.append(level)
            new_codes.append(level_codes)
        if (len(new_levels) == 1):
            return Index(new_levels[0].take(new_codes[0]))._format_native_types()
        else:
            mi = MultiIndex(levels=new_levels, codes=new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)
            return mi._values

    def format(self, name=None, formatter=None, na_rep=None, names=False, space=2, sparsify=None, adjoin=True):
        if (name is not None):
            names = name
        if (len(self) == 0):
            return []
        stringified_levels = []
        for (lev, level_codes) in zip(self.levels, self.codes):
            na = (na_rep if (na_rep is not None) else _get_na_rep(lev.dtype.type))
            if (len(lev) > 0):
                formatted = lev.take(level_codes).format(formatter=formatter)
                mask = (level_codes == (- 1))
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()
            else:
                formatted = [pprint_thing((na if isna(x) else x), escape_chars=('\t', '\r', '\n')) for x in algos.take_1d(lev._values, level_codes)]
            stringified_levels.append(formatted)
        result_levels = []
        for (lev, lev_name) in zip(stringified_levels, self.names):
            level = []
            if names:
                level.append((pprint_thing(lev_name, escape_chars=('\t', '\r', '\n')) if (lev_name is not None) else ''))
            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)
        if (sparsify is None):
            sparsify = get_option('display.multi_sparse')
        if sparsify:
            sentinel = ''
            assert (isinstance(sparsify, bool) or (sparsify is lib.no_default))
            if (sparsify in [False, lib.no_default]):
                sentinel = sparsify
            result_levels = sparsify_labels(result_levels, start=int(names), sentinel=sentinel)
        if adjoin:
            from pandas.io.formats.format import get_adjustment
            adj = get_adjustment()
            return adj.adjoin(space, *result_levels).split('\n')
        else:
            return result_levels

    def _get_names(self):
        return FrozenList(self._names)

    def _set_names(self, names, level=None, validate=True):
        '\n        Set new names on index. Each name has to be a hashable type.\n\n        Parameters\n        ----------\n        values : str or sequence\n            name(s) to set\n        level : int, level name, or sequence of int/level names (default None)\n            If the index is a MultiIndex (hierarchical), level(s) to set (None\n            for all levels).  Otherwise level must be None\n        validate : boolean, default True\n            validate that the names match level lengths\n\n        Raises\n        ------\n        TypeError if each name is not hashable.\n\n        Notes\n        -----\n        sets names on levels. WARNING: mutates!\n\n        Note that you generally want to set this *after* changing levels, so\n        that it only acts on copies\n        '
        if ((names is not None) and (not is_list_like(names))):
            raise ValueError('Names should be list-like for a MultiIndex')
        names = list(names)
        if validate:
            if ((level is not None) and (len(names) != len(level))):
                raise ValueError('Length of names must match length of level.')
            if ((level is None) and (len(names) != self.nlevels)):
                raise ValueError('Length of names must match number of levels in MultiIndex.')
        if (level is None):
            level = range(self.nlevels)
        else:
            level = [self._get_level_number(lev) for lev in level]
        for (lev, name) in zip(level, names):
            if (name is not None):
                if (not is_hashable(name)):
                    raise TypeError(f'{type(self).__name__}.name must be a hashable type')
            self._names[lev] = name
        self._reset_cache()
    names = property(fset=_set_names, fget=_get_names, doc="\n        Names of levels in MultiIndex.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays(\n        ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])\n        >>> mi\n        MultiIndex([(1, 3, 5),\n                    (2, 4, 6)],\n                   names=['x', 'y', 'z'])\n        >>> mi.names\n        FrozenList(['x', 'y', 'z'])\n        ")

    @doc(Index._get_grouper_for_level)
    def _get_grouper_for_level(self, mapper, level):
        indexer = self.codes[level]
        level_index = self.levels[level]
        if (mapper is not None):
            level_values = self.levels[level].take(indexer)
            grouper = level_values.map(mapper)
            return (grouper, None, None)
        (codes, uniques) = algos.factorize(indexer, sort=True)
        if ((len(uniques) > 0) and (uniques[0] == (- 1))):
            mask = (indexer != (- 1))
            (ok_codes, uniques) = algos.factorize(indexer[mask], sort=True)
            codes = np.empty(len(indexer), dtype=indexer.dtype)
            codes[mask] = ok_codes
            codes[(~ mask)] = (- 1)
        if (len(uniques) < len(level_index)):
            level_index = level_index.take(uniques)
        else:
            level_index = level_index.copy()
        if level_index._can_hold_na:
            grouper = level_index.take(codes, fill_value=True)
        else:
            grouper = level_index.take(codes)
        return (grouper, codes, level_index)

    @cache_readonly
    def inferred_type(self):
        return 'mixed'

    def _get_level_number(self, level):
        count = self.names.count(level)
        if ((count > 1) and (not is_integer(level))):
            raise ValueError(f'The name {level} occurs multiple times, use a level number')
        try:
            level = self.names.index(level)
        except ValueError as err:
            if (not is_integer(level)):
                raise KeyError(f'Level {level} not found') from err
            elif (level < 0):
                level += self.nlevels
                if (level < 0):
                    orig_level = (level - self.nlevels)
                    raise IndexError(f'Too many levels: Index has only {self.nlevels} levels, {orig_level} is not a valid level number') from err
            elif (level >= self.nlevels):
                raise IndexError(f'Too many levels: Index has only {self.nlevels} levels, not {(level + 1)}') from err
        return level

    @property
    def _has_complex_internals(self):
        return True

    @cache_readonly
    def is_monotonic_increasing(self):
        '\n        return if the index is monotonic increasing (only equal or\n        increasing) values.\n        '
        if any((((- 1) in code) for code in self.codes)):
            return False
        if all((level.is_monotonic for level in self.levels)):
            return libalgos.is_lexsorted([x.astype('int64', copy=False) for x in self.codes])
        values = [self._get_level_values(i)._values for i in reversed(range(len(self.levels)))]
        try:
            sort_order = np.lexsort(values)
            return Index(sort_order).is_monotonic
        except TypeError:
            return Index(self._values).is_monotonic

    @cache_readonly
    def is_monotonic_decreasing(self):
        '\n        return if the index is monotonic decreasing (only equal or\n        decreasing) values.\n        '
        return self[::(- 1)].is_monotonic_increasing

    @cache_readonly
    def _inferred_type_levels(self):
        ' return a list of the inferred types, one for each level '
        return [i.inferred_type for i in self.levels]

    @doc(Index.duplicated)
    def duplicated(self, keep='first'):
        shape = map(len, self.levels)
        ids = get_group_index(self.codes, shape, sort=False, xnull=False)
        return duplicated_int64(ids, keep)

    def fillna(self, value=None, downcast=None):
        '\n        fillna is not implemented for MultiIndex\n        '
        raise NotImplementedError('isna is not defined for MultiIndex')

    @doc(Index.dropna)
    def dropna(self, how='any'):
        nans = [(level_codes == (- 1)) for level_codes in self.codes]
        if (how == 'any'):
            indexer = np.any(nans, axis=0)
        elif (how == 'all'):
            indexer = np.all(nans, axis=0)
        else:
            raise ValueError(f'invalid how option: {how}')
        new_codes = [level_codes[(~ indexer)] for level_codes in self.codes]
        return self.set_codes(codes=new_codes)

    def _get_level_values(self, level, unique=False):
        '\n        Return vector of label values for requested level,\n        equal to the length of the index\n\n        **this is an internal method**\n\n        Parameters\n        ----------\n        level : int level\n        unique : bool, default False\n            if True, drop duplicated values\n\n        Returns\n        -------\n        values : ndarray\n        '
        lev = self.levels[level]
        level_codes = self.codes[level]
        name = self._names[level]
        if unique:
            level_codes = algos.unique(level_codes)
        filled = algos.take_1d(lev._values, level_codes, fill_value=lev._na_value)
        return lev._shallow_copy(filled, name=name)

    def get_level_values(self, level):
        "\n        Return vector of label values for requested level.\n\n        Length of returned vector is equal to the length of the index.\n\n        Parameters\n        ----------\n        level : int or str\n            ``level`` is either the integer position of the level in the\n            MultiIndex, or the name of the level.\n\n        Returns\n        -------\n        values : Index\n            Values is a level of this MultiIndex converted to\n            a single :class:`Index` (or subclass thereof).\n\n        Examples\n        --------\n        Create a MultiIndex:\n\n        >>> mi = pd.MultiIndex.from_arrays((list('abc'), list('def')))\n        >>> mi.names = ['level_1', 'level_2']\n\n        Get level values by supplying level as either integer or name:\n\n        >>> mi.get_level_values(0)\n        Index(['a', 'b', 'c'], dtype='object', name='level_1')\n        >>> mi.get_level_values('level_2')\n        Index(['d', 'e', 'f'], dtype='object', name='level_2')\n        "
        level = self._get_level_number(level)
        values = self._get_level_values(level)
        return values

    @doc(Index.unique)
    def unique(self, level=None):
        if (level is None):
            return super().unique()
        else:
            level = self._get_level_number(level)
            return self._get_level_values(level=level, unique=True)

    def to_frame(self, index=True, name=None):
        "\n        Create a DataFrame with the levels of the MultiIndex as columns.\n\n        Column ordering is determined by the DataFrame constructor with data as\n        a dict.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        index : bool, default True\n            Set the index of the returned DataFrame as the original MultiIndex.\n\n        name : list / sequence of str, optional\n            The passed names should substitute index level names.\n\n        Returns\n        -------\n        DataFrame : a DataFrame containing the original MultiIndex data.\n\n        See Also\n        --------\n        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous\n            tabular data.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([['a', 'b'], ['c', 'd']])\n        >>> mi\n        MultiIndex([('a', 'c'),\n                    ('b', 'd')],\n                   )\n\n        >>> df = mi.to_frame()\n        >>> df\n             0  1\n        a c  a  c\n        b d  b  d\n\n        >>> df = mi.to_frame(index=False)\n        >>> df\n           0  1\n        0  a  c\n        1  b  d\n\n        >>> df = mi.to_frame(name=['x', 'y'])\n        >>> df\n             x  y\n        a c  a  c\n        b d  b  d\n        "
        from pandas import DataFrame
        if (name is not None):
            if (not is_list_like(name)):
                raise TypeError("'name' must be a list / sequence of column names.")
            if (len(name) != len(self.levels)):
                raise ValueError("'name' should have same length as number of levels on index.")
            idx_names = name
        else:
            idx_names = self.names
        result = DataFrame({(level if (lvlname is None) else lvlname): self._get_level_values(level) for (lvlname, level) in zip(idx_names, range(len(self.levels)))}, copy=False)
        if index:
            result.index = self
        return result

    def to_flat_index(self):
        "\n        Convert a MultiIndex to an Index of Tuples containing the level values.\n\n        .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        pd.Index\n            Index with the MultiIndex data represented in Tuples.\n\n        Notes\n        -----\n        This method will simply return the caller if called by anything other\n        than a MultiIndex.\n\n        Examples\n        --------\n        >>> index = pd.MultiIndex.from_product(\n        ...     [['foo', 'bar'], ['baz', 'qux']],\n        ...     names=['a', 'b'])\n        >>> index.to_flat_index()\n        Index([('foo', 'baz'), ('foo', 'qux'),\n               ('bar', 'baz'), ('bar', 'qux')],\n              dtype='object')\n        "
        return Index(self._values, tupleize_cols=False)

    @property
    def _is_all_dates(self):
        return False

    def is_lexsorted(self):
        "\n        Return True if the codes are lexicographically sorted.\n\n        Returns\n        -------\n        bool\n\n        Examples\n        --------\n        In the below examples, the first level of the MultiIndex is sorted because\n        a<b<c, so there is no need to look at the next level.\n\n        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'], ['d', 'e', 'f']]).is_lexsorted()\n        True\n        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'], ['d', 'f', 'e']]).is_lexsorted()\n        True\n\n        In case there is a tie, the lexicographical sorting looks\n        at the next level of the MultiIndex.\n\n        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'b', 'c']]).is_lexsorted()\n        True\n        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'c', 'b']]).is_lexsorted()\n        False\n        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],\n        ...                            ['aa', 'bb', 'aa', 'bb']]).is_lexsorted()\n        True\n        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],\n        ...                            ['bb', 'aa', 'aa', 'bb']]).is_lexsorted()\n        False\n        "
        return (self.lexsort_depth == self.nlevels)

    @cache_readonly
    def lexsort_depth(self):
        if (self.sortorder is not None):
            return self.sortorder
        return self._lexsort_depth()

    def _lexsort_depth(self):
        '\n        Compute and return the lexsort_depth, the number of levels of the\n        MultiIndex that are sorted lexically\n\n        Returns\n        -------\n        int\n        '
        int64_codes = [ensure_int64(level_codes) for level_codes in self.codes]
        for k in range(self.nlevels, 0, (- 1)):
            if libalgos.is_lexsorted(int64_codes[:k]):
                return k
        return 0

    def _sort_levels_monotonic(self):
        "\n        This is an *internal* function.\n\n        Create a new MultiIndex from the current to monotonically sorted\n        items IN the levels. This does not actually make the entire MultiIndex\n        monotonic, JUST the levels.\n\n        The resulting MultiIndex will have the same outward\n        appearance, meaning the same .values and ordering. It will also\n        be .equals() to the original.\n\n        Returns\n        -------\n        MultiIndex\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],\n        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])\n        >>> mi\n        MultiIndex([('a', 'bb'),\n                    ('a', 'aa'),\n                    ('b', 'bb'),\n                    ('b', 'aa')],\n                   )\n\n        >>> mi.sort_values()\n        MultiIndex([('a', 'aa'),\n                    ('a', 'bb'),\n                    ('b', 'aa'),\n                    ('b', 'bb')],\n                   )\n        "
        if (self.is_lexsorted() and self.is_monotonic):
            return self
        new_levels = []
        new_codes = []
        for (lev, level_codes) in zip(self.levels, self.codes):
            if (not lev.is_monotonic):
                try:
                    indexer = lev.argsort()
                except TypeError:
                    pass
                else:
                    lev = lev.take(indexer)
                    indexer = ensure_int64(indexer)
                    ri = lib.get_reverse_indexer(indexer, len(indexer))
                    level_codes = algos.take_1d(ri, level_codes)
            new_levels.append(lev)
            new_codes.append(level_codes)
        return MultiIndex(new_levels, new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)

    def remove_unused_levels(self):
        "\n        Create new MultiIndex from current that removes unused levels.\n\n        Unused level(s) means levels that are not expressed in the\n        labels. The resulting MultiIndex will have the same outward\n        appearance, meaning the same .values and ordering. It will\n        also be .equals() to the original.\n\n        Returns\n        -------\n        MultiIndex\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_product([range(2), list('ab')])\n        >>> mi\n        MultiIndex([(0, 'a'),\n                    (0, 'b'),\n                    (1, 'a'),\n                    (1, 'b')],\n                   )\n\n        >>> mi[2:]\n        MultiIndex([(1, 'a'),\n                    (1, 'b')],\n                   )\n\n        The 0 from the first level is not represented\n        and can be removed\n\n        >>> mi2 = mi[2:].remove_unused_levels()\n        >>> mi2.levels\n        FrozenList([[1], ['a', 'b']])\n        "
        new_levels = []
        new_codes = []
        changed = False
        for (lev, level_codes) in zip(self.levels, self.codes):
            uniques = (np.where((np.bincount((level_codes + 1)) > 0))[0] - 1)
            has_na = int((len(uniques) and (uniques[0] == (- 1))))
            if (len(uniques) != (len(lev) + has_na)):
                if (lev.isna().any() and (len(uniques) == len(lev))):
                    break
                changed = True
                uniques = algos.unique(level_codes)
                if has_na:
                    na_idx = np.where((uniques == (- 1)))[0]
                    uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]
                code_mapping = np.zeros((len(lev) + has_na))
                code_mapping[uniques] = (np.arange(len(uniques)) - has_na)
                level_codes = code_mapping[level_codes]
                lev = lev.take(uniques[has_na:])
            new_levels.append(lev)
            new_codes.append(level_codes)
        result = self.view()
        if changed:
            result._reset_identity()
            result._set_levels(new_levels, validate=False)
            result._set_codes(new_codes, validate=False)
        return result

    def __reduce__(self):
        'Necessary for making this object picklable'
        d = {'levels': list(self.levels), 'codes': list(self.codes), 'sortorder': self.sortorder, 'names': list(self.names)}
        return (ibase._new_Index, (type(self), d), None)

    def __getitem__(self, key):
        if is_scalar(key):
            key = com.cast_scalar_indexer(key, warn_float=True)
            retval = []
            for (lev, level_codes) in zip(self.levels, self.codes):
                if (level_codes[key] == (- 1)):
                    retval.append(np.nan)
                else:
                    retval.append(lev[level_codes[key]])
            return tuple(retval)
        else:
            if com.is_bool_indexer(key):
                key = np.asarray(key, dtype=bool)
                sortorder = self.sortorder
            else:
                sortorder = None
                if isinstance(key, Index):
                    key = np.asarray(key)
            new_codes = [level_codes[key] for level_codes in self.codes]
            return MultiIndex(levels=self.levels, codes=new_codes, names=self.names, sortorder=sortorder, verify_integrity=False)

    @Appender((_index_shared_docs['take'] % _index_doc_kwargs))
    def take(self, indices, axis=0, allow_fill=True, fill_value=None, **kwargs):
        nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)
        na_value = (- 1)
        if allow_fill:
            taken = [lab.take(indices) for lab in self.codes]
            mask = (indices == (- 1))
            if mask.any():
                masked = []
                for new_label in taken:
                    label_values = new_label
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked
        else:
            taken = [lab.take(indices) for lab in self.codes]
        return MultiIndex(levels=self.levels, codes=taken, names=self.names, verify_integrity=False)

    def append(self, other):
        '\n        Append a collection of Index options together\n\n        Parameters\n        ----------\n        other : Index or list/tuple of indices\n\n        Returns\n        -------\n        appended : Index\n        '
        if (not isinstance(other, (list, tuple))):
            other = [other]
        if all(((isinstance(o, MultiIndex) and (o.nlevels >= self.nlevels)) for o in other)):
            arrays = []
            for i in range(self.nlevels):
                label = self._get_level_values(i)
                appended = [o._get_level_values(i) for o in other]
                arrays.append(label.append(appended))
            return MultiIndex.from_arrays(arrays, names=self.names)
        to_concat = ((self._values,) + tuple((k._values for k in other)))
        new_tuples = np.concatenate(to_concat)
        try:
            return MultiIndex.from_tuples(new_tuples, names=self.names)
        except (TypeError, IndexError):
            return Index(new_tuples)

    def argsort(self, *args, **kwargs):
        return self._values.argsort(*args, **kwargs)

    @Appender((_index_shared_docs['repeat'] % _index_doc_kwargs))
    def repeat(self, repeats, axis=None):
        nv.validate_repeat((), {'axis': axis})
        repeats = ensure_platform_int(repeats)
        return MultiIndex(levels=self.levels, codes=[level_codes.view(np.ndarray).astype(np.intp).repeat(repeats) for level_codes in self.codes], names=self.names, sortorder=self.sortorder, verify_integrity=False)

    def where(self, cond, other=None):
        raise NotImplementedError('.where is not supported for MultiIndex operations')

    def drop(self, codes, level=None, errors='raise'):
        "\n        Make new MultiIndex with passed list of codes deleted\n\n        Parameters\n        ----------\n        codes : array-like\n            Must be a list of tuples when level is not specified\n        level : int or level name, default None\n        errors : str, default 'raise'\n\n        Returns\n        -------\n        dropped : MultiIndex\n        "
        if (level is not None):
            return self._drop_from_level(codes, level, errors)
        if (not isinstance(codes, (np.ndarray, Index))):
            try:
                codes = com.index_labels_to_array(codes, dtype=object)
            except ValueError:
                pass
        inds = []
        for level_codes in codes:
            try:
                loc = self.get_loc(level_codes)
                if isinstance(loc, int):
                    inds.append(loc)
                elif isinstance(loc, slice):
                    step = (loc.step if (loc.step is not None) else 1)
                    inds.extend(range(loc.start, loc.stop, step))
                elif com.is_bool_indexer(loc):
                    if (self.lexsort_depth == 0):
                        warnings.warn('dropping on a non-lexsorted multi-index without a level parameter may impact performance.', PerformanceWarning, stacklevel=3)
                    loc = loc.nonzero()[0]
                    inds.extend(loc)
                else:
                    msg = f'unsupported indexer of type {type(loc)}'
                    raise AssertionError(msg)
            except KeyError:
                if (errors != 'ignore'):
                    raise
        return self.delete(inds)

    def _drop_from_level(self, codes, level, errors='raise'):
        codes = com.index_labels_to_array(codes)
        i = self._get_level_number(level)
        index = self.levels[i]
        values = index.get_indexer(codes)
        nan_codes = isna(codes)
        values[(np.equal(nan_codes, False) & (values == (- 1)))] = (- 2)
        if (index.shape[0] == self.shape[0]):
            values[np.equal(nan_codes, True)] = (- 2)
        not_found = codes[(values == (- 2))]
        if ((len(not_found) != 0) and (errors != 'ignore')):
            raise KeyError(f'labels {not_found} not found in level')
        mask = (~ algos.isin(self.codes[i], values))
        return self[mask]

    def swaplevel(self, i=(- 2), j=(- 1)):
        "\n        Swap level i with level j.\n\n        Calling this method does not change the ordering of the values.\n\n        Parameters\n        ----------\n        i : int, str, default -2\n            First level of index to be swapped. Can pass level name as string.\n            Type of parameters can be mixed.\n        j : int, str, default -1\n            Second level of index to be swapped. Can pass level name as string.\n            Type of parameters can be mixed.\n\n        Returns\n        -------\n        MultiIndex\n            A new MultiIndex.\n\n        See Also\n        --------\n        Series.swaplevel : Swap levels i and j in a MultiIndex.\n        Dataframe.swaplevel : Swap levels i and j in a MultiIndex on a\n            particular axis.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],\n        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])\n        >>> mi\n        MultiIndex([('a', 'bb'),\n                    ('a', 'aa'),\n                    ('b', 'bb'),\n                    ('b', 'aa')],\n                   )\n        >>> mi.swaplevel(0, 1)\n        MultiIndex([('bb', 'a'),\n                    ('aa', 'a'),\n                    ('bb', 'b'),\n                    ('aa', 'b')],\n                   )\n        "
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)
        i = self._get_level_number(i)
        j = self._get_level_number(j)
        (new_levels[i], new_levels[j]) = (new_levels[j], new_levels[i])
        (new_codes[i], new_codes[j]) = (new_codes[j], new_codes[i])
        (new_names[i], new_names[j]) = (new_names[j], new_names[i])
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    def reorder_levels(self, order):
        "\n        Rearrange levels using input order. May not drop or duplicate levels.\n\n        Parameters\n        ----------\n        order : list of int or list of str\n            List representing new level order. Reference level by number\n            (position) or by key (label).\n\n        Returns\n        -------\n        MultiIndex\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=['x', 'y'])\n        >>> mi\n        MultiIndex([(1, 3),\n                    (2, 4)],\n                   names=['x', 'y'])\n\n        >>> mi.reorder_levels(order=[1, 0])\n        MultiIndex([(3, 1),\n                    (4, 2)],\n                   names=['y', 'x'])\n\n        >>> mi.reorder_levels(order=['y', 'x'])\n        MultiIndex([(3, 1),\n                    (4, 2)],\n                   names=['y', 'x'])\n        "
        order = [self._get_level_number(i) for i in order]
        if (len(order) != self.nlevels):
            raise AssertionError(f'Length of order must be same as number of levels ({self.nlevels}), got {len(order)}')
        new_levels = [self.levels[i] for i in order]
        new_codes = [self.codes[i] for i in order]
        new_names = [self.names[i] for i in order]
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    def _get_codes_for_sorting(self):
        '\n        we are categorizing our codes by using the\n        available categories (all, not just observed)\n        excluding any missing ones (-1); this is in preparation\n        for sorting, where we need to disambiguate that -1 is not\n        a valid valid\n        '

        def cats(level_codes):
            return np.arange(((np.array(level_codes).max() + 1) if len(level_codes) else 0), dtype=level_codes.dtype)
        return [Categorical.from_codes(level_codes, cats(level_codes), ordered=True) for level_codes in self.codes]

    def sortlevel(self, level=0, ascending=True, sort_remaining=True):
        '\n        Sort MultiIndex at the requested level.\n\n        The result will respect the original ordering of the associated\n        factor at that level.\n\n        Parameters\n        ----------\n        level : list-like, int or str, default 0\n            If a string is given, must be a name of the level.\n            If list-like must be names or ints of levels.\n        ascending : bool, default True\n            False to sort in descending order.\n            Can also be a list to specify a directed ordering.\n        sort_remaining : sort by the remaining levels after level\n\n        Returns\n        -------\n        sorted_index : pd.MultiIndex\n            Resulting index.\n        indexer : np.ndarray\n            Indices of output values in original index.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([[0, 0], [2, 1]])\n        >>> mi\n        MultiIndex([(0, 2),\n                    (0, 1)],\n                   )\n\n        >>> mi.sortlevel()\n        (MultiIndex([(0, 1),\n                    (0, 2)],\n                   ), array([1, 0]))\n\n        >>> mi.sortlevel(sort_remaining=False)\n        (MultiIndex([(0, 2),\n                    (0, 1)],\n                   ), array([0, 1]))\n\n        >>> mi.sortlevel(1)\n        (MultiIndex([(0, 1),\n                    (0, 2)],\n                   ), array([1, 0]))\n\n        >>> mi.sortlevel(1, ascending=False)\n        (MultiIndex([(0, 2),\n                    (0, 1)],\n                   ), array([0, 1]))\n        '
        if isinstance(level, (str, int)):
            level = [level]
        level = [self._get_level_number(lev) for lev in level]
        sortorder = None
        if isinstance(ascending, list):
            if (not (len(level) == len(ascending))):
                raise ValueError('level must have same length as ascending')
            indexer = lexsort_indexer([self.codes[lev] for lev in level], orders=ascending)
        else:
            codes = list(self.codes)
            shape = list(self.levshape)
            primary = tuple((codes[lev] for lev in level))
            primshp = tuple((shape[lev] for lev in level))
            for lev in sorted(level, reverse=True):
                codes.pop(lev)
                shape.pop(lev)
            if sort_remaining:
                primary += (primary + tuple(codes))
                primshp += (primshp + tuple(shape))
            else:
                sortorder = level[0]
            indexer = indexer_from_factorized(primary, primshp, compress=False)
            if (not ascending):
                indexer = indexer[::(- 1)]
        indexer = ensure_platform_int(indexer)
        new_codes = [level_codes.take(indexer) for level_codes in self.codes]
        new_index = MultiIndex(codes=new_codes, levels=self.levels, names=self.names, sortorder=sortorder, verify_integrity=False)
        return (new_index, indexer)

    def reindex(self, target, method=None, level=None, limit=None, tolerance=None):
        "\n        Create index with target's values (move/add/delete values as necessary)\n\n        Returns\n        -------\n        new_index : pd.MultiIndex\n            Resulting index\n        indexer : np.ndarray or None\n            Indices of output values in original index.\n\n        "
        preserve_names = (not hasattr(target, 'names'))
        if (level is not None):
            if (method is not None):
                raise TypeError('Fill method not supported if level passed')
            target = ibase.ensure_has_len(target)
            if ((len(target) == 0) and (not isinstance(target, Index))):
                idx = self.levels[level]
                attrs = idx._get_attributes_dict()
                attrs.pop('freq', None)
                target = type(idx)._simple_new(np.empty(0, dtype=idx.dtype), **attrs)
            else:
                target = ensure_index(target)
            (target, indexer, _) = self._join_level(target, level, how='right', return_indexers=True, keep_order=False)
        else:
            target = ensure_index(target)
            if self.equals(target):
                indexer = None
            elif self.is_unique:
                indexer = self.get_indexer(target, method=method, limit=limit, tolerance=tolerance)
            else:
                raise ValueError('cannot handle a non-unique multi-index!')
        if (not isinstance(target, MultiIndex)):
            if (indexer is None):
                target = self
            elif (indexer >= 0).all():
                target = self.take(indexer)
            else:
                target = MultiIndex.from_tuples(target)
        if (preserve_names and (target.nlevels == self.nlevels) and (target.names != self.names)):
            target = target.copy(deep=False)
            target.names = self.names
        return (target, indexer)

    def _check_indexing_error(self, key):
        if ((not is_hashable(key)) or is_iterator(key)):
            raise InvalidIndexError(key)

    def _should_fallback_to_positional(self):
        '\n        Should integer key(s) be treated as positional?\n        '
        return self.levels[0]._should_fallback_to_positional()

    def _get_values_for_loc(self, series, loc, key):
        '\n        Do a positional lookup on the given Series, returning either a scalar\n        or a Series.\n\n        Assumes that `series.index is self`\n        '
        new_values = series._values[loc]
        if is_scalar(loc):
            return new_values
        if ((len(new_values) == 1) and (not (self.nlevels > 1))):
            return new_values[0]
        new_index = self[loc]
        new_index = maybe_droplevels(new_index, key)
        new_ser = series._constructor(new_values, index=new_index, name=series.name)
        return new_ser.__finalize__(series)

    def _convert_listlike_indexer(self, keyarr):
        '\n        Parameters\n        ----------\n        keyarr : list-like\n            Indexer to convert.\n\n        Returns\n        -------\n        tuple (indexer, keyarr)\n            indexer is an ndarray or None if cannot convert\n            keyarr are tuple-safe keys\n        '
        (indexer, keyarr) = super()._convert_listlike_indexer(keyarr)
        if ((indexer is None) and len(keyarr) and (not isinstance(keyarr[0], tuple))):
            level = 0
            (_, indexer) = self.reindex(keyarr, level=level)
            if (indexer is None):
                indexer = np.arange(len(self))
            check = self.levels[0].get_indexer(keyarr)
            mask = (check == (- 1))
            if mask.any():
                raise KeyError(f'{keyarr[mask]} not in index')
        return (indexer, keyarr)

    def _get_partial_string_timestamp_match_key(self, key):
        '\n        Translate any partial string timestamp matches in key, returning the\n        new key.\n\n        Only relevant for MultiIndex.\n        '
        if (isinstance(key, str) and self.levels[0]._supports_partial_string_indexing):
            key = ((key,) + ((slice(None),) * (len(self.levels) - 1)))
        if isinstance(key, tuple):
            new_key = []
            for (i, component) in enumerate(key):
                if (isinstance(component, str) and self.levels[i]._supports_partial_string_indexing):
                    new_key.append(slice(component, component, None))
                else:
                    new_key.append(component)
            key = tuple(new_key)
        return key

    def _get_indexer(self, target, method=None, limit=None, tolerance=None):
        if (not len(target)):
            return ensure_platform_int(np.array([]))
        if (not isinstance(target, MultiIndex)):
            try:
                target = MultiIndex.from_tuples(target)
            except (TypeError, ValueError):
                if (method is None):
                    return Index(self._values).get_indexer(target, method=method, limit=limit, tolerance=tolerance)
        if ((method == 'pad') or (method == 'backfill')):
            if (tolerance is not None):
                raise NotImplementedError('tolerance not implemented yet for MultiIndex')
            indexer = self._engine.get_indexer(values=self._values, target=target, method=method, limit=limit)
        elif (method == 'nearest'):
            raise NotImplementedError("method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365")
        else:
            indexer = self._engine.get_indexer(target)
        return ensure_platform_int(indexer)

    def get_slice_bound(self, label, side, kind):
        '\n        For an ordered MultiIndex, compute slice bound\n        that corresponds to given label.\n\n        Returns leftmost (one-past-the-rightmost if `side==\'right\') position\n        of given label.\n\n        Parameters\n        ----------\n        label : object or tuple of objects\n        side : {\'left\', \'right\'}\n        kind : {\'loc\', \'getitem\'}\n\n        Returns\n        -------\n        int\n            Index of label.\n\n        Notes\n        -----\n        This method only works if level 0 index of the MultiIndex is lexsorted.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([list(\'abbc\'), list(\'gefd\')])\n\n        Get the locations from the leftmost \'b\' in the first level\n        until the end of the multiindex:\n\n        >>> mi.get_slice_bound(\'b\', side="left", kind="loc")\n        1\n\n        Like above, but if you get the locations from the rightmost\n        \'b\' in the first level and \'f\' in the second level:\n\n        >>> mi.get_slice_bound((\'b\',\'f\'), side="right", kind="loc")\n        3\n\n        See Also\n        --------\n        MultiIndex.get_loc : Get location for a label or a tuple of labels.\n        MultiIndex.get_locs : Get location for a label/slice/list/mask or a\n                              sequence of such.\n        '
        if (not isinstance(label, tuple)):
            label = (label,)
        return self._partial_tup_index(label, side=side)

    def slice_locs(self, start=None, end=None, step=None, kind=None):
        "\n        For an ordered MultiIndex, compute the slice locations for input\n        labels.\n\n        The input labels can be tuples representing partial levels, e.g. for a\n        MultiIndex with 3 levels, you can pass a single value (corresponding to\n        the first level), or a 1-, 2-, or 3-tuple.\n\n        Parameters\n        ----------\n        start : label or tuple, default None\n            If None, defaults to the beginning\n        end : label or tuple\n            If None, defaults to the end\n        step : int or None\n            Slice step\n        kind : string, optional, defaults None\n\n        Returns\n        -------\n        (start, end) : (int, int)\n\n        Notes\n        -----\n        This method only works if the MultiIndex is properly lexsorted. So,\n        if only the first 2 levels of a 3-level MultiIndex are lexsorted,\n        you can only pass two levels to ``.slice_locs``.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([list('abbd'), list('deff')],\n        ...                                names=['A', 'B'])\n\n        Get the slice locations from the beginning of 'b' in the first level\n        until the end of the multiindex:\n\n        >>> mi.slice_locs(start='b')\n        (1, 4)\n\n        Like above, but stop at the end of 'b' in the first level and 'f' in\n        the second level:\n\n        >>> mi.slice_locs(start='b', end=('b', 'f'))\n        (1, 3)\n\n        See Also\n        --------\n        MultiIndex.get_loc : Get location for a label or a tuple of labels.\n        MultiIndex.get_locs : Get location for a label/slice/list/mask or a\n                              sequence of such.\n        "
        return super().slice_locs(start, end, step, kind=kind)

    def _partial_tup_index(self, tup, side='left'):
        if (len(tup) > self.lexsort_depth):
            raise UnsortedIndexError(f'Key length ({len(tup)}) was greater than MultiIndex lexsort depth ({self.lexsort_depth})')
        n = len(tup)
        (start, end) = (0, len(self))
        zipped = zip(tup, self.levels, self.codes)
        for (k, (lab, lev, labs)) in enumerate(zipped):
            section = labs[start:end]
            if ((lab not in lev) and (not isna(lab))):
                if (not lev.is_type_compatible(lib.infer_dtype([lab], skipna=False))):
                    raise TypeError(f'Level type mismatch: {lab}')
                loc = lev.searchsorted(lab, side=side)
                if ((side == 'right') and (loc >= 0)):
                    loc -= 1
                return (start + section.searchsorted(loc, side=side))
            idx = self._get_loc_single_level_index(lev, lab)
            if (isinstance(idx, slice) and (k < (n - 1))):
                start = idx.start
                end = idx.stop
            elif (k < (n - 1)):
                end = (start + section.searchsorted(idx, side='right'))
                start = (start + section.searchsorted(idx, side='left'))
            elif isinstance(idx, slice):
                idx = idx.start
                return (start + section.searchsorted(idx, side=side))
            else:
                return (start + section.searchsorted(idx, side=side))

    def _get_loc_single_level_index(self, level_index, key):
        '\n        If key is NA value, location of index unify as -1.\n\n        Parameters\n        ----------\n        level_index: Index\n        key : label\n\n        Returns\n        -------\n        loc : int\n            If key is NA value, loc is -1\n            Else, location of key in index.\n\n        See Also\n        --------\n        Index.get_loc : The get_loc method for (single-level) index.\n        '
        if (is_scalar(key) and isna(key)):
            return (- 1)
        else:
            return level_index.get_loc(key)

    def get_loc(self, key, method=None):
        "\n        Get location for a label or a tuple of labels.\n\n        The location is returned as an integer/slice or boolean\n        mask.\n\n        Parameters\n        ----------\n        key : label or tuple of labels (one for each level)\n        method : None\n\n        Returns\n        -------\n        loc : int, slice object or boolean mask\n            If the key is past the lexsort depth, the return may be a\n            boolean mask array, otherwise it is always a slice or int.\n\n        See Also\n        --------\n        Index.get_loc : The get_loc method for (single-level) index.\n        MultiIndex.slice_locs : Get slice location given start label(s) and\n                                end label(s).\n        MultiIndex.get_locs : Get location for a label/slice/list/mask or a\n                              sequence of such.\n\n        Notes\n        -----\n        The key cannot be a slice, list of same-level labels, a boolean mask,\n        or a sequence of such. If you want to use those, use\n        :meth:`MultiIndex.get_locs` instead.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])\n\n        >>> mi.get_loc('b')\n        slice(1, 3, None)\n\n        >>> mi.get_loc(('b', 'e'))\n        1\n        "
        if (method is not None):
            raise NotImplementedError('only the default get_loc method is currently supported for MultiIndex')
        hash(key)

        def _maybe_to_slice(loc):
            'convert integer indexer to boolean mask or slice if possible'
            if ((not isinstance(loc, np.ndarray)) or (loc.dtype != np.intp)):
                return loc
            loc = lib.maybe_indices_to_slice(loc, len(self))
            if isinstance(loc, slice):
                return loc
            mask = np.empty(len(self), dtype='bool')
            mask.fill(False)
            mask[loc] = True
            return mask
        if (not isinstance(key, tuple)):
            loc = self._get_level_indexer(key, level=0)
            return _maybe_to_slice(loc)
        keylen = len(key)
        if (self.nlevels < keylen):
            raise KeyError(f'Key length ({keylen}) exceeds index depth ({self.nlevels})')
        if ((keylen == self.nlevels) and self.is_unique):
            return self._engine.get_loc(key)
        i = self.lexsort_depth
        (lead_key, follow_key) = (key[:i], key[i:])
        (start, stop) = (self.slice_locs(lead_key, lead_key) if lead_key else (0, len(self)))
        if (start == stop):
            raise KeyError(key)
        if (not follow_key):
            return slice(start, stop)
        warnings.warn('indexing past lexsort depth may impact performance.', PerformanceWarning, stacklevel=10)
        loc = np.arange(start, stop, dtype=np.intp)
        for (i, k) in enumerate(follow_key, len(lead_key)):
            mask = (self.codes[i][loc] == self._get_loc_single_level_index(self.levels[i], k))
            if (not mask.all()):
                loc = loc[mask]
            if (not len(loc)):
                raise KeyError(key)
        return (_maybe_to_slice(loc) if (len(loc) != (stop - start)) else slice(start, stop))

    def get_loc_level(self, key, level=0, drop_level=True):
        "\n        Get location and sliced index for requested label(s)/level(s).\n\n        Parameters\n        ----------\n        key : label or sequence of labels\n        level : int/level name or list thereof, optional\n        drop_level : bool, default True\n            If ``False``, the resulting index will not drop any level.\n\n        Returns\n        -------\n        loc : A 2-tuple where the elements are:\n              Element 0: int, slice object or boolean array\n              Element 1: The resulting sliced multiindex/index. If the key\n              contains all levels, this will be ``None``.\n\n        See Also\n        --------\n        MultiIndex.get_loc  : Get location for a label or a tuple of labels.\n        MultiIndex.get_locs : Get location for a label/slice/list/mask or a\n                              sequence of such.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')],\n        ...                                names=['A', 'B'])\n\n        >>> mi.get_loc_level('b')\n        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))\n\n        >>> mi.get_loc_level('e', level='B')\n        (array([False,  True, False]), Index(['b'], dtype='object', name='A'))\n\n        >>> mi.get_loc_level(['b', 'e'])\n        (1, None)\n        "
        if (not isinstance(level, (list, tuple))):
            level = self._get_level_number(level)
        else:
            level = [self._get_level_number(lev) for lev in level]
        return self._get_loc_level(key, level=level, drop_level=drop_level)

    def _get_loc_level(self, key, level=0, drop_level=True):
        '\n        get_loc_level but with `level` known to be positional, not name-based.\n        '

        def maybe_mi_droplevels(indexer, levels, drop_level: bool):
            if (not drop_level):
                return self[indexer]
            orig_index = new_index = self[indexer]
            for i in sorted(levels, reverse=True):
                try:
                    new_index = new_index._drop_level_numbers([i])
                except ValueError:
                    return orig_index
            return new_index
        if isinstance(level, (tuple, list)):
            if (len(key) != len(level)):
                raise AssertionError('Key for location must have same length as number of levels')
            result = None
            for (lev, k) in zip(level, key):
                (loc, new_index) = self._get_loc_level(k, level=lev)
                if isinstance(loc, slice):
                    mask = np.zeros(len(self), dtype=bool)
                    mask[loc] = True
                    loc = mask
                result = (loc if (result is None) else (result & loc))
            return (result, maybe_mi_droplevels(result, level, drop_level))
        if isinstance(key, list):
            key = tuple(key)
        if (isinstance(key, tuple) and (level == 0)):
            try:
                if (key in self.levels[0]):
                    indexer = self._get_level_indexer(key, level=level)
                    new_index = maybe_mi_droplevels(indexer, [0], drop_level)
                    return (indexer, new_index)
            except (TypeError, InvalidIndexError):
                pass
            if (not any((isinstance(k, slice) for k in key))):

                def partial_selection(key, indexer=None):
                    if (indexer is None):
                        indexer = self.get_loc(key)
                    ilevels = [i for i in range(len(key)) if (key[i] != slice(None, None))]
                    return (indexer, maybe_mi_droplevels(indexer, ilevels, drop_level))
                if ((len(key) == self.nlevels) and self.is_unique):
                    try:
                        return (self._engine.get_loc(key), None)
                    except KeyError as e:
                        raise KeyError(key) from e
                else:
                    return partial_selection(key)
            else:
                indexer = None
                for (i, k) in enumerate(key):
                    if (not isinstance(k, slice)):
                        k = self._get_level_indexer(k, level=i)
                        if isinstance(k, slice):
                            if ((k.start == 0) and (k.stop == len(self))):
                                k = slice(None, None)
                        else:
                            k_index = k
                    if isinstance(k, slice):
                        if (k == slice(None, None)):
                            continue
                        else:
                            raise TypeError(key)
                    if (indexer is None):
                        indexer = k_index
                    else:
                        indexer &= k_index
                if (indexer is None):
                    indexer = slice(None, None)
                ilevels = [i for i in range(len(key)) if (key[i] != slice(None, None))]
                return (indexer, maybe_mi_droplevels(indexer, ilevels, drop_level))
        else:
            indexer = self._get_level_indexer(key, level=level)
            return (indexer, maybe_mi_droplevels(indexer, [level], drop_level))

    def _get_level_indexer(self, key, level=0, indexer=None):
        level_index = self.levels[level]
        level_codes = self.codes[level]

        def convert_indexer(start, stop, step, indexer=indexer, codes=level_codes):
            if ((step is not None) and (step < 0)):
                (start, stop) = ((stop - 1), (start - 1))
            r = np.arange(start, stop, step)
            if ((indexer is not None) and (len(indexer) != len(codes))):
                from pandas import Series
                mapper = Series(indexer)
                indexer = codes.take(ensure_platform_int(indexer))
                result = Series(Index(indexer).isin(r).nonzero()[0])
                m = result.map(mapper)
                m = np.asarray(m)
            else:
                m = np.zeros(len(codes), dtype=bool)
                m[np.in1d(codes, r, assume_unique=Index(codes).is_unique)] = True
            return m
        if isinstance(key, slice):
            try:
                if (key.start is not None):
                    start = level_index.get_loc(key.start)
                else:
                    start = 0
                if (key.stop is not None):
                    stop = level_index.get_loc(key.stop)
                elif isinstance(start, slice):
                    stop = len(level_index)
                else:
                    stop = (len(level_index) - 1)
                step = key.step
            except KeyError:
                start = stop = level_index.slice_indexer(key.start, key.stop, key.step, kind='loc')
                step = start.step
            if (isinstance(start, slice) or isinstance(stop, slice)):
                start = getattr(start, 'start', start)
                stop = getattr(stop, 'stop', stop)
                return convert_indexer(start, stop, step)
            elif ((level > 0) or (self.lexsort_depth == 0) or (step is not None)):
                return convert_indexer(start, (stop + 1), step)
            else:
                i = level_codes.searchsorted(start, side='left')
                j = level_codes.searchsorted(stop, side='right')
                return slice(i, j, step)
        else:
            idx = self._get_loc_single_level_index(level_index, key)
            if ((level > 0) or (self.lexsort_depth == 0)):
                locs = np.array((level_codes == idx), dtype=bool, copy=False)
                if (not locs.any()):
                    raise KeyError(key)
                return locs
            if isinstance(idx, slice):
                start = idx.start
                end = idx.stop
            else:
                start = level_codes.searchsorted(idx, side='left')
                end = level_codes.searchsorted(idx, side='right')
            if (start == end):
                raise KeyError(key)
            return slice(start, end)

    def get_locs(self, seq):
        "\n        Get location for a sequence of labels.\n\n        Parameters\n        ----------\n        seq : label, slice, list, mask or a sequence of such\n           You should use one of the above for each level.\n           If a level should not be used, set it to ``slice(None)``.\n\n        Returns\n        -------\n        numpy.ndarray\n            NumPy array of integers suitable for passing to iloc.\n\n        See Also\n        --------\n        MultiIndex.get_loc : Get location for a label or a tuple of labels.\n        MultiIndex.slice_locs : Get slice location given start label(s) and\n                                end label(s).\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])\n\n        >>> mi.get_locs('b')  # doctest: +SKIP\n        array([1, 2], dtype=int64)\n\n        >>> mi.get_locs([slice(None), ['e', 'f']])  # doctest: +SKIP\n        array([1, 2], dtype=int64)\n\n        >>> mi.get_locs([[True, False, True], slice('e', 'f')])  # doctest: +SKIP\n        array([2], dtype=int64)\n        "
        true_slices = [i for (i, s) in enumerate(com.is_true_slices(seq)) if s]
        if (true_slices and (true_slices[(- 1)] >= self.lexsort_depth)):
            raise UnsortedIndexError(f'MultiIndex slicing requires the index to be lexsorted: slicing on levels {true_slices}, lexsort depth {self.lexsort_depth}')
        n = len(self)
        indexer = None

        def _convert_to_indexer(r) -> Int64Index:
            if isinstance(r, slice):
                m = np.zeros(n, dtype=bool)
                m[r] = True
                r = m.nonzero()[0]
            elif com.is_bool_indexer(r):
                if (len(r) != n):
                    raise ValueError('cannot index with a boolean indexer that is not the same length as the index')
                r = r.nonzero()[0]
            return Int64Index(r)

        def _update_indexer(idxr: Optional[Index], indexer: Optional[Index], key) -> Index:
            if (indexer is None):
                indexer = Index(np.arange(n))
            if (idxr is None):
                return indexer
            indexer_intersection = indexer.intersection(idxr)
            if (indexer_intersection.empty and (not idxr.empty) and (not indexer.empty)):
                raise KeyError(key)
            return indexer_intersection
        for (i, k) in enumerate(seq):
            if com.is_bool_indexer(k):
                k = np.asarray(k)
                indexer = _update_indexer(_convert_to_indexer(k), indexer=indexer, key=seq)
            elif is_list_like(k):
                indexers: Optional[Int64Index] = None
                for x in k:
                    try:
                        idxrs = _convert_to_indexer(self._get_level_indexer(x, level=i, indexer=indexer))
                        indexers = (idxrs if (indexers is None) else indexers).union(idxrs, sort=False)
                    except KeyError:
                        continue
                if (indexers is not None):
                    indexer = _update_indexer(indexers, indexer=indexer, key=seq)
                else:
                    return np.array([], dtype=np.int64)
            elif com.is_null_slice(k):
                indexer = _update_indexer(None, indexer=indexer, key=seq)
            elif isinstance(k, slice):
                indexer = _update_indexer(_convert_to_indexer(self._get_level_indexer(k, level=i, indexer=indexer)), indexer=indexer, key=seq)
            else:
                indexer = _update_indexer(_convert_to_indexer(self.get_loc_level(k, level=i, drop_level=False)[0]), indexer=indexer, key=seq)
        if (indexer is None):
            return np.array([], dtype=np.int64)
        assert isinstance(indexer, Int64Index), type(indexer)
        indexer = self._reorder_indexer(seq, indexer)
        return indexer._values

    def _reorder_indexer(self, seq, indexer):
        '\n        Reorder an indexer of a MultiIndex (self) so that the label are in the\n        same order as given in seq\n\n        Parameters\n        ----------\n        seq : label/slice/list/mask or a sequence of such\n        indexer: an Int64Index indexer of self\n\n        Returns\n        -------\n        indexer : a sorted Int64Index indexer of self ordered as seq\n        '
        if self.is_lexsorted():
            need_sort = False
            for (i, k) in enumerate(seq):
                if is_list_like(k):
                    if (not need_sort):
                        k_codes = self.levels[i].get_indexer(k)
                        k_codes = k_codes[(k_codes >= 0)]
                        need_sort = (k_codes[:(- 1)] > k_codes[1:]).any()
                elif (isinstance(k, slice) and (k.step is not None) and (k.step < 0)):
                    need_sort = True
            if (not need_sort):
                return indexer
        n = len(self)
        keys: Tuple[(np.ndarray, ...)] = ()
        for (i, k) in enumerate(seq):
            if is_scalar(k):
                k = [k]
            if com.is_bool_indexer(k):
                new_order = np.arange(n)[indexer]
            elif is_list_like(k):
                key_order_map = (np.ones(len(self.levels[i]), dtype=np.uint64) * len(self.levels[i]))
                level_indexer = self.levels[i].get_indexer(k)
                level_indexer = level_indexer[(level_indexer >= 0)]
                key_order_map[level_indexer] = np.arange(len(level_indexer))
                new_order = key_order_map[self.codes[i][indexer]]
            elif (isinstance(k, slice) and (k.step is not None) and (k.step < 0)):
                new_order = np.arange(n)[k][indexer]
            elif (isinstance(k, slice) and (k.start is None) and (k.stop is None)):
                new_order = np.ones((n,))[indexer]
            else:
                new_order = np.arange(n)[indexer]
            keys = ((new_order,) + keys)
        ind = np.lexsort(keys)
        return indexer[ind]

    def truncate(self, before=None, after=None):
        '\n        Slice index between two labels / tuples, return new MultiIndex\n\n        Parameters\n        ----------\n        before : label or tuple, can be partial. Default None\n            None defaults to start\n        after : label or tuple, can be partial. Default None\n            None defaults to end\n\n        Returns\n        -------\n        truncated : MultiIndex\n        '
        if (after and before and (after < before)):
            raise ValueError('after < before')
        (i, j) = self.levels[0].slice_locs(before, after)
        (left, right) = self.slice_locs(before, after)
        new_levels = list(self.levels)
        new_levels[0] = new_levels[0][i:j]
        new_codes = [level_codes[left:right] for level_codes in self.codes]
        new_codes[0] = (new_codes[0] - i)
        return MultiIndex(levels=new_levels, codes=new_codes, names=self._names, verify_integrity=False)

    def equals(self, other):
        '\n        Determines if two MultiIndex objects have the same labeling information\n        (the levels themselves do not necessarily have to be the same)\n\n        See Also\n        --------\n        equal_levels\n        '
        if self.is_(other):
            return True
        if (not isinstance(other, Index)):
            return False
        if (len(self) != len(other)):
            return False
        if (not isinstance(other, MultiIndex)):
            if (not self._should_compare(other)):
                return False
            return array_equivalent(self._values, other._values)
        if (self.nlevels != other.nlevels):
            return False
        for i in range(self.nlevels):
            self_codes = self.codes[i]
            other_codes = other.codes[i]
            self_mask = (self_codes == (- 1))
            other_mask = (other_codes == (- 1))
            if (not np.array_equal(self_mask, other_mask)):
                return False
            self_codes = self_codes[(~ self_mask)]
            self_values = algos.take_nd(np.asarray(self.levels[i]._values), self_codes, allow_fill=False)
            other_codes = other_codes[(~ other_mask)]
            other_values = algos.take_nd(np.asarray(other.levels[i]._values), other_codes, allow_fill=False)
            if ((len(self_values) == 0) and (len(other_values) == 0)):
                continue
            if (not array_equivalent(self_values, other_values)):
                return False
        return True

    def equal_levels(self, other):
        '\n        Return True if the levels of both MultiIndex objects are the same\n\n        '
        if (self.nlevels != other.nlevels):
            return False
        for i in range(self.nlevels):
            if (not self.levels[i].equals(other.levels[i])):
                return False
        return True

    def _union(self, other, sort):
        (other, result_names) = self._convert_can_do_setop(other)
        rvals = other._values.astype(object, copy=False)
        uniq_tuples = lib.fast_unique_multiple([self._values, rvals], sort=sort)
        return MultiIndex.from_arrays(zip(*uniq_tuples), sortorder=0, names=result_names)

    def _is_comparable_dtype(self, dtype):
        return is_object_dtype(dtype)

    def _get_reconciled_name_object(self, other):
        '\n        If the result of a set operation will be self,\n        return self, unless the names change, in which\n        case make a shallow copy of self.\n        '
        names = self._maybe_match_names(other)
        if (self.names != names):
            return self.rename(names)
        return self

    def _maybe_match_names(self, other):
        '\n        Try to find common names to attach to the result of an operation between\n        a and b.  Return a consensus list of names if they match at least partly\n        or list of None if they have completely different names.\n        '
        if (len(self.names) != len(other.names)):
            return ([None] * len(self.names))
        names = []
        for (a_name, b_name) in zip(self.names, other.names):
            if (a_name == b_name):
                names.append(a_name)
            else:
                names.append(None)
        return names

    def _intersection(self, other, sort=False):
        (other, result_names) = self._convert_can_do_setop(other)
        lvals = self._values
        rvals = other._values.astype(object, copy=False)
        uniq_tuples = None
        if (self.is_monotonic and other.is_monotonic):
            try:
                inner_tuples = self._inner_indexer(lvals, rvals)[0]
                sort = False
            except TypeError:
                pass
            else:
                uniq_tuples = algos.unique(inner_tuples)
        if (uniq_tuples is None):
            other_uniq = set(rvals)
            seen = set()
            uniq_tuples = [x for x in lvals if ((x in other_uniq) and (not ((x in seen) or seen.add(x))))]
        if (sort is None):
            uniq_tuples = sorted(uniq_tuples)
        if (len(uniq_tuples) == 0):
            return MultiIndex(levels=self.levels, codes=([[]] * self.nlevels), names=result_names, verify_integrity=False)
        else:
            return MultiIndex.from_arrays(zip(*uniq_tuples), sortorder=0, names=result_names)

    def _difference(self, other, sort):
        (other, result_names) = self._convert_can_do_setop(other)
        this = self._get_unique_index()
        indexer = this.get_indexer(other)
        indexer = indexer.take((indexer != (- 1)).nonzero()[0])
        label_diff = np.setdiff1d(np.arange(this.size), indexer, assume_unique=True)
        difference = this._values.take(label_diff)
        if (sort is None):
            difference = sorted(difference)
        if (len(difference) == 0):
            return MultiIndex(levels=([[]] * self.nlevels), codes=([[]] * self.nlevels), names=result_names, verify_integrity=False)
        else:
            return MultiIndex.from_tuples(difference, sortorder=0, names=result_names)

    def _convert_can_do_setop(self, other):
        result_names = self.names
        if (not isinstance(other, Index)):
            if (len(other) == 0):
                return (self[:0], self.names)
            else:
                msg = 'other must be a MultiIndex or a list of tuples'
                try:
                    other = MultiIndex.from_tuples(other, names=self.names)
                except (ValueError, TypeError) as err:
                    raise TypeError(msg) from err
        else:
            result_names = get_unanimous_names(self, other)
        return (other, result_names)

    def symmetric_difference(self, other, result_name=None, sort=None):
        tups = Index.symmetric_difference(self, other, result_name, sort)
        if (len(tups) == 0):
            return type(self)(levels=[[] for _ in range(self.nlevels)], codes=[[] for _ in range(self.nlevels)], names=tups.name)
        return type(self).from_tuples(tups, names=tups.name)

    @doc(Index.astype)
    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_categorical_dtype(dtype):
            msg = '> 1 ndim Categorical are not supported at this time'
            raise NotImplementedError(msg)
        elif (not is_object_dtype(dtype)):
            raise TypeError('Setting a MultiIndex dtype to anything other than object is not supported')
        elif (copy is True):
            return self._shallow_copy()
        return self

    def _validate_fill_value(self, item):
        if (not isinstance(item, tuple)):
            item = ((item,) + (('',) * (self.nlevels - 1)))
        elif (len(item) != self.nlevels):
            raise ValueError('Item must have length equal to number of levels.')
        return item

    def insert(self, loc, item):
        '\n        Make new MultiIndex inserting new item at location\n\n        Parameters\n        ----------\n        loc : int\n        item : tuple\n            Must be same length as number of levels in the MultiIndex\n\n        Returns\n        -------\n        new_index : Index\n        '
        item = self._validate_fill_value(item)
        new_levels = []
        new_codes = []
        for (k, level, level_codes) in zip(item, self.levels, self.codes):
            if (k not in level):
                lev_loc = len(level)
                try:
                    level = level.insert(lev_loc, k)
                except TypeError:
                    level = level.astype(object).insert(lev_loc, k)
            else:
                lev_loc = level.get_loc(k)
            new_levels.append(level)
            new_codes.append(np.insert(ensure_int64(level_codes), loc, lev_loc))
        return MultiIndex(levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False)

    def delete(self, loc):
        '\n        Make new index with passed location deleted\n\n        Returns\n        -------\n        new_index : MultiIndex\n        '
        new_codes = [np.delete(level_codes, loc) for level_codes in self.codes]
        return MultiIndex(levels=self.levels, codes=new_codes, names=self.names, verify_integrity=False)

    @doc(Index.isin)
    def isin(self, values, level=None):
        if (level is None):
            values = MultiIndex.from_tuples(values, names=self.names)._values
            return algos.isin(self._values, values)
        else:
            num = self._get_level_number(level)
            levs = self.get_level_values(num)
            if (levs.size == 0):
                return np.zeros(len(levs), dtype=np.bool_)
            return levs.isin(values)
    __add__ = make_invalid_op('__add__')
    __radd__ = make_invalid_op('__radd__')
    __iadd__ = make_invalid_op('__iadd__')
    __sub__ = make_invalid_op('__sub__')
    __rsub__ = make_invalid_op('__rsub__')
    __isub__ = make_invalid_op('__isub__')
    __pow__ = make_invalid_op('__pow__')
    __rpow__ = make_invalid_op('__rpow__')
    __mul__ = make_invalid_op('__mul__')
    __rmul__ = make_invalid_op('__rmul__')
    __floordiv__ = make_invalid_op('__floordiv__')
    __rfloordiv__ = make_invalid_op('__rfloordiv__')
    __truediv__ = make_invalid_op('__truediv__')
    __rtruediv__ = make_invalid_op('__rtruediv__')
    __mod__ = make_invalid_op('__mod__')
    __rmod__ = make_invalid_op('__rmod__')
    __divmod__ = make_invalid_op('__divmod__')
    __rdivmod__ = make_invalid_op('__rdivmod__')
    __neg__ = make_invalid_op('__neg__')
    __pos__ = make_invalid_op('__pos__')
    __abs__ = make_invalid_op('__abs__')
    __inv__ = make_invalid_op('__inv__')

def sparsify_labels(label_list, start=0, sentinel=''):
    pivoted = list(zip(*label_list))
    k = len(label_list)
    result = pivoted[:(start + 1)]
    prev = pivoted[start]
    for cur in pivoted[(start + 1):]:
        sparse_cur = []
        for (i, (p, t)) in enumerate(zip(prev, cur)):
            if (i == (k - 1)):
                sparse_cur.append(t)
                result.append(sparse_cur)
                break
            if (p == t):
                sparse_cur.append(sentinel)
            else:
                sparse_cur.extend(cur[i:])
                result.append(sparse_cur)
                break
        prev = cur
    return list(zip(*result))

def _get_na_rep(dtype):
    return {np.datetime64: 'NaT', np.timedelta64: 'NaT'}.get(dtype, 'NaN')

def maybe_droplevels(index, key):
    '\n    Attempt to drop level or levels from the given index.\n\n    Parameters\n    ----------\n    index: Index\n    key : scalar or tuple\n\n    Returns\n    -------\n    Index\n    '
    original_index = index
    if isinstance(key, tuple):
        for _ in key:
            try:
                index = index._drop_level_numbers([0])
            except ValueError:
                return original_index
    else:
        try:
            index = index._drop_level_numbers([0])
        except ValueError:
            pass
    return index

def _coerce_indexer_frozen(array_like, categories, copy=False):
    '\n    Coerce the array_like indexer to the smallest integer dtype that can encode all\n    of the given categories.\n\n    Parameters\n    ----------\n    array_like : array-like\n    categories : array-like\n    copy : bool\n\n    Returns\n    -------\n    np.ndarray\n        Non-writeable.\n    '
    array_like = coerce_indexer_dtype(array_like, categories)
    if copy:
        array_like = array_like.copy()
    array_like.flags.writeable = False
    return array_like
