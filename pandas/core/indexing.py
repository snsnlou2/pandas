
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Hashable, List, Sequence, Tuple, Union
import warnings
import numpy as np
from pandas._config.config import option_context
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.errors import AbstractMethodError, InvalidIndexError
from pandas.util._decorators import doc
from pandas.core.dtypes.common import is_array_like, is_hashable, is_integer, is_iterator, is_list_like, is_numeric_dtype, is_object_dtype, is_scalar, is_sequence
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import ABCDataFrame, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.missing import infer_fill_value, isna
import pandas.core.common as com
from pandas.core.construction import array as pd_array
from pandas.core.indexers import check_array_indexer, is_list_like_indexer, length_of_indexer
from pandas.core.indexes.api import Index
if TYPE_CHECKING:
    from pandas import DataFrame, Series
_NS = slice(None, None)

class _IndexSlice():
    "\n    Create an object to more easily perform multi-index slicing.\n\n    See Also\n    --------\n    MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.\n\n    Notes\n    -----\n    See :ref:`Defined Levels <advanced.shown_levels>`\n    for further info on slicing a MultiIndex.\n\n    Examples\n    --------\n    >>> midx = pd.MultiIndex.from_product([['A0','A1'], ['B0','B1','B2','B3']])\n    >>> columns = ['foo', 'bar']\n    >>> dfmi = pd.DataFrame(np.arange(16).reshape((len(midx), len(columns))),\n    ...                     index=midx, columns=columns)\n\n    Using the default slice command:\n\n    >>> dfmi.loc[(slice(None), slice('B0', 'B1')), :]\n               foo  bar\n        A0 B0    0    1\n           B1    2    3\n        A1 B0    8    9\n           B1   10   11\n\n    Using the IndexSlice class for a more intuitive command:\n\n    >>> idx = pd.IndexSlice\n    >>> dfmi.loc[idx[:, 'B0':'B1'], :]\n               foo  bar\n        A0 B0    0    1\n           B1    2    3\n        A1 B0    8    9\n           B1   10   11\n    "

    def __getitem__(self, arg):
        return arg
IndexSlice = _IndexSlice()

class IndexingError(Exception):
    pass

class IndexingMixin():
    '\n    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.\n    '

    @property
    def iloc(self):
        "\n        Purely integer-location based indexing for selection by position.\n\n        ``.iloc[]`` is primarily integer position based (from ``0`` to\n        ``length-1`` of the axis), but may also be used with a boolean\n        array.\n\n        Allowed inputs are:\n\n        - An integer, e.g. ``5``.\n        - A list or array of integers, e.g. ``[4, 3, 0]``.\n        - A slice object with ints, e.g. ``1:7``.\n        - A boolean array.\n        - A ``callable`` function with one argument (the calling Series or\n          DataFrame) and that returns valid output for indexing (one of the above).\n          This is useful in method chains, when you don't have a reference to the\n          calling object, but would like to base your selection on some value.\n\n        ``.iloc`` will raise ``IndexError`` if a requested indexer is\n        out-of-bounds, except *slice* indexers which allow out-of-bounds\n        indexing (this conforms with python/numpy *slice* semantics).\n\n        See more at :ref:`Selection by Position <indexing.integer>`.\n\n        See Also\n        --------\n        DataFrame.iat : Fast integer location scalar accessor.\n        DataFrame.loc : Purely label-location based indexer for selection by label.\n        Series.iloc : Purely integer-location based indexing for\n                       selection by position.\n\n        Examples\n        --------\n        >>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},\n        ...           {'a': 100, 'b': 200, 'c': 300, 'd': 400},\n        ...           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]\n        >>> df = pd.DataFrame(mydict)\n        >>> df\n              a     b     c     d\n        0     1     2     3     4\n        1   100   200   300   400\n        2  1000  2000  3000  4000\n\n        **Indexing just the rows**\n\n        With a scalar integer.\n\n        >>> type(df.iloc[0])\n        <class 'pandas.core.series.Series'>\n        >>> df.iloc[0]\n        a    1\n        b    2\n        c    3\n        d    4\n        Name: 0, dtype: int64\n\n        With a list of integers.\n\n        >>> df.iloc[[0]]\n           a  b  c  d\n        0  1  2  3  4\n        >>> type(df.iloc[[0]])\n        <class 'pandas.core.frame.DataFrame'>\n\n        >>> df.iloc[[0, 1]]\n             a    b    c    d\n        0    1    2    3    4\n        1  100  200  300  400\n\n        With a `slice` object.\n\n        >>> df.iloc[:3]\n              a     b     c     d\n        0     1     2     3     4\n        1   100   200   300   400\n        2  1000  2000  3000  4000\n\n        With a boolean mask the same length as the index.\n\n        >>> df.iloc[[True, False, True]]\n              a     b     c     d\n        0     1     2     3     4\n        2  1000  2000  3000  4000\n\n        With a callable, useful in method chains. The `x` passed\n        to the ``lambda`` is the DataFrame being sliced. This selects\n        the rows whose index label even.\n\n        >>> df.iloc[lambda x: x.index % 2 == 0]\n              a     b     c     d\n        0     1     2     3     4\n        2  1000  2000  3000  4000\n\n        **Indexing both axes**\n\n        You can mix the indexer types for the index and columns. Use ``:`` to\n        select the entire axis.\n\n        With scalar integers.\n\n        >>> df.iloc[0, 1]\n        2\n\n        With lists of integers.\n\n        >>> df.iloc[[0, 2], [1, 3]]\n              b     d\n        0     2     4\n        2  2000  4000\n\n        With `slice` objects.\n\n        >>> df.iloc[1:3, 0:3]\n              a     b     c\n        1   100   200   300\n        2  1000  2000  3000\n\n        With a boolean array whose length matches the columns.\n\n        >>> df.iloc[:, [True, False, True, False]]\n              a     c\n        0     1     3\n        1   100   300\n        2  1000  3000\n\n        With a callable function that expects the Series or DataFrame.\n\n        >>> df.iloc[:, lambda df: [0, 2]]\n              a     c\n        0     1     3\n        1   100   300\n        2  1000  3000\n        "
        return _iLocIndexer('iloc', self)

    @property
    def loc(self):
        '\n        Access a group of rows and columns by label(s) or a boolean array.\n\n        ``.loc[]`` is primarily label based, but may also be used with a\n        boolean array.\n\n        Allowed inputs are:\n\n        - A single label, e.g. ``5`` or ``\'a\'``, (note that ``5`` is\n          interpreted as a *label* of the index, and **never** as an\n          integer position along the index).\n        - A list or array of labels, e.g. ``[\'a\', \'b\', \'c\']``.\n        - A slice object with labels, e.g. ``\'a\':\'f\'``.\n\n          .. warning:: Note that contrary to usual python slices, **both** the\n              start and the stop are included\n\n        - A boolean array of the same length as the axis being sliced,\n          e.g. ``[True, False, True]``.\n        - An alignable boolean Series. The index of the key will be aligned before\n          masking.\n        - An alignable Index. The Index of the returned selection will be the input.\n        - A ``callable`` function with one argument (the calling Series or\n          DataFrame) and that returns valid output for indexing (one of the above)\n\n        See more at :ref:`Selection by Label <indexing.label>`.\n\n        Raises\n        ------\n        KeyError\n            If any items are not found.\n        IndexingError\n            If an indexed key is passed and its index is unalignable to the frame index.\n\n        See Also\n        --------\n        DataFrame.at : Access a single value for a row/column label pair.\n        DataFrame.iloc : Access group of rows and columns by integer position(s).\n        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the\n            Series/DataFrame.\n        Series.loc : Access group of values using labels.\n\n        Examples\n        --------\n        **Getting values**\n\n        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],\n        ...      index=[\'cobra\', \'viper\', \'sidewinder\'],\n        ...      columns=[\'max_speed\', \'shield\'])\n        >>> df\n                    max_speed  shield\n        cobra               1       2\n        viper               4       5\n        sidewinder          7       8\n\n        Single label. Note this returns the row as a Series.\n\n        >>> df.loc[\'viper\']\n        max_speed    4\n        shield       5\n        Name: viper, dtype: int64\n\n        List of labels. Note using ``[[]]`` returns a DataFrame.\n\n        >>> df.loc[[\'viper\', \'sidewinder\']]\n                    max_speed  shield\n        viper               4       5\n        sidewinder          7       8\n\n        Single label for row and column\n\n        >>> df.loc[\'cobra\', \'shield\']\n        2\n\n        Slice with labels for row and single label for column. As mentioned\n        above, note that both the start and stop of the slice are included.\n\n        >>> df.loc[\'cobra\':\'viper\', \'max_speed\']\n        cobra    1\n        viper    4\n        Name: max_speed, dtype: int64\n\n        Boolean list with the same length as the row axis\n\n        >>> df.loc[[False, False, True]]\n                    max_speed  shield\n        sidewinder          7       8\n\n        Alignable boolean Series:\n\n        >>> df.loc[pd.Series([False, True, False],\n        ...        index=[\'viper\', \'sidewinder\', \'cobra\'])]\n                    max_speed  shield\n        sidewinder          7       8\n\n        Index (same behavior as ``df.reindex``)\n\n        >>> df.loc[pd.Index(["cobra", "viper"], name="foo")]\n               max_speed  shield\n        foo\n        cobra          1       2\n        viper          4       5\n\n        Conditional that returns a boolean Series\n\n        >>> df.loc[df[\'shield\'] > 6]\n                    max_speed  shield\n        sidewinder          7       8\n\n        Conditional that returns a boolean Series with column labels specified\n\n        >>> df.loc[df[\'shield\'] > 6, [\'max_speed\']]\n                    max_speed\n        sidewinder          7\n\n        Callable that returns a boolean Series\n\n        >>> df.loc[lambda df: df[\'shield\'] == 8]\n                    max_speed  shield\n        sidewinder          7       8\n\n        **Setting values**\n\n        Set value for all items matching the list of labels\n\n        >>> df.loc[[\'viper\', \'sidewinder\'], [\'shield\']] = 50\n        >>> df\n                    max_speed  shield\n        cobra               1       2\n        viper               4      50\n        sidewinder          7      50\n\n        Set value for an entire row\n\n        >>> df.loc[\'cobra\'] = 10\n        >>> df\n                    max_speed  shield\n        cobra              10      10\n        viper               4      50\n        sidewinder          7      50\n\n        Set value for an entire column\n\n        >>> df.loc[:, \'max_speed\'] = 30\n        >>> df\n                    max_speed  shield\n        cobra              30      10\n        viper              30      50\n        sidewinder         30      50\n\n        Set value for rows matching callable condition\n\n        >>> df.loc[df[\'shield\'] > 35] = 0\n        >>> df\n                    max_speed  shield\n        cobra              30      10\n        viper               0       0\n        sidewinder          0       0\n\n        **Getting values on a DataFrame with an index that has integer labels**\n\n        Another example using integers for the index\n\n        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],\n        ...      index=[7, 8, 9], columns=[\'max_speed\', \'shield\'])\n        >>> df\n           max_speed  shield\n        7          1       2\n        8          4       5\n        9          7       8\n\n        Slice with integer labels for rows. As mentioned above, note that both\n        the start and stop of the slice are included.\n\n        >>> df.loc[7:9]\n           max_speed  shield\n        7          1       2\n        8          4       5\n        9          7       8\n\n        **Getting values with a MultiIndex**\n\n        A number of examples using a DataFrame with a MultiIndex\n\n        >>> tuples = [\n        ...    (\'cobra\', \'mark i\'), (\'cobra\', \'mark ii\'),\n        ...    (\'sidewinder\', \'mark i\'), (\'sidewinder\', \'mark ii\'),\n        ...    (\'viper\', \'mark ii\'), (\'viper\', \'mark iii\')\n        ... ]\n        >>> index = pd.MultiIndex.from_tuples(tuples)\n        >>> values = [[12, 2], [0, 4], [10, 20],\n        ...         [1, 4], [7, 1], [16, 36]]\n        >>> df = pd.DataFrame(values, columns=[\'max_speed\', \'shield\'], index=index)\n        >>> df\n                             max_speed  shield\n        cobra      mark i           12       2\n                   mark ii           0       4\n        sidewinder mark i           10      20\n                   mark ii           1       4\n        viper      mark ii           7       1\n                   mark iii         16      36\n\n        Single label. Note this returns a DataFrame with a single index.\n\n        >>> df.loc[\'cobra\']\n                 max_speed  shield\n        mark i          12       2\n        mark ii          0       4\n\n        Single index tuple. Note this returns a Series.\n\n        >>> df.loc[(\'cobra\', \'mark ii\')]\n        max_speed    0\n        shield       4\n        Name: (cobra, mark ii), dtype: int64\n\n        Single label for row and column. Similar to passing in a tuple, this\n        returns a Series.\n\n        >>> df.loc[\'cobra\', \'mark i\']\n        max_speed    12\n        shield        2\n        Name: (cobra, mark i), dtype: int64\n\n        Single tuple. Note using ``[[]]`` returns a DataFrame.\n\n        >>> df.loc[[(\'cobra\', \'mark ii\')]]\n                       max_speed  shield\n        cobra mark ii          0       4\n\n        Single tuple for the index with a single label for the column\n\n        >>> df.loc[(\'cobra\', \'mark i\'), \'shield\']\n        2\n\n        Slice from index tuple to single label\n\n        >>> df.loc[(\'cobra\', \'mark i\'):\'viper\']\n                             max_speed  shield\n        cobra      mark i           12       2\n                   mark ii           0       4\n        sidewinder mark i           10      20\n                   mark ii           1       4\n        viper      mark ii           7       1\n                   mark iii         16      36\n\n        Slice from index tuple to index tuple\n\n        >>> df.loc[(\'cobra\', \'mark i\'):(\'viper\', \'mark ii\')]\n                            max_speed  shield\n        cobra      mark i          12       2\n                   mark ii          0       4\n        sidewinder mark i          10      20\n                   mark ii          1       4\n        viper      mark ii          7       1\n        '
        return _LocIndexer('loc', self)

    @property
    def at(self):
        "\n        Access a single value for a row/column label pair.\n\n        Similar to ``loc``, in that both provide label-based lookups. Use\n        ``at`` if you only need to get or set a single value in a DataFrame\n        or Series.\n\n        Raises\n        ------\n        KeyError\n            If 'label' does not exist in DataFrame.\n\n        See Also\n        --------\n        DataFrame.iat : Access a single value for a row/column pair by integer\n            position.\n        DataFrame.loc : Access a group of rows and columns by label(s).\n        Series.at : Access a single value using a label.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],\n        ...                   index=[4, 5, 6], columns=['A', 'B', 'C'])\n        >>> df\n            A   B   C\n        4   0   2   3\n        5   0   4   1\n        6  10  20  30\n\n        Get value at specified row/column pair\n\n        >>> df.at[4, 'B']\n        2\n\n        Set value at specified row/column pair\n\n        >>> df.at[4, 'B'] = 10\n        >>> df.at[4, 'B']\n        10\n\n        Get value within a Series\n\n        >>> df.loc[5].at['B']\n        4\n        "
        return _AtIndexer('at', self)

    @property
    def iat(self):
        "\n        Access a single value for a row/column pair by integer position.\n\n        Similar to ``iloc``, in that both provide integer-based lookups. Use\n        ``iat`` if you only need to get or set a single value in a DataFrame\n        or Series.\n\n        Raises\n        ------\n        IndexError\n            When integer position is out of bounds.\n\n        See Also\n        --------\n        DataFrame.at : Access a single value for a row/column label pair.\n        DataFrame.loc : Access a group of rows and columns by label(s).\n        DataFrame.iloc : Access a group of rows and columns by integer position(s).\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],\n        ...                   columns=['A', 'B', 'C'])\n        >>> df\n            A   B   C\n        0   0   2   3\n        1   0   4   1\n        2  10  20  30\n\n        Get value at specified row/column pair\n\n        >>> df.iat[1, 2]\n        1\n\n        Set value at specified row/column pair\n\n        >>> df.iat[1, 2] = 10\n        >>> df.iat[1, 2]\n        10\n\n        Get value within a series\n\n        >>> df.loc[0].iat[1]\n        2\n        "
        return _iAtIndexer('iat', self)

class _LocationIndexer(NDFrameIndexerBase):
    axis = None

    def __call__(self, axis=None):
        new_self = type(self)(self.name, self.obj)
        if (axis is not None):
            axis = self.obj._get_axis_number(axis)
        new_self.axis = axis
        return new_self

    def _get_setitem_indexer(self, key):
        '\n        Convert a potentially-label-based key into a positional indexer.\n        '
        if (self.name == 'loc'):
            self._ensure_listlike_indexer(key)
        if (self.axis is not None):
            return self._convert_tuple(key, is_setter=True)
        ax = self.obj._get_axis(0)
        if (isinstance(ax, ABCMultiIndex) and (self.name != 'iloc')):
            with suppress(TypeError, KeyError, InvalidIndexError):
                return ax.get_loc(key)
        if isinstance(key, tuple):
            with suppress(IndexingError):
                return self._convert_tuple(key, is_setter=True)
        if isinstance(key, range):
            return list(key)
        try:
            return self._convert_to_indexer(key, axis=0, is_setter=True)
        except TypeError as e:
            if ('cannot do' in str(e)):
                raise
            elif ('unhashable type' in str(e)):
                raise
            raise IndexingError(key) from e

    def _ensure_listlike_indexer(self, key, axis=None, value=None):
        '\n        Ensure that a list-like of column labels are all present by adding them if\n        they do not already exist.\n\n        Parameters\n        ----------\n        key : list-like of column labels\n            Target labels.\n        axis : key axis if known\n        '
        column_axis = 1
        if (self.ndim != 2):
            return
        if (isinstance(key, tuple) and (not isinstance(self.obj.index, ABCMultiIndex))):
            key = key[column_axis]
            axis = column_axis
        if ((axis == column_axis) and (not isinstance(self.obj.columns, ABCMultiIndex)) and is_list_like_indexer(key) and (not com.is_bool_indexer(key)) and all((is_hashable(k) for k in key))):
            keys = self.obj.columns.union(key, sort=False)
            self.obj._mgr = self.obj._mgr.reindex_axis(keys, axis=0, copy=False, consolidate=False, only_slice=True)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
        else:
            key = com.apply_if_callable(key, self.obj)
        indexer = self._get_setitem_indexer(key)
        self._has_valid_setitem_indexer(key)
        iloc = (self if (self.name == 'iloc') else self.obj.iloc)
        iloc._setitem_with_indexer(indexer, value, self.name)

    def _validate_key(self, key, axis):
        '\n        Ensure that key is valid for current indexer.\n\n        Parameters\n        ----------\n        key : scalar, slice or list-like\n            Key requested.\n        axis : int\n            Dimension on which the indexing is being made.\n\n        Raises\n        ------\n        TypeError\n            If the key (or some element of it) has wrong type.\n        IndexError\n            If the key (or some element of it) is out of bounds.\n        KeyError\n            If the key was not found.\n        '
        raise AbstractMethodError(self)

    def _has_valid_tuple(self, key):
        '\n        Check the key for valid keys across my indexer.\n        '
        self._validate_key_length(key)
        for (i, k) in enumerate(key):
            try:
                self._validate_key(k, i)
            except ValueError as err:
                raise ValueError(f'Location based indexing can only have [{self._valid_types}] types') from err

    def _is_nested_tuple_indexer(self, tup):
        '\n        Returns\n        -------\n        bool\n        '
        if any((isinstance(ax, ABCMultiIndex) for ax in self.obj.axes)):
            return any((is_nested_tuple(tup, ax) for ax in self.obj.axes))
        return False

    def _convert_tuple(self, key, is_setter=False):
        keyidx = []
        if (self.axis is not None):
            axis = self.obj._get_axis_number(self.axis)
            for i in range(self.ndim):
                if (i == axis):
                    keyidx.append(self._convert_to_indexer(key, axis=axis, is_setter=is_setter))
                else:
                    keyidx.append(slice(None))
        else:
            self._validate_key_length(key)
            for (i, k) in enumerate(key):
                idx = self._convert_to_indexer(k, axis=i, is_setter=is_setter)
                keyidx.append(idx)
        return tuple(keyidx)

    def _validate_key_length(self, key):
        if (len(key) > self.ndim):
            raise IndexingError('Too many indexers')

    def _getitem_tuple_same_dim(self, tup):
        '\n        Index with indexers that should return an object of the same dimension\n        as self.obj.\n\n        This is only called after a failed call to _getitem_lowerdim.\n        '
        retval = self.obj
        for (i, key) in enumerate(tup):
            if com.is_null_slice(key):
                continue
            retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
            assert (retval.ndim == self.ndim)
        return retval

    def _getitem_lowerdim(self, tup):
        if (self.axis is not None):
            axis = self.obj._get_axis_number(self.axis)
            return self._getitem_axis(tup, axis=axis)
        if self._is_nested_tuple_indexer(tup):
            return self._getitem_nested_tuple(tup)
        ax0 = self.obj._get_axis(0)
        if (isinstance(ax0, ABCMultiIndex) and (self.name != 'iloc')):
            with suppress(IndexingError):
                return self._handle_lowerdim_multi_index_axis0(tup)
        self._validate_key_length(tup)
        for (i, key) in enumerate(tup):
            if is_label_like(key):
                section = self._getitem_axis(key, axis=i)
                if (section.ndim == self.ndim):
                    new_key = ((tup[:i] + (_NS,)) + tup[(i + 1):])
                else:
                    new_key = (tup[:i] + tup[(i + 1):])
                    if (len(new_key) == 1):
                        new_key = new_key[0]
                if com.is_null_slice(new_key):
                    return section
                return getattr(section, self.name)[new_key]
        raise IndexingError('not applicable')

    def _getitem_nested_tuple(self, tup):
        if (len(tup) > self.ndim):
            if (self.name != 'loc'):
                raise ValueError('Too many indices')
            if ((self.ndim == 1) or (not any((isinstance(x, slice) for x in tup)))):
                with suppress(IndexingError):
                    return self._handle_lowerdim_multi_index_axis0(tup)
            axis = (self.axis or 0)
            return self._getitem_axis(tup, axis=axis)
        obj = self.obj
        axis = 0
        for key in tup:
            if com.is_null_slice(key):
                axis += 1
                continue
            current_ndim = obj.ndim
            obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
            axis += 1
            if (is_scalar(obj) or (not hasattr(obj, 'ndim'))):
                break
            if (obj.ndim < current_ndim):
                axis -= 1
        return obj

    def _convert_to_indexer(self, key, axis, is_setter=False):
        raise AbstractMethodError(self)

    def __getitem__(self, key):
        if (type(key) is tuple):
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
            if self._is_scalar_access(key):
                with suppress(KeyError, IndexError, AttributeError):
                    return self.obj._get_value(*key, takeable=self._takeable)
            return self._getitem_tuple(key)
        else:
            axis = (self.axis or 0)
            maybe_callable = com.apply_if_callable(key, self.obj)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _is_scalar_access(self, key):
        raise NotImplementedError()

    def _getitem_tuple(self, tup):
        raise AbstractMethodError(self)

    def _getitem_axis(self, key, axis):
        raise NotImplementedError()

    def _has_valid_setitem_indexer(self, indexer):
        raise AbstractMethodError(self)

    def _getbool_axis(self, key, axis):
        labels = self.obj._get_axis(axis)
        key = check_bool_indexer(labels, key)
        inds = key.nonzero()[0]
        return self.obj._take_with_is_copy(inds, axis=axis)

@doc(IndexingMixin.loc)
class _LocIndexer(_LocationIndexer):
    _takeable = False
    _valid_types = 'labels (MUST BE IN THE INDEX), slices of labels (BOTH endpoints included! Can be slices of integers if the index is integers), listlike of labels, boolean'

    @doc(_LocationIndexer._validate_key)
    def _validate_key(self, key, axis):
        pass

    def _has_valid_setitem_indexer(self, indexer):
        return True

    def _is_scalar_access(self, key):
        '\n        Returns\n        -------\n        bool\n        '
        if (len(key) != self.ndim):
            return False
        for (i, k) in enumerate(key):
            if (not is_scalar(k)):
                return False
            ax = self.obj.axes[i]
            if isinstance(ax, ABCMultiIndex):
                return False
            if (isinstance(k, str) and ax._supports_partial_string_indexing):
                return False
            if (not ax.is_unique):
                return False
        return True

    def _multi_take_opportunity(self, tup):
        '\n        Check whether there is the possibility to use ``_multi_take``.\n\n        Currently the limit is that all axes being indexed, must be indexed with\n        list-likes.\n\n        Parameters\n        ----------\n        tup : tuple\n            Tuple of indexers, one per axis.\n\n        Returns\n        -------\n        bool\n            Whether the current indexing,\n            can be passed through `_multi_take`.\n        '
        if (not all((is_list_like_indexer(x) for x in tup))):
            return False
        if any((com.is_bool_indexer(x) for x in tup)):
            return False
        return True

    def _multi_take(self, tup):
        '\n        Create the indexers for the passed tuple of keys, and\n        executes the take operation. This allows the take operation to be\n        executed all at once, rather than once for each dimension.\n        Improving efficiency.\n\n        Parameters\n        ----------\n        tup : tuple\n            Tuple of indexers, one per axis.\n\n        Returns\n        -------\n        values: same type as the object being indexed\n        '
        d = {axis: self._get_listlike_indexer(key, axis) for (key, axis) in zip(tup, self.obj._AXIS_ORDERS)}
        return self.obj._reindex_with_indexers(d, copy=True, allow_dups=True)

    def _getitem_iterable(self, key, axis):
        '\n        Index current object with an iterable collection of keys.\n\n        Parameters\n        ----------\n        key : iterable\n            Targeted labels.\n        axis: int\n            Dimension on which the indexing is being made.\n\n        Raises\n        ------\n        KeyError\n            If no key was found. Will change in the future to raise if not all\n            keys were found.\n\n        Returns\n        -------\n        scalar, DataFrame, or Series: indexed value(s).\n        '
        self._validate_key(key, axis)
        (keyarr, indexer) = self._get_listlike_indexer(key, axis, raise_missing=False)
        return self.obj._reindex_with_indexers({axis: [keyarr, indexer]}, copy=True, allow_dups=True)

    def _getitem_tuple(self, tup):
        with suppress(IndexingError):
            return self._getitem_lowerdim(tup)
        self._has_valid_tuple(tup)
        if self._multi_take_opportunity(tup):
            return self._multi_take(tup)
        return self._getitem_tuple_same_dim(tup)

    def _get_label(self, label, axis):
        return self.obj.xs(label, axis=axis)

    def _handle_lowerdim_multi_index_axis0(self, tup):
        axis = (self.axis or 0)
        try:
            return self._get_label(tup, axis=axis)
        except (TypeError, InvalidIndexError):
            pass
        except KeyError as ek:
            if (self.ndim < len(tup) <= self.obj.index.nlevels):
                raise ek
        raise IndexingError('No label returned')

    def _getitem_axis(self, key, axis):
        key = item_from_zerodim(key)
        if is_iterator(key):
            key = list(key)
        labels = self.obj._get_axis(axis)
        key = labels._get_partial_string_timestamp_match_key(key)
        if isinstance(key, slice):
            self._validate_key(key, axis)
            return self._get_slice_axis(key, axis=axis)
        elif com.is_bool_indexer(key):
            return self._getbool_axis(key, axis=axis)
        elif is_list_like_indexer(key):
            if (not (isinstance(key, tuple) and isinstance(labels, ABCMultiIndex))):
                if (hasattr(key, 'ndim') and (key.ndim > 1)):
                    raise ValueError('Cannot index with multidimensional key')
                return self._getitem_iterable(key, axis=axis)
            if is_nested_tuple(key, labels):
                locs = labels.get_locs(key)
                indexer = ([slice(None)] * self.ndim)
                indexer[axis] = locs
                return self.obj.iloc[tuple(indexer)]
        self._validate_key(key, axis)
        return self._get_label(key, axis=axis)

    def _get_slice_axis(self, slice_obj, axis):
        '\n        This is pretty simple as we just have to deal with labels.\n        '
        obj = self.obj
        if (not need_slice(slice_obj)):
            return obj.copy(deep=False)
        labels = obj._get_axis(axis)
        indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop, slice_obj.step, kind='loc')
        if isinstance(indexer, slice):
            return self.obj._slice(indexer, axis=axis)
        else:
            return self.obj.take(indexer, axis=axis)

    def _convert_to_indexer(self, key, axis, is_setter=False):
        "\n        Convert indexing key into something we can use to do actual fancy\n        indexing on a ndarray.\n\n        Examples\n        ix[:5] -> slice(0, 5)\n        ix[[1,2,3]] -> [1,2,3]\n        ix[['foo', 'bar', 'baz']] -> [i, j, k] (indices of foo, bar, baz)\n\n        Going by Zen of Python?\n        'In the face of ambiguity, refuse the temptation to guess.'\n        raise AmbiguousIndexError with integer labels?\n        - No, prefer label-based indexing\n        "
        labels = self.obj._get_axis(axis)
        if isinstance(key, slice):
            return labels._convert_slice_indexer(key, kind='loc')
        is_int_index = labels.is_integer()
        is_int_positional = (is_integer(key) and (not is_int_index))
        if (is_scalar(key) or isinstance(labels, ABCMultiIndex)):
            try:
                return labels.get_loc(key)
            except LookupError:
                if (isinstance(key, tuple) and isinstance(labels, ABCMultiIndex)):
                    if (len(key) == labels.nlevels):
                        return {'key': key}
                    raise
            except InvalidIndexError:
                if (not isinstance(labels, ABCMultiIndex)):
                    raise
            except TypeError:
                pass
            except ValueError:
                if (not is_int_positional):
                    raise
        if is_int_positional:
            return {'key': key}
        if is_nested_tuple(key, labels):
            return labels.get_locs(key)
        elif is_list_like_indexer(key):
            if com.is_bool_indexer(key):
                key = check_bool_indexer(labels, key)
                (inds,) = key.nonzero()
                return inds
            else:
                return self._get_listlike_indexer(key, axis, raise_missing=True)[1]
        else:
            try:
                return labels.get_loc(key)
            except LookupError:
                if (not is_list_like_indexer(key)):
                    return {'key': key}
                raise

    def _get_listlike_indexer(self, key, axis, raise_missing=False):
        "\n        Transform a list-like of keys into a new index and an indexer.\n\n        Parameters\n        ----------\n        key : list-like\n            Targeted labels.\n        axis: int\n            Dimension on which the indexing is being made.\n        raise_missing: bool, default False\n            Whether to raise a KeyError if some labels were not found.\n            Will be removed in the future, and then this method will always behave as\n            if ``raise_missing=True``.\n\n        Raises\n        ------\n        KeyError\n            If at least one key was requested but none was found, and\n            raise_missing=True.\n\n        Returns\n        -------\n        keyarr: Index\n            New index (coinciding with 'key' if the axis is unique).\n        values : array-like\n            Indexer for the return object, -1 denotes keys not found.\n        "
        ax = self.obj._get_axis(axis)
        (indexer, keyarr) = ax._convert_listlike_indexer(key)
        if ((indexer is not None) and (indexer != (- 1)).all()):
            return (ax[indexer], indexer)
        if ax._index_as_unique:
            indexer = ax.get_indexer_for(keyarr)
            keyarr = ax.reindex(keyarr)[0]
        else:
            (keyarr, indexer, new_indexer) = ax._reindex_non_unique(keyarr)
        self._validate_read_indexer(keyarr, indexer, axis, raise_missing=raise_missing)
        return (keyarr, indexer)

    def _validate_read_indexer(self, key, indexer, axis, raise_missing=False):
        '\n        Check that indexer can be used to return a result.\n\n        e.g. at least one element was found,\n        unless the list of keys was actually empty.\n\n        Parameters\n        ----------\n        key : list-like\n            Targeted labels (only used to show correct error message).\n        indexer: array-like of booleans\n            Indices corresponding to the key,\n            (with -1 indicating not found).\n        axis: int\n            Dimension on which the indexing is being made.\n        raise_missing: bool\n            Whether to raise a KeyError if some labels are not found. Will be\n            removed in the future, and then this method will always behave as\n            if raise_missing=True.\n\n        Raises\n        ------\n        KeyError\n            If at least one key was requested but none was found, and\n            raise_missing=True.\n        '
        if (len(key) == 0):
            return
        missing_mask = (indexer < 0)
        missing = missing_mask.sum()
        if missing:
            if (missing == len(indexer)):
                axis_name = self.obj._get_axis_name(axis)
                raise KeyError(f'None of [{key}] are in the [{axis_name}]')
            ax = self.obj._get_axis(axis)
            if raise_missing:
                not_found = list((set(key) - set(ax)))
                raise KeyError(f'{not_found} not in index')
            not_found = key[missing_mask]
            with option_context('display.max_seq_items', 10, 'display.width', 80):
                raise KeyError(f'Passing list-likes to .loc or [] with any missing labels is no longer supported. The following labels were missing: {not_found}. See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike')

@doc(IndexingMixin.iloc)
class _iLocIndexer(_LocationIndexer):
    _valid_types = 'integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array'
    _takeable = True

    def _validate_key(self, key, axis):
        if com.is_bool_indexer(key):
            if (hasattr(key, 'index') and isinstance(key.index, Index)):
                if (key.index.inferred_type == 'integer'):
                    raise NotImplementedError('iLocation based boolean indexing on an integer type is not available')
                raise ValueError('iLocation based boolean indexing cannot use an indexable as a mask')
            return
        if isinstance(key, slice):
            return
        elif is_integer(key):
            self._validate_integer(key, axis)
        elif isinstance(key, tuple):
            raise IndexingError('Too many indexers')
        elif is_list_like_indexer(key):
            arr = np.array(key)
            len_axis = len(self.obj._get_axis(axis))
            if (not is_numeric_dtype(arr.dtype)):
                raise IndexError(f'.iloc requires numeric indexers, got {arr}')
            if (len(arr) and ((arr.max() >= len_axis) or (arr.min() < (- len_axis)))):
                raise IndexError('positional indexers are out-of-bounds')
        else:
            raise ValueError(f'Can only index by location with a [{self._valid_types}]')

    def _has_valid_setitem_indexer(self, indexer):
        '\n        Validate that a positional indexer cannot enlarge its target\n        will raise if needed, does not modify the indexer externally.\n\n        Returns\n        -------\n        bool\n        '
        if isinstance(indexer, dict):
            raise IndexError('iloc cannot enlarge its target object')
        if (not isinstance(indexer, tuple)):
            indexer = _tuplify(self.ndim, indexer)
        for (ax, i) in zip(self.obj.axes, indexer):
            if isinstance(i, slice):
                pass
            elif is_list_like_indexer(i):
                pass
            elif is_integer(i):
                if (i >= len(ax)):
                    raise IndexError('iloc cannot enlarge its target object')
            elif isinstance(i, dict):
                raise IndexError('iloc cannot enlarge its target object')
        return True

    def _is_scalar_access(self, key):
        '\n        Returns\n        -------\n        bool\n        '
        if (len(key) != self.ndim):
            return False
        for k in key:
            if (not is_integer(k)):
                return False
        return True

    def _validate_integer(self, key, axis):
        "\n        Check that 'key' is a valid position in the desired axis.\n\n        Parameters\n        ----------\n        key : int\n            Requested position.\n        axis : int\n            Desired axis.\n\n        Raises\n        ------\n        IndexError\n            If 'key' is not a valid position in axis 'axis'.\n        "
        len_axis = len(self.obj._get_axis(axis))
        if ((key >= len_axis) or (key < (- len_axis))):
            raise IndexError('single positional indexer is out-of-bounds')

    def _getitem_tuple(self, tup):
        self._has_valid_tuple(tup)
        with suppress(IndexingError):
            return self._getitem_lowerdim(tup)
        return self._getitem_tuple_same_dim(tup)

    def _get_list_axis(self, key, axis):
        '\n        Return Series values by list or array of integers.\n\n        Parameters\n        ----------\n        key : list-like positional indexer\n        axis : int\n\n        Returns\n        -------\n        Series object\n\n        Notes\n        -----\n        `axis` can only be zero.\n        '
        try:
            return self.obj._take_with_is_copy(key, axis=axis)
        except IndexError as err:
            raise IndexError('positional indexers are out-of-bounds') from err

    def _getitem_axis(self, key, axis):
        if isinstance(key, slice):
            return self._get_slice_axis(key, axis=axis)
        if isinstance(key, list):
            key = np.asarray(key)
        if com.is_bool_indexer(key):
            self._validate_key(key, axis)
            return self._getbool_axis(key, axis=axis)
        elif is_list_like_indexer(key):
            return self._get_list_axis(key, axis=axis)
        else:
            key = item_from_zerodim(key)
            if (not is_integer(key)):
                raise TypeError('Cannot index by location index with a non-integer key')
            self._validate_integer(key, axis)
            return self.obj._ixs(key, axis=axis)

    def _get_slice_axis(self, slice_obj, axis):
        obj = self.obj
        if (not need_slice(slice_obj)):
            return obj.copy(deep=False)
        labels = obj._get_axis(axis)
        labels._validate_positional_slice(slice_obj)
        return self.obj._slice(slice_obj, axis=axis)

    def _convert_to_indexer(self, key, axis, is_setter=False):
        '\n        Much simpler as we only have to deal with our valid types.\n        '
        return key

    def _get_setitem_indexer(self, key):
        return key

    def _setitem_with_indexer(self, indexer, value, name='iloc'):
        '\n        _setitem_with_indexer is for setting values on a Series/DataFrame\n        using positional indexers.\n\n        If the relevant keys are not present, the Series/DataFrame may be\n        expanded.\n\n        This method is currently broken when dealing with non-unique Indexes,\n        since it goes from positional indexers back to labels when calling\n        BlockManager methods, see GH#12991, GH#22046, GH#15686.\n        '
        info_axis = self.obj._info_axis_number
        take_split_path = (not self.obj._mgr.is_single_block)
        if ((not take_split_path) and self.obj._mgr.blocks):
            if (self.ndim > 1):
                val = (list(value.values()) if isinstance(value, dict) else value)
                blk = self.obj._mgr.blocks[0]
                take_split_path = (not blk._can_hold_element(val))
        if (isinstance(indexer, tuple) and (len(indexer) == len(self.obj.axes))):
            for (i, ax) in zip(indexer, self.obj.axes):
                if (isinstance(ax, ABCMultiIndex) and (not (is_integer(i) or com.is_null_slice(i)))):
                    take_split_path = True
                    break
        if isinstance(indexer, tuple):
            nindexer = []
            for (i, idx) in enumerate(indexer):
                if isinstance(idx, dict):
                    (key, _) = convert_missing_indexer(idx)
                    if ((self.ndim > 1) and (i == info_axis)):
                        if (not len(self.obj)):
                            if (not is_list_like_indexer(value)):
                                raise ValueError('cannot set a frame with no defined index and a scalar')
                            self.obj[key] = value
                            return
                        if com.is_null_slice(indexer[0]):
                            self.obj[key] = value
                        else:
                            self.obj[key] = infer_fill_value(value)
                        new_indexer = convert_from_missing_indexer_tuple(indexer, self.obj.axes)
                        self._setitem_with_indexer(new_indexer, value, name)
                        return
                    index = self.obj._get_axis(i)
                    labels = index.insert(len(index), key)
                    self.obj._mgr = self.obj.reindex(labels, axis=i)._mgr
                    self.obj._maybe_update_cacher(clear=True)
                    self.obj._is_copy = None
                    nindexer.append(labels.get_loc(key))
                else:
                    nindexer.append(idx)
            indexer = tuple(nindexer)
        else:
            (indexer, missing) = convert_missing_indexer(indexer)
            if missing:
                self._setitem_with_indexer_missing(indexer, value)
                return
        if take_split_path:
            self._setitem_with_indexer_split_path(indexer, value, name)
        else:
            self._setitem_single_block(indexer, value, name)

    def _setitem_with_indexer_split_path(self, indexer, value, name):
        '\n        Setitem column-wise.\n        '
        assert (self.ndim == 2)
        if (not isinstance(indexer, tuple)):
            indexer = _tuplify(self.ndim, indexer)
        if (len(indexer) > self.ndim):
            raise IndexError('too many indices for array')
        if (isinstance(indexer[0], np.ndarray) and (indexer[0].ndim > 2)):
            raise ValueError('Cannot set values with ndim > 2')
        if ((isinstance(value, ABCSeries) and (name != 'iloc')) or isinstance(value, dict)):
            from pandas import Series
            value = self._align_series(indexer, Series(value))
        info_axis = indexer[1]
        ilocs = self._ensure_iterable_column_indexer(info_axis)
        pi = indexer[0]
        lplane_indexer = length_of_indexer(pi, self.obj.index)
        if (is_list_like_indexer(value) and (getattr(value, 'ndim', 1) > 0)):
            if isinstance(value, ABCDataFrame):
                self._setitem_with_indexer_frame_value(indexer, value, name)
            elif (np.ndim(value) == 2):
                self._setitem_with_indexer_2d_value(indexer, value)
            elif ((len(ilocs) == 1) and (lplane_indexer == len(value)) and (not is_scalar(pi))):
                self._setitem_single_column(ilocs[0], value, pi)
            elif ((len(ilocs) == 1) and (0 != lplane_indexer != len(value))):
                if ((len(value) == 1) and (not is_integer(info_axis))):
                    return self._setitem_with_indexer((pi, info_axis[0]), value[0])
                raise ValueError('Must have equal len keys and value when setting with an iterable')
            elif ((lplane_indexer == 0) and (len(value) == len(self.obj.index))):
                pass
            elif (len(ilocs) == len(value)):
                for (loc, v) in zip(ilocs, value):
                    self._setitem_single_column(loc, v, pi)
            elif ((len(ilocs) == 1) and com.is_null_slice(pi) and (len(self.obj) == 0)):
                self._setitem_single_column(ilocs[0], value, pi)
            else:
                raise ValueError('Must have equal len keys and value when setting with an iterable')
        else:
            for loc in ilocs:
                self._setitem_single_column(loc, value, pi)

    def _setitem_with_indexer_2d_value(self, indexer, value):
        pi = indexer[0]
        ilocs = self._ensure_iterable_column_indexer(indexer[1])
        value = np.array(value, dtype=object)
        if (len(ilocs) != value.shape[1]):
            raise ValueError('Must have equal len keys and value when setting with an ndarray')
        for (i, loc) in enumerate(ilocs):
            self._setitem_single_column(loc, value[:, i].tolist(), pi)

    def _setitem_with_indexer_frame_value(self, indexer, value, name):
        ilocs = self._ensure_iterable_column_indexer(indexer[1])
        sub_indexer = list(indexer)
        pi = indexer[0]
        multiindex_indexer = isinstance(self.obj.columns, ABCMultiIndex)
        unique_cols = value.columns.is_unique
        if (name == 'iloc'):
            for (i, loc) in enumerate(ilocs):
                val = value.iloc[:, i]
                self._setitem_single_column(loc, val, pi)
        elif ((not unique_cols) and value.columns.equals(self.obj.columns)):
            for loc in ilocs:
                item = self.obj.columns[loc]
                if (item in value):
                    sub_indexer[1] = item
                    val = self._align_series(tuple(sub_indexer), value.iloc[:, loc], multiindex_indexer)
                else:
                    val = np.nan
                self._setitem_single_column(loc, val, pi)
        elif (not unique_cols):
            raise ValueError('Setting with non-unique columns is not allowed.')
        else:
            for loc in ilocs:
                item = self.obj.columns[loc]
                if (item in value):
                    sub_indexer[1] = item
                    val = self._align_series(tuple(sub_indexer), value[item], multiindex_indexer)
                else:
                    val = np.nan
                self._setitem_single_column(loc, val, pi)

    def _setitem_single_column(self, loc, value, plane_indexer):
        '\n\n        Parameters\n        ----------\n        loc : int\n            Indexer for column position\n        plane_indexer : int, slice, listlike[int]\n            The indexer we use for setitem along axis=0.\n        '
        pi = plane_indexer
        ser = self.obj._ixs(loc, axis=1)
        if (com.is_null_slice(pi) or com.is_full_slice(pi, len(self.obj))):
            ser = value
        else:
            ser = ser.copy()
            ser._mgr = ser._mgr.setitem(indexer=(pi,), value=value)
            ser._maybe_update_cacher(clear=True)
        self.obj._iset_item(loc, ser)

    def _setitem_single_block(self, indexer, value, name):
        '\n        _setitem_with_indexer for the case when we have a single Block.\n        '
        from pandas import Series
        info_axis = self.obj._info_axis_number
        item_labels = self.obj._get_axis(info_axis)
        if isinstance(indexer, tuple):
            if ((len(indexer) > info_axis) and is_integer(indexer[info_axis]) and all((com.is_null_slice(idx) for (i, idx) in enumerate(indexer) if (i != info_axis))) and item_labels.is_unique):
                self.obj[item_labels[indexer[info_axis]]] = value
                return
            indexer = maybe_convert_ix(*indexer)
        if ((isinstance(value, ABCSeries) and (name != 'iloc')) or isinstance(value, dict)):
            value = self._align_series(indexer, Series(value))
        elif (isinstance(value, ABCDataFrame) and (name != 'iloc')):
            value = self._align_frame(indexer, value)
        self.obj._check_is_chained_assignment_possible()
        self.obj._consolidate_inplace()
        self.obj._mgr = self.obj._mgr.setitem(indexer=indexer, value=value)
        self.obj._maybe_update_cacher(clear=True)

    def _setitem_with_indexer_missing(self, indexer, value):
        '\n        Insert new row(s) or column(s) into the Series or DataFrame.\n        '
        from pandas import Series
        if (self.ndim == 1):
            index = self.obj.index
            new_index = index.insert(len(index), indexer)
            if index.is_unique:
                new_indexer = index.get_indexer([new_index[(- 1)]])
                if (new_indexer != (- 1)).any():
                    return self._setitem_with_indexer(new_indexer, value, 'loc')
            new_values = Series([value])._values
            if len(self.obj._values):
                new_values = concat_compat([self.obj._values, new_values])
            self.obj._mgr = self.obj._constructor(new_values, index=new_index, name=self.obj.name)._mgr
            self.obj._maybe_update_cacher(clear=True)
        elif (self.ndim == 2):
            if (not len(self.obj.columns)):
                raise ValueError('cannot set a frame with no defined columns')
            if isinstance(value, ABCSeries):
                value = value.reindex(index=self.obj.columns, copy=True)
                value.name = indexer
            elif isinstance(value, dict):
                value = Series(value, index=self.obj.columns, name=indexer, dtype=object)
            else:
                if is_list_like_indexer(value):
                    if (len(value) != len(self.obj.columns)):
                        raise ValueError('cannot set a row with mismatched columns')
                value = Series(value, index=self.obj.columns, name=indexer)
            self.obj._mgr = self.obj.append(value)._mgr
            self.obj._maybe_update_cacher(clear=True)

    def _ensure_iterable_column_indexer(self, column_indexer):
        '\n        Ensure that our column indexer is something that can be iterated over.\n        '
        if is_integer(column_indexer):
            ilocs = [column_indexer]
        elif isinstance(column_indexer, slice):
            ri = Index(range(len(self.obj.columns)))
            ilocs = ri[column_indexer]
        else:
            ilocs = column_indexer
        return ilocs

    def _align_series(self, indexer, ser, multiindex_indexer=False):
        '\n        Parameters\n        ----------\n        indexer : tuple, slice, scalar\n            Indexer used to get the locations that will be set to `ser`.\n        ser : pd.Series\n            Values to assign to the locations specified by `indexer`.\n        multiindex_indexer : boolean, optional\n            Defaults to False. Should be set to True if `indexer` was from\n            a `pd.MultiIndex`, to avoid unnecessary broadcasting.\n\n        Returns\n        -------\n        `np.array` of `ser` broadcast to the appropriate shape for assignment\n        to the locations selected by `indexer`\n        '
        if isinstance(indexer, (slice, np.ndarray, list, Index)):
            indexer = (indexer,)
        if isinstance(indexer, tuple):

            def ravel(i):
                return (i.ravel() if isinstance(i, np.ndarray) else i)
            indexer = tuple(map(ravel, indexer))
            aligners = [(not com.is_null_slice(idx)) for idx in indexer]
            sum_aligners = sum(aligners)
            single_aligner = (sum_aligners == 1)
            is_frame = (self.ndim == 2)
            obj = self.obj
            if is_frame:
                single_aligner = (single_aligner and aligners[0])
            if ((sum_aligners == self.ndim) and all((is_sequence(_) for _ in indexer))):
                ser = ser.reindex(obj.axes[0][indexer[0]], copy=True)._values
                if ((len(indexer) > 1) and (not multiindex_indexer)):
                    len_indexer = len(indexer[1])
                    ser = np.tile(ser, len_indexer).reshape(len_indexer, (- 1)).T
                return ser
            for (i, idx) in enumerate(indexer):
                ax = obj.axes[i]
                if (is_sequence(idx) or isinstance(idx, slice)):
                    if (single_aligner and com.is_null_slice(idx)):
                        continue
                    new_ix = ax[idx]
                    if (not is_list_like_indexer(new_ix)):
                        new_ix = Index([new_ix])
                    else:
                        new_ix = Index(new_ix)
                    if (ser.index.equals(new_ix) or (not len(new_ix))):
                        return ser._values.copy()
                    return ser.reindex(new_ix)._values
                elif single_aligner:
                    ax = self.obj.axes[1]
                    if (ser.index.equals(ax) or (not len(ax))):
                        return ser._values.copy()
                    return ser.reindex(ax)._values
        elif is_scalar(indexer):
            ax = self.obj._get_axis(1)
            if ser.index.equals(ax):
                return ser._values.copy()
            return ser.reindex(ax)._values
        raise ValueError('Incompatible indexer with Series')

    def _align_frame(self, indexer, df):
        is_frame = (self.ndim == 2)
        if isinstance(indexer, tuple):
            (idx, cols) = (None, None)
            sindexers = []
            for (i, ix) in enumerate(indexer):
                ax = self.obj.axes[i]
                if (is_sequence(ix) or isinstance(ix, slice)):
                    if isinstance(ix, np.ndarray):
                        ix = ix.ravel()
                    if (idx is None):
                        idx = ax[ix]
                    elif (cols is None):
                        cols = ax[ix]
                    else:
                        break
                else:
                    sindexers.append(i)
            if ((idx is not None) and (cols is not None)):
                if (df.index.equals(idx) and df.columns.equals(cols)):
                    val = df.copy()._values
                else:
                    val = df.reindex(idx, columns=cols)._values
                return val
        elif ((isinstance(indexer, slice) or is_list_like_indexer(indexer)) and is_frame):
            ax = self.obj.index[indexer]
            if df.index.equals(ax):
                val = df.copy()._values
            else:
                if (isinstance(ax, ABCMultiIndex) and isinstance(df.index, ABCMultiIndex) and (ax.nlevels != df.index.nlevels)):
                    raise TypeError('cannot align on a multi-index with out specifying the join levels')
                val = df.reindex(index=ax)._values
            return val
        raise ValueError('Incompatible indexer with DataFrame')

class _ScalarAccessIndexer(NDFrameIndexerBase):
    '\n    Access scalars quickly.\n    '

    def _convert_key(self, key, is_setter=False):
        raise AbstractMethodError(self)

    def __getitem__(self, key):
        if (not isinstance(key, tuple)):
            if (not is_list_like_indexer(key)):
                key = (key,)
            else:
                raise ValueError('Invalid call for scalar access (getting)!')
        key = self._convert_key(key)
        return self.obj._get_value(*key, takeable=self._takeable)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
        else:
            key = com.apply_if_callable(key, self.obj)
        if (not isinstance(key, tuple)):
            key = _tuplify(self.ndim, key)
        key = list(self._convert_key(key, is_setter=True))
        if (len(key) != self.ndim):
            raise ValueError('Not enough indexers for scalar access (setting)!')
        self.obj._set_value(*key, value=value, takeable=self._takeable)

@doc(IndexingMixin.at)
class _AtIndexer(_ScalarAccessIndexer):
    _takeable = False

    def _convert_key(self, key, is_setter=False):
        "\n        Require they keys to be the same type as the index. (so we don't\n        fallback)\n        "
        if ((self.ndim == 1) and (len(key) > 1)):
            key = (key,)
        if is_setter:
            return list(key)
        return key

    @property
    def _axes_are_unique(self):
        assert (self.ndim == 2)
        return (self.obj.index.is_unique and self.obj.columns.is_unique)

    def __getitem__(self, key):
        if ((self.ndim == 2) and (not self._axes_are_unique)):
            if ((not isinstance(key, tuple)) or (not all((is_scalar(x) for x in key)))):
                raise ValueError('Invalid call for scalar access (getting)!')
            return self.obj.loc[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if ((self.ndim == 2) and (not self._axes_are_unique)):
            if ((not isinstance(key, tuple)) or (not all((is_scalar(x) for x in key)))):
                raise ValueError('Invalid call for scalar access (setting)!')
            self.obj.loc[key] = value
            return
        return super().__setitem__(key, value)

@doc(IndexingMixin.iat)
class _iAtIndexer(_ScalarAccessIndexer):
    _takeable = True

    def _convert_key(self, key, is_setter=False):
        '\n        Require integer args. (and convert to label arguments)\n        '
        for (a, i) in zip(self.obj.axes, key):
            if (not is_integer(i)):
                raise ValueError('iAt based indexing can only have integer indexers')
        return key

def _tuplify(ndim, loc):
    '\n    Given an indexer for the first dimension, create an equivalent tuple\n    for indexing over all dimensions.\n\n    Parameters\n    ----------\n    ndim : int\n    loc : object\n\n    Returns\n    -------\n    tuple\n    '
    _tup: List[Union[(Hashable, slice)]]
    _tup = [slice(None, None) for _ in range(ndim)]
    _tup[0] = loc
    return tuple(_tup)

def convert_to_index_sliceable(obj, key):
    '\n    If we are index sliceable, then return my slicer, otherwise return None.\n    '
    idx = obj.index
    if isinstance(key, slice):
        return idx._convert_slice_indexer(key, kind='getitem')
    elif isinstance(key, str):
        if (key in obj.columns):
            return None
        if idx._supports_partial_string_indexing:
            try:
                res = idx._get_string_slice(key)
                warnings.warn('Indexing a DataFrame with a datetimelike index using a single string to slice the rows, like `frame[string]`, is deprecated and will be removed in a future version. Use `frame.loc[string]` instead.', FutureWarning, stacklevel=3)
                return res
            except (KeyError, ValueError, NotImplementedError):
                return None
    return None

def check_bool_indexer(index, key):
    '\n    Check if key is a valid boolean indexer for an object with such index and\n    perform reindexing or conversion if needed.\n\n    This function assumes that is_bool_indexer(key) == True.\n\n    Parameters\n    ----------\n    index : Index\n        Index of the object on which the indexing is done.\n    key : list-like\n        Boolean indexer to check.\n\n    Returns\n    -------\n    np.array\n        Resulting key.\n\n    Raises\n    ------\n    IndexError\n        If the key does not have the same length as index.\n    IndexingError\n        If the index of the key is unalignable to index.\n    '
    result = key
    if (isinstance(key, ABCSeries) and (not key.index.equals(index))):
        result = result.reindex(index)
        mask = isna(result._values)
        if mask.any():
            raise IndexingError('Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).')
        return result.astype(bool)._values
    if is_object_dtype(key):
        result = np.asarray(result, dtype=bool)
    elif (not is_array_like(result)):
        result = pd_array(result, dtype=bool)
    return check_array_indexer(index, result)

def convert_missing_indexer(indexer):
    '\n    Reverse convert a missing indexer, which is a dict\n    return the scalar indexer and a boolean indicating if we converted\n    '
    if isinstance(indexer, dict):
        indexer = indexer['key']
        if isinstance(indexer, bool):
            raise KeyError('cannot use a single bool to index into setitem')
        return (indexer, True)
    return (indexer, False)

def convert_from_missing_indexer_tuple(indexer, axes):
    "\n    Create a filtered indexer that doesn't have any missing indexers.\n    "

    def get_indexer(_i, _idx):
        return (axes[_i].get_loc(_idx['key']) if isinstance(_idx, dict) else _idx)
    return tuple((get_indexer(_i, _idx) for (_i, _idx) in enumerate(indexer)))

def maybe_convert_ix(*args):
    '\n    We likely want to take the cross-product.\n    '
    for arg in args:
        if (not isinstance(arg, (np.ndarray, list, ABCSeries, Index))):
            return args
    return np.ix_(*args)

def is_nested_tuple(tup, labels):
    '\n    Returns\n    -------\n    bool\n    '
    if (not isinstance(tup, tuple)):
        return False
    for k in tup:
        if (is_list_like(k) or isinstance(k, slice)):
            return isinstance(labels, ABCMultiIndex)
    return False

def is_label_like(key):
    '\n    Returns\n    -------\n    bool\n    '
    return ((not isinstance(key, slice)) and (not is_list_like_indexer(key)))

def need_slice(obj):
    '\n    Returns\n    -------\n    bool\n    '
    return ((obj.start is not None) or (obj.stop is not None) or ((obj.step is not None) and (obj.step != 1)))

def non_reducing_slice(slice_):
    "\n    Ensure that a slice doesn't reduce to a Series or Scalar.\n\n    Any user-passed `subset` should have this called on it\n    to make sure we're always working with DataFrames.\n    "
    kinds = (ABCSeries, np.ndarray, Index, list, str)
    if isinstance(slice_, kinds):
        slice_ = IndexSlice[:, slice_]

    def pred(part) -> bool:
        '\n        Returns\n        -------\n        bool\n            True if slice does *not* reduce,\n            False if `part` is a tuple.\n        '
        return ((isinstance(part, slice) or is_list_like(part)) and (not isinstance(part, tuple)))
    if (not is_list_like(slice_)):
        if (not isinstance(slice_, slice)):
            slice_ = [[slice_]]
        else:
            slice_ = [slice_]
    else:
        slice_ = [(part if pred(part) else [part]) for part in slice_]
    return tuple(slice_)

def maybe_numeric_slice(df, slice_, include_bool=False):
    "\n    Want nice defaults for background_gradient that don't break\n    with non-numeric data. But if slice_ is passed go with that.\n    "
    if (slice_ is None):
        dtypes = [np.number]
        if include_bool:
            dtypes.append(bool)
        slice_ = IndexSlice[:, df.select_dtypes(include=dtypes).columns]
    return slice_
