
from __future__ import annotations
import collections
from datetime import timedelta
import functools
import gc
import json
import operator
import pickle
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Hashable, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union, cast
import warnings
import weakref
import numpy as np
from pandas._config import config
from pandas._libs import lib
from pandas._libs.tslibs import Period, Tick, Timestamp, to_offset
from pandas._typing import Axis, CompressionOptions, FilePathOrBuffer, FrameOrSeries, IndexKeyFunc, IndexLabel, JSONSerializable, Label, Level, Renamer, StorageOptions, TimedeltaConvertibleTypes, TimestampConvertibleTypes, ValueKeyFunc, final
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError, InvalidIndexError
from pandas.util._decorators import doc, rewrite_axis_style_signature
from pandas.util._validators import validate_bool_kwarg, validate_fillna_kwargs, validate_percentile
from pandas.core.dtypes.common import ensure_int64, ensure_object, ensure_str, is_bool, is_bool_dtype, is_datetime64_any_dtype, is_datetime64tz_dtype, is_dict_like, is_dtype_equal, is_extension_array_dtype, is_float, is_list_like, is_number, is_numeric_dtype, is_object_dtype, is_re_compilable, is_scalar, is_timedelta64_dtype, pandas_dtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import isna, notna
import pandas as pd
from pandas.core import arraylike, indexing, missing, nanops
import pandas.core.algorithms as algos
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject, SelectionMixin
import pandas.core.common as com
from pandas.core.construction import create_series_with_explicit_dtype, extract_array
from pandas.core.flags import Flags
from pandas.core.indexes import base as ibase
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex, PeriodIndex, RangeIndex, ensure_index
from pandas.core.internals import BlockManager
from pandas.core.missing import find_valid_index
from pandas.core.ops import align_method_FRAME
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import Expanding, ExponentialMovingWindow, Rolling, Window
from pandas.io.formats import format as fmt
from pandas.io.formats.format import DataFrameFormatter, DataFrameRenderer, format_percentiles
from pandas.io.formats.printing import pprint_thing
if TYPE_CHECKING:
    from pandas._libs.tslibs import BaseOffset
    from pandas.core.frame import DataFrame
    from pandas.core.resample import Resampler
    from pandas.core.series import Series
    from pandas.core.window.indexers import BaseIndexer
_shared_docs = {**_shared_docs}
_shared_doc_kwargs = {'axes': 'keywords for axes', 'klass': 'Series/DataFrame', 'axes_single_arg': 'int or labels for object', 'args_transpose': 'axes to permute (int or label for object)', 'inplace': '\n    inplace : boolean, default False\n        If True, performs operation inplace and returns None.', 'optional_by': '\n        by : str or list of str\n            Name or list of names to sort by', 'replace_iloc': '\n    This differs from updating with ``.loc`` or ``.iloc``, which require\n    you to specify a location to update with some value.'}
bool_t = bool

class NDFrame(PandasObject, SelectionMixin, indexing.IndexingMixin):
    '\n    N-dimensional analogue of DataFrame. Store multi-dimensional in a\n    size-mutable, labeled data structure\n\n    Parameters\n    ----------\n    data : BlockManager\n    axes : list\n    copy : bool, default False\n    '
    _internal_names = ['_mgr', '_cacher', '_item_cache', '_cache', '_is_copy', '_subtyp', '_name', '_index', '_default_kind', '_default_fill_value', '_metadata', '__array_struct__', '__array_interface__', '_flags']
    _internal_names_set = set(_internal_names)
    _accessors = set()
    _hidden_attrs = frozenset(['_AXIS_NAMES', '_AXIS_NUMBERS', 'get_values', 'tshift'])
    _metadata = []
    _is_copy = None

    def __init__(self, data, copy=False, attrs=None):
        object.__setattr__(self, '_is_copy', None)
        object.__setattr__(self, '_mgr', data)
        object.__setattr__(self, '_item_cache', {})
        if (attrs is None):
            attrs = {}
        else:
            attrs = dict(attrs)
        object.__setattr__(self, '_attrs', attrs)
        object.__setattr__(self, '_flags', Flags(self, allows_duplicate_labels=True))

    @classmethod
    def _init_mgr(cls, mgr, axes, dtype=None, copy=False):
        ' passed a manager and a axes dict '
        for (a, axe) in axes.items():
            if (axe is not None):
                axe = ensure_index(axe)
                bm_axis = cls._get_block_manager_axis(a)
                mgr = mgr.reindex_axis(axe, axis=bm_axis, copy=False)
        if copy:
            mgr = mgr.copy()
        if (dtype is not None):
            if ((len(mgr.blocks) > 1) or (mgr.blocks[0].values.dtype != dtype)):
                mgr = mgr.astype(dtype=dtype)
        return mgr

    @property
    def attrs(self):
        '\n        Dictionary of global attributes of this dataset.\n\n        .. warning::\n\n           attrs is experimental and may change without warning.\n\n        See Also\n        --------\n        DataFrame.flags : Global flags applying to this object.\n        '
        if (self._attrs is None):
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        self._attrs = dict(value)

    @final
    @property
    def flags(self):
        '\n        Get the properties associated with this pandas object.\n\n        The available flags are\n\n        * :attr:`Flags.allows_duplicate_labels`\n\n        See Also\n        --------\n        Flags : Flags that apply to pandas objects.\n        DataFrame.attrs : Global metadata applying to this dataset.\n\n        Notes\n        -----\n        "Flags" differ from "metadata". Flags reflect properties of the\n        pandas object (the Series or DataFrame). Metadata refer to properties\n        of the dataset, and should be stored in :attr:`DataFrame.attrs`.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2]})\n        >>> df.flags\n        <Flags(allows_duplicate_labels=True)>\n\n        Flags can be get or set using ``.``\n\n        >>> df.flags.allows_duplicate_labels\n        True\n        >>> df.flags.allows_duplicate_labels = False\n\n        Or by slicing with a key\n\n        >>> df.flags["allows_duplicate_labels"]\n        False\n        >>> df.flags["allows_duplicate_labels"] = True\n        '
        return self._flags

    @final
    def set_flags(self, *, copy=False, allows_duplicate_labels=None):
        '\n        Return a new object with updated flags.\n\n        Parameters\n        ----------\n        allows_duplicate_labels : bool, optional\n            Whether the returned object allows duplicate labels.\n\n        Returns\n        -------\n        Series or DataFrame\n            The same type as the caller.\n\n        See Also\n        --------\n        DataFrame.attrs : Global metadata applying to this dataset.\n        DataFrame.flags : Global flags applying to this object.\n\n        Notes\n        -----\n        This method returns a new object that\'s a view on the same data\n        as the input. Mutating the input or the output values will be reflected\n        in the other.\n\n        This method is intended to be used in method chains.\n\n        "Flags" differ from "metadata". Flags reflect properties of the\n        pandas object (the Series or DataFrame). Metadata refer to properties\n        of the dataset, and should be stored in :attr:`DataFrame.attrs`.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2]})\n        >>> df.flags.allows_duplicate_labels\n        True\n        >>> df2 = df.set_flags(allows_duplicate_labels=False)\n        >>> df2.flags.allows_duplicate_labels\n        False\n        '
        df = self.copy(deep=copy)
        if (allows_duplicate_labels is not None):
            df.flags['allows_duplicate_labels'] = allows_duplicate_labels
        return df

    @final
    @classmethod
    def _validate_dtype(cls, dtype):
        ' validate the passed dtype '
        if (dtype is not None):
            dtype = pandas_dtype(dtype)
            if (dtype.kind == 'V'):
                raise NotImplementedError(f'compound dtypes are not implemented in the {cls.__name__} constructor')
        return dtype

    @property
    def _constructor(self):
        '\n        Used when a manipulation result has the same dimensions as the\n        original.\n        '
        raise AbstractMethodError(self)

    @property
    def _constructor_sliced(self):
        '\n        Used when a manipulation result has one lower dimension(s) as the\n        original, such as DataFrame single columns slicing.\n        '
        raise AbstractMethodError(self)

    @property
    def _constructor_expanddim(self):
        '\n        Used when a manipulation result has one higher dimension as the\n        original, such as Series.to_frame()\n        '
        raise NotImplementedError

    @final
    @property
    def _data(self):
        return self._mgr
    _stat_axis_number = 0
    _stat_axis_name = 'index'
    _ix = None
    _AXIS_TO_AXIS_NUMBER = {0: 0, 'index': 0, 'rows': 0}

    @property
    def _AXIS_NUMBERS(self):
        '.. deprecated:: 1.1.0'
        warnings.warn('_AXIS_NUMBERS has been deprecated.', FutureWarning, stacklevel=3)
        return {'index': 0}

    @property
    def _AXIS_NAMES(self):
        '.. deprecated:: 1.1.0'
        warnings.warn('_AXIS_NAMES has been deprecated.', FutureWarning, stacklevel=3)
        return {0: 'index'}

    @final
    def _construct_axes_dict(self, axes=None, **kwargs):
        'Return an axes dictionary for myself.'
        d = {a: self._get_axis(a) for a in (axes or self._AXIS_ORDERS)}
        d.update(kwargs)
        return d

    @final
    @classmethod
    def _construct_axes_from_arguments(cls, args, kwargs, require_all=False, sentinel=None):
        '\n        Construct and returns axes if supplied in args/kwargs.\n\n        If require_all, raise if all axis arguments are not supplied\n        return a tuple of (axes, kwargs).\n\n        sentinel specifies the default parameter when an axis is not\n        supplied; useful to distinguish when a user explicitly passes None\n        in scenarios where None has special meaning.\n        '
        args = list(args)
        for a in cls._AXIS_ORDERS:
            if (a not in kwargs):
                try:
                    kwargs[a] = args.pop(0)
                except IndexError as err:
                    if require_all:
                        raise TypeError('not enough/duplicate arguments specified!') from err
        axes = {a: kwargs.pop(a, sentinel) for a in cls._AXIS_ORDERS}
        return (axes, kwargs)

    @final
    @classmethod
    def _get_axis_number(cls, axis):
        try:
            return cls._AXIS_TO_AXIS_NUMBER[axis]
        except KeyError:
            raise ValueError(f'No axis named {axis} for object type {cls.__name__}')

    @final
    @classmethod
    def _get_axis_name(cls, axis):
        axis_number = cls._get_axis_number(axis)
        return cls._AXIS_ORDERS[axis_number]

    @final
    def _get_axis(self, axis):
        axis_number = self._get_axis_number(axis)
        assert (axis_number in {0, 1})
        return (self.index if (axis_number == 0) else self.columns)

    @final
    @classmethod
    def _get_block_manager_axis(cls, axis):
        'Map the axis to the block_manager axis.'
        axis = cls._get_axis_number(axis)
        if cls._AXIS_REVERSED:
            m = (cls._AXIS_LEN - 1)
            return (m - axis)
        return axis

    @final
    def _get_axis_resolvers(self, axis):
        axis_index = getattr(self, axis)
        d = {}
        prefix = axis[0]
        for (i, name) in enumerate(axis_index.names):
            if (name is not None):
                key = level = name
            else:
                key = f'{prefix}level_{i}'
                level = i
            level_values = axis_index.get_level_values(level)
            s = level_values.to_series()
            s.index = axis_index
            d[key] = s
        if isinstance(axis_index, MultiIndex):
            dindex = axis_index
        else:
            dindex = axis_index.to_series()
        d[axis] = dindex
        return d

    @final
    def _get_index_resolvers(self):
        from pandas.core.computation.parsing import clean_column_name
        d: Dict[(str, Union[(Series, MultiIndex)])] = {}
        for axis_name in self._AXIS_ORDERS:
            d.update(self._get_axis_resolvers(axis_name))
        return {clean_column_name(k): v for (k, v) in d.items() if (not isinstance(k, int))}

    @final
    def _get_cleaned_column_resolvers(self):
        "\n        Return the special character free column resolvers of a dataframe.\n\n        Column names with special characters are 'cleaned up' so that they can\n        be referred to by backtick quoting.\n        Used in :meth:`DataFrame.eval`.\n        "
        from pandas.core.computation.parsing import clean_column_name
        if isinstance(self, ABCSeries):
            return {clean_column_name(self.name): self}
        return {clean_column_name(k): v for (k, v) in self.items() if (not isinstance(k, int))}

    @property
    def _info_axis(self):
        return getattr(self, self._info_axis_name)

    @property
    def _stat_axis(self):
        return getattr(self, self._stat_axis_name)

    @property
    def shape(self):
        '\n        Return a tuple of axis dimensions\n        '
        return tuple((len(self._get_axis(a)) for a in self._AXIS_ORDERS))

    @property
    def axes(self):
        '\n        Return index label(s) of the internal NDFrame\n        '
        return [self._get_axis(a) for a in self._AXIS_ORDERS]

    @property
    def ndim(self):
        "\n        Return an int representing the number of axes / array dimensions.\n\n        Return 1 if Series. Otherwise return 2 if DataFrame.\n\n        See Also\n        --------\n        ndarray.ndim : Number of array dimensions.\n\n        Examples\n        --------\n        >>> s = pd.Series({'a': 1, 'b': 2, 'c': 3})\n        >>> s.ndim\n        1\n\n        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})\n        >>> df.ndim\n        2\n        "
        return self._mgr.ndim

    @property
    def size(self):
        "\n        Return an int representing the number of elements in this object.\n\n        Return the number of rows if Series. Otherwise return the number of\n        rows times number of columns if DataFrame.\n\n        See Also\n        --------\n        ndarray.size : Number of elements in the array.\n\n        Examples\n        --------\n        >>> s = pd.Series({'a': 1, 'b': 2, 'c': 3})\n        >>> s.size\n        3\n\n        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})\n        >>> df.size\n        4\n        "
        return np.prod(self.shape)

    @final
    @property
    def _selected_obj(self):
        ' internal compat with SelectionMixin '
        return self

    @final
    @property
    def _obj_with_exclusions(self):
        ' internal compat with SelectionMixin '
        return self

    def set_axis(self, labels, axis=0, inplace=False):
        '\n        Assign desired index to given axis.\n\n        Indexes for%(extended_summary_sub)s row labels can be changed by assigning\n        a list-like or Index.\n\n        Parameters\n        ----------\n        labels : list-like, Index\n            The values for the new index.\n\n        axis : %(axes_single_arg)s, default 0\n            The axis to update. The value 0 identifies the rows%(axis_description_sub)s.\n\n        inplace : bool, default False\n            Whether to return a new %(klass)s instance.\n\n        Returns\n        -------\n        renamed : %(klass)s or None\n            An object of type %(klass)s or None if ``inplace=True``.\n\n        See Also\n        --------\n        %(klass)s.rename_axis : Alter the name of the index%(see_also_sub)s.\n        '
        self._check_inplace_and_allows_duplicate_labels(inplace)
        return self._set_axis_nocheck(labels, axis, inplace)

    @final
    def _set_axis_nocheck(self, labels, axis, inplace):
        if inplace:
            setattr(self, self._get_axis_name(axis), labels)
        else:
            obj = self.copy()
            obj.set_axis(labels, axis=axis, inplace=True)
            return obj

    def _set_axis(self, axis, labels):
        labels = ensure_index(labels)
        self._mgr.set_axis(axis, labels)
        self._clear_item_cache()

    @final
    def swapaxes(self, axis1, axis2, copy=True):
        '\n        Interchange axes and swap values axes appropriately.\n\n        Returns\n        -------\n        y : same as input\n        '
        i = self._get_axis_number(axis1)
        j = self._get_axis_number(axis2)
        if (i == j):
            if copy:
                return self.copy()
            return self
        mapping = {i: j, j: i}
        new_axes = (self._get_axis(mapping.get(k, k)) for k in range(self._AXIS_LEN))
        new_values = self.values.swapaxes(i, j)
        if copy:
            new_values = new_values.copy()
        return self._constructor(new_values, *new_axes).__finalize__(self, method='swapaxes')

    @final
    def droplevel(self, level, axis=0):
        "\n        Return DataFrame with requested index / column level(s) removed.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        level : int, str, or list-like\n            If a string is given, must be the name of a level\n            If list-like, elements must be names or positional indexes\n            of levels.\n\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            Axis along which the level(s) is removed:\n\n            * 0 or 'index': remove level(s) in column.\n            * 1 or 'columns': remove level(s) in row.\n\n        Returns\n        -------\n        DataFrame\n            DataFrame with requested index / column level(s) removed.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([\n        ...     [1, 2, 3, 4],\n        ...     [5, 6, 7, 8],\n        ...     [9, 10, 11, 12]\n        ... ]).set_index([0, 1]).rename_axis(['a', 'b'])\n\n        >>> df.columns = pd.MultiIndex.from_tuples([\n        ...     ('c', 'e'), ('d', 'f')\n        ... ], names=['level_1', 'level_2'])\n\n        >>> df\n        level_1   c   d\n        level_2   e   f\n        a b\n        1 2      3   4\n        5 6      7   8\n        9 10    11  12\n\n        >>> df.droplevel('a')\n        level_1   c   d\n        level_2   e   f\n        b\n        2        3   4\n        6        7   8\n        10      11  12\n\n        >>> df.droplevel('level_2', axis=1)\n        level_1   c   d\n        a b\n        1 2      3   4\n        5 6      7   8\n        9 10    11  12\n        "
        labels = self._get_axis(axis)
        new_labels = labels.droplevel(level)
        result = self.set_axis(new_labels, axis=axis, inplace=False)
        return result

    def pop(self, item):
        result = self[item]
        del self[item]
        if (self.ndim == 2):
            result._reset_cacher()
        return result

    @final
    def squeeze(self, axis=None):
        "\n        Squeeze 1 dimensional axis objects into scalars.\n\n        Series or DataFrames with a single element are squeezed to a scalar.\n        DataFrames with a single column or a single row are squeezed to a\n        Series. Otherwise the object is unchanged.\n\n        This method is most useful when you don't know if your\n        object is a Series or DataFrame, but you do know it has just a single\n        column. In that case you can safely call `squeeze` to ensure you have a\n        Series.\n\n        Parameters\n        ----------\n        axis : {0 or 'index', 1 or 'columns', None}, default None\n            A specific axis to squeeze. By default, all length-1 axes are\n            squeezed.\n\n        Returns\n        -------\n        DataFrame, Series, or scalar\n            The projection after squeezing `axis` or all the axes.\n\n        See Also\n        --------\n        Series.iloc : Integer-location based indexing for selecting scalars.\n        DataFrame.iloc : Integer-location based indexing for selecting Series.\n        Series.to_frame : Inverse of DataFrame.squeeze for a\n            single-column DataFrame.\n\n        Examples\n        --------\n        >>> primes = pd.Series([2, 3, 5, 7])\n\n        Slicing might produce a Series with a single value:\n\n        >>> even_primes = primes[primes % 2 == 0]\n        >>> even_primes\n        0    2\n        dtype: int64\n\n        >>> even_primes.squeeze()\n        2\n\n        Squeezing objects with more than one value in every axis does nothing:\n\n        >>> odd_primes = primes[primes % 2 == 1]\n        >>> odd_primes\n        1    3\n        2    5\n        3    7\n        dtype: int64\n\n        >>> odd_primes.squeeze()\n        1    3\n        2    5\n        3    7\n        dtype: int64\n\n        Squeezing is even more effective when used with DataFrames.\n\n        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])\n        >>> df\n           a  b\n        0  1  2\n        1  3  4\n\n        Slicing a single column will produce a DataFrame with the columns\n        having only one value:\n\n        >>> df_a = df[['a']]\n        >>> df_a\n           a\n        0  1\n        1  3\n\n        So the columns can be squeezed down, resulting in a Series:\n\n        >>> df_a.squeeze('columns')\n        0    1\n        1    3\n        Name: a, dtype: int64\n\n        Slicing a single row from a single column will produce a single\n        scalar DataFrame:\n\n        >>> df_0a = df.loc[df.index < 1, ['a']]\n        >>> df_0a\n           a\n        0  1\n\n        Squeezing the rows produces a single scalar Series:\n\n        >>> df_0a.squeeze('rows')\n        a    1\n        Name: 0, dtype: int64\n\n        Squeezing all axes will project directly into a scalar:\n\n        >>> df_0a.squeeze()\n        1\n        "
        axis = (range(self._AXIS_LEN) if (axis is None) else (self._get_axis_number(axis),))
        return self.iloc[tuple(((0 if ((i in axis) and (len(a) == 1)) else slice(None)) for (i, a) in enumerate(self.axes)))]

    def rename(self, mapper=None, *, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore'):
        '\n        Alter axes input function or functions. Function / dict values must be\n        unique (1-to-1). Labels not contained in a dict / Series will be left\n        as-is. Extra labels listed don\'t throw an error. Alternatively, change\n        ``Series.name`` with a scalar value (Series only).\n\n        Parameters\n        ----------\n        %(axes)s : scalar, list-like, dict-like or function, optional\n            Scalar or list-like will alter the ``Series.name`` attribute,\n            and raise on DataFrame.\n            dict-like or functions are transformations to apply to\n            that axis\' values\n        copy : bool, default True\n            Also copy underlying data.\n        inplace : bool, default False\n            Whether to return a new {klass}. If True then value of copy is\n            ignored.\n        level : int or level name, default None\n            In case of a MultiIndex, only rename labels in the specified\n            level.\n        errors : {\'ignore\', \'raise\'}, default \'ignore\'\n            If \'raise\', raise a `KeyError` when a dict-like `mapper`, `index`,\n            or `columns` contains labels that are not present in the Index\n            being transformed.\n            If \'ignore\', existing keys will be renamed and extra keys will be\n            ignored.\n\n        Returns\n        -------\n        renamed : {klass} (new object)\n\n        Raises\n        ------\n        KeyError\n            If any of the labels is not found in the selected axis and\n            "errors=\'raise\'".\n\n        See Also\n        --------\n        NDFrame.rename_axis\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3])\n        >>> s\n        0    1\n        1    2\n        2    3\n        dtype: int64\n        >>> s.rename("my_name") # scalar, changes Series.name\n        0    1\n        1    2\n        2    3\n        Name: my_name, dtype: int64\n        >>> s.rename(lambda x: x ** 2)  # function, changes labels\n        0    1\n        1    2\n        4    3\n        dtype: int64\n        >>> s.rename({1: 3, 2: 5})  # mapping, changes labels\n        0    1\n        3    2\n        5    3\n        dtype: int64\n\n        Since ``DataFrame`` doesn\'t have a ``.name`` attribute,\n        only mapping-type arguments are allowed.\n\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})\n        >>> df.rename(2)\n        Traceback (most recent call last):\n        ...\n        TypeError: \'int\' object is not callable\n\n        ``DataFrame.rename`` supports two calling conventions\n\n        * ``(index=index_mapper, columns=columns_mapper, ...)``\n        * ``(mapper, axis={\'index\', \'columns\'}, ...)``\n\n        We *highly* recommend using keyword arguments to clarify your\n        intent.\n\n        >>> df.rename(index=str, columns={"A": "a", "B": "c"})\n           a  c\n        0  1  4\n        1  2  5\n        2  3  6\n\n        >>> df.rename(index=str, columns={"A": "a", "C": "c"})\n           a  B\n        0  1  4\n        1  2  5\n        2  3  6\n\n        Using axis-style parameters\n\n        >>> df.rename(str.lower, axis=\'columns\')\n           a  b\n        0  1  4\n        1  2  5\n        2  3  6\n\n        >>> df.rename({1: 2, 2: 4}, axis=\'index\')\n           A  B\n        0  1  4\n        2  2  5\n        4  3  6\n\n        See the :ref:`user guide <basics.rename>` for more.\n        '
        if ((mapper is None) and (index is None) and (columns is None)):
            raise TypeError('must pass an index to rename')
        if ((index is not None) or (columns is not None)):
            if (axis is not None):
                raise TypeError("Cannot specify both 'axis' and any of 'index' or 'columns'")
            elif (mapper is not None):
                raise TypeError("Cannot specify both 'mapper' and any of 'index' or 'columns'")
        elif (axis and (self._get_axis_number(axis) == 1)):
            columns = mapper
        else:
            index = mapper
        self._check_inplace_and_allows_duplicate_labels(inplace)
        result = (self if inplace else self.copy(deep=copy))
        for (axis_no, replacements) in enumerate((index, columns)):
            if (replacements is None):
                continue
            ax = self._get_axis(axis_no)
            f = com.get_rename_function(replacements)
            if (level is not None):
                level = ax._get_level_number(level)
            if (not callable(replacements)):
                indexer = ax.get_indexer_for(replacements)
                if ((errors == 'raise') and len(indexer[(indexer == (- 1))])):
                    missing_labels = [label for (index, label) in enumerate(replacements) if (indexer[index] == (- 1))]
                    raise KeyError(f'{missing_labels} not found in axis')
            new_index = ax._transform_index(f, level)
            result._set_axis_nocheck(new_index, axis=axis_no, inplace=True)
            result._clear_item_cache()
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result.__finalize__(self, method='rename')

    @rewrite_axis_style_signature('mapper', [('copy', True), ('inplace', False)])
    def rename_axis(self, mapper=lib.no_default, **kwargs):
        '\n        Set the name of the axis for the index or columns.\n\n        Parameters\n        ----------\n        mapper : scalar, list-like, optional\n            Value to set the axis name attribute.\n        index, columns : scalar, list-like, dict-like or function, optional\n            A scalar, list-like, dict-like or functions transformations to\n            apply to that axis\' values.\n            Note that the ``columns`` parameter is not allowed if the\n            object is a Series. This parameter only apply for DataFrame\n            type objects.\n\n            Use either ``mapper`` and ``axis`` to\n            specify the axis to target with ``mapper``, or ``index``\n            and/or ``columns``.\n\n            .. versionchanged:: 0.24.0\n\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            The axis to rename.\n        copy : bool, default True\n            Also copy underlying data.\n        inplace : bool, default False\n            Modifies the object directly, instead of creating a new Series\n            or DataFrame.\n\n        Returns\n        -------\n        Series, DataFrame, or None\n            The same type as the caller or None if ``inplace=True``.\n\n        See Also\n        --------\n        Series.rename : Alter Series index labels or name.\n        DataFrame.rename : Alter DataFrame index labels or name.\n        Index.rename : Set new names on index.\n\n        Notes\n        -----\n        ``DataFrame.rename_axis`` supports two calling conventions\n\n        * ``(index=index_mapper, columns=columns_mapper, ...)``\n        * ``(mapper, axis={\'index\', \'columns\'}, ...)``\n\n        The first calling convention will only modify the names of\n        the index and/or the names of the Index object that is the columns.\n        In this case, the parameter ``copy`` is ignored.\n\n        The second calling convention will modify the names of the\n        corresponding index if mapper is a list or a scalar.\n        However, if mapper is dict-like or a function, it will use the\n        deprecated behavior of modifying the axis *labels*.\n\n        We *highly* recommend using keyword arguments to clarify your\n        intent.\n\n        Examples\n        --------\n        **Series**\n\n        >>> s = pd.Series(["dog", "cat", "monkey"])\n        >>> s\n        0       dog\n        1       cat\n        2    monkey\n        dtype: object\n        >>> s.rename_axis("animal")\n        animal\n        0    dog\n        1    cat\n        2    monkey\n        dtype: object\n\n        **DataFrame**\n\n        >>> df = pd.DataFrame({"num_legs": [4, 4, 2],\n        ...                    "num_arms": [0, 0, 2]},\n        ...                   ["dog", "cat", "monkey"])\n        >>> df\n                num_legs  num_arms\n        dog            4         0\n        cat            4         0\n        monkey         2         2\n        >>> df = df.rename_axis("animal")\n        >>> df\n                num_legs  num_arms\n        animal\n        dog            4         0\n        cat            4         0\n        monkey         2         2\n        >>> df = df.rename_axis("limbs", axis="columns")\n        >>> df\n        limbs   num_legs  num_arms\n        animal\n        dog            4         0\n        cat            4         0\n        monkey         2         2\n\n        **MultiIndex**\n\n        >>> df.index = pd.MultiIndex.from_product([[\'mammal\'],\n        ...                                        [\'dog\', \'cat\', \'monkey\']],\n        ...                                       names=[\'type\', \'name\'])\n        >>> df\n        limbs          num_legs  num_arms\n        type   name\n        mammal dog            4         0\n               cat            4         0\n               monkey         2         2\n\n        >>> df.rename_axis(index={\'type\': \'class\'})\n        limbs          num_legs  num_arms\n        class  name\n        mammal dog            4         0\n               cat            4         0\n               monkey         2         2\n\n        >>> df.rename_axis(columns=str.upper)\n        LIMBS          num_legs  num_arms\n        type   name\n        mammal dog            4         0\n               cat            4         0\n               monkey         2         2\n        '
        (axes, kwargs) = self._construct_axes_from_arguments((), kwargs, sentinel=lib.no_default)
        copy = kwargs.pop('copy', True)
        inplace = kwargs.pop('inplace', False)
        axis = kwargs.pop('axis', 0)
        if (axis is not None):
            axis = self._get_axis_number(axis)
        if kwargs:
            raise TypeError(f'rename_axis() got an unexpected keyword argument "{list(kwargs.keys())[0]}"')
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (mapper is not lib.no_default):
            non_mapper = (is_scalar(mapper) or (is_list_like(mapper) and (not is_dict_like(mapper))))
            if non_mapper:
                return self._set_axis_name(mapper, axis=axis, inplace=inplace)
            else:
                raise ValueError('Use `.rename` to alter labels with a mapper.')
        else:
            result = (self if inplace else self.copy(deep=copy))
            for axis in range(self._AXIS_LEN):
                v = axes.get(self._get_axis_name(axis))
                if (v is lib.no_default):
                    continue
                non_mapper = (is_scalar(v) or (is_list_like(v) and (not is_dict_like(v))))
                if non_mapper:
                    newnames = v
                else:
                    f = com.get_rename_function(v)
                    curnames = self._get_axis(axis).names
                    newnames = [f(name) for name in curnames]
                result._set_axis_name(newnames, axis=axis, inplace=True)
            if (not inplace):
                return result

    @final
    def _set_axis_name(self, name, axis=0, inplace=False):
        '\n        Set the name(s) of the axis.\n\n        Parameters\n        ----------\n        name : str or list of str\n            Name(s) to set.\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            The axis to set the label. The value 0 or \'index\' specifies index,\n            and the value 1 or \'columns\' specifies columns.\n        inplace : bool, default False\n            If `True`, do operation inplace and return None.\n\n        Returns\n        -------\n        Series, DataFrame, or None\n            The same type as the caller or `None` if `inplace` is `True`.\n\n        See Also\n        --------\n        DataFrame.rename : Alter the axis labels of :class:`DataFrame`.\n        Series.rename : Alter the index labels or set the index name\n            of :class:`Series`.\n        Index.rename : Set the name of :class:`Index` or :class:`MultiIndex`.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"num_legs": [4, 4, 2]},\n        ...                   ["dog", "cat", "monkey"])\n        >>> df\n                num_legs\n        dog            4\n        cat            4\n        monkey         2\n        >>> df._set_axis_name("animal")\n                num_legs\n        animal\n        dog            4\n        cat            4\n        monkey         2\n        >>> df.index = pd.MultiIndex.from_product(\n        ...                [["mammal"], [\'dog\', \'cat\', \'monkey\']])\n        >>> df._set_axis_name(["type", "name"])\n                       num_legs\n        type   name\n        mammal dog        4\n               cat        4\n               monkey     2\n        '
        axis = self._get_axis_number(axis)
        idx = self._get_axis(axis).set_names(name)
        inplace = validate_bool_kwarg(inplace, 'inplace')
        renamed = (self if inplace else self.copy())
        renamed.set_axis(idx, axis=axis, inplace=True)
        if (not inplace):
            return renamed

    @final
    def _indexed_same(self, other):
        return all((self._get_axis(a).equals(other._get_axis(a)) for a in self._AXIS_ORDERS))

    @final
    def equals(self, other):
        '\n        Test whether two objects contain the same elements.\n\n        This function allows two Series or DataFrames to be compared against\n        each other to see if they have the same shape and elements. NaNs in\n        the same location are considered equal.\n\n        The row/column index do not need to have the same type, as long\n        as the values are considered equal. Corresponding columns must be of\n        the same dtype.\n\n        Parameters\n        ----------\n        other : Series or DataFrame\n            The other Series or DataFrame to be compared with the first.\n\n        Returns\n        -------\n        bool\n            True if all elements are the same in both objects, False\n            otherwise.\n\n        See Also\n        --------\n        Series.eq : Compare two Series objects of the same length\n            and return a Series where each element is True if the element\n            in each Series is equal, False otherwise.\n        DataFrame.eq : Compare two DataFrame objects of the same shape and\n            return a DataFrame where each element is True if the respective\n            element in each DataFrame is equal, False otherwise.\n        testing.assert_series_equal : Raises an AssertionError if left and\n            right are not equal. Provides an easy interface to ignore\n            inequality in dtypes, indexes and precision among others.\n        testing.assert_frame_equal : Like assert_series_equal, but targets\n            DataFrames.\n        numpy.array_equal : Return True if two arrays have the same shape\n            and elements, False otherwise.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({1: [10], 2: [20]})\n        >>> df\n            1   2\n        0  10  20\n\n        DataFrames df and exactly_equal have the same types and values for\n        their elements and column labels, which will return True.\n\n        >>> exactly_equal = pd.DataFrame({1: [10], 2: [20]})\n        >>> exactly_equal\n            1   2\n        0  10  20\n        >>> df.equals(exactly_equal)\n        True\n\n        DataFrames df and different_column_type have the same element\n        types and values, but have different types for the column labels,\n        which will still return True.\n\n        >>> different_column_type = pd.DataFrame({1.0: [10], 2.0: [20]})\n        >>> different_column_type\n           1.0  2.0\n        0   10   20\n        >>> df.equals(different_column_type)\n        True\n\n        DataFrames df and different_data_type have different types for the\n        same values for their elements, and will return False even though\n        their column labels are the same values and types.\n\n        >>> different_data_type = pd.DataFrame({1: [10.0], 2: [20.0]})\n        >>> different_data_type\n              1     2\n        0  10.0  20.0\n        >>> df.equals(different_data_type)\n        False\n        '
        if (not (isinstance(other, type(self)) or isinstance(self, type(other)))):
            return False
        other = cast(NDFrame, other)
        return self._mgr.equals(other._mgr)

    @final
    def __neg__(self):
        values = self._values
        if is_bool_dtype(values):
            arr = operator.inv(values)
        elif (is_numeric_dtype(values) or is_timedelta64_dtype(values) or is_object_dtype(values)):
            arr = operator.neg(values)
        else:
            raise TypeError(f'Unary negative expects numeric dtype, not {values.dtype}')
        return self.__array_wrap__(arr)

    @final
    def __pos__(self):
        values = self._values
        if is_bool_dtype(values):
            arr = values
        elif (is_numeric_dtype(values) or is_timedelta64_dtype(values) or is_object_dtype(values)):
            arr = operator.pos(values)
        else:
            raise TypeError(f'Unary plus expects bool, numeric, timedelta, or object dtype, not {values.dtype}')
        return self.__array_wrap__(arr)

    @final
    def __invert__(self):
        if (not self.size):
            return self
        new_data = self._mgr.apply(operator.invert)
        result = self._constructor(new_data).__finalize__(self, method='__invert__')
        return result

    @final
    def __nonzero__(self):
        raise ValueError(f'The truth value of a {type(self).__name__} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().')
    __bool__ = __nonzero__

    @final
    def bool(self):
        "\n        Return the bool of a single element Series or DataFrame.\n\n        This must be a boolean scalar value, either True or False. It will raise a\n        ValueError if the Series or DataFrame does not have exactly 1 element, or that\n        element is not boolean (integer values 0 and 1 will also raise an exception).\n\n        Returns\n        -------\n        bool\n            The value in the Series or DataFrame.\n\n        See Also\n        --------\n        Series.astype : Change the data type of a Series, including to boolean.\n        DataFrame.astype : Change the data type of a DataFrame, including to boolean.\n        numpy.bool_ : NumPy boolean data type, used by pandas for boolean values.\n\n        Examples\n        --------\n        The method will only work for single element objects with a boolean value:\n\n        >>> pd.Series([True]).bool()\n        True\n        >>> pd.Series([False]).bool()\n        False\n\n        >>> pd.DataFrame({'col': [True]}).bool()\n        True\n        >>> pd.DataFrame({'col': [False]}).bool()\n        False\n        "
        v = self.squeeze()
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        elif is_scalar(v):
            raise ValueError(f'bool cannot act on a non-boolean single element {type(self).__name__}')
        self.__nonzero__()

    @final
    def __abs__(self):
        return self.abs()

    @final
    def __round__(self, decimals=0):
        return self.round(decimals)

    @final
    def _is_level_reference(self, key, axis=0):
        '\n        Test whether a key is a level reference for a given axis.\n\n        To be considered a level reference, `key` must be a string that:\n          - (axis=0): Matches the name of an index level and does NOT match\n            a column label.\n          - (axis=1): Matches the name of a column level and does NOT match\n            an index label.\n\n        Parameters\n        ----------\n        key : str\n            Potential level name for the given axis\n        axis : int, default 0\n            Axis that levels are associated with (0 for index, 1 for columns)\n\n        Returns\n        -------\n        is_level : bool\n        '
        axis = self._get_axis_number(axis)
        return ((key is not None) and is_hashable(key) and (key in self.axes[axis].names) and (not self._is_label_reference(key, axis=axis)))

    @final
    def _is_label_reference(self, key, axis=0):
        '\n        Test whether a key is a label reference for a given axis.\n\n        To be considered a label reference, `key` must be a string that:\n          - (axis=0): Matches a column label\n          - (axis=1): Matches an index label\n\n        Parameters\n        ----------\n        key: str\n            Potential label name\n        axis: int, default 0\n            Axis perpendicular to the axis that labels are associated with\n            (0 means search for column labels, 1 means search for index labels)\n\n        Returns\n        -------\n        is_label: bool\n        '
        axis = self._get_axis_number(axis)
        other_axes = (ax for ax in range(self._AXIS_LEN) if (ax != axis))
        return ((key is not None) and is_hashable(key) and any(((key in self.axes[ax]) for ax in other_axes)))

    @final
    def _is_label_or_level_reference(self, key, axis=0):
        '\n        Test whether a key is a label or level reference for a given axis.\n\n        To be considered either a label or a level reference, `key` must be a\n        string that:\n          - (axis=0): Matches a column label or an index level\n          - (axis=1): Matches an index label or a column level\n\n        Parameters\n        ----------\n        key: str\n            Potential label or level name\n        axis: int, default 0\n            Axis that levels are associated with (0 for index, 1 for columns)\n\n        Returns\n        -------\n        is_label_or_level: bool\n        '
        return (self._is_level_reference(key, axis=axis) or self._is_label_reference(key, axis=axis))

    @final
    def _check_label_or_level_ambiguity(self, key, axis=0):
        '\n        Check whether `key` is ambiguous.\n\n        By ambiguous, we mean that it matches both a level of the input\n        `axis` and a label of the other axis.\n\n        Parameters\n        ----------\n        key: str or object\n            Label or level name.\n        axis: int, default 0\n            Axis that levels are associated with (0 for index, 1 for columns).\n\n        Raises\n        ------\n        ValueError: `key` is ambiguous\n        '
        axis = self._get_axis_number(axis)
        other_axes = (ax for ax in range(self._AXIS_LEN) if (ax != axis))
        if ((key is not None) and is_hashable(key) and (key in self.axes[axis].names) and any(((key in self.axes[ax]) for ax in other_axes))):
            (level_article, level_type) = (('an', 'index') if (axis == 0) else ('a', 'column'))
            (label_article, label_type) = (('a', 'column') if (axis == 0) else ('an', 'index'))
            msg = f"'{key}' is both {level_article} {level_type} level and {label_article} {label_type} label, which is ambiguous."
            raise ValueError(msg)

    @final
    def _get_label_or_level_values(self, key, axis=0):
        "\n        Return a 1-D array of values associated with `key`, a label or level\n        from the given `axis`.\n\n        Retrieval logic:\n          - (axis=0): Return column values if `key` matches a column label.\n            Otherwise return index level values if `key` matches an index\n            level.\n          - (axis=1): Return row values if `key` matches an index label.\n            Otherwise return column level values if 'key' matches a column\n            level\n\n        Parameters\n        ----------\n        key: str\n            Label or level name.\n        axis: int, default 0\n            Axis that levels are associated with (0 for index, 1 for columns)\n\n        Returns\n        -------\n        values: np.ndarray\n\n        Raises\n        ------\n        KeyError\n            if `key` matches neither a label nor a level\n        ValueError\n            if `key` matches multiple labels\n        FutureWarning\n            if `key` is ambiguous. This will become an ambiguity error in a\n            future version\n        "
        axis = self._get_axis_number(axis)
        other_axes = [ax for ax in range(self._AXIS_LEN) if (ax != axis)]
        if self._is_label_reference(key, axis=axis):
            self._check_label_or_level_ambiguity(key, axis=axis)
            values = self.xs(key, axis=other_axes[0])._values
        elif self._is_level_reference(key, axis=axis):
            values = self.axes[axis].get_level_values(key)._values
        else:
            raise KeyError(key)
        if (values.ndim > 1):
            if (other_axes and isinstance(self._get_axis(other_axes[0]), MultiIndex)):
                multi_message = '\nFor a multi-index, the label must be a tuple with elements corresponding to each level.'
            else:
                multi_message = ''
            label_axis_name = ('column' if (axis == 0) else 'index')
            raise ValueError(f"The {label_axis_name} label '{key}' is not unique.{multi_message}")
        return values

    @final
    def _drop_labels_or_levels(self, keys, axis=0):
        '\n        Drop labels and/or levels for the given `axis`.\n\n        For each key in `keys`:\n          - (axis=0): If key matches a column label then drop the column.\n            Otherwise if key matches an index level then drop the level.\n          - (axis=1): If key matches an index label then drop the row.\n            Otherwise if key matches a column level then drop the level.\n\n        Parameters\n        ----------\n        keys: str or list of str\n            labels or levels to drop\n        axis: int, default 0\n            Axis that levels are associated with (0 for index, 1 for columns)\n\n        Returns\n        -------\n        dropped: DataFrame\n\n        Raises\n        ------\n        ValueError\n            if any `keys` match neither a label nor a level\n        '
        axis = self._get_axis_number(axis)
        keys = com.maybe_make_list(keys)
        invalid_keys = [k for k in keys if (not self._is_label_or_level_reference(k, axis=axis))]
        if invalid_keys:
            raise ValueError(f'The following keys are not valid labels or levels for axis {axis}: {invalid_keys}')
        levels_to_drop = [k for k in keys if self._is_level_reference(k, axis=axis)]
        labels_to_drop = [k for k in keys if (not self._is_level_reference(k, axis=axis))]
        dropped = self.copy()
        if (axis == 0):
            if levels_to_drop:
                dropped.reset_index(levels_to_drop, drop=True, inplace=True)
            if labels_to_drop:
                dropped.drop(labels_to_drop, axis=1, inplace=True)
        else:
            if levels_to_drop:
                if isinstance(dropped.columns, MultiIndex):
                    dropped.columns = dropped.columns.droplevel(levels_to_drop)
                else:
                    dropped.columns = RangeIndex(dropped.columns.size)
            if labels_to_drop:
                dropped.drop(labels_to_drop, axis=0, inplace=True)
        return dropped

    def __hash__(self):
        raise TypeError(f'{repr(type(self).__name__)} objects are mutable, thus they cannot be hashed')

    def __iter__(self):
        '\n        Iterate over info axis.\n\n        Returns\n        -------\n        iterator\n            Info axis as iterator.\n        '
        return iter(self._info_axis)

    def keys(self):
        "\n        Get the 'info axis' (see Indexing for more).\n\n        This is index for Series, columns for DataFrame.\n\n        Returns\n        -------\n        Index\n            Info axis.\n        "
        return self._info_axis

    def items(self):
        '\n        Iterate over (label, values) on info axis\n\n        This is index for Series and columns for DataFrame.\n\n        Returns\n        -------\n        Generator\n        '
        for h in self._info_axis:
            (yield (h, self[h]))

    @doc(items)
    def iteritems(self):
        return self.items()

    def __len__(self):
        'Returns length of info axis'
        return len(self._info_axis)

    @final
    def __contains__(self, key):
        'True if the key is in the info axis'
        return (key in self._info_axis)

    @property
    def empty(self):
        "\n        Indicator whether DataFrame is empty.\n\n        True if DataFrame is entirely empty (no items), meaning any of the\n        axes are of length 0.\n\n        Returns\n        -------\n        bool\n            If DataFrame is empty, return True, if not return False.\n\n        See Also\n        --------\n        Series.dropna : Return series without null values.\n        DataFrame.dropna : Return DataFrame with labels on given axis omitted\n            where (all or any) data are missing.\n\n        Notes\n        -----\n        If DataFrame contains only NaNs, it is still not considered empty. See\n        the example below.\n\n        Examples\n        --------\n        An example of an actual empty DataFrame. Notice the index is empty:\n\n        >>> df_empty = pd.DataFrame({'A' : []})\n        >>> df_empty\n        Empty DataFrame\n        Columns: [A]\n        Index: []\n        >>> df_empty.empty\n        True\n\n        If we only have NaNs in our DataFrame, it is not considered empty! We\n        will need to drop the NaNs to make the DataFrame empty:\n\n        >>> df = pd.DataFrame({'A' : [np.nan]})\n        >>> df\n            A\n        0 NaN\n        >>> df.empty\n        False\n        >>> df.dropna().empty\n        True\n        "
        return any(((len(self._get_axis(a)) == 0) for a in self._AXIS_ORDERS))
    __array_priority__ = 1000

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)

    def __array_wrap__(self, result, context=None):
        '\n        Gets called after a ufunc and other functions.\n\n        Parameters\n        ----------\n        result: np.ndarray\n            The result of the ufunc or other function called on the NumPy array\n            returned by __array__\n        context: tuple of (func, tuple, int)\n            This parameter is returned by ufuncs as a 3-element tuple: (name of the\n            ufunc, arguments of the ufunc, domain of the ufunc), but is not set by\n            other numpy functions.q\n\n        Notes\n        -----\n        Series implements __array_ufunc_ so this not called for ufunc on Series.\n        '
        result = lib.item_from_zerodim(result)
        if is_scalar(result):
            return result
        d = self._construct_axes_dict(self._AXIS_ORDERS, copy=False)
        return self._constructor(result, **d).__finalize__(self, method='__array_wrap__')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)

    @final
    def __getstate__(self):
        meta = {k: getattr(self, k, None) for k in self._metadata}
        return {'_mgr': self._mgr, '_typ': self._typ, '_metadata': self._metadata, 'attrs': self.attrs, '_flags': {k: self.flags[k] for k in self.flags._keys}, **meta}

    @final
    def __setstate__(self, state):
        if isinstance(state, BlockManager):
            self._mgr = state
        elif isinstance(state, dict):
            if (('_data' in state) and ('_mgr' not in state)):
                state['_mgr'] = state.pop('_data')
            typ = state.get('_typ')
            if (typ is not None):
                attrs = state.get('_attrs', {})
                object.__setattr__(self, '_attrs', attrs)
                flags = state.get('_flags', {'allows_duplicate_labels': True})
                object.__setattr__(self, '_flags', Flags(self, **flags))
                meta = set((self._internal_names + self._metadata))
                for k in list(meta):
                    if ((k in state) and (k != '_flags')):
                        v = state[k]
                        object.__setattr__(self, k, v)
                for (k, v) in state.items():
                    if (k not in meta):
                        object.__setattr__(self, k, v)
            else:
                raise NotImplementedError('Pre-0.12 pickles are no longer supported')
        elif (len(state) == 2):
            raise NotImplementedError('Pre-0.12 pickles are no longer supported')
        self._item_cache = {}

    def __repr__(self):
        prepr = f"[{','.join(map(pprint_thing, self))}]"
        return f'{type(self).__name__}({prepr})'

    @final
    def _repr_latex_(self):
        '\n        Returns a LaTeX representation for a particular object.\n        Mainly for use with nbconvert (jupyter notebook conversion to pdf).\n        '
        if config.get_option('display.latex.repr'):
            return self.to_latex()
        else:
            return None

    @final
    def _repr_data_resource_(self):
        '\n        Not a real Jupyter special repr method, but we use the same\n        naming convention.\n        '
        if config.get_option('display.html.table_schema'):
            data = self.head(config.get_option('display.max_rows'))
            as_json = data.to_json(orient='table')
            as_json = cast(str, as_json)
            payload = json.loads(as_json, object_pairs_hook=collections.OrderedDict)
            return payload

    @final
    @doc(klass='object', storage_options=_shared_docs['storage_options'])
    def to_excel(self, excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None, storage_options=None):
        '\n        Write {klass} to an Excel sheet.\n\n        To write a single {klass} to an Excel .xlsx file it is only necessary to\n        specify a target file name. To write to multiple sheets it is necessary to\n        create an `ExcelWriter` object with a target file name, and specify a sheet\n        in the file to write to.\n\n        Multiple sheets may be written to by specifying unique `sheet_name`.\n        With all data written to the file it is necessary to save the changes.\n        Note that creating an `ExcelWriter` object with a file name that already\n        exists will result in the contents of the existing file being erased.\n\n        Parameters\n        ----------\n        excel_writer : path-like, file-like, or ExcelWriter object\n            File path or existing ExcelWriter.\n        sheet_name : str, default \'Sheet1\'\n            Name of sheet which will contain DataFrame.\n        na_rep : str, default \'\'\n            Missing data representation.\n        float_format : str, optional\n            Format string for floating point numbers. For example\n            ``float_format="%.2f"`` will format 0.1234 to 0.12.\n        columns : sequence or list of str, optional\n            Columns to write.\n        header : bool or list of str, default True\n            Write out the column names. If a list of string is given it is\n            assumed to be aliases for the column names.\n        index : bool, default True\n            Write row names (index).\n        index_label : str or sequence, optional\n            Column label for index column(s) if desired. If not specified, and\n            `header` and `index` are True, then the index names are used. A\n            sequence should be given if the DataFrame uses MultiIndex.\n        startrow : int, default 0\n            Upper left cell row to dump data frame.\n        startcol : int, default 0\n            Upper left cell column to dump data frame.\n        engine : str, optional\n            Write engine to use, \'openpyxl\' or \'xlsxwriter\'. You can also set this\n            via the options ``io.excel.xlsx.writer``, ``io.excel.xls.writer``, and\n            ``io.excel.xlsm.writer``.\n\n            .. deprecated:: 1.2.0\n\n                As the `xlwt <https://pypi.org/project/xlwt/>`__ package is no longer\n                maintained, the ``xlwt`` engine will be removed in a future version\n                of pandas.\n\n        merge_cells : bool, default True\n            Write MultiIndex and Hierarchical Rows as merged cells.\n        encoding : str, optional\n            Encoding of the resulting excel file. Only necessary for xlwt,\n            other writers support unicode natively.\n        inf_rep : str, default \'inf\'\n            Representation for infinity (there is no native representation for\n            infinity in Excel).\n        verbose : bool, default True\n            Display more information in the error logs.\n        freeze_panes : tuple of int (length 2), optional\n            Specifies the one-based bottommost row and rightmost column that\n            is to be frozen.\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n\n        See Also\n        --------\n        to_csv : Write DataFrame to a comma-separated values (csv) file.\n        ExcelWriter : Class for writing DataFrame objects into excel sheets.\n        read_excel : Read an Excel file into a pandas DataFrame.\n        read_csv : Read a comma-separated values (csv) file into DataFrame.\n\n        Notes\n        -----\n        For compatibility with :meth:`~DataFrame.to_csv`,\n        to_excel serializes lists and dicts to strings before writing.\n\n        Once a workbook has been saved it is not possible write further data\n        without rewriting the whole workbook.\n\n        Examples\n        --------\n\n        Create, write to and save a workbook:\n\n        >>> df1 = pd.DataFrame([[\'a\', \'b\'], [\'c\', \'d\']],\n        ...                    index=[\'row 1\', \'row 2\'],\n        ...                    columns=[\'col 1\', \'col 2\'])\n        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP\n\n        To specify the sheet name:\n\n        >>> df1.to_excel("output.xlsx",\n        ...              sheet_name=\'Sheet_name_1\')  # doctest: +SKIP\n\n        If you wish to write to more than one sheet in the workbook, it is\n        necessary to specify an ExcelWriter object:\n\n        >>> df2 = df1.copy()\n        >>> with pd.ExcelWriter(\'output.xlsx\') as writer:  # doctest: +SKIP\n        ...     df1.to_excel(writer, sheet_name=\'Sheet_name_1\')\n        ...     df2.to_excel(writer, sheet_name=\'Sheet_name_2\')\n\n        ExcelWriter can also be used to append to an existing Excel file:\n\n        >>> with pd.ExcelWriter(\'output.xlsx\',\n        ...                     mode=\'a\') as writer:  # doctest: +SKIP\n        ...     df.to_excel(writer, sheet_name=\'Sheet_name_3\')\n\n        To set the library that is used to write the Excel file,\n        you can pass the `engine` keyword (the default engine is\n        automatically chosen depending on the file extension):\n\n        >>> df1.to_excel(\'output1.xlsx\', engine=\'xlsxwriter\')  # doctest: +SKIP\n        '
        df = (self if isinstance(self, ABCDataFrame) else self.to_frame())
        from pandas.io.formats.excel import ExcelFormatter
        formatter = ExcelFormatter(df, na_rep=na_rep, cols=columns, header=header, float_format=float_format, index=index, index_label=index_label, merge_cells=merge_cells, inf_rep=inf_rep)
        formatter.write(excel_writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol, freeze_panes=freeze_panes, engine=engine, storage_options=storage_options)

    @final
    @doc(storage_options=_shared_docs['storage_options'])
    def to_json(self, path_or_buf=None, orient=None, date_format=None, double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=True, indent=None, storage_options=None):
        '\n        Convert the object to a JSON string.\n\n        Note NaN\'s and None will be converted to null and datetime objects\n        will be converted to UNIX timestamps.\n\n        Parameters\n        ----------\n        path_or_buf : str or file handle, optional\n            File path or object. If not specified, the result is returned as\n            a string.\n        orient : str\n            Indication of expected JSON string format.\n\n            * Series:\n\n                - default is \'index\'\n                - allowed values are: {{\'split\', \'records\', \'index\', \'table\'}}.\n\n            * DataFrame:\n\n                - default is \'columns\'\n                - allowed values are: {{\'split\', \'records\', \'index\', \'columns\',\n                  \'values\', \'table\'}}.\n\n            * The format of the JSON string:\n\n                - \'split\' : dict like {{\'index\' -> [index], \'columns\' -> [columns],\n                  \'data\' -> [values]}}\n                - \'records\' : list like [{{column -> value}}, ... , {{column -> value}}]\n                - \'index\' : dict like {{index -> {{column -> value}}}}\n                - \'columns\' : dict like {{column -> {{index -> value}}}}\n                - \'values\' : just the values array\n                - \'table\' : dict like {{\'schema\': {{schema}}, \'data\': {{data}}}}\n\n                Describing the data, where data component is like ``orient=\'records\'``.\n\n        date_format : {{None, \'epoch\', \'iso\'}}\n            Type of date conversion. \'epoch\' = epoch milliseconds,\n            \'iso\' = ISO8601. The default depends on the `orient`. For\n            ``orient=\'table\'``, the default is \'iso\'. For all other orients,\n            the default is \'epoch\'.\n        double_precision : int, default 10\n            The number of decimal places to use when encoding\n            floating point values.\n        force_ascii : bool, default True\n            Force encoded string to be ASCII.\n        date_unit : str, default \'ms\' (milliseconds)\n            The time unit to encode to, governs timestamp and ISO8601\n            precision.  One of \'s\', \'ms\', \'us\', \'ns\' for second, millisecond,\n            microsecond, and nanosecond respectively.\n        default_handler : callable, default None\n            Handler to call if object cannot otherwise be converted to a\n            suitable format for JSON. Should receive a single argument which is\n            the object to convert and return a serialisable object.\n        lines : bool, default False\n            If \'orient\' is \'records\' write out line delimited json format. Will\n            throw ValueError if incorrect \'orient\' since others are not list\n            like.\n\n        compression : {{\'infer\', \'gzip\', \'bz2\', \'zip\', \'xz\', None}}\n\n            A string representing the compression to use in the output file,\n            only used when the first argument is a filename. By default, the\n            compression is inferred from the filename.\n\n            .. versionchanged:: 0.24.0\n               \'infer\' option added and set to default\n        index : bool, default True\n            Whether to include the index values in the JSON string. Not\n            including the index (``index=False``) is only supported when\n            orient is \'split\' or \'table\'.\n        indent : int, optional\n           Length of whitespace used to indent each record.\n\n           .. versionadded:: 1.0.0\n\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n\n        Returns\n        -------\n        None or str\n            If path_or_buf is None, returns the resulting json format as a\n            string. Otherwise returns None.\n\n        See Also\n        --------\n        read_json : Convert a JSON string to pandas object.\n\n        Notes\n        -----\n        The behavior of ``indent=0`` varies from the stdlib, which does not\n        indent the output but does insert newlines. Currently, ``indent=0``\n        and the default ``indent=None`` are equivalent in pandas, though this\n        may change in a future release.\n\n        ``orient=\'table\'`` contains a \'pandas_version\' field under \'schema\'.\n        This stores the version of `pandas` used in the latest revision of the\n        schema.\n\n        Examples\n        --------\n        >>> import json\n        >>> df = pd.DataFrame(\n        ...     [["a", "b"], ["c", "d"]],\n        ...     index=["row 1", "row 2"],\n        ...     columns=["col 1", "col 2"],\n        ... )\n\n        >>> result = df.to_json(orient="split")\n        >>> parsed = json.loads(result)\n        >>> json.dumps(parsed, indent=4)  # doctest: +SKIP\n        {{\n            "columns": [\n                "col 1",\n                "col 2"\n            ],\n            "index": [\n                "row 1",\n                "row 2"\n            ],\n            "data": [\n                [\n                    "a",\n                    "b"\n                ],\n                [\n                    "c",\n                    "d"\n                ]\n            ]\n        }}\n\n        Encoding/decoding a Dataframe using ``\'records\'`` formatted JSON.\n        Note that index labels are not preserved with this encoding.\n\n        >>> result = df.to_json(orient="records")\n        >>> parsed = json.loads(result)\n        >>> json.dumps(parsed, indent=4)  # doctest: +SKIP\n        [\n            {{\n                "col 1": "a",\n                "col 2": "b"\n            }},\n            {{\n                "col 1": "c",\n                "col 2": "d"\n            }}\n        ]\n\n        Encoding/decoding a Dataframe using ``\'index\'`` formatted JSON:\n\n        >>> result = df.to_json(orient="index")\n        >>> parsed = json.loads(result)\n        >>> json.dumps(parsed, indent=4)  # doctest: +SKIP\n        {{\n            "row 1": {{\n                "col 1": "a",\n                "col 2": "b"\n            }},\n            "row 2": {{\n                "col 1": "c",\n                "col 2": "d"\n            }}\n        }}\n\n        Encoding/decoding a Dataframe using ``\'columns\'`` formatted JSON:\n\n        >>> result = df.to_json(orient="columns")\n        >>> parsed = json.loads(result)\n        >>> json.dumps(parsed, indent=4)  # doctest: +SKIP\n        {{\n            "col 1": {{\n                "row 1": "a",\n                "row 2": "c"\n            }},\n            "col 2": {{\n                "row 1": "b",\n                "row 2": "d"\n            }}\n        }}\n\n        Encoding/decoding a Dataframe using ``\'values\'`` formatted JSON:\n\n        >>> result = df.to_json(orient="values")\n        >>> parsed = json.loads(result)\n        >>> json.dumps(parsed, indent=4)  # doctest: +SKIP\n        [\n            [\n                "a",\n                "b"\n            ],\n            [\n                "c",\n                "d"\n            ]\n        ]\n\n        Encoding with Table Schema:\n\n        >>> result = df.to_json(orient="table")\n        >>> parsed = json.loads(result)\n        >>> json.dumps(parsed, indent=4)  # doctest: +SKIP\n        {{\n            "schema": {{\n                "fields": [\n                    {{\n                        "name": "index",\n                        "type": "string"\n                    }},\n                    {{\n                        "name": "col 1",\n                        "type": "string"\n                    }},\n                    {{\n                        "name": "col 2",\n                        "type": "string"\n                    }}\n                ],\n                "primaryKey": [\n                    "index"\n                ],\n                "pandas_version": "0.20.0"\n            }},\n            "data": [\n                {{\n                    "index": "row 1",\n                    "col 1": "a",\n                    "col 2": "b"\n                }},\n                {{\n                    "index": "row 2",\n                    "col 1": "c",\n                    "col 2": "d"\n                }}\n            ]\n        }}\n        '
        from pandas.io import json
        if ((date_format is None) and (orient == 'table')):
            date_format = 'iso'
        elif (date_format is None):
            date_format = 'epoch'
        config.is_nonnegative_int(indent)
        indent = (indent or 0)
        return json.to_json(path_or_buf=path_or_buf, obj=self, orient=orient, date_format=date_format, double_precision=double_precision, force_ascii=force_ascii, date_unit=date_unit, default_handler=default_handler, lines=lines, compression=compression, index=index, indent=indent, storage_options=storage_options)

    @final
    def to_hdf(self, path_or_buf, key, mode='a', complevel=None, complib=None, append=False, format=None, index=True, min_itemsize=None, nan_rep=None, dropna=None, data_columns=None, errors='strict', encoding='UTF-8'):
        '\n        Write the contained data to an HDF5 file using HDFStore.\n\n        Hierarchical Data Format (HDF) is self-describing, allowing an\n        application to interpret the structure and contents of a file with\n        no outside information. One HDF file can hold a mix of related objects\n        which can be accessed as a group or as individual objects.\n\n        In order to add another DataFrame or Series to an existing HDF file\n        please use append mode and a different a key.\n\n        .. warning::\n\n           One can store a subclass of ``DataFrame`` or ``Series`` to HDF5,\n           but the type of the subclass is lost upon storing.\n\n        For more information see the :ref:`user guide <io.hdf5>`.\n\n        Parameters\n        ----------\n        path_or_buf : str or pandas.HDFStore\n            File path or HDFStore object.\n        key : str\n            Identifier for the group in the store.\n        mode : {\'a\', \'w\', \'r+\'}, default \'a\'\n            Mode to open file:\n\n            - \'w\': write, a new file is created (an existing file with\n              the same name would be deleted).\n            - \'a\': append, an existing file is opened for reading and\n              writing, and if the file does not exist it is created.\n            - \'r+\': similar to \'a\', but the file must already exist.\n        complevel : {0-9}, optional\n            Specifies a compression level for data.\n            A value of 0 disables compression.\n        complib : {\'zlib\', \'lzo\', \'bzip2\', \'blosc\'}, default \'zlib\'\n            Specifies the compression library to be used.\n            As of v0.20.2 these additional compressors for Blosc are supported\n            (default if no compressor specified: \'blosc:blosclz\'):\n            {\'blosc:blosclz\', \'blosc:lz4\', \'blosc:lz4hc\', \'blosc:snappy\',\n            \'blosc:zlib\', \'blosc:zstd\'}.\n            Specifying a compression library which is not available issues\n            a ValueError.\n        append : bool, default False\n            For Table formats, append the input data to the existing.\n        format : {\'fixed\', \'table\', None}, default \'fixed\'\n            Possible values:\n\n            - \'fixed\': Fixed format. Fast writing/reading. Not-appendable,\n              nor searchable.\n            - \'table\': Table format. Write as a PyTables Table structure\n              which may perform worse but allow more flexible operations\n              like searching / selecting subsets of the data.\n            - If None, pd.get_option(\'io.hdf.default_format\') is checked,\n              followed by fallback to "fixed"\n        errors : str, default \'strict\'\n            Specifies how encoding and decoding errors are to be handled.\n            See the errors argument for :func:`open` for a full list\n            of options.\n        encoding : str, default "UTF-8"\n        min_itemsize : dict or int, optional\n            Map column names to minimum string sizes for columns.\n        nan_rep : Any, optional\n            How to represent null values as str.\n            Not allowed with append=True.\n        data_columns : list of columns or True, optional\n            List of columns to create as indexed data columns for on-disk\n            queries, or True to use all columns. By default only the axes\n            of the object are indexed. See :ref:`io.hdf5-query-data-columns`.\n            Applicable only to format=\'table\'.\n\n        See Also\n        --------\n        read_hdf : Read from HDF file.\n        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.\n        DataFrame.to_sql : Write to a sql table.\n        DataFrame.to_feather : Write out feather-format for DataFrames.\n        DataFrame.to_csv : Write out to a csv file.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'A\': [1, 2, 3], \'B\': [4, 5, 6]},\n        ...                   index=[\'a\', \'b\', \'c\'])\n        >>> df.to_hdf(\'data.h5\', key=\'df\', mode=\'w\')\n\n        We can add another object to the same file:\n\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> s.to_hdf(\'data.h5\', key=\'s\')\n\n        Reading from HDF file:\n\n        >>> pd.read_hdf(\'data.h5\', \'df\')\n        A  B\n        a  1  4\n        b  2  5\n        c  3  6\n        >>> pd.read_hdf(\'data.h5\', \'s\')\n        0    1\n        1    2\n        2    3\n        3    4\n        dtype: int64\n\n        Deleting file with data:\n\n        >>> import os\n        >>> os.remove(\'data.h5\')\n        '
        from pandas.io import pytables
        pytables.to_hdf(path_or_buf, key, self, mode=mode, complevel=complevel, complib=complib, append=append, format=format, index=index, min_itemsize=min_itemsize, nan_rep=nan_rep, dropna=dropna, data_columns=data_columns, errors=errors, encoding=encoding)

    @final
    def to_sql(self, name, con, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None, method=None):
        '\n        Write records stored in a DataFrame to a SQL database.\n\n        Databases supported by SQLAlchemy [1]_ are supported. Tables can be\n        newly created, appended to, or overwritten.\n\n        Parameters\n        ----------\n        name : str\n            Name of SQL table.\n        con : sqlalchemy.engine.(Engine or Connection) or sqlite3.Connection\n            Using SQLAlchemy makes it possible to use any DB supported by that\n            library. Legacy support is provided for sqlite3.Connection objects. The user\n            is responsible for engine disposal and connection closure for the SQLAlchemy\n            connectable See `here                 <https://docs.sqlalchemy.org/en/13/core/connections.html>`_.\n\n        schema : str, optional\n            Specify the schema (if database flavor supports this). If None, use\n            default schema.\n        if_exists : {\'fail\', \'replace\', \'append\'}, default \'fail\'\n            How to behave if the table already exists.\n\n            * fail: Raise a ValueError.\n            * replace: Drop the table before inserting new values.\n            * append: Insert new values to the existing table.\n\n        index : bool, default True\n            Write DataFrame index as a column. Uses `index_label` as the column\n            name in the table.\n        index_label : str or sequence, default None\n            Column label for index column(s). If None is given (default) and\n            `index` is True, then the index names are used.\n            A sequence should be given if the DataFrame uses MultiIndex.\n        chunksize : int, optional\n            Specify the number of rows in each batch to be written at a time.\n            By default, all rows will be written at once.\n        dtype : dict or scalar, optional\n            Specifying the datatype for columns. If a dictionary is used, the\n            keys should be the column names and the values should be the\n            SQLAlchemy types or strings for the sqlite3 legacy mode. If a\n            scalar is provided, it will be applied to all columns.\n        method : {None, \'multi\', callable}, optional\n            Controls the SQL insertion clause used:\n\n            * None : Uses standard SQL ``INSERT`` clause (one per row).\n            * \'multi\': Pass multiple values in a single ``INSERT`` clause.\n            * callable with signature ``(pd_table, conn, keys, data_iter)``.\n\n            Details and a sample callable implementation can be found in the\n            section :ref:`insert method <io.sql.method>`.\n\n            .. versionadded:: 0.24.0\n\n        Raises\n        ------\n        ValueError\n            When the table already exists and `if_exists` is \'fail\' (the\n            default).\n\n        See Also\n        --------\n        read_sql : Read a DataFrame from a table.\n\n        Notes\n        -----\n        Timezone aware datetime columns will be written as\n        ``Timestamp with timezone`` type with SQLAlchemy if supported by the\n        database. Otherwise, the datetimes will be stored as timezone unaware\n        timestamps local to the original timezone.\n\n        .. versionadded:: 0.24.0\n\n        References\n        ----------\n        .. [1] https://docs.sqlalchemy.org\n        .. [2] https://www.python.org/dev/peps/pep-0249/\n\n        Examples\n        --------\n        Create an in-memory SQLite database.\n\n        >>> from sqlalchemy import create_engine\n        >>> engine = create_engine(\'sqlite://\', echo=False)\n\n        Create a table from scratch with 3 rows.\n\n        >>> df = pd.DataFrame({\'name\' : [\'User 1\', \'User 2\', \'User 3\']})\n        >>> df\n             name\n        0  User 1\n        1  User 2\n        2  User 3\n\n        >>> df.to_sql(\'users\', con=engine)\n        >>> engine.execute("SELECT * FROM users").fetchall()\n        [(0, \'User 1\'), (1, \'User 2\'), (2, \'User 3\')]\n\n        An `sqlalchemy.engine.Connection` can also be passed to `con`:\n\n        >>> with engine.begin() as connection:\n        ...     df1 = pd.DataFrame({\'name\' : [\'User 4\', \'User 5\']})\n        ...     df1.to_sql(\'users\', con=connection, if_exists=\'append\')\n\n        This is allowed to support operations that require that the same\n        DBAPI connection is used for the entire operation.\n\n        >>> df2 = pd.DataFrame({\'name\' : [\'User 6\', \'User 7\']})\n        >>> df2.to_sql(\'users\', con=engine, if_exists=\'append\')\n        >>> engine.execute("SELECT * FROM users").fetchall()\n        [(0, \'User 1\'), (1, \'User 2\'), (2, \'User 3\'),\n         (0, \'User 4\'), (1, \'User 5\'), (0, \'User 6\'),\n         (1, \'User 7\')]\n\n        Overwrite the table with just ``df2``.\n\n        >>> df2.to_sql(\'users\', con=engine, if_exists=\'replace\',\n        ...            index_label=\'id\')\n        >>> engine.execute("SELECT * FROM users").fetchall()\n        [(0, \'User 6\'), (1, \'User 7\')]\n\n        Specify the dtype (especially useful for integers with missing values).\n        Notice that while pandas is forced to store the data as floating point,\n        the database supports nullable integers. When fetching the data with\n        Python, we get back integer scalars.\n\n        >>> df = pd.DataFrame({"A": [1, None, 2]})\n        >>> df\n             A\n        0  1.0\n        1  NaN\n        2  2.0\n\n        >>> from sqlalchemy.types import Integer\n        >>> df.to_sql(\'integers\', con=engine, index=False,\n        ...           dtype={"A": Integer()})\n\n        >>> engine.execute("SELECT * FROM integers").fetchall()\n        [(1,), (None,), (2,)]\n        '
        from pandas.io import sql
        sql.to_sql(self, name, con, schema=schema, if_exists=if_exists, index=index, index_label=index_label, chunksize=chunksize, dtype=dtype, method=method)

    @final
    @doc(storage_options=_shared_docs['storage_options'])
    def to_pickle(self, path, compression='infer', protocol=pickle.HIGHEST_PROTOCOL, storage_options=None):
        '\n        Pickle (serialize) object to file.\n\n        Parameters\n        ----------\n        path : str\n            File path where the pickled object will be stored.\n        compression : {{\'infer\', \'gzip\', \'bz2\', \'zip\', \'xz\', None}},         default \'infer\'\n            A string representing the compression to use in the output file. By\n            default, infers from the file extension in specified path.\n            Compression mode may be any of the following possible\n            values: {{‘infer’, ‘gzip’, ‘bz2’, ‘zip’, ‘xz’, None}}. If compression\n            mode is ‘infer’ and path_or_buf is path-like, then detect\n            compression mode from the following extensions:\n            ‘.gz’, ‘.bz2’, ‘.zip’ or ‘.xz’. (otherwise no compression).\n            If dict given and mode is ‘zip’ or inferred as ‘zip’, other entries\n            passed as additional compression options.\n        protocol : int\n            Int which indicates which protocol should be used by the pickler,\n            default HIGHEST_PROTOCOL (see [1]_ paragraph 12.1.2). The possible\n            values are 0, 1, 2, 3, 4, 5. A negative value for the protocol\n            parameter is equivalent to setting its value to HIGHEST_PROTOCOL.\n\n            .. [1] https://docs.python.org/3/library/pickle.html.\n\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n\n        See Also\n        --------\n        read_pickle : Load pickled pandas object (or any object) from file.\n        DataFrame.to_hdf : Write DataFrame to an HDF5 file.\n        DataFrame.to_sql : Write DataFrame to a SQL database.\n        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.\n\n        Examples\n        --------\n        >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})\n        >>> original_df\n           foo  bar\n        0    0    5\n        1    1    6\n        2    2    7\n        3    3    8\n        4    4    9\n        >>> original_df.to_pickle("./dummy.pkl")\n\n        >>> unpickled_df = pd.read_pickle("./dummy.pkl")\n        >>> unpickled_df\n           foo  bar\n        0    0    5\n        1    1    6\n        2    2    7\n        3    3    8\n        4    4    9\n\n        >>> import os\n        >>> os.remove("./dummy.pkl")\n        '
        from pandas.io.pickle import to_pickle
        to_pickle(self, path, compression=compression, protocol=protocol, storage_options=storage_options)

    @final
    def to_clipboard(self, excel=True, sep=None, **kwargs):
        "\n        Copy object to the system clipboard.\n\n        Write a text representation of object to the system clipboard.\n        This can be pasted into Excel, for example.\n\n        Parameters\n        ----------\n        excel : bool, default True\n            Produce output in a csv format for easy pasting into excel.\n\n            - True, use the provided separator for csv pasting.\n            - False, write a string representation of the object to the clipboard.\n\n        sep : str, default ``'\\t'``\n            Field delimiter.\n        **kwargs\n            These parameters will be passed to DataFrame.to_csv.\n\n        See Also\n        --------\n        DataFrame.to_csv : Write a DataFrame to a comma-separated values\n            (csv) file.\n        read_clipboard : Read text from clipboard and pass to read_table.\n\n        Notes\n        -----\n        Requirements for your platform.\n\n          - Linux : `xclip`, or `xsel` (with `PyQt4` modules)\n          - Windows : none\n          - OS X : none\n\n        Examples\n        --------\n        Copy the contents of a DataFrame to the clipboard.\n\n        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])\n\n        >>> df.to_clipboard(sep=',')  # doctest: +SKIP\n        ... # Wrote the following to the system clipboard:\n        ... # ,A,B,C\n        ... # 0,1,2,3\n        ... # 1,4,5,6\n\n        We can omit the index by passing the keyword `index` and setting\n        it to false.\n\n        >>> df.to_clipboard(sep=',', index=False)  # doctest: +SKIP\n        ... # Wrote the following to the system clipboard:\n        ... # A,B,C\n        ... # 1,2,3\n        ... # 4,5,6\n        "
        from pandas.io import clipboards
        clipboards.to_clipboard(self, excel=excel, sep=sep, **kwargs)

    @final
    def to_xarray(self):
        "\n        Return an xarray object from the pandas object.\n\n        Returns\n        -------\n        xarray.DataArray or xarray.Dataset\n            Data in the pandas structure converted to Dataset if the object is\n            a DataFrame, or a DataArray if the object is a Series.\n\n        See Also\n        --------\n        DataFrame.to_hdf : Write DataFrame to an HDF5 file.\n        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.\n\n        Notes\n        -----\n        See the `xarray docs <https://xarray.pydata.org/en/stable/>`__\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([('falcon', 'bird', 389.0, 2),\n        ...                    ('parrot', 'bird', 24.0, 2),\n        ...                    ('lion', 'mammal', 80.5, 4),\n        ...                    ('monkey', 'mammal', np.nan, 4)],\n        ...                   columns=['name', 'class', 'max_speed',\n        ...                            'num_legs'])\n        >>> df\n             name   class  max_speed  num_legs\n        0  falcon    bird      389.0         2\n        1  parrot    bird       24.0         2\n        2    lion  mammal       80.5         4\n        3  monkey  mammal        NaN         4\n\n        >>> df.to_xarray()\n        <xarray.Dataset>\n        Dimensions:    (index: 4)\n        Coordinates:\n          * index      (index) int64 0 1 2 3\n        Data variables:\n            name       (index) object 'falcon' 'parrot' 'lion' 'monkey'\n            class      (index) object 'bird' 'bird' 'mammal' 'mammal'\n            max_speed  (index) float64 389.0 24.0 80.5 nan\n            num_legs   (index) int64 2 2 4 4\n\n        >>> df['max_speed'].to_xarray()\n        <xarray.DataArray 'max_speed' (index: 4)>\n        array([389. ,  24. ,  80.5,   nan])\n        Coordinates:\n          * index    (index) int64 0 1 2 3\n\n        >>> dates = pd.to_datetime(['2018-01-01', '2018-01-01',\n        ...                         '2018-01-02', '2018-01-02'])\n        >>> df_multiindex = pd.DataFrame({'date': dates,\n        ...                               'animal': ['falcon', 'parrot',\n        ...                                          'falcon', 'parrot'],\n        ...                               'speed': [350, 18, 361, 15]})\n        >>> df_multiindex = df_multiindex.set_index(['date', 'animal'])\n\n        >>> df_multiindex\n                           speed\n        date       animal\n        2018-01-01 falcon    350\n                   parrot     18\n        2018-01-02 falcon    361\n                   parrot     15\n\n        >>> df_multiindex.to_xarray()\n        <xarray.Dataset>\n        Dimensions:  (animal: 2, date: 2)\n        Coordinates:\n          * date     (date) datetime64[ns] 2018-01-01 2018-01-02\n          * animal   (animal) object 'falcon' 'parrot'\n        Data variables:\n            speed    (date, animal) int64 350 18 361 15\n        "
        xarray = import_optional_dependency('xarray')
        if (self.ndim == 1):
            return xarray.DataArray.from_series(self)
        else:
            return xarray.Dataset.from_dataframe(self)

    @final
    @doc(returns=fmt.return_docstring)
    def to_latex(self, buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None, caption=None, label=None, position=None):
        '\n        Render object to a LaTeX tabular, longtable, or nested table/tabular.\n\n        Requires ``\\usepackage{{booktabs}}``.  The output can be copy/pasted\n        into a main LaTeX document or read from an external file\n        with ``\\input{{table.tex}}``.\n\n        .. versionchanged:: 1.0.0\n           Added caption and label arguments.\n\n        .. versionchanged:: 1.2.0\n           Added position argument, changed meaning of caption argument.\n\n        Parameters\n        ----------\n        buf : str, Path or StringIO-like, optional, default None\n            Buffer to write to. If None, the output is returned as a string.\n        columns : list of label, optional\n            The subset of columns to write. Writes all columns by default.\n        col_space : int, optional\n            The minimum width of each column.\n        header : bool or list of str, default True\n            Write out the column names. If a list of strings is given,\n            it is assumed to be aliases for the column names.\n        index : bool, default True\n            Write row names (index).\n        na_rep : str, default \'NaN\'\n            Missing data representation.\n        formatters : list of functions or dict of {{str: function}}, optional\n            Formatter functions to apply to columns\' elements by position or\n            name. The result of each function must be a unicode string.\n            List must be of length equal to the number of columns.\n        float_format : one-parameter function or str, optional, default None\n            Formatter for floating point numbers. For example\n            ``float_format="%.2f"`` and ``float_format="{{:0.2f}}".format`` will\n            both result in 0.1234 being formatted as 0.12.\n        sparsify : bool, optional\n            Set to False for a DataFrame with a hierarchical index to print\n            every multiindex key at each row. By default, the value will be\n            read from the config module.\n        index_names : bool, default True\n            Prints the names of the indexes.\n        bold_rows : bool, default False\n            Make the row labels bold in the output.\n        column_format : str, optional\n            The columns format as specified in `LaTeX table format\n            <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g. \'rcl\' for 3\n            columns. By default, \'l\' will be used for all columns except\n            columns of numbers, which default to \'r\'.\n        longtable : bool, optional\n            By default, the value will be read from the pandas config\n            module. Use a longtable environment instead of tabular. Requires\n            adding a \\usepackage{{longtable}} to your LaTeX preamble.\n        escape : bool, optional\n            By default, the value will be read from the pandas config\n            module. When set to False prevents from escaping latex special\n            characters in column names.\n        encoding : str, optional\n            A string representing the encoding to use in the output file,\n            defaults to \'utf-8\'.\n        decimal : str, default \'.\'\n            Character recognized as decimal separator, e.g. \',\' in Europe.\n        multicolumn : bool, default True\n            Use \\multicolumn to enhance MultiIndex columns.\n            The default will be read from the config module.\n        multicolumn_format : str, default \'l\'\n            The alignment for multicolumns, similar to `column_format`\n            The default will be read from the config module.\n        multirow : bool, default False\n            Use \\multirow to enhance MultiIndex rows. Requires adding a\n            \\usepackage{{multirow}} to your LaTeX preamble. Will print\n            centered labels (instead of top-aligned) across the contained\n            rows, separating groups via clines. The default will be read\n            from the pandas config module.\n        caption : str or tuple, optional\n            Tuple (full_caption, short_caption),\n            which results in ``\\caption[short_caption]{{full_caption}}``;\n            if a single string is passed, no short caption will be set.\n\n            .. versionadded:: 1.0.0\n\n            .. versionchanged:: 1.2.0\n               Optionally allow caption to be a tuple ``(full_caption, short_caption)``.\n\n        label : str, optional\n            The LaTeX label to be placed inside ``\\label{{}}`` in the output.\n            This is used with ``\\ref{{}}`` in the main ``.tex`` file.\n\n            .. versionadded:: 1.0.0\n        position : str, optional\n            The LaTeX positional argument for tables, to be placed after\n            ``\\begin{{}}`` in the output.\n\n            .. versionadded:: 1.2.0\n        {returns}\n        See Also\n        --------\n        DataFrame.to_string : Render a DataFrame to a console-friendly\n            tabular output.\n        DataFrame.to_html : Render a DataFrame as an HTML table.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(dict(name=[\'Raphael\', \'Donatello\'],\n        ...                   mask=[\'red\', \'purple\'],\n        ...                   weapon=[\'sai\', \'bo staff\']))\n        >>> print(df.to_latex(index=False))  # doctest: +NORMALIZE_WHITESPACE\n        \\begin{{tabular}}{{lll}}\n         \\toprule\n               name &    mask &    weapon \\\\\n         \\midrule\n            Raphael &     red &       sai \\\\\n          Donatello &  purple &  bo staff \\\\\n        \\bottomrule\n        \\end{{tabular}}\n        '
        if (self.ndim == 1):
            self = self.to_frame()
        if (longtable is None):
            longtable = config.get_option('display.latex.longtable')
        if (escape is None):
            escape = config.get_option('display.latex.escape')
        if (multicolumn is None):
            multicolumn = config.get_option('display.latex.multicolumn')
        if (multicolumn_format is None):
            multicolumn_format = config.get_option('display.latex.multicolumn_format')
        if (multirow is None):
            multirow = config.get_option('display.latex.multirow')
        self = cast('DataFrame', self)
        formatter = DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, header=header, index=index, formatters=formatters, float_format=float_format, bold_rows=bold_rows, sparsify=sparsify, index_names=index_names, escape=escape, decimal=decimal)
        return DataFrameRenderer(formatter).to_latex(buf=buf, column_format=column_format, longtable=longtable, encoding=encoding, multicolumn=multicolumn, multicolumn_format=multicolumn_format, multirow=multirow, caption=caption, label=label, position=position)

    @final
    @doc(storage_options=_shared_docs['storage_options'])
    def to_csv(self, path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None):
        '\n        Write object to a comma-separated values (csv) file.\n\n        .. versionchanged:: 0.24.0\n            The order of arguments for Series was changed.\n\n        Parameters\n        ----------\n        path_or_buf : str or file handle, default None\n            File path or object, if None is provided the result is returned as\n            a string.  If a non-binary file object is passed, it should be opened\n            with `newline=\'\'`, disabling universal newlines. If a binary\n            file object is passed, `mode` might need to contain a `\'b\'`.\n\n            .. versionchanged:: 0.24.0\n\n               Was previously named "path" for Series.\n\n            .. versionchanged:: 1.2.0\n\n               Support for binary file objects was introduced.\n\n        sep : str, default \',\'\n            String of length 1. Field delimiter for the output file.\n        na_rep : str, default \'\'\n            Missing data representation.\n        float_format : str, default None\n            Format string for floating point numbers.\n        columns : sequence, optional\n            Columns to write.\n        header : bool or list of str, default True\n            Write out the column names. If a list of strings is given it is\n            assumed to be aliases for the column names.\n\n            .. versionchanged:: 0.24.0\n\n               Previously defaulted to False for Series.\n\n        index : bool, default True\n            Write row names (index).\n        index_label : str or sequence, or False, default None\n            Column label for index column(s) if desired. If None is given, and\n            `header` and `index` are True, then the index names are used. A\n            sequence should be given if the object uses MultiIndex. If\n            False do not print fields for index names. Use index_label=False\n            for easier importing in R.\n        mode : str\n            Python write mode, default \'w\'.\n        encoding : str, optional\n            A string representing the encoding to use in the output file,\n            defaults to \'utf-8\'. `encoding` is not supported if `path_or_buf`\n            is a non-binary file object.\n        compression : str or dict, default \'infer\'\n            If str, represents compression mode. If dict, value at \'method\' is\n            the compression mode. Compression mode may be any of the following\n            possible values: {{\'infer\', \'gzip\', \'bz2\', \'zip\', \'xz\', None}}. If\n            compression mode is \'infer\' and `path_or_buf` is path-like, then\n            detect compression mode from the following extensions: \'.gz\',\n            \'.bz2\', \'.zip\' or \'.xz\'. (otherwise no compression). If dict given\n            and mode is one of {{\'zip\', \'gzip\', \'bz2\'}}, or inferred as\n            one of the above, other entries passed as\n            additional compression options.\n\n            .. versionchanged:: 1.0.0\n\n               May now be a dict with key \'method\' as compression mode\n               and other entries as additional compression options if\n               compression mode is \'zip\'.\n\n            .. versionchanged:: 1.1.0\n\n               Passing compression options as keys in dict is\n               supported for compression modes \'gzip\' and \'bz2\'\n               as well as \'zip\'.\n\n            .. versionchanged:: 1.2.0\n\n                Compression is supported for binary file objects.\n\n            .. versionchanged:: 1.2.0\n\n                Previous versions forwarded dict entries for \'gzip\' to\n                `gzip.open` instead of `gzip.GzipFile` which prevented\n                setting `mtime`.\n\n        quoting : optional constant from csv module\n            Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`\n            then floats are converted to strings and thus csv.QUOTE_NONNUMERIC\n            will treat them as non-numeric.\n        quotechar : str, default \'\\"\'\n            String of length 1. Character used to quote fields.\n        line_terminator : str, optional\n            The newline character or character sequence to use in the output\n            file. Defaults to `os.linesep`, which depends on the OS in which\n            this method is called (\'\\n\' for linux, \'\\r\\n\' for Windows, i.e.).\n\n            .. versionchanged:: 0.24.0\n        chunksize : int or None\n            Rows to write at a time.\n        date_format : str, default None\n            Format string for datetime objects.\n        doublequote : bool, default True\n            Control quoting of `quotechar` inside a field.\n        escapechar : str, default None\n            String of length 1. Character used to escape `sep` and `quotechar`\n            when appropriate.\n        decimal : str, default \'.\'\n            Character recognized as decimal separator. E.g. use \',\' for\n            European data.\n        errors : str, default \'strict\'\n            Specifies how encoding and decoding errors are to be handled.\n            See the errors argument for :func:`open` for a full list\n            of options.\n\n            .. versionadded:: 1.1.0\n\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n\n        Returns\n        -------\n        None or str\n            If path_or_buf is None, returns the resulting csv format as a\n            string. Otherwise returns None.\n\n        See Also\n        --------\n        read_csv : Load a CSV file into a DataFrame.\n        to_excel : Write DataFrame to an Excel file.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({{\'name\': [\'Raphael\', \'Donatello\'],\n        ...                    \'mask\': [\'red\', \'purple\'],\n        ...                    \'weapon\': [\'sai\', \'bo staff\']}})\n        >>> df.to_csv(index=False)\n        \'name,mask,weapon\\nRaphael,red,sai\\nDonatello,purple,bo staff\\n\'\n\n        Create \'out.zip\' containing \'out.csv\'\n\n        >>> compression_opts = dict(method=\'zip\',\n        ...                         archive_name=\'out.csv\')  # doctest: +SKIP\n        >>> df.to_csv(\'out.zip\', index=False,\n        ...           compression=compression_opts)  # doctest: +SKIP\n        '
        df = (self if isinstance(self, ABCDataFrame) else self.to_frame())
        formatter = DataFrameFormatter(frame=df, header=header, index=index, na_rep=na_rep, float_format=float_format, decimal=decimal)
        return DataFrameRenderer(formatter).to_csv(path_or_buf, line_terminator=line_terminator, sep=sep, encoding=encoding, errors=errors, compression=compression, quoting=quoting, columns=columns, index_label=index_label, mode=mode, chunksize=chunksize, quotechar=quotechar, date_format=date_format, doublequote=doublequote, escapechar=escapechar, storage_options=storage_options)

    @final
    def _set_as_cached(self, item, cacher):
        '\n        Set the _cacher attribute on the calling object with a weakref to\n        cacher.\n        '
        self._cacher = (item, weakref.ref(cacher))

    @final
    def _reset_cacher(self):
        '\n        Reset the cacher.\n        '
        if hasattr(self, '_cacher'):
            del self._cacher

    @final
    def _maybe_cache_changed(self, item, value):
        '\n        The object has called back to us saying maybe it has changed.\n        '
        loc = self._info_axis.get_loc(item)
        self._mgr.iset(loc, value)

    @final
    @property
    def _is_cached(self):
        'Return boolean indicating if self is cached or not.'
        return (getattr(self, '_cacher', None) is not None)

    @final
    def _get_cacher(self):
        'return my cacher or None'
        cacher = getattr(self, '_cacher', None)
        if (cacher is not None):
            cacher = cacher[1]()
        return cacher

    @final
    def _maybe_update_cacher(self, clear=False, verify_is_copy=True):
        '\n        See if we need to update our parent cacher if clear, then clear our\n        cache.\n\n        Parameters\n        ----------\n        clear : bool, default False\n            Clear the item cache.\n        verify_is_copy : bool, default True\n            Provide is_copy checks.\n        '
        cacher = getattr(self, '_cacher', None)
        if (cacher is not None):
            ref = cacher[1]()
            if (ref is None):
                del self._cacher
            elif (len(self) == len(ref)):
                ref._maybe_cache_changed(cacher[0], self)
            else:
                ref._item_cache.pop(cacher[0], None)
        if verify_is_copy:
            self._check_setitem_copy(stacklevel=5, t='referent')
        if clear:
            self._clear_item_cache()

    @final
    def _clear_item_cache(self):
        self._item_cache.clear()

    def take(self, indices, axis=0, is_copy=None, **kwargs):
        "\n        Return the elements in the given *positional* indices along an axis.\n\n        This means that we are not indexing according to actual values in\n        the index attribute of the object. We are indexing according to the\n        actual position of the element in the object.\n\n        Parameters\n        ----------\n        indices : array-like\n            An array of ints indicating which positions to take.\n        axis : {0 or 'index', 1 or 'columns', None}, default 0\n            The axis on which to select elements. ``0`` means that we are\n            selecting rows, ``1`` means that we are selecting columns.\n        is_copy : bool\n            Before pandas 1.0, ``is_copy=False`` can be specified to ensure\n            that the return value is an actual copy. Starting with pandas 1.0,\n            ``take`` always returns a copy, and the keyword is therefore\n            deprecated.\n\n            .. deprecated:: 1.0.0\n        **kwargs\n            For compatibility with :meth:`numpy.take`. Has no effect on the\n            output.\n\n        Returns\n        -------\n        taken : same type as caller\n            An array-like containing the elements taken from the object.\n\n        See Also\n        --------\n        DataFrame.loc : Select a subset of a DataFrame by labels.\n        DataFrame.iloc : Select a subset of a DataFrame by positions.\n        numpy.take : Take elements from an array along an axis.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([('falcon', 'bird', 389.0),\n        ...                    ('parrot', 'bird', 24.0),\n        ...                    ('lion', 'mammal', 80.5),\n        ...                    ('monkey', 'mammal', np.nan)],\n        ...                   columns=['name', 'class', 'max_speed'],\n        ...                   index=[0, 2, 3, 1])\n        >>> df\n             name   class  max_speed\n        0  falcon    bird      389.0\n        2  parrot    bird       24.0\n        3    lion  mammal       80.5\n        1  monkey  mammal        NaN\n\n        Take elements at positions 0 and 3 along the axis 0 (default).\n\n        Note how the actual indices selected (0 and 1) do not correspond to\n        our selected indices 0 and 3. That's because we are selecting the 0th\n        and 3rd rows, not rows whose indices equal 0 and 3.\n\n        >>> df.take([0, 3])\n             name   class  max_speed\n        0  falcon    bird      389.0\n        1  monkey  mammal        NaN\n\n        Take elements at indices 1 and 2 along the axis 1 (column selection).\n\n        >>> df.take([1, 2], axis=1)\n            class  max_speed\n        0    bird      389.0\n        2    bird       24.0\n        3  mammal       80.5\n        1  mammal        NaN\n\n        We may take elements using negative integers for positive indices,\n        starting from the end of the object, just like with Python lists.\n\n        >>> df.take([-1, -2])\n             name   class  max_speed\n        1  monkey  mammal        NaN\n        3    lion  mammal       80.5\n        "
        if (is_copy is not None):
            warnings.warn("is_copy is deprecated and will be removed in a future version. 'take' always returns a copy, so there is no need to specify this.", FutureWarning, stacklevel=2)
        nv.validate_take((), kwargs)
        self._consolidate_inplace()
        new_data = self._mgr.take(indices, axis=self._get_block_manager_axis(axis), verify=True)
        return self._constructor(new_data).__finalize__(self, method='take')

    @final
    def _take_with_is_copy(self, indices, axis=0):
        '\n        Internal version of the `take` method that sets the `_is_copy`\n        attribute to keep track of the parent dataframe (using in indexing\n        for the SettingWithCopyWarning).\n\n        See the docstring of `take` for full explanation of the parameters.\n        '
        result = self.take(indices=indices, axis=axis)
        if (not result._get_axis(axis).equals(self._get_axis(axis))):
            result._set_is_copy(self)
        return result

    @final
    def xs(self, key, axis=0, level=None, drop_level=True):
        "\n        Return cross-section from the Series/DataFrame.\n\n        This method takes a `key` argument to select data at a particular\n        level of a MultiIndex.\n\n        Parameters\n        ----------\n        key : label or tuple of label\n            Label contained in the index, or partially in a MultiIndex.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            Axis to retrieve cross-section on.\n        level : object, defaults to first n levels (n=1 or len(key))\n            In case of a key partially contained in a MultiIndex, indicate\n            which levels are used. Levels can be referred by label or position.\n        drop_level : bool, default True\n            If False, returns object with same levels as self.\n\n        Returns\n        -------\n        Series or DataFrame\n            Cross-section from the original Series or DataFrame\n            corresponding to the selected index levels.\n\n        See Also\n        --------\n        DataFrame.loc : Access a group of rows and columns\n            by label(s) or a boolean array.\n        DataFrame.iloc : Purely integer-location based indexing\n            for selection by position.\n\n        Notes\n        -----\n        `xs` can not be used to set values.\n\n        MultiIndex Slicers is a generic way to get/set values on\n        any level or levels.\n        It is a superset of `xs` functionality, see\n        :ref:`MultiIndex Slicers <advanced.mi_slicers>`.\n\n        Examples\n        --------\n        >>> d = {'num_legs': [4, 4, 2, 2],\n        ...      'num_wings': [0, 0, 2, 2],\n        ...      'class': ['mammal', 'mammal', 'mammal', 'bird'],\n        ...      'animal': ['cat', 'dog', 'bat', 'penguin'],\n        ...      'locomotion': ['walks', 'walks', 'flies', 'walks']}\n        >>> df = pd.DataFrame(data=d)\n        >>> df = df.set_index(['class', 'animal', 'locomotion'])\n        >>> df\n                                   num_legs  num_wings\n        class  animal  locomotion\n        mammal cat     walks              4          0\n               dog     walks              4          0\n               bat     flies              2          2\n        bird   penguin walks              2          2\n\n        Get values at specified index\n\n        >>> df.xs('mammal')\n                           num_legs  num_wings\n        animal locomotion\n        cat    walks              4          0\n        dog    walks              4          0\n        bat    flies              2          2\n\n        Get values at several indexes\n\n        >>> df.xs(('mammal', 'dog'))\n                    num_legs  num_wings\n        locomotion\n        walks              4          0\n\n        Get values at specified index and level\n\n        >>> df.xs('cat', level=1)\n                           num_legs  num_wings\n        class  locomotion\n        mammal walks              4          0\n\n        Get values at several indexes and levels\n\n        >>> df.xs(('bird', 'walks'),\n        ...       level=[0, 'locomotion'])\n                 num_legs  num_wings\n        animal\n        penguin         2          2\n\n        Get values at specified column and axis\n\n        >>> df.xs('num_wings', axis=1)\n        class   animal   locomotion\n        mammal  cat      walks         0\n                dog      walks         0\n                bat      flies         2\n        bird    penguin  walks         2\n        Name: num_wings, dtype: int64\n        "
        axis = self._get_axis_number(axis)
        labels = self._get_axis(axis)
        if (level is not None):
            if (not isinstance(labels, MultiIndex)):
                raise TypeError('Index must be a MultiIndex')
            (loc, new_ax) = labels.get_loc_level(key, level=level, drop_level=drop_level)
            _indexer = ([slice(None)] * self.ndim)
            _indexer[axis] = loc
            indexer = tuple(_indexer)
            result = self.iloc[indexer]
            setattr(result, result._get_axis_name(axis), new_ax)
            return result
        if (axis == 1):
            if drop_level:
                return self[key]
            index = self.columns
        else:
            index = self.index
        self._consolidate_inplace()
        if isinstance(index, MultiIndex):
            try:
                (loc, new_index) = index._get_loc_level(key, level=0, drop_level=drop_level)
            except TypeError as e:
                raise TypeError(f'Expected label or tuple of labels, got {key}') from e
        else:
            loc = index.get_loc(key)
            if isinstance(loc, np.ndarray):
                if (loc.dtype == np.bool_):
                    (inds,) = loc.nonzero()
                    return self._take_with_is_copy(inds, axis=axis)
                else:
                    return self._take_with_is_copy(loc, axis=axis)
            if (not is_scalar(loc)):
                new_index = index[loc]
        if (is_scalar(loc) and (axis == 0)):
            if (self.ndim == 1):
                return self._values[loc]
            new_values = self._mgr.fast_xs(loc)
            result = self._constructor_sliced(new_values, index=self.columns, name=self.index[loc], dtype=new_values.dtype)
        elif is_scalar(loc):
            result = self.iloc[:, slice(loc, (loc + 1))]
        elif (axis == 1):
            result = self.iloc[:, loc]
        else:
            result = self.iloc[loc]
            result.index = new_index
        result._set_is_copy(self, copy=(not result._is_view))
        return result

    def __getitem__(self, item):
        raise AbstractMethodError(self)

    @final
    def _get_item_cache(self, item):
        'Return the cached item, item represents a label indexer.'
        cache = self._item_cache
        res = cache.get(item)
        if (res is None):
            loc = self.columns.get_loc(item)
            values = self._mgr.iget(loc)
            res = self._box_col_values(values, loc).__finalize__(self)
            cache[item] = res
            res._set_as_cached(item, self)
            res._is_copy = self._is_copy
        return res

    def _slice(self, slobj, axis=0):
        '\n        Construct a slice of this container.\n\n        Slicing with this method is *always* positional.\n        '
        assert isinstance(slobj, slice), type(slobj)
        axis = self._get_block_manager_axis(axis)
        result = self._constructor(self._mgr.get_slice(slobj, axis=axis))
        result = result.__finalize__(self)
        is_copy = ((axis != 0) or result._is_view)
        result._set_is_copy(self, copy=is_copy)
        return result

    @final
    def _set_is_copy(self, ref, copy=True):
        if (not copy):
            self._is_copy = None
        else:
            assert (ref is not None)
            self._is_copy = weakref.ref(ref)

    @final
    def _check_is_chained_assignment_possible(self):
        '\n        Check if we are a view, have a cacher, and are of mixed type.\n        If so, then force a setitem_copy check.\n\n        Should be called just near setting a value\n\n        Will return a boolean if it we are a view and are cached, but a\n        single-dtype meaning that the cacher should be updated following\n        setting.\n        '
        if (self._is_view and self._is_cached):
            ref = self._get_cacher()
            if ((ref is not None) and ref._is_mixed_type):
                self._check_setitem_copy(stacklevel=4, t='referent', force=True)
            return True
        elif self._is_copy:
            self._check_setitem_copy(stacklevel=4, t='referent')
        return False

    @final
    def _check_setitem_copy(self, stacklevel=4, t='setting', force=False):
        "\n\n        Parameters\n        ----------\n        stacklevel : int, default 4\n           the level to show of the stack when the error is output\n        t : str, the type of setting error\n        force : bool, default False\n           If True, then force showing an error.\n\n        validate if we are doing a setitem on a chained copy.\n\n        If you call this function, be sure to set the stacklevel such that the\n        user will see the error *at the level of setting*\n\n        It is technically possible to figure out that we are setting on\n        a copy even WITH a multi-dtyped pandas object. In other words, some\n        blocks may be views while other are not. Currently _is_view will ALWAYS\n        return False for multi-blocks to avoid having to handle this case.\n\n        df = DataFrame(np.arange(0,9), columns=['count'])\n        df['group'] = 'b'\n\n        # This technically need not raise SettingWithCopy if both are view\n        # (which is not # generally guaranteed but is usually True.  However,\n        # this is in general not a good practice and we recommend using .loc.\n        df.iloc[0:5]['group'] = 'a'\n\n        "
        if (not (force or self._is_copy)):
            return
        value = config.get_option('mode.chained_assignment')
        if (value is None):
            return
        if ((self._is_copy is not None) and (not isinstance(self._is_copy, str))):
            r = self._is_copy()
            if ((not gc.get_referents(r)) or ((r is not None) and (r.shape == self.shape))):
                self._is_copy = None
                return
        if isinstance(self._is_copy, str):
            t = self._is_copy
        elif (t == 'referent'):
            t = '\nA value is trying to be set on a copy of a slice from a DataFrame\n\nSee the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy'
        else:
            t = '\nA value is trying to be set on a copy of a slice from a DataFrame.\nTry using .loc[row_indexer,col_indexer] = value instead\n\nSee the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy'
        if (value == 'raise'):
            raise com.SettingWithCopyError(t)
        elif (value == 'warn'):
            warnings.warn(t, com.SettingWithCopyWarning, stacklevel=stacklevel)

    def __delitem__(self, key):
        '\n        Delete item\n        '
        deleted = False
        maybe_shortcut = False
        if ((self.ndim == 2) and isinstance(self.columns, MultiIndex)):
            try:
                maybe_shortcut = (key not in self.columns._engine)
            except TypeError:
                pass
        if maybe_shortcut:
            if (not isinstance(key, tuple)):
                key = (key,)
            for col in self.columns:
                if (isinstance(col, tuple) and (col[:len(key)] == key)):
                    del self[col]
                    deleted = True
        if (not deleted):
            loc = self.axes[(- 1)].get_loc(key)
            self._mgr.idelete(loc)
        try:
            del self._item_cache[key]
        except KeyError:
            pass

    @final
    def _check_inplace_and_allows_duplicate_labels(self, inplace):
        if (inplace and (not self.flags.allows_duplicate_labels)):
            raise ValueError("Cannot specify 'inplace=True' when 'self.flags.allows_duplicate_labels' is False.")

    @final
    def get(self, key, default=None):
        '\n        Get item from object for given key (ex: DataFrame column).\n\n        Returns default value if not found.\n\n        Parameters\n        ----------\n        key : object\n\n        Returns\n        -------\n        value : same type as items contained in object\n        '
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    @final
    @property
    def _is_view(self):
        'Return boolean indicating if self is view of another array '
        return self._mgr.is_view

    @final
    def reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None):
        "\n        Return an object with matching indices as other object.\n\n        Conform the object to the same index on all axes. Optional\n        filling logic, placing NaN in locations having no value\n        in the previous index. A new object is produced unless the\n        new index is equivalent to the current one and copy=False.\n\n        Parameters\n        ----------\n        other : Object of the same data type\n            Its row and column indices are used to define the new indices\n            of this object.\n        method : {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}\n            Method to use for filling holes in reindexed DataFrame.\n            Please note: this is only applicable to DataFrames/Series with a\n            monotonically increasing/decreasing index.\n\n            * None (default): don't fill gaps\n            * pad / ffill: propagate last valid observation forward to next\n              valid\n            * backfill / bfill: use next valid observation to fill gap\n            * nearest: use nearest valid observations to fill gap.\n\n        copy : bool, default True\n            Return a new object, even if the passed indexes are the same.\n        limit : int, default None\n            Maximum number of consecutive labels to fill for inexact matches.\n        tolerance : optional\n            Maximum distance between original and new labels for inexact\n            matches. The values of the index at the matching locations must\n            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.\n\n            Tolerance may be a scalar value, which applies the same tolerance\n            to all values, or list-like, which applies variable tolerance per\n            element. List-like includes list, tuple, array, Series, and must be\n            the same size as the index and its dtype must exactly match the\n            index's type.\n\n        Returns\n        -------\n        Series or DataFrame\n            Same type as caller, but with changed indices on each axis.\n\n        See Also\n        --------\n        DataFrame.set_index : Set row labels.\n        DataFrame.reset_index : Remove row labels or move them to new columns.\n        DataFrame.reindex : Change to new indices or expand indices.\n\n        Notes\n        -----\n        Same as calling\n        ``.reindex(index=other.index, columns=other.columns,...)``.\n\n        Examples\n        --------\n        >>> df1 = pd.DataFrame([[24.3, 75.7, 'high'],\n        ...                     [31, 87.8, 'high'],\n        ...                     [22, 71.6, 'medium'],\n        ...                     [35, 95, 'medium']],\n        ...                    columns=['temp_celsius', 'temp_fahrenheit',\n        ...                             'windspeed'],\n        ...                    index=pd.date_range(start='2014-02-12',\n        ...                                        end='2014-02-15', freq='D'))\n\n        >>> df1\n                    temp_celsius  temp_fahrenheit windspeed\n        2014-02-12          24.3             75.7      high\n        2014-02-13          31.0             87.8      high\n        2014-02-14          22.0             71.6    medium\n        2014-02-15          35.0             95.0    medium\n\n        >>> df2 = pd.DataFrame([[28, 'low'],\n        ...                     [30, 'low'],\n        ...                     [35.1, 'medium']],\n        ...                    columns=['temp_celsius', 'windspeed'],\n        ...                    index=pd.DatetimeIndex(['2014-02-12', '2014-02-13',\n        ...                                            '2014-02-15']))\n\n        >>> df2\n                    temp_celsius windspeed\n        2014-02-12          28.0       low\n        2014-02-13          30.0       low\n        2014-02-15          35.1    medium\n\n        >>> df2.reindex_like(df1)\n                    temp_celsius  temp_fahrenheit windspeed\n        2014-02-12          28.0              NaN       low\n        2014-02-13          30.0              NaN       low\n        2014-02-14           NaN              NaN       NaN\n        2014-02-15          35.1              NaN    medium\n        "
        d = other._construct_axes_dict(axes=self._AXIS_ORDERS, method=method, copy=copy, limit=limit, tolerance=tolerance)
        return self.reindex(**d)

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (labels is not None):
            if ((index is not None) or (columns is not None)):
                raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
            axis_name = self._get_axis_name(axis)
            axes = {axis_name: labels}
        elif ((index is not None) or (columns is not None)):
            (axes, _) = self._construct_axes_from_arguments((index, columns), {})
        else:
            raise ValueError("Need to specify at least one of 'labels', 'index' or 'columns'")
        obj = self
        for (axis, labels) in axes.items():
            if (labels is not None):
                obj = obj._drop_axis(labels, axis, level=level, errors=errors)
        if inplace:
            self._update_inplace(obj)
        else:
            return obj

    @final
    def _drop_axis(self, labels, axis, level=None, errors='raise'):
        "\n        Drop labels from specified axis. Used in the ``drop`` method\n        internally.\n\n        Parameters\n        ----------\n        labels : single label or list-like\n        axis : int or axis name\n        level : int or level name, default None\n            For MultiIndex\n        errors : {'ignore', 'raise'}, default 'raise'\n            If 'ignore', suppress error and existing labels are dropped.\n\n        "
        axis = self._get_axis_number(axis)
        axis_name = self._get_axis_name(axis)
        axis = self._get_axis(axis)
        if axis.is_unique:
            if (level is not None):
                if (not isinstance(axis, MultiIndex)):
                    raise AssertionError('axis must be a MultiIndex')
                new_axis = axis.drop(labels, level=level, errors=errors)
            else:
                new_axis = axis.drop(labels, errors=errors)
            result = self.reindex(**{axis_name: new_axis})
        else:
            labels = ensure_object(com.index_labels_to_array(labels))
            if (level is not None):
                if (not isinstance(axis, MultiIndex)):
                    raise AssertionError('axis must be a MultiIndex')
                indexer = (~ axis.get_level_values(level).isin(labels))
                if ((errors == 'raise') and indexer.all()):
                    raise KeyError(f'{labels} not found in axis')
            elif (isinstance(axis, MultiIndex) and (labels.dtype == 'object')):
                indexer = (~ axis.get_level_values(0).isin(labels))
            else:
                indexer = (~ axis.isin(labels))
                labels_missing = (axis.get_indexer_for(labels) == (- 1)).any()
                if ((errors == 'raise') and labels_missing):
                    raise KeyError(f'{labels} not found in axis')
            slicer = ([slice(None)] * self.ndim)
            slicer[self._get_axis_number(axis_name)] = indexer
            result = self.loc[tuple(slicer)]
        return result

    @final
    def _update_inplace(self, result, verify_is_copy=True):
        '\n        Replace self internals with result.\n\n        Parameters\n        ----------\n        result : same type as self\n        verify_is_copy : bool, default True\n            Provide is_copy checks.\n        '
        self._reset_cache()
        self._clear_item_cache()
        self._mgr = result._mgr
        self._maybe_update_cacher(verify_is_copy=verify_is_copy)

    @final
    def add_prefix(self, prefix):
        "\n        Prefix labels with string `prefix`.\n\n        For Series, the row labels are prefixed.\n        For DataFrame, the column labels are prefixed.\n\n        Parameters\n        ----------\n        prefix : str\n            The string to add before each label.\n\n        Returns\n        -------\n        Series or DataFrame\n            New Series or DataFrame with updated labels.\n\n        See Also\n        --------\n        Series.add_suffix: Suffix row labels with string `suffix`.\n        DataFrame.add_suffix: Suffix column labels with string `suffix`.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> s\n        0    1\n        1    2\n        2    3\n        3    4\n        dtype: int64\n\n        >>> s.add_prefix('item_')\n        item_0    1\n        item_1    2\n        item_2    3\n        item_3    4\n        dtype: int64\n\n        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})\n        >>> df\n           A  B\n        0  1  3\n        1  2  4\n        2  3  5\n        3  4  6\n\n        >>> df.add_prefix('col_')\n             col_A  col_B\n        0       1       3\n        1       2       4\n        2       3       5\n        3       4       6\n        "
        f = functools.partial('{prefix}{}'.format, prefix=prefix)
        mapper = {self._info_axis_name: f}
        return self.rename(**mapper)

    @final
    def add_suffix(self, suffix):
        "\n        Suffix labels with string `suffix`.\n\n        For Series, the row labels are suffixed.\n        For DataFrame, the column labels are suffixed.\n\n        Parameters\n        ----------\n        suffix : str\n            The string to add after each label.\n\n        Returns\n        -------\n        Series or DataFrame\n            New Series or DataFrame with updated labels.\n\n        See Also\n        --------\n        Series.add_prefix: Prefix row labels with string `prefix`.\n        DataFrame.add_prefix: Prefix column labels with string `prefix`.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> s\n        0    1\n        1    2\n        2    3\n        3    4\n        dtype: int64\n\n        >>> s.add_suffix('_item')\n        0_item    1\n        1_item    2\n        2_item    3\n        3_item    4\n        dtype: int64\n\n        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})\n        >>> df\n           A  B\n        0  1  3\n        1  2  4\n        2  3  5\n        3  4  6\n\n        >>> df.add_suffix('_col')\n             A_col  B_col\n        0       1       3\n        1       2       4\n        2       3       5\n        3       4       6\n        "
        f = functools.partial('{}{suffix}'.format, suffix=suffix)
        mapper = {self._info_axis_name: f}
        return self.rename(**mapper)

    def sort_values(self, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None):
        '\n        Sort by the values along either axis.\n\n        Parameters\n        ----------%(optional_by)s\n        axis : %(axes_single_arg)s, default 0\n             Axis to be sorted.\n        ascending : bool or list of bool, default True\n             Sort ascending vs. descending. Specify list for multiple sort\n             orders.  If this is a list of bools, must match the length of\n             the by.\n        inplace : bool, default False\n             If True, perform operation in-place.\n        kind : {\'quicksort\', \'mergesort\', \'heapsort\', \'stable\'}, default \'quicksort\'\n             Choice of sorting algorithm. See also :func:`numpy.sort` for more\n             information. `mergesort` and `stable` are the only stable algorithms. For\n             DataFrames, this option is only applied when sorting on a single\n             column or label.\n        na_position : {\'first\', \'last\'}, default \'last\'\n             Puts NaNs at the beginning if `first`; `last` puts NaNs at the\n             end.\n        ignore_index : bool, default False\n             If True, the resulting axis will be labeled 0, 1, …, n - 1.\n\n             .. versionadded:: 1.0.0\n\n        key : callable, optional\n            Apply the key function to the values\n            before sorting. This is similar to the `key` argument in the\n            builtin :meth:`sorted` function, with the notable difference that\n            this `key` function should be *vectorized*. It should expect a\n            ``Series`` and return a Series with the same shape as the input.\n            It will be applied to each column in `by` independently.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        DataFrame or None\n            DataFrame with sorted values or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.sort_index : Sort a DataFrame by the index.\n        Series.sort_values : Similar method for a Series.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\n        ...     \'col1\': [\'A\', \'A\', \'B\', np.nan, \'D\', \'C\'],\n        ...     \'col2\': [2, 1, 9, 8, 7, 4],\n        ...     \'col3\': [0, 1, 9, 4, 2, 3],\n        ...     \'col4\': [\'a\', \'B\', \'c\', \'D\', \'e\', \'F\']\n        ... })\n        >>> df\n          col1  col2  col3 col4\n        0    A     2     0    a\n        1    A     1     1    B\n        2    B     9     9    c\n        3  NaN     8     4    D\n        4    D     7     2    e\n        5    C     4     3    F\n\n        Sort by col1\n\n        >>> df.sort_values(by=[\'col1\'])\n          col1  col2  col3 col4\n        0    A     2     0    a\n        1    A     1     1    B\n        2    B     9     9    c\n        5    C     4     3    F\n        4    D     7     2    e\n        3  NaN     8     4    D\n\n        Sort by multiple columns\n\n        >>> df.sort_values(by=[\'col1\', \'col2\'])\n          col1  col2  col3 col4\n        1    A     1     1    B\n        0    A     2     0    a\n        2    B     9     9    c\n        5    C     4     3    F\n        4    D     7     2    e\n        3  NaN     8     4    D\n\n        Sort Descending\n\n        >>> df.sort_values(by=\'col1\', ascending=False)\n          col1  col2  col3 col4\n        4    D     7     2    e\n        5    C     4     3    F\n        2    B     9     9    c\n        0    A     2     0    a\n        1    A     1     1    B\n        3  NaN     8     4    D\n\n        Putting NAs first\n\n        >>> df.sort_values(by=\'col1\', ascending=False, na_position=\'first\')\n          col1  col2  col3 col4\n        3  NaN     8     4    D\n        4    D     7     2    e\n        5    C     4     3    F\n        2    B     9     9    c\n        0    A     2     0    a\n        1    A     1     1    B\n\n        Sorting with a key function\n\n        >>> df.sort_values(by=\'col4\', key=lambda col: col.str.lower())\n           col1  col2  col3 col4\n        0    A     2     0    a\n        1    A     1     1    B\n        2    B     9     9    c\n        3  NaN     8     4    D\n        4    D     7     2    e\n        5    C     4     3    F\n\n        Natural sort with the key argument,\n        using the `natsort <https://github.com/SethMMorton/natsort>` package.\n\n        >>> df = pd.DataFrame({\n        ...    "time": [\'0hr\', \'128hr\', \'72hr\', \'48hr\', \'96hr\'],\n        ...    "value": [10, 20, 30, 40, 50]\n        ... })\n        >>> df\n            time  value\n        0    0hr     10\n        1  128hr     20\n        2   72hr     30\n        3   48hr     40\n        4   96hr     50\n        >>> from natsort import index_natsorted\n        >>> df.sort_values(\n        ...    by="time",\n        ...    key=lambda x: np.argsort(index_natsorted(df["time"]))\n        ... )\n            time  value\n        0    0hr     10\n        3   48hr     40\n        2   72hr     30\n        4   96hr     50\n        1  128hr     20\n        '
        raise AbstractMethodError(self)

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None):
        inplace = validate_bool_kwarg(inplace, 'inplace')
        axis = self._get_axis_number(axis)
        target = self._get_axis(axis)
        indexer = get_indexer_indexer(target, level, ascending, kind, na_position, sort_remaining, key)
        if (indexer is None):
            if inplace:
                return
            else:
                return self.copy()
        baxis = self._get_block_manager_axis(axis)
        new_data = self._mgr.take(indexer, axis=baxis, verify=False)
        new_data.axes[baxis] = new_data.axes[baxis]._sort_levels_monotonic()
        if ignore_index:
            axis = (1 if isinstance(self, ABCDataFrame) else 0)
            new_data.axes[axis] = ibase.default_index(len(indexer))
        result = self._constructor(new_data)
        if inplace:
            return self._update_inplace(result)
        else:
            return result.__finalize__(self, method='sort_index')

    @doc(klass=_shared_doc_kwargs['klass'], axes=_shared_doc_kwargs['axes'], optional_labels='', optional_axis='')
    def reindex(self, *args, **kwargs):
        '\n        Conform {klass} to new index with optional filling logic.\n\n        Places NA/NaN in locations having no value in the previous index. A new object\n        is produced unless the new index is equivalent to the current one and\n        ``copy=False``.\n\n        Parameters\n        ----------\n        {optional_labels}\n        {axes} : array-like, optional\n            New labels / index to conform to, should be specified using\n            keywords. Preferably an Index object to avoid duplicating data.\n        {optional_axis}\n        method : {{None, \'backfill\'/\'bfill\', \'pad\'/\'ffill\', \'nearest\'}}\n            Method to use for filling holes in reindexed DataFrame.\n            Please note: this is only applicable to DataFrames/Series with a\n            monotonically increasing/decreasing index.\n\n            * None (default): don\'t fill gaps\n            * pad / ffill: Propagate last valid observation forward to next\n              valid.\n            * backfill / bfill: Use next valid observation to fill gap.\n            * nearest: Use nearest valid observations to fill gap.\n\n        copy : bool, default True\n            Return a new object, even if the passed indexes are the same.\n        level : int or name\n            Broadcast across a level, matching Index values on the\n            passed MultiIndex level.\n        fill_value : scalar, default np.NaN\n            Value to use for missing values. Defaults to NaN, but can be any\n            "compatible" value.\n        limit : int, default None\n            Maximum number of consecutive elements to forward or backward fill.\n        tolerance : optional\n            Maximum distance between original and new labels for inexact\n            matches. The values of the index at the matching locations most\n            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.\n\n            Tolerance may be a scalar value, which applies the same tolerance\n            to all values, or list-like, which applies variable tolerance per\n            element. List-like includes list, tuple, array, Series, and must be\n            the same size as the index and its dtype must exactly match the\n            index\'s type.\n\n        Returns\n        -------\n        {klass} with changed index.\n\n        See Also\n        --------\n        DataFrame.set_index : Set row labels.\n        DataFrame.reset_index : Remove row labels or move them to new columns.\n        DataFrame.reindex_like : Change to same indices as other DataFrame.\n\n        Examples\n        --------\n        ``DataFrame.reindex`` supports two calling conventions\n\n        * ``(index=index_labels, columns=column_labels, ...)``\n        * ``(labels, axis={{\'index\', \'columns\'}}, ...)``\n\n        We *highly* recommend using keyword arguments to clarify your\n        intent.\n\n        Create a dataframe with some fictional data.\n\n        >>> index = [\'Firefox\', \'Chrome\', \'Safari\', \'IE10\', \'Konqueror\']\n        >>> df = pd.DataFrame({{\'http_status\': [200, 200, 404, 404, 301],\n        ...                   \'response_time\': [0.04, 0.02, 0.07, 0.08, 1.0]}},\n        ...                   index=index)\n        >>> df\n                   http_status  response_time\n        Firefox            200           0.04\n        Chrome             200           0.02\n        Safari             404           0.07\n        IE10               404           0.08\n        Konqueror          301           1.00\n\n        Create a new index and reindex the dataframe. By default\n        values in the new index that do not have corresponding\n        records in the dataframe are assigned ``NaN``.\n\n        >>> new_index = [\'Safari\', \'Iceweasel\', \'Comodo Dragon\', \'IE10\',\n        ...              \'Chrome\']\n        >>> df.reindex(new_index)\n                       http_status  response_time\n        Safari               404.0           0.07\n        Iceweasel              NaN            NaN\n        Comodo Dragon          NaN            NaN\n        IE10                 404.0           0.08\n        Chrome               200.0           0.02\n\n        We can fill in the missing values by passing a value to\n        the keyword ``fill_value``. Because the index is not monotonically\n        increasing or decreasing, we cannot use arguments to the keyword\n        ``method`` to fill the ``NaN`` values.\n\n        >>> df.reindex(new_index, fill_value=0)\n                       http_status  response_time\n        Safari                 404           0.07\n        Iceweasel                0           0.00\n        Comodo Dragon            0           0.00\n        IE10                   404           0.08\n        Chrome                 200           0.02\n\n        >>> df.reindex(new_index, fill_value=\'missing\')\n                      http_status response_time\n        Safari                404          0.07\n        Iceweasel         missing       missing\n        Comodo Dragon     missing       missing\n        IE10                  404          0.08\n        Chrome                200          0.02\n\n        We can also reindex the columns.\n\n        >>> df.reindex(columns=[\'http_status\', \'user_agent\'])\n                   http_status  user_agent\n        Firefox            200         NaN\n        Chrome             200         NaN\n        Safari             404         NaN\n        IE10               404         NaN\n        Konqueror          301         NaN\n\n        Or we can use "axis-style" keyword arguments\n\n        >>> df.reindex([\'http_status\', \'user_agent\'], axis="columns")\n                   http_status  user_agent\n        Firefox            200         NaN\n        Chrome             200         NaN\n        Safari             404         NaN\n        IE10               404         NaN\n        Konqueror          301         NaN\n\n        To further illustrate the filling functionality in\n        ``reindex``, we will create a dataframe with a\n        monotonically increasing index (for example, a sequence\n        of dates).\n\n        >>> date_index = pd.date_range(\'1/1/2010\', periods=6, freq=\'D\')\n        >>> df2 = pd.DataFrame({{"prices": [100, 101, np.nan, 100, 89, 88]}},\n        ...                    index=date_index)\n        >>> df2\n                    prices\n        2010-01-01   100.0\n        2010-01-02   101.0\n        2010-01-03     NaN\n        2010-01-04   100.0\n        2010-01-05    89.0\n        2010-01-06    88.0\n\n        Suppose we decide to expand the dataframe to cover a wider\n        date range.\n\n        >>> date_index2 = pd.date_range(\'12/29/2009\', periods=10, freq=\'D\')\n        >>> df2.reindex(date_index2)\n                    prices\n        2009-12-29     NaN\n        2009-12-30     NaN\n        2009-12-31     NaN\n        2010-01-01   100.0\n        2010-01-02   101.0\n        2010-01-03     NaN\n        2010-01-04   100.0\n        2010-01-05    89.0\n        2010-01-06    88.0\n        2010-01-07     NaN\n\n        The index entries that did not have a value in the original data frame\n        (for example, \'2009-12-29\') are by default filled with ``NaN``.\n        If desired, we can fill in the missing values using one of several\n        options.\n\n        For example, to back-propagate the last valid value to fill the ``NaN``\n        values, pass ``bfill`` as an argument to the ``method`` keyword.\n\n        >>> df2.reindex(date_index2, method=\'bfill\')\n                    prices\n        2009-12-29   100.0\n        2009-12-30   100.0\n        2009-12-31   100.0\n        2010-01-01   100.0\n        2010-01-02   101.0\n        2010-01-03     NaN\n        2010-01-04   100.0\n        2010-01-05    89.0\n        2010-01-06    88.0\n        2010-01-07     NaN\n\n        Please note that the ``NaN`` value present in the original dataframe\n        (at index value 2010-01-03) will not be filled by any of the\n        value propagation schemes. This is because filling while reindexing\n        does not look at dataframe values, but only compares the original and\n        desired indexes. If you do want to fill in the ``NaN`` values present\n        in the original dataframe, use the ``fillna()`` method.\n\n        See the :ref:`user guide <basics.reindexing>` for more.\n        '
        (axes, kwargs) = self._construct_axes_from_arguments(args, kwargs)
        method = missing.clean_reindex_fill_method(kwargs.pop('method', None))
        level = kwargs.pop('level', None)
        copy = kwargs.pop('copy', True)
        limit = kwargs.pop('limit', None)
        tolerance = kwargs.pop('tolerance', None)
        fill_value = kwargs.pop('fill_value', None)
        kwargs.pop('axis', None)
        if kwargs:
            raise TypeError(f'reindex() got an unexpected keyword argument "{list(kwargs.keys())[0]}"')
        self._consolidate_inplace()
        if all((self._get_axis(axis).identical(ax) for (axis, ax) in axes.items() if (ax is not None))):
            if copy:
                return self.copy()
            return self
        if self._needs_reindex_multi(axes, method, level):
            return self._reindex_multi(axes, copy, fill_value)
        return self._reindex_axes(axes, level, limit, tolerance, method, fill_value, copy).__finalize__(self, method='reindex')

    @final
    def _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy):
        'Perform the reindex for all the axes.'
        obj = self
        for a in self._AXIS_ORDERS:
            labels = axes[a]
            if (labels is None):
                continue
            ax = self._get_axis(a)
            (new_index, indexer) = ax.reindex(labels, level=level, limit=limit, tolerance=tolerance, method=method)
            axis = self._get_axis_number(a)
            obj = obj._reindex_with_indexers({axis: [new_index, indexer]}, fill_value=fill_value, copy=copy, allow_dups=False)
        return obj

    @final
    def _needs_reindex_multi(self, axes, method, level):
        'Check if we do need a multi reindex.'
        return ((com.count_not_none(*axes.values()) == self._AXIS_LEN) and (method is None) and (level is None) and (not self._is_mixed_type))

    def _reindex_multi(self, axes, copy, fill_value):
        raise AbstractMethodError(self)

    @final
    def _reindex_with_indexers(self, reindexers, fill_value=None, copy=False, allow_dups=False):
        'allow_dups indicates an internal call here '
        new_data = self._mgr
        for axis in sorted(reindexers.keys()):
            (index, indexer) = reindexers[axis]
            baxis = self._get_block_manager_axis(axis)
            if (index is None):
                continue
            index = ensure_index(index)
            if (indexer is not None):
                indexer = ensure_int64(indexer)
            new_data = new_data.reindex_indexer(index, indexer, axis=baxis, fill_value=fill_value, allow_dups=allow_dups, copy=copy)
            copy = False
        if (copy and (new_data is self._mgr)):
            new_data = new_data.copy()
        return self._constructor(new_data).__finalize__(self)

    def filter(self, items=None, like=None, regex=None, axis=None):
        '\n        Subset the dataframe rows or columns according to the specified index labels.\n\n        Note that this routine does not filter a dataframe on its\n        contents. The filter is applied to the labels of the index.\n\n        Parameters\n        ----------\n        items : list-like\n            Keep labels from axis which are in items.\n        like : str\n            Keep labels from axis for which "like in label == True".\n        regex : str (regular expression)\n            Keep labels from axis for which re.search(regex, label) == True.\n        axis : {0 or ‘index’, 1 or ‘columns’, None}, default None\n            The axis to filter on, expressed either as an index (int)\n            or axis name (str). By default this is the info axis,\n            \'index\' for Series, \'columns\' for DataFrame.\n\n        Returns\n        -------\n        same type as input object\n\n        See Also\n        --------\n        DataFrame.loc : Access a group of rows and columns\n            by label(s) or a boolean array.\n\n        Notes\n        -----\n        The ``items``, ``like``, and ``regex`` parameters are\n        enforced to be mutually exclusive.\n\n        ``axis`` defaults to the info axis that is used when indexing\n        with ``[]``.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.array(([1, 2, 3], [4, 5, 6])),\n        ...                   index=[\'mouse\', \'rabbit\'],\n        ...                   columns=[\'one\', \'two\', \'three\'])\n        >>> df\n                one  two  three\n        mouse     1    2      3\n        rabbit    4    5      6\n\n        >>> # select columns by name\n        >>> df.filter(items=[\'one\', \'three\'])\n                 one  three\n        mouse     1      3\n        rabbit    4      6\n\n        >>> # select columns by regular expression\n        >>> df.filter(regex=\'e$\', axis=1)\n                 one  three\n        mouse     1      3\n        rabbit    4      6\n\n        >>> # select rows containing \'bbi\'\n        >>> df.filter(like=\'bbi\', axis=0)\n                 one  two  three\n        rabbit    4    5      6\n        '
        nkw = com.count_not_none(items, like, regex)
        if (nkw > 1):
            raise TypeError('Keyword arguments `items`, `like`, or `regex` are mutually exclusive')
        if (axis is None):
            axis = self._info_axis_name
        labels = self._get_axis(axis)
        if (items is not None):
            name = self._get_axis_name(axis)
            return self.reindex(**{name: [r for r in items if (r in labels)]})
        elif like:

            def f(x) -> bool:
                assert (like is not None)
                return (like in ensure_str(x))
            values = labels.map(f)
            return self.loc(axis=axis)[values]
        elif regex:

            def f(x) -> bool:
                return (matcher.search(ensure_str(x)) is not None)
            matcher = re.compile(regex)
            values = labels.map(f)
            return self.loc(axis=axis)[values]
        else:
            raise TypeError('Must pass either `items`, `like`, or `regex`')

    @final
    def head(self, n=5):
        "\n        Return the first `n` rows.\n\n        This function returns the first `n` rows for the object based\n        on position. It is useful for quickly testing if your object\n        has the right type of data in it.\n\n        For negative values of `n`, this function returns all rows except\n        the last `n` rows, equivalent to ``df[:-n]``.\n\n        Parameters\n        ----------\n        n : int, default 5\n            Number of rows to select.\n\n        Returns\n        -------\n        same type as caller\n            The first `n` rows of the caller object.\n\n        See Also\n        --------\n        DataFrame.tail: Returns the last `n` rows.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',\n        ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})\n        >>> df\n              animal\n        0  alligator\n        1        bee\n        2     falcon\n        3       lion\n        4     monkey\n        5     parrot\n        6      shark\n        7      whale\n        8      zebra\n\n        Viewing the first 5 lines\n\n        >>> df.head()\n              animal\n        0  alligator\n        1        bee\n        2     falcon\n        3       lion\n        4     monkey\n\n        Viewing the first `n` lines (three in this case)\n\n        >>> df.head(3)\n              animal\n        0  alligator\n        1        bee\n        2     falcon\n\n        For negative values of `n`\n\n        >>> df.head(-3)\n              animal\n        0  alligator\n        1        bee\n        2     falcon\n        3       lion\n        4     monkey\n        5     parrot\n        "
        return self.iloc[:n]

    @final
    def tail(self, n=5):
        "\n        Return the last `n` rows.\n\n        This function returns last `n` rows from the object based on\n        position. It is useful for quickly verifying data, for example,\n        after sorting or appending rows.\n\n        For negative values of `n`, this function returns all rows except\n        the first `n` rows, equivalent to ``df[n:]``.\n\n        Parameters\n        ----------\n        n : int, default 5\n            Number of rows to select.\n\n        Returns\n        -------\n        type of caller\n            The last `n` rows of the caller object.\n\n        See Also\n        --------\n        DataFrame.head : The first `n` rows of the caller object.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',\n        ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})\n        >>> df\n              animal\n        0  alligator\n        1        bee\n        2     falcon\n        3       lion\n        4     monkey\n        5     parrot\n        6      shark\n        7      whale\n        8      zebra\n\n        Viewing the last 5 lines\n\n        >>> df.tail()\n           animal\n        4  monkey\n        5  parrot\n        6   shark\n        7   whale\n        8   zebra\n\n        Viewing the last `n` lines (three in this case)\n\n        >>> df.tail(3)\n          animal\n        6  shark\n        7  whale\n        8  zebra\n\n        For negative values of `n`\n\n        >>> df.tail(-3)\n           animal\n        3    lion\n        4  monkey\n        5  parrot\n        6   shark\n        7   whale\n        8   zebra\n        "
        if (n == 0):
            return self.iloc[0:0]
        return self.iloc[(- n):]

    @final
    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None):
        "\n        Return a random sample of items from an axis of object.\n\n        You can use `random_state` for reproducibility.\n\n        Parameters\n        ----------\n        n : int, optional\n            Number of items from axis to return. Cannot be used with `frac`.\n            Default = 1 if `frac` = None.\n        frac : float, optional\n            Fraction of axis items to return. Cannot be used with `n`.\n        replace : bool, default False\n            Allow or disallow sampling of the same row more than once.\n        weights : str or ndarray-like, optional\n            Default 'None' results in equal probability weighting.\n            If passed a Series, will align with target object on index. Index\n            values in weights not found in sampled object will be ignored and\n            index values in sampled object not in weights will be assigned\n            weights of zero.\n            If called on a DataFrame, will accept the name of a column\n            when axis = 0.\n            Unless weights are a Series, weights must be same length as axis\n            being sampled.\n            If weights do not sum to 1, they will be normalized to sum to 1.\n            Missing values in the weights column will be treated as zero.\n            Infinite values not allowed.\n        random_state : int, array-like, BitGenerator, np.random.RandomState, optional\n            If int, array-like, or BitGenerator (NumPy>=1.17), seed for\n            random number generator\n            If np.random.RandomState, use as numpy RandomState object.\n\n            .. versionchanged:: 1.1.0\n\n                array-like and BitGenerator (for NumPy>=1.17) object now passed to\n                np.random.RandomState() as seed\n\n        axis : {0 or ‘index’, 1 or ‘columns’, None}, default None\n            Axis to sample. Accepts axis number or name. Default is stat axis\n            for given data type (0 for Series and DataFrames).\n\n        Returns\n        -------\n        Series or DataFrame\n            A new object of same type as caller containing `n` items randomly\n            sampled from the caller object.\n\n        See Also\n        --------\n        DataFrameGroupBy.sample: Generates random samples from each group of a\n            DataFrame object.\n        SeriesGroupBy.sample: Generates random samples from each group of a\n            Series object.\n        numpy.random.choice: Generates a random sample from a given 1-D numpy\n            array.\n\n        Notes\n        -----\n        If `frac` > 1, `replacement` should be set to `True`.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'num_legs': [2, 4, 8, 0],\n        ...                    'num_wings': [2, 0, 0, 0],\n        ...                    'num_specimen_seen': [10, 2, 1, 8]},\n        ...                   index=['falcon', 'dog', 'spider', 'fish'])\n        >>> df\n                num_legs  num_wings  num_specimen_seen\n        falcon         2          2                 10\n        dog            4          0                  2\n        spider         8          0                  1\n        fish           0          0                  8\n\n        Extract 3 random elements from the ``Series`` ``df['num_legs']``:\n        Note that we use `random_state` to ensure the reproducibility of\n        the examples.\n\n        >>> df['num_legs'].sample(n=3, random_state=1)\n        fish      0\n        spider    8\n        falcon    2\n        Name: num_legs, dtype: int64\n\n        A random 50% sample of the ``DataFrame`` with replacement:\n\n        >>> df.sample(frac=0.5, replace=True, random_state=1)\n              num_legs  num_wings  num_specimen_seen\n        dog          4          0                  2\n        fish         0          0                  8\n\n        An upsample sample of the ``DataFrame`` with replacement:\n        Note that `replace` parameter has to be `True` for `frac` parameter > 1.\n\n        >>> df.sample(frac=2, replace=True, random_state=1)\n                num_legs  num_wings  num_specimen_seen\n        dog            4          0                  2\n        fish           0          0                  8\n        falcon         2          2                 10\n        falcon         2          2                 10\n        fish           0          0                  8\n        dog            4          0                  2\n        fish           0          0                  8\n        dog            4          0                  2\n\n        Using a DataFrame column as weights. Rows with larger value in the\n        `num_specimen_seen` column are more likely to be sampled.\n\n        >>> df.sample(n=2, weights='num_specimen_seen', random_state=1)\n                num_legs  num_wings  num_specimen_seen\n        falcon         2          2                 10\n        fish           0          0                  8\n        "
        if (axis is None):
            axis = self._stat_axis_number
        axis = self._get_axis_number(axis)
        axis_length = self.shape[axis]
        rs = com.random_state(random_state)
        if (weights is not None):
            if isinstance(weights, ABCSeries):
                weights = weights.reindex(self.axes[axis])
            if isinstance(weights, str):
                if isinstance(self, ABCDataFrame):
                    if (axis == 0):
                        try:
                            weights = self[weights]
                        except KeyError as err:
                            raise KeyError('String passed to weights not a valid column') from err
                    else:
                        raise ValueError('Strings can only be passed to weights when sampling from rows on a DataFrame')
                else:
                    raise ValueError('Strings cannot be passed as weights when sampling from a Series.')
            weights = pd.Series(weights, dtype='float64')
            if (len(weights) != axis_length):
                raise ValueError('Weights and axis to be sampled must be of same length')
            if ((weights == np.inf).any() or (weights == (- np.inf)).any()):
                raise ValueError('weight vector may not include `inf` values')
            if (weights < 0).any():
                raise ValueError('weight vector many not include negative values')
            weights = weights.fillna(0)
            if (weights.sum() != 1):
                if (weights.sum() != 0):
                    weights = (weights / weights.sum())
                else:
                    raise ValueError('Invalid weights: weights sum to zero')
            weights = weights._values
        if ((n is None) and (frac is None)):
            n = 1
        elif ((frac is not None) and (frac > 1) and (not replace)):
            raise ValueError('Replace has to be set to `True` when upsampling the population `frac` > 1.')
        elif ((n is not None) and (frac is None) and ((n % 1) != 0)):
            raise ValueError('Only integers accepted as `n` values')
        elif ((n is None) and (frac is not None)):
            n = int(round((frac * axis_length)))
        elif ((n is not None) and (frac is not None)):
            raise ValueError('Please enter a value for `frac` OR `n`, not both')
        if (n < 0):
            raise ValueError('A negative number of rows requested. Please provide positive value.')
        locs = rs.choice(axis_length, size=n, replace=replace, p=weights)
        return self.take(locs, axis=axis)

    @final
    @doc(klass=_shared_doc_kwargs['klass'])
    def pipe(self, func, *args, **kwargs):
        "\n        Apply func(self, \\*args, \\*\\*kwargs).\n\n        Parameters\n        ----------\n        func : function\n            Function to apply to the {klass}.\n            ``args``, and ``kwargs`` are passed into ``func``.\n            Alternatively a ``(callable, data_keyword)`` tuple where\n            ``data_keyword`` is a string indicating the keyword of\n            ``callable`` that expects the {klass}.\n        args : iterable, optional\n            Positional arguments passed into ``func``.\n        kwargs : mapping, optional\n            A dictionary of keyword arguments passed into ``func``.\n\n        Returns\n        -------\n        object : the return type of ``func``.\n\n        See Also\n        --------\n        DataFrame.apply : Apply a function along input axis of DataFrame.\n        DataFrame.applymap : Apply a function elementwise on a whole DataFrame.\n        Series.map : Apply a mapping correspondence on a\n            :class:`~pandas.Series`.\n\n        Notes\n        -----\n        Use ``.pipe`` when chaining together functions that expect\n        Series, DataFrames or GroupBy objects. Instead of writing\n\n        >>> func(g(h(df), arg1=a), arg2=b, arg3=c)  # doctest: +SKIP\n\n        You can write\n\n        >>> (df.pipe(h)\n        ...    .pipe(g, arg1=a)\n        ...    .pipe(func, arg2=b, arg3=c)\n        ... )  # doctest: +SKIP\n\n        If you have a function that takes the data as (say) the second\n        argument, pass a tuple indicating which keyword expects the\n        data. For example, suppose ``f`` takes its data as ``arg2``:\n\n        >>> (df.pipe(h)\n        ...    .pipe(g, arg1=a)\n        ...    .pipe((func, 'arg2'), arg1=a, arg3=c)\n        ...  )  # doctest: +SKIP\n        "
        return com.pipe(self, func, *args, **kwargs)

    @final
    def __finalize__(self, other, method=None, **kwargs):
        '\n        Propagate metadata from other to self.\n\n        Parameters\n        ----------\n        other : the object from which to get the attributes that we are going\n            to propagate\n        method : str, optional\n            A passed method name providing context on where ``__finalize__``\n            was called.\n\n            .. warning::\n\n               The value passed as `method` are not currently considered\n               stable across pandas releases.\n        '
        if isinstance(other, NDFrame):
            for name in other.attrs:
                self.attrs[name] = other.attrs[name]
            self.flags.allows_duplicate_labels = other.flags.allows_duplicate_labels
            for name in (set(self._metadata) & set(other._metadata)):
                assert isinstance(name, str)
                object.__setattr__(self, name, getattr(other, name, None))
        if (method == 'concat'):
            allows_duplicate_labels = all((x.flags.allows_duplicate_labels for x in other.objs))
            self.flags.allows_duplicate_labels = allows_duplicate_labels
        return self

    def __getattr__(self, name):
        '\n        After regular attribute access, try looking up the name\n        This allows simpler access to columns for interactive use.\n        '
        if ((name in self._internal_names_set) or (name in self._metadata) or (name in self._accessors)):
            return object.__getattribute__(self, name)
        else:
            if self._info_axis._can_hold_identifiers_and_holds_name(name):
                return self[name]
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        '\n        After regular attribute access, try setting the name\n        This allows simpler access to columns for interactive use.\n        '
        try:
            object.__getattribute__(self, name)
            return object.__setattr__(self, name, value)
        except AttributeError:
            pass
        if (name in self._internal_names_set):
            object.__setattr__(self, name, value)
        elif (name in self._metadata):
            object.__setattr__(self, name, value)
        else:
            try:
                existing = getattr(self, name)
                if isinstance(existing, Index):
                    object.__setattr__(self, name, value)
                elif (name in self._info_axis):
                    self[name] = value
                else:
                    object.__setattr__(self, name, value)
            except (AttributeError, TypeError):
                if (isinstance(self, ABCDataFrame) and is_list_like(value)):
                    warnings.warn("Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access", stacklevel=2)
                object.__setattr__(self, name, value)

    @final
    def _dir_additions(self):
        '\n        add the string-like attributes from the info_axis.\n        If info_axis is a MultiIndex, its first level values are used.\n        '
        additions = super()._dir_additions()
        if self._info_axis._can_hold_strings:
            additions.update(self._info_axis._dir_additions_for_owner)
        return additions

    @final
    def _protect_consolidate(self, f):
        '\n        Consolidate _mgr -- if the blocks have changed, then clear the\n        cache\n        '
        blocks_before = len(self._mgr.blocks)
        result = f()
        if (len(self._mgr.blocks) != blocks_before):
            self._clear_item_cache()
        return result

    @final
    def _consolidate_inplace(self):
        'Consolidate data in place and return None'

        def f():
            self._mgr = self._mgr.consolidate()
        self._protect_consolidate(f)

    @final
    def _consolidate(self):
        '\n        Compute NDFrame with "consolidated" internals (data of each dtype\n        grouped together in a single ndarray).\n\n        Returns\n        -------\n        consolidated : same type as caller\n        '
        f = (lambda : self._mgr.consolidate())
        cons_data = self._protect_consolidate(f)
        return self._constructor(cons_data).__finalize__(self)

    @final
    @property
    def _is_mixed_type(self):
        if self._mgr.is_single_block:
            return False
        if self._mgr.any_extension_types:
            return True
        return (self.dtypes.nunique() > 1)

    @final
    def _check_inplace_setting(self, value):
        ' check whether we allow in-place setting with this type of value '
        if self._is_mixed_type:
            if (not self._mgr.is_numeric_mixed_type):
                if (is_float(value) and np.isnan(value)):
                    return True
                raise TypeError('Cannot do inplace boolean setting on mixed-types with a non np.nan value')
        return True

    @final
    def _get_numeric_data(self):
        return self._constructor(self._mgr.get_numeric_data()).__finalize__(self)

    @final
    def _get_bool_data(self):
        return self._constructor(self._mgr.get_bool_data()).__finalize__(self)

    @property
    def values(self):
        "\n        Return a Numpy representation of the DataFrame.\n\n        .. warning::\n\n           We recommend using :meth:`DataFrame.to_numpy` instead.\n\n        Only the values in the DataFrame will be returned, the axes labels\n        will be removed.\n\n        Returns\n        -------\n        numpy.ndarray\n            The values of the DataFrame.\n\n        See Also\n        --------\n        DataFrame.to_numpy : Recommended alternative to this method.\n        DataFrame.index : Retrieve the index labels.\n        DataFrame.columns : Retrieving the column names.\n\n        Notes\n        -----\n        The dtype will be a lower-common-denominator dtype (implicit\n        upcasting); that is to say if the dtypes (even of numeric types)\n        are mixed, the one that accommodates all will be chosen. Use this\n        with care if you are not dealing with the blocks.\n\n        e.g. If the dtypes are float16 and float32, dtype will be upcast to\n        float32.  If dtypes are int32 and uint8, dtype will be upcast to\n        int32. By :func:`numpy.find_common_type` convention, mixing int64\n        and uint64 will result in a float64 dtype.\n\n        Examples\n        --------\n        A DataFrame where all columns are the same type (e.g., int64) results\n        in an array of the same type.\n\n        >>> df = pd.DataFrame({'age':    [ 3,  29],\n        ...                    'height': [94, 170],\n        ...                    'weight': [31, 115]})\n        >>> df\n           age  height  weight\n        0    3      94      31\n        1   29     170     115\n        >>> df.dtypes\n        age       int64\n        height    int64\n        weight    int64\n        dtype: object\n        >>> df.values\n        array([[  3,  94,  31],\n               [ 29, 170, 115]])\n\n        A DataFrame with mixed type columns(e.g., str/object, int64, float32)\n        results in an ndarray of the broadest type that accommodates these\n        mixed types (e.g., object).\n\n        >>> df2 = pd.DataFrame([('parrot',   24.0, 'second'),\n        ...                     ('lion',     80.5, 1),\n        ...                     ('monkey', np.nan, None)],\n        ...                   columns=('name', 'max_speed', 'rank'))\n        >>> df2.dtypes\n        name          object\n        max_speed    float64\n        rank          object\n        dtype: object\n        >>> df2.values\n        array([['parrot', 24.0, 'second'],\n               ['lion', 80.5, 1],\n               ['monkey', nan, None]], dtype=object)\n        "
        self._consolidate_inplace()
        return self._mgr.as_array(transpose=self._AXIS_REVERSED)

    @property
    def _values(self):
        'internal implementation'
        return self.values

    @property
    def dtypes(self):
        "\n        Return the dtypes in the DataFrame.\n\n        This returns a Series with the data type of each column.\n        The result's index is the original DataFrame's columns. Columns\n        with mixed types are stored with the ``object`` dtype. See\n        :ref:`the User Guide <basics.dtypes>` for more.\n\n        Returns\n        -------\n        pandas.Series\n            The data type of each column.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'float': [1.0],\n        ...                    'int': [1],\n        ...                    'datetime': [pd.Timestamp('20180310')],\n        ...                    'string': ['foo']})\n        >>> df.dtypes\n        float              float64\n        int                  int64\n        datetime    datetime64[ns]\n        string              object\n        dtype: object\n        "
        data = self._mgr.get_dtypes()
        return self._constructor_sliced(data, index=self._info_axis, dtype=np.object_)

    @final
    def _to_dict_of_blocks(self, copy=True):
        '\n        Return a dict of dtype -> Constructor Types that\n        each is a homogeneous dtype.\n\n        Internal ONLY\n        '
        return {k: self._constructor(v).__finalize__(self) for (k, v) in self._mgr.to_dict(copy=copy).items()}

    def astype(self, dtype, copy=True, errors='raise'):
        "\n        Cast a pandas object to a specified dtype ``dtype``.\n\n        Parameters\n        ----------\n        dtype : data type, or dict of column name -> data type\n            Use a numpy.dtype or Python type to cast entire pandas object to\n            the same type. Alternatively, use {col: dtype, ...}, where col is a\n            column label and dtype is a numpy.dtype or Python type to cast one\n            or more of the DataFrame's columns to column-specific types.\n        copy : bool, default True\n            Return a copy when ``copy=True`` (be very careful setting\n            ``copy=False`` as changes to values then may propagate to other\n            pandas objects).\n        errors : {'raise', 'ignore'}, default 'raise'\n            Control raising of exceptions on invalid data for provided dtype.\n\n            - ``raise`` : allow exceptions to be raised\n            - ``ignore`` : suppress exceptions. On error return original object.\n\n        Returns\n        -------\n        casted : same type as caller\n\n        See Also\n        --------\n        to_datetime : Convert argument to datetime.\n        to_timedelta : Convert argument to timedelta.\n        to_numeric : Convert argument to a numeric type.\n        numpy.ndarray.astype : Cast a numpy array to a specified type.\n\n        Examples\n        --------\n        Create a DataFrame:\n\n        >>> d = {'col1': [1, 2], 'col2': [3, 4]}\n        >>> df = pd.DataFrame(data=d)\n        >>> df.dtypes\n        col1    int64\n        col2    int64\n        dtype: object\n\n        Cast all columns to int32:\n\n        >>> df.astype('int32').dtypes\n        col1    int32\n        col2    int32\n        dtype: object\n\n        Cast col1 to int32 using a dictionary:\n\n        >>> df.astype({'col1': 'int32'}).dtypes\n        col1    int32\n        col2    int64\n        dtype: object\n\n        Create a series:\n\n        >>> ser = pd.Series([1, 2], dtype='int32')\n        >>> ser\n        0    1\n        1    2\n        dtype: int32\n        >>> ser.astype('int64')\n        0    1\n        1    2\n        dtype: int64\n\n        Convert to categorical type:\n\n        >>> ser.astype('category')\n        0    1\n        1    2\n        dtype: category\n        Categories (2, int64): [1, 2]\n\n        Convert to ordered categorical type with custom ordering:\n\n        >>> cat_dtype = pd.api.types.CategoricalDtype(\n        ...     categories=[2, 1], ordered=True)\n        >>> ser.astype(cat_dtype)\n        0    1\n        1    2\n        dtype: category\n        Categories (2, int64): [2 < 1]\n\n        Note that using ``copy=False`` and changing data on a new\n        pandas object may propagate changes:\n\n        >>> s1 = pd.Series([1, 2])\n        >>> s2 = s1.astype('int64', copy=False)\n        >>> s2[0] = 10\n        >>> s1  # note that s1[0] has changed too\n        0    10\n        1     2\n        dtype: int64\n\n        Create a series of dates:\n\n        >>> ser_date = pd.Series(pd.date_range('20200101', periods=3))\n        >>> ser_date\n        0   2020-01-01\n        1   2020-01-02\n        2   2020-01-03\n        dtype: datetime64[ns]\n\n        Datetimes are localized to UTC first before\n        converting to the specified timezone:\n\n        >>> ser_date.astype('datetime64[ns, US/Eastern]')\n        0   2019-12-31 19:00:00-05:00\n        1   2020-01-01 19:00:00-05:00\n        2   2020-01-02 19:00:00-05:00\n        dtype: datetime64[ns, US/Eastern]\n        "
        if is_dict_like(dtype):
            if (self.ndim == 1):
                if ((len(dtype) > 1) or (self.name not in dtype)):
                    raise KeyError('Only the Series name can be used for the key in Series dtype mappings.')
                new_type = dtype[self.name]
                return self.astype(new_type, copy, errors)
            for col_name in dtype.keys():
                if (col_name not in self):
                    raise KeyError('Only a column name can be used for the key in a dtype mappings argument.')
            results = []
            for (col_name, col) in self.items():
                if (col_name in dtype):
                    results.append(col.astype(dtype=dtype[col_name], copy=copy, errors=errors))
                else:
                    results.append((col.copy() if copy else col))
        elif (is_extension_array_dtype(dtype) and (self.ndim > 1)):
            results = [self.iloc[:, i].astype(dtype, copy=copy) for i in range(len(self.columns))]
        else:
            new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
            return self._constructor(new_data).__finalize__(self, method='astype')
        if (not results):
            return self.copy()
        result = pd.concat(results, axis=1, copy=False)
        result.columns = self.columns
        return result

    @final
    def copy(self, deep=True):
        '\n        Make a copy of this object\'s indices and data.\n\n        When ``deep=True`` (default), a new object will be created with a\n        copy of the calling object\'s data and indices. Modifications to\n        the data or indices of the copy will not be reflected in the\n        original object (see notes below).\n\n        When ``deep=False``, a new object will be created without copying\n        the calling object\'s data or index (only references to the data\n        and index are copied). Any changes to the data of the original\n        will be reflected in the shallow copy (and vice versa).\n\n        Parameters\n        ----------\n        deep : bool, default True\n            Make a deep copy, including a copy of the data and the indices.\n            With ``deep=False`` neither the indices nor the data are copied.\n\n        Returns\n        -------\n        copy : Series or DataFrame\n            Object type matches caller.\n\n        Notes\n        -----\n        When ``deep=True``, data is copied but actual Python objects\n        will not be copied recursively, only the reference to the object.\n        This is in contrast to `copy.deepcopy` in the Standard Library,\n        which recursively copies object data (see examples below).\n\n        While ``Index`` objects are copied when ``deep=True``, the underlying\n        numpy array is not copied for performance reasons. Since ``Index`` is\n        immutable, the underlying data can be safely shared and a copy\n        is not needed.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2], index=["a", "b"])\n        >>> s\n        a    1\n        b    2\n        dtype: int64\n\n        >>> s_copy = s.copy()\n        >>> s_copy\n        a    1\n        b    2\n        dtype: int64\n\n        **Shallow copy versus default (deep) copy:**\n\n        >>> s = pd.Series([1, 2], index=["a", "b"])\n        >>> deep = s.copy()\n        >>> shallow = s.copy(deep=False)\n\n        Shallow copy shares data and index with original.\n\n        >>> s is shallow\n        False\n        >>> s.values is shallow.values and s.index is shallow.index\n        True\n\n        Deep copy has own copy of data and index.\n\n        >>> s is deep\n        False\n        >>> s.values is deep.values or s.index is deep.index\n        False\n\n        Updates to the data shared by shallow copy and original is reflected\n        in both; deep copy remains unchanged.\n\n        >>> s[0] = 3\n        >>> shallow[1] = 4\n        >>> s\n        a    3\n        b    4\n        dtype: int64\n        >>> shallow\n        a    3\n        b    4\n        dtype: int64\n        >>> deep\n        a    1\n        b    2\n        dtype: int64\n\n        Note that when copying an object containing Python objects, a deep copy\n        will copy the data, but will not do so recursively. Updating a nested\n        data object will be reflected in the deep copy.\n\n        >>> s = pd.Series([[1, 2], [3, 4]])\n        >>> deep = s.copy()\n        >>> s[0][0] = 10\n        >>> s\n        0    [10, 2]\n        1     [3, 4]\n        dtype: object\n        >>> deep\n        0    [10, 2]\n        1     [3, 4]\n        dtype: object\n        '
        data = self._mgr.copy(deep=deep)
        self._clear_item_cache()
        return self._constructor(data).__finalize__(self, method='copy')

    @final
    def __copy__(self, deep=True):
        return self.copy(deep=deep)

    @final
    def __deepcopy__(self, memo=None):
        '\n        Parameters\n        ----------\n        memo, default None\n            Standard signature. Unused\n        '
        return self.copy(deep=True)

    @final
    def _convert(self, datetime=False, numeric=False, timedelta=False):
        '\n        Attempt to infer better dtype for object columns\n\n        Parameters\n        ----------\n        datetime : bool, default False\n            If True, convert to date where possible.\n        numeric : bool, default False\n            If True, attempt to convert to numbers (including strings), with\n            unconvertible values becoming NaN.\n        timedelta : bool, default False\n            If True, convert to timedelta where possible.\n\n        Returns\n        -------\n        converted : same as input object\n        '
        validate_bool_kwarg(datetime, 'datetime')
        validate_bool_kwarg(numeric, 'numeric')
        validate_bool_kwarg(timedelta, 'timedelta')
        return self._constructor(self._mgr.convert(datetime=datetime, numeric=numeric, timedelta=timedelta, copy=True)).__finalize__(self)

    @final
    def infer_objects(self):
        '\n        Attempt to infer better dtypes for object columns.\n\n        Attempts soft conversion of object-dtyped\n        columns, leaving non-object and unconvertible\n        columns unchanged. The inference rules are the\n        same as during normal Series/DataFrame construction.\n\n        Returns\n        -------\n        converted : same type as input object\n\n        See Also\n        --------\n        to_datetime : Convert argument to datetime.\n        to_timedelta : Convert argument to timedelta.\n        to_numeric : Convert argument to numeric type.\n        convert_dtypes : Convert argument to best possible dtype.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": ["a", 1, 2, 3]})\n        >>> df = df.iloc[1:]\n        >>> df\n           A\n        1  1\n        2  2\n        3  3\n\n        >>> df.dtypes\n        A    object\n        dtype: object\n\n        >>> df.infer_objects().dtypes\n        A    int64\n        dtype: object\n        '
        return self._constructor(self._mgr.convert(datetime=True, numeric=False, timedelta=True, copy=True)).__finalize__(self, method='infer_objects')

    @final
    def convert_dtypes(self, infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True):
        '\n        Convert columns to best possible dtypes using dtypes supporting ``pd.NA``.\n\n        .. versionadded:: 1.0.0\n\n        Parameters\n        ----------\n        infer_objects : bool, default True\n            Whether object dtypes should be converted to the best possible types.\n        convert_string : bool, default True\n            Whether object dtypes should be converted to ``StringDtype()``.\n        convert_integer : bool, default True\n            Whether, if possible, conversion can be done to integer extension types.\n        convert_boolean : bool, defaults True\n            Whether object dtypes should be converted to ``BooleanDtypes()``.\n        convert_floating : bool, defaults True\n            Whether, if possible, conversion can be done to floating extension types.\n            If `convert_integer` is also True, preference will be give to integer\n            dtypes if the floats can be faithfully casted to integers.\n\n            .. versionadded:: 1.2.0\n\n        Returns\n        -------\n        Series or DataFrame\n            Copy of input object with new dtype.\n\n        See Also\n        --------\n        infer_objects : Infer dtypes of objects.\n        to_datetime : Convert argument to datetime.\n        to_timedelta : Convert argument to timedelta.\n        to_numeric : Convert argument to a numeric type.\n\n        Notes\n        -----\n        By default, ``convert_dtypes`` will attempt to convert a Series (or each\n        Series in a DataFrame) to dtypes that support ``pd.NA``. By using the options\n        ``convert_string``, ``convert_integer``, ``convert_boolean`` and\n        ``convert_boolean``, it is possible to turn off individual conversions\n        to ``StringDtype``, the integer extension types, ``BooleanDtype``\n        or floating extension types, respectively.\n\n        For object-dtyped columns, if ``infer_objects`` is ``True``, use the inference\n        rules as during normal Series/DataFrame construction.  Then, if possible,\n        convert to ``StringDtype``, ``BooleanDtype`` or an appropriate integer\n        or floating extension type, otherwise leave as ``object``.\n\n        If the dtype is integer, convert to an appropriate integer extension type.\n\n        If the dtype is numeric, and consists of all integers, convert to an\n        appropriate integer extension type. Otherwise, convert to an\n        appropriate floating extension type.\n\n        .. versionchanged:: 1.2\n            Starting with pandas 1.2, this method also converts float columns\n            to the nullable floating extension type.\n\n        In the future, as new dtypes are added that support ``pd.NA``, the results\n        of this method will change to support those new dtypes.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(\n        ...     {\n        ...         "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),\n        ...         "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),\n        ...         "c": pd.Series([True, False, np.nan], dtype=np.dtype("O")),\n        ...         "d": pd.Series(["h", "i", np.nan], dtype=np.dtype("O")),\n        ...         "e": pd.Series([10, np.nan, 20], dtype=np.dtype("float")),\n        ...         "f": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),\n        ...     }\n        ... )\n\n        Start with a DataFrame with default dtypes.\n\n        >>> df\n           a  b      c    d     e      f\n        0  1  x   True    h  10.0    NaN\n        1  2  y  False    i   NaN  100.5\n        2  3  z    NaN  NaN  20.0  200.0\n\n        >>> df.dtypes\n        a      int32\n        b     object\n        c     object\n        d     object\n        e    float64\n        f    float64\n        dtype: object\n\n        Convert the DataFrame to use best possible dtypes.\n\n        >>> dfn = df.convert_dtypes()\n        >>> dfn\n           a  b      c     d     e      f\n        0  1  x   True     h    10   <NA>\n        1  2  y  False     i  <NA>  100.5\n        2  3  z   <NA>  <NA>    20  200.0\n\n        >>> dfn.dtypes\n        a      Int32\n        b     string\n        c    boolean\n        d     string\n        e      Int64\n        f    Float64\n        dtype: object\n\n        Start with a Series of strings and missing data represented by ``np.nan``.\n\n        >>> s = pd.Series(["a", "b", np.nan])\n        >>> s\n        0      a\n        1      b\n        2    NaN\n        dtype: object\n\n        Obtain a Series with dtype ``StringDtype``.\n\n        >>> s.convert_dtypes()\n        0       a\n        1       b\n        2    <NA>\n        dtype: string\n        '
        if (self.ndim == 1):
            return self._convert_dtypes(infer_objects, convert_string, convert_integer, convert_boolean, convert_floating)
        else:
            results = [col._convert_dtypes(infer_objects, convert_string, convert_integer, convert_boolean, convert_floating) for (col_name, col) in self.items()]
            result = pd.concat(results, axis=1, copy=False)
            return result

    @doc(**_shared_doc_kwargs)
    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        "\n        Fill NA/NaN values using the specified method.\n\n        Parameters\n        ----------\n        value : scalar, dict, Series, or DataFrame\n            Value to use to fill holes (e.g. 0), alternately a\n            dict/Series/DataFrame of values specifying which value to use for\n            each index (for a Series) or column (for a DataFrame).  Values not\n            in the dict/Series/DataFrame will not be filled. This value cannot\n            be a list.\n        method : {{'backfill', 'bfill', 'pad', 'ffill', None}}, default None\n            Method to use for filling holes in reindexed Series\n            pad / ffill: propagate last valid observation forward to next valid\n            backfill / bfill: use next valid observation to fill gap.\n        axis : {axes_single_arg}\n            Axis along which to fill missing values.\n        inplace : bool, default False\n            If True, fill in-place. Note: this will modify any\n            other views on this object (e.g., a no-copy slice for a column in a\n            DataFrame).\n        limit : int, default None\n            If method is specified, this is the maximum number of consecutive\n            NaN values to forward/backward fill. In other words, if there is\n            a gap with more than this number of consecutive NaNs, it will only\n            be partially filled. If method is not specified, this is the\n            maximum number of entries along the entire axis where NaNs will be\n            filled. Must be greater than 0 if not None.\n        downcast : dict, default is None\n            A dict of item->dtype of what to downcast if possible,\n            or the string 'infer' which will try to downcast to an appropriate\n            equal type (e.g. float64 to int64 if possible).\n\n        Returns\n        -------\n        {klass} or None\n            Object with missing values filled or None if ``inplace=True``.\n\n        See Also\n        --------\n        interpolate : Fill NaN values using interpolation.\n        reindex : Conform object to new index.\n        asfreq : Convert TimeSeries to specified frequency.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([[np.nan, 2, np.nan, 0],\n        ...                    [3, 4, np.nan, 1],\n        ...                    [np.nan, np.nan, np.nan, 5],\n        ...                    [np.nan, 3, np.nan, 4]],\n        ...                   columns=list('ABCD'))\n        >>> df\n             A    B   C  D\n        0  NaN  2.0 NaN  0\n        1  3.0  4.0 NaN  1\n        2  NaN  NaN NaN  5\n        3  NaN  3.0 NaN  4\n\n        Replace all NaN elements with 0s.\n\n        >>> df.fillna(0)\n            A   B   C   D\n        0   0.0 2.0 0.0 0\n        1   3.0 4.0 0.0 1\n        2   0.0 0.0 0.0 5\n        3   0.0 3.0 0.0 4\n\n        We can also propagate non-null values forward or backward.\n\n        >>> df.fillna(method='ffill')\n            A   B   C   D\n        0   NaN 2.0 NaN 0\n        1   3.0 4.0 NaN 1\n        2   3.0 4.0 NaN 5\n        3   3.0 3.0 NaN 4\n\n        Replace all NaN elements in column 'A', 'B', 'C', and 'D', with 0, 1,\n        2, and 3 respectively.\n\n        >>> values = {{'A': 0, 'B': 1, 'C': 2, 'D': 3}}\n        >>> df.fillna(value=values)\n            A   B   C   D\n        0   0.0 2.0 2.0 0\n        1   3.0 4.0 2.0 1\n        2   0.0 1.0 2.0 5\n        3   0.0 3.0 2.0 4\n\n        Only replace the first NaN element.\n\n        >>> df.fillna(value=values, limit=1)\n            A   B   C   D\n        0   0.0 2.0 2.0 0\n        1   3.0 4.0 NaN 1\n        2   NaN 1.0 NaN 5\n        3   NaN 3.0 NaN 4\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        (value, method) = validate_fillna_kwargs(value, method)
        self._consolidate_inplace()
        if (axis is None):
            axis = 0
        axis = self._get_axis_number(axis)
        if (value is None):
            if ((not self._mgr.is_single_block) and (axis == 1)):
                if inplace:
                    raise NotImplementedError()
                result = self.T.fillna(method=method, limit=limit).T
                result._mgr = result._mgr.downcast()
                return result
            new_data = self._mgr.interpolate(method=method, axis=axis, limit=limit, inplace=inplace, coerce=True, downcast=downcast)
        elif (self.ndim == 1):
            if isinstance(value, (dict, ABCSeries)):
                value = create_series_with_explicit_dtype(value, dtype_if_empty=object)
                value = value.reindex(self.index, copy=False)
                value = value._values
            elif (not is_list_like(value)):
                pass
            else:
                raise TypeError(f'"value" parameter must be a scalar, dict or Series, but you passed a "{type(value).__name__}"')
            new_data = self._mgr.fillna(value=value, limit=limit, inplace=inplace, downcast=downcast)
        elif isinstance(value, (dict, ABCSeries)):
            if (axis == 1):
                raise NotImplementedError('Currently only can fill with dict/Series column by column')
            result = (self if inplace else self.copy())
            for (k, v) in value.items():
                if (k not in result):
                    continue
                obj = result[k]
                obj.fillna(v, limit=limit, inplace=True, downcast=downcast)
            return (result if (not inplace) else None)
        elif (not is_list_like(value)):
            new_data = self._mgr.fillna(value=value, limit=limit, inplace=inplace, downcast=downcast)
        elif (isinstance(value, ABCDataFrame) and (self.ndim == 2)):
            new_data = self.where(self.notna(), value)._data
        else:
            raise ValueError(f'invalid fill value with a {type(value)}')
        result = self._constructor(new_data)
        if inplace:
            return self._update_inplace(result)
        else:
            return result.__finalize__(self, method='fillna')

    @final
    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        "\n        Synonym for :meth:`DataFrame.fillna` with ``method='ffill'``.\n\n        Returns\n        -------\n        {klass} or None\n            Object with missing values filled or None if ``inplace=True``.\n        "
        return self.fillna(method='ffill', axis=axis, inplace=inplace, limit=limit, downcast=downcast)
    pad = ffill

    @final
    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        "\n        Synonym for :meth:`DataFrame.fillna` with ``method='bfill'``.\n\n        Returns\n        -------\n        {klass} or None\n            Object with missing values filled or None if ``inplace=True``.\n        "
        return self.fillna(method='bfill', axis=axis, inplace=inplace, limit=limit, downcast=downcast)
    backfill = bfill

    @doc(_shared_docs['replace'], klass=_shared_doc_kwargs['klass'], inplace=_shared_doc_kwargs['inplace'], replace_iloc=_shared_doc_kwargs['replace_iloc'])
    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        if (not (is_scalar(to_replace) or is_re_compilable(to_replace) or is_list_like(to_replace))):
            raise TypeError(f"Expecting 'to_replace' to be either a scalar, array-like, dict or None, got invalid type {repr(type(to_replace).__name__)}")
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if ((not is_bool(regex)) and (to_replace is not None)):
            raise ValueError("'to_replace' must be 'None' if 'regex' is not a bool")
        self._consolidate_inplace()
        if (value is None):
            if ((not is_dict_like(to_replace)) and (not is_dict_like(regex))):
                to_replace = [to_replace]
            if isinstance(to_replace, (tuple, list)):
                if isinstance(self, ABCDataFrame):
                    from pandas import Series
                    return self.apply(Series._replace_single, args=(to_replace, method, inplace, limit))
                self = cast('Series', self)
                return self._replace_single(to_replace, method, inplace, limit)
            if (not is_dict_like(to_replace)):
                if (not is_dict_like(regex)):
                    raise TypeError('If "to_replace" and "value" are both None and "to_replace" is not a list, then regex must be a mapping')
                to_replace = regex
                regex = True
            items = list(to_replace.items())
            if items:
                (keys, values) = zip(*items)
            else:
                (keys, values) = ([], [])
            are_mappings = [is_dict_like(v) for v in values]
            if any(are_mappings):
                if (not all(are_mappings)):
                    raise TypeError('If a nested mapping is passed, all values of the top level mapping must be mappings')
                to_rep_dict = {}
                value_dict = {}
                for (k, v) in items:
                    (keys, values) = (list(zip(*v.items())) or ([], []))
                    to_rep_dict[k] = list(keys)
                    value_dict[k] = list(values)
                (to_replace, value) = (to_rep_dict, value_dict)
            else:
                (to_replace, value) = (keys, values)
            return self.replace(to_replace, value, inplace=inplace, limit=limit, regex=regex)
        else:
            if (not self.size):
                if inplace:
                    return
                return self.copy()
            if is_dict_like(to_replace):
                if is_dict_like(value):
                    mapping = {col: (to_replace[col], value[col]) for col in to_replace.keys() if ((col in value.keys()) and (col in self))}
                    return self._replace_columnwise(mapping, inplace, regex)
                elif (not is_list_like(value)):
                    if (self.ndim == 1):
                        raise ValueError('Series.replace cannot use dict-like to_replace and non-None value')
                    mapping = {col: (to_rep, value) for (col, to_rep) in to_replace.items()}
                    return self._replace_columnwise(mapping, inplace, regex)
                else:
                    raise TypeError('value argument must be scalar, dict, or Series')
            elif is_list_like(to_replace):
                if (not is_list_like(value)):
                    value = ([value] * len(to_replace))
                if (len(to_replace) != len(value)):
                    raise ValueError(f'Replacement lists must match in length. Expecting {len(to_replace)} got {len(value)} ')
                new_data = self._mgr.replace_list(src_list=to_replace, dest_list=value, inplace=inplace, regex=regex)
            elif (to_replace is None):
                if (not (is_re_compilable(regex) or is_list_like(regex) or is_dict_like(regex))):
                    raise TypeError(f"'regex' must be a string or a compiled regular expression or a list or dict of strings or regular expressions, you passed a {repr(type(regex).__name__)}")
                return self.replace(regex, value, inplace=inplace, limit=limit, regex=True)
            elif is_dict_like(value):
                if (self.ndim == 1):
                    raise ValueError('Series.replace cannot use dict-value and non-None to_replace')
                mapping = {col: (to_replace, val) for (col, val) in value.items()}
                return self._replace_columnwise(mapping, inplace, regex)
            elif (not is_list_like(value)):
                new_data = self._mgr.replace(to_replace=to_replace, value=value, inplace=inplace, regex=regex)
            else:
                raise TypeError(f'Invalid "to_replace" type: {repr(type(to_replace).__name__)}')
        result = self._constructor(new_data)
        if inplace:
            return self._update_inplace(result)
        else:
            return result.__finalize__(self, method='replace')

    @final
    def interpolate(self, method='linear', axis=0, limit=None, inplace=False, limit_direction=None, limit_area=None, downcast=None, **kwargs):
        '\n        Fill NaN values using an interpolation method.\n\n        Please note that only ``method=\'linear\'`` is supported for\n        DataFrame/Series with a MultiIndex.\n\n        Parameters\n        ----------\n        method : str, default \'linear\'\n            Interpolation technique to use. One of:\n\n            * \'linear\': Ignore the index and treat the values as equally\n              spaced. This is the only method supported on MultiIndexes.\n            * \'time\': Works on daily and higher resolution data to interpolate\n              given length of interval.\n            * \'index\', \'values\': use the actual numerical values of the index.\n            * \'pad\': Fill in NaNs using existing values.\n            * \'nearest\', \'zero\', \'slinear\', \'quadratic\', \'cubic\', \'spline\',\n              \'barycentric\', \'polynomial\': Passed to\n              `scipy.interpolate.interp1d`. These methods use the numerical\n              values of the index.  Both \'polynomial\' and \'spline\' require that\n              you also specify an `order` (int), e.g.\n              ``df.interpolate(method=\'polynomial\', order=5)``.\n            * \'krogh\', \'piecewise_polynomial\', \'spline\', \'pchip\', \'akima\',\n              \'cubicspline\': Wrappers around the SciPy interpolation methods of\n              similar names. See `Notes`.\n            * \'from_derivatives\': Refers to\n              `scipy.interpolate.BPoly.from_derivatives` which\n              replaces \'piecewise_polynomial\' interpolation method in\n              scipy 0.18.\n\n        axis : {{0 or \'index\', 1 or \'columns\', None}}, default None\n            Axis to interpolate along.\n        limit : int, optional\n            Maximum number of consecutive NaNs to fill. Must be greater than\n            0.\n        inplace : bool, default False\n            Update the data in place if possible.\n        limit_direction : {{\'forward\', \'backward\', \'both\'}}, Optional\n            Consecutive NaNs will be filled in this direction.\n\n            If limit is specified:\n                * If \'method\' is \'pad\' or \'ffill\', \'limit_direction\' must be \'forward\'.\n                * If \'method\' is \'backfill\' or \'bfill\', \'limit_direction\' must be\n                  \'backwards\'.\n\n            If \'limit\' is not specified:\n                * If \'method\' is \'backfill\' or \'bfill\', the default is \'backward\'\n                * else the default is \'forward\'\n\n            .. versionchanged:: 1.1.0\n                raises ValueError if `limit_direction` is \'forward\' or \'both\' and\n                    method is \'backfill\' or \'bfill\'.\n                raises ValueError if `limit_direction` is \'backward\' or \'both\' and\n                    method is \'pad\' or \'ffill\'.\n\n        limit_area : {{`None`, \'inside\', \'outside\'}}, default None\n            If limit is specified, consecutive NaNs will be filled with this\n            restriction.\n\n            * ``None``: No fill restriction.\n            * \'inside\': Only fill NaNs surrounded by valid values\n              (interpolate).\n            * \'outside\': Only fill NaNs outside valid values (extrapolate).\n\n        downcast : optional, \'infer\' or None, defaults to None\n            Downcast dtypes if possible.\n        ``**kwargs`` : optional\n            Keyword arguments to pass on to the interpolating function.\n\n        Returns\n        -------\n        Series or DataFrame or None\n            Returns the same object type as the caller, interpolated at\n            some or all ``NaN`` values or None if ``inplace=True``.\n\n        See Also\n        --------\n        fillna : Fill missing values using different methods.\n        scipy.interpolate.Akima1DInterpolator : Piecewise cubic polynomials\n            (Akima interpolator).\n        scipy.interpolate.BPoly.from_derivatives : Piecewise polynomial in the\n            Bernstein basis.\n        scipy.interpolate.interp1d : Interpolate a 1-D function.\n        scipy.interpolate.KroghInterpolator : Interpolate polynomial (Krogh\n            interpolator).\n        scipy.interpolate.PchipInterpolator : PCHIP 1-d monotonic cubic\n            interpolation.\n        scipy.interpolate.CubicSpline : Cubic spline data interpolator.\n\n        Notes\n        -----\n        The \'krogh\', \'piecewise_polynomial\', \'spline\', \'pchip\' and \'akima\'\n        methods are wrappers around the respective SciPy implementations of\n        similar names. These use the actual numerical values of the index.\n        For more information on their behavior, see the\n        `SciPy documentation\n        <https://docs.scipy.org/doc/scipy/reference/interpolate.html#univariate-interpolation>`__\n        and `SciPy tutorial\n        <https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html>`__.\n\n        Examples\n        --------\n        Filling in ``NaN`` in a :class:`~pandas.Series` via linear\n        interpolation.\n\n        >>> s = pd.Series([0, 1, np.nan, 3])\n        >>> s\n        0    0.0\n        1    1.0\n        2    NaN\n        3    3.0\n        dtype: float64\n        >>> s.interpolate()\n        0    0.0\n        1    1.0\n        2    2.0\n        3    3.0\n        dtype: float64\n\n        Filling in ``NaN`` in a Series by padding, but filling at most two\n        consecutive ``NaN`` at a time.\n\n        >>> s = pd.Series([np.nan, "single_one", np.nan,\n        ...                "fill_two_more", np.nan, np.nan, np.nan,\n        ...                4.71, np.nan])\n        >>> s\n        0              NaN\n        1       single_one\n        2              NaN\n        3    fill_two_more\n        4              NaN\n        5              NaN\n        6              NaN\n        7             4.71\n        8              NaN\n        dtype: object\n        >>> s.interpolate(method=\'pad\', limit=2)\n        0              NaN\n        1       single_one\n        2       single_one\n        3    fill_two_more\n        4    fill_two_more\n        5    fill_two_more\n        6              NaN\n        7             4.71\n        8             4.71\n        dtype: object\n\n        Filling in ``NaN`` in a Series via polynomial interpolation or splines:\n        Both \'polynomial\' and \'spline\' methods require that you also specify\n        an ``order`` (int).\n\n        >>> s = pd.Series([0, 2, np.nan, 8])\n        >>> s.interpolate(method=\'polynomial\', order=2)\n        0    0.000000\n        1    2.000000\n        2    4.666667\n        3    8.000000\n        dtype: float64\n\n        Fill the DataFrame forward (that is, going down) along each column\n        using linear interpolation.\n\n        Note how the last entry in column \'a\' is interpolated differently,\n        because there is no entry after it to use for interpolation.\n        Note how the first entry in column \'b\' remains ``NaN``, because there\n        is no entry before it to use for interpolation.\n\n        >>> df = pd.DataFrame([(0.0, np.nan, -1.0, 1.0),\n        ...                    (np.nan, 2.0, np.nan, np.nan),\n        ...                    (2.0, 3.0, np.nan, 9.0),\n        ...                    (np.nan, 4.0, -4.0, 16.0)],\n        ...                   columns=list(\'abcd\'))\n        >>> df\n             a    b    c     d\n        0  0.0  NaN -1.0   1.0\n        1  NaN  2.0  NaN   NaN\n        2  2.0  3.0  NaN   9.0\n        3  NaN  4.0 -4.0  16.0\n        >>> df.interpolate(method=\'linear\', limit_direction=\'forward\', axis=0)\n             a    b    c     d\n        0  0.0  NaN -1.0   1.0\n        1  1.0  2.0 -2.0   5.0\n        2  2.0  3.0 -3.0   9.0\n        3  2.0  4.0 -4.0  16.0\n\n        Using polynomial interpolation.\n\n        >>> df[\'d\'].interpolate(method=\'polynomial\', order=2)\n        0     1.0\n        1     4.0\n        2     9.0\n        3    16.0\n        Name: d, dtype: float64\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        axis = self._get_axis_number(axis)
        fillna_methods = ['ffill', 'bfill', 'pad', 'backfill']
        should_transpose = ((axis == 1) and (method not in fillna_methods))
        obj = (self.T if should_transpose else self)
        if obj.empty:
            return self.copy()
        if (method not in fillna_methods):
            axis = self._info_axis_number
        if (isinstance(obj.index, MultiIndex) and (method != 'linear')):
            raise ValueError('Only `method=linear` interpolation is supported on MultiIndexes.')
        if (limit_direction is None):
            limit_direction = ('backward' if (method in ('backfill', 'bfill')) else 'forward')
        else:
            if ((method in ('pad', 'ffill')) and (limit_direction != 'forward')):
                raise ValueError(f"`limit_direction` must be 'forward' for method `{method}`")
            if ((method in ('backfill', 'bfill')) and (limit_direction != 'backward')):
                raise ValueError(f"`limit_direction` must be 'backward' for method `{method}`")
        if ((obj.ndim == 2) and np.all((obj.dtypes == np.dtype(object)))):
            raise TypeError('Cannot interpolate with all object-dtype columns in the DataFrame. Try setting at least one column to a numeric dtype.')
        if (method == 'linear'):
            index = np.arange(len(obj.index))
            index = Index(index)
        else:
            index = obj.index
            methods = {'index', 'values', 'nearest', 'time'}
            is_numeric_or_datetime = (is_numeric_dtype(index.dtype) or is_datetime64_any_dtype(index.dtype) or is_timedelta64_dtype(index.dtype))
            if ((method not in methods) and (not is_numeric_or_datetime)):
                raise ValueError(f'Index column must be numeric or datetime type when using {method} method other than linear. Try setting a numeric or datetime index column before interpolating.')
        if isna(index).any():
            raise NotImplementedError('Interpolation with NaNs in the index has not been implemented. Try filling those NaNs before interpolating.')
        new_data = obj._mgr.interpolate(method=method, axis=axis, index=index, limit=limit, limit_direction=limit_direction, limit_area=limit_area, inplace=inplace, downcast=downcast, **kwargs)
        result = self._constructor(new_data)
        if should_transpose:
            result = result.T
        if inplace:
            return self._update_inplace(result)
        else:
            return result.__finalize__(self, method='interpolate')

    @final
    def asof(self, where, subset=None):
        "\n        Return the last row(s) without any NaNs before `where`.\n\n        The last row (for each element in `where`, if list) without any\n        NaN is taken.\n        In case of a :class:`~pandas.DataFrame`, the last row without NaN\n        considering only the subset of columns (if not `None`)\n\n        If there is no good value, NaN is returned for a Series or\n        a Series of NaN values for a DataFrame\n\n        Parameters\n        ----------\n        where : date or array-like of dates\n            Date(s) before which the last row(s) are returned.\n        subset : str or array-like of str, default `None`\n            For DataFrame, if not `None`, only use these columns to\n            check for NaNs.\n\n        Returns\n        -------\n        scalar, Series, or DataFrame\n\n            The return can be:\n\n            * scalar : when `self` is a Series and `where` is a scalar\n            * Series: when `self` is a Series and `where` is an array-like,\n              or when `self` is a DataFrame and `where` is a scalar\n            * DataFrame : when `self` is a DataFrame and `where` is an\n              array-like\n\n            Return scalar, Series, or DataFrame.\n\n        See Also\n        --------\n        merge_asof : Perform an asof merge. Similar to left join.\n\n        Notes\n        -----\n        Dates are assumed to be sorted. Raises if this is not the case.\n\n        Examples\n        --------\n        A Series and a scalar `where`.\n\n        >>> s = pd.Series([1, 2, np.nan, 4], index=[10, 20, 30, 40])\n        >>> s\n        10    1.0\n        20    2.0\n        30    NaN\n        40    4.0\n        dtype: float64\n\n        >>> s.asof(20)\n        2.0\n\n        For a sequence `where`, a Series is returned. The first value is\n        NaN, because the first element of `where` is before the first\n        index value.\n\n        >>> s.asof([5, 20])\n        5     NaN\n        20    2.0\n        dtype: float64\n\n        Missing values are not considered. The following is ``2.0``, not\n        NaN, even though NaN is at the index location for ``30``.\n\n        >>> s.asof(30)\n        2.0\n\n        Take all columns into consideration\n\n        >>> df = pd.DataFrame({'a': [10, 20, 30, 40, 50],\n        ...                    'b': [None, None, None, None, 500]},\n        ...                   index=pd.DatetimeIndex(['2018-02-27 09:01:00',\n        ...                                           '2018-02-27 09:02:00',\n        ...                                           '2018-02-27 09:03:00',\n        ...                                           '2018-02-27 09:04:00',\n        ...                                           '2018-02-27 09:05:00']))\n        >>> df.asof(pd.DatetimeIndex(['2018-02-27 09:03:30',\n        ...                           '2018-02-27 09:04:30']))\n                              a   b\n        2018-02-27 09:03:30 NaN NaN\n        2018-02-27 09:04:30 NaN NaN\n\n        Take a single column into consideration\n\n        >>> df.asof(pd.DatetimeIndex(['2018-02-27 09:03:30',\n        ...                           '2018-02-27 09:04:30']),\n        ...         subset=['a'])\n                                 a   b\n        2018-02-27 09:03:30   30.0 NaN\n        2018-02-27 09:04:30   40.0 NaN\n        "
        if isinstance(where, str):
            where = Timestamp(where)
        if (not self.index.is_monotonic):
            raise ValueError('asof requires a sorted index')
        is_series = isinstance(self, ABCSeries)
        if is_series:
            if (subset is not None):
                raise ValueError('subset is not valid for Series')
        else:
            if (subset is None):
                subset = self.columns
            if (not is_list_like(subset)):
                subset = [subset]
        is_list = is_list_like(where)
        if (not is_list):
            start = self.index[0]
            if isinstance(self.index, PeriodIndex):
                where = Period(where, freq=self.index.freq)
            if (where < start):
                if (not is_series):
                    return self._constructor_sliced(index=self.columns, name=where, dtype=np.float64)
                return np.nan
            if is_series:
                loc = self.index.searchsorted(where, side='right')
                if (loc > 0):
                    loc -= 1
                values = self._values
                while ((loc > 0) and isna(values[loc])):
                    loc -= 1
                return values[loc]
        if (not isinstance(where, Index)):
            where = (Index(where) if is_list else Index([where]))
        nulls = (self.isna() if is_series else self[subset].isna().any(1))
        if nulls.all():
            if is_series:
                self = cast('Series', self)
                return self._constructor(np.nan, index=where, name=self.name)
            elif is_list:
                self = cast('DataFrame', self)
                return self._constructor(np.nan, index=where, columns=self.columns)
            else:
                self = cast('DataFrame', self)
                return self._constructor_sliced(np.nan, index=self.columns, name=where[0])
        locs = self.index.asof_locs(where, (~ nulls._values))
        missing = (locs == (- 1))
        data = self.take(locs)
        data.index = where
        data.loc[missing] = np.nan
        return (data if is_list else data.iloc[(- 1)])

    @doc(klass=_shared_doc_kwargs['klass'])
    def isna(self):
        "\n        Detect missing values.\n\n        Return a boolean same-sized object indicating if the values are NA.\n        NA values, such as None or :attr:`numpy.NaN`, gets mapped to True\n        values.\n        Everything else gets mapped to False values. Characters such as empty\n        strings ``''`` or :attr:`numpy.inf` are not considered NA values\n        (unless you set ``pandas.options.mode.use_inf_as_na = True``).\n\n        Returns\n        -------\n        {klass}\n            Mask of bool values for each element in {klass} that\n            indicates whether an element is an NA value.\n\n        See Also\n        --------\n        {klass}.isnull : Alias of isna.\n        {klass}.notna : Boolean inverse of isna.\n        {klass}.dropna : Omit axes labels with missing values.\n        isna : Top-level isna.\n\n        Examples\n        --------\n        Show which entries in a DataFrame are NA.\n\n        >>> df = pd.DataFrame(dict(age=[5, 6, np.NaN],\n        ...                    born=[pd.NaT, pd.Timestamp('1939-05-27'),\n        ...                          pd.Timestamp('1940-04-25')],\n        ...                    name=['Alfred', 'Batman', ''],\n        ...                    toy=[None, 'Batmobile', 'Joker']))\n        >>> df\n           age       born    name        toy\n        0  5.0        NaT  Alfred       None\n        1  6.0 1939-05-27  Batman  Batmobile\n        2  NaN 1940-04-25              Joker\n\n        >>> df.isna()\n             age   born   name    toy\n        0  False   True  False   True\n        1  False  False  False  False\n        2   True  False  False  False\n\n        Show which entries in a Series are NA.\n\n        >>> ser = pd.Series([5, 6, np.NaN])\n        >>> ser\n        0    5.0\n        1    6.0\n        2    NaN\n        dtype: float64\n\n        >>> ser.isna()\n        0    False\n        1    False\n        2     True\n        dtype: bool\n        "
        return isna(self).__finalize__(self, method='isna')

    @doc(isna, klass=_shared_doc_kwargs['klass'])
    def isnull(self):
        return isna(self).__finalize__(self, method='isnull')

    @doc(klass=_shared_doc_kwargs['klass'])
    def notna(self):
        "\n        Detect existing (non-missing) values.\n\n        Return a boolean same-sized object indicating if the values are not NA.\n        Non-missing values get mapped to True. Characters such as empty\n        strings ``''`` or :attr:`numpy.inf` are not considered NA values\n        (unless you set ``pandas.options.mode.use_inf_as_na = True``).\n        NA values, such as None or :attr:`numpy.NaN`, get mapped to False\n        values.\n\n        Returns\n        -------\n        {klass}\n            Mask of bool values for each element in {klass} that\n            indicates whether an element is not an NA value.\n\n        See Also\n        --------\n        {klass}.notnull : Alias of notna.\n        {klass}.isna : Boolean inverse of notna.\n        {klass}.dropna : Omit axes labels with missing values.\n        notna : Top-level notna.\n\n        Examples\n        --------\n        Show which entries in a DataFrame are not NA.\n\n        >>> df = pd.DataFrame(dict(age=[5, 6, np.NaN],\n        ...                    born=[pd.NaT, pd.Timestamp('1939-05-27'),\n        ...                          pd.Timestamp('1940-04-25')],\n        ...                    name=['Alfred', 'Batman', ''],\n        ...                    toy=[None, 'Batmobile', 'Joker']))\n        >>> df\n           age       born    name        toy\n        0  5.0        NaT  Alfred       None\n        1  6.0 1939-05-27  Batman  Batmobile\n        2  NaN 1940-04-25              Joker\n\n        >>> df.notna()\n             age   born  name    toy\n        0   True  False  True  False\n        1   True   True  True   True\n        2  False   True  True   True\n\n        Show which entries in a Series are not NA.\n\n        >>> ser = pd.Series([5, 6, np.NaN])\n        >>> ser\n        0    5.0\n        1    6.0\n        2    NaN\n        dtype: float64\n\n        >>> ser.notna()\n        0     True\n        1     True\n        2    False\n        dtype: bool\n        "
        return notna(self).__finalize__(self, method='notna')

    @doc(notna, klass=_shared_doc_kwargs['klass'])
    def notnull(self):
        return notna(self).__finalize__(self, method='notnull')

    @final
    def _clip_with_scalar(self, lower, upper, inplace=False):
        if (((lower is not None) and np.any(isna(lower))) or ((upper is not None) and np.any(isna(upper)))):
            raise ValueError('Cannot use an NA value as a clip threshold')
        result = self
        mask = isna(self._values)
        with np.errstate(all='ignore'):
            if (upper is not None):
                subset = (self.to_numpy() <= upper)
                result = result.where(subset, upper, axis=None, inplace=False)
            if (lower is not None):
                subset = (self.to_numpy() >= lower)
                result = result.where(subset, lower, axis=None, inplace=False)
        if np.any(mask):
            result[mask] = np.nan
        if inplace:
            return self._update_inplace(result)
        else:
            return result

    @final
    def _clip_with_one_bound(self, threshold, method, axis, inplace):
        if (axis is not None):
            axis = self._get_axis_number(axis)
        if (is_scalar(threshold) and is_number(threshold)):
            if (method.__name__ == 'le'):
                return self._clip_with_scalar(None, threshold, inplace=inplace)
            return self._clip_with_scalar(threshold, None, inplace=inplace)
        subset = (method(threshold, axis=axis) | isna(self))
        if ((not isinstance(threshold, ABCSeries)) and is_list_like(threshold)):
            if isinstance(self, ABCSeries):
                threshold = self._constructor(threshold, index=self.index)
            else:
                threshold = align_method_FRAME(self, threshold, axis, flex=None)[1]
        return self.where(subset, threshold, axis=axis, inplace=inplace)

    @final
    def clip(self, lower=None, upper=None, axis=None, inplace=False, *args, **kwargs):
        "\n        Trim values at input threshold(s).\n\n        Assigns values outside boundary to boundary values. Thresholds\n        can be singular values or array like, and in the latter case\n        the clipping is performed element-wise in the specified axis.\n\n        Parameters\n        ----------\n        lower : float or array_like, default None\n            Minimum threshold value. All values below this\n            threshold will be set to it.\n        upper : float or array_like, default None\n            Maximum threshold value. All values above this\n            threshold will be set to it.\n        axis : int or str axis name, optional\n            Align object with lower and upper along the given axis.\n        inplace : bool, default False\n            Whether to perform the operation in place on the data.\n        *args, **kwargs\n            Additional keywords have no effect but might be accepted\n            for compatibility with numpy.\n\n        Returns\n        -------\n        Series or DataFrame or None\n            Same type as calling object with the values outside the\n            clip boundaries replaced or None if ``inplace=True``.\n\n        See Also\n        --------\n        Series.clip : Trim values at input threshold in series.\n        DataFrame.clip : Trim values at input threshold in dataframe.\n        numpy.clip : Clip (limit) the values in an array.\n\n        Examples\n        --------\n        >>> data = {'col_0': [9, -3, 0, -1, 5], 'col_1': [-2, -7, 6, 8, -5]}\n        >>> df = pd.DataFrame(data)\n        >>> df\n           col_0  col_1\n        0      9     -2\n        1     -3     -7\n        2      0      6\n        3     -1      8\n        4      5     -5\n\n        Clips per column using lower and upper thresholds:\n\n        >>> df.clip(-4, 6)\n           col_0  col_1\n        0      6     -2\n        1     -3     -4\n        2      0      6\n        3     -1      6\n        4      5     -4\n\n        Clips using specific lower and upper thresholds per column element:\n\n        >>> t = pd.Series([2, -4, -1, 6, 3])\n        >>> t\n        0    2\n        1   -4\n        2   -1\n        3    6\n        4    3\n        dtype: int64\n\n        >>> df.clip(t, t + 4, axis=0)\n           col_0  col_1\n        0      6      2\n        1     -3     -4\n        2      0      3\n        3      6      8\n        4      5      3\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        axis = nv.validate_clip_with_axis(axis, args, kwargs)
        if (axis is not None):
            axis = self._get_axis_number(axis)
        if ((not is_list_like(lower)) and np.any(isna(lower))):
            lower = None
        if ((not is_list_like(upper)) and np.any(isna(upper))):
            upper = None
        if ((lower is not None) and (upper is not None)):
            if (is_scalar(lower) and is_scalar(upper)):
                (lower, upper) = (min(lower, upper), max(lower, upper))
        if (((lower is None) or (is_scalar(lower) and is_number(lower))) and ((upper is None) or (is_scalar(upper) and is_number(upper)))):
            return self._clip_with_scalar(lower, upper, inplace=inplace)
        result = self
        if (lower is not None):
            result = result._clip_with_one_bound(lower, method=self.ge, axis=axis, inplace=inplace)
        if (upper is not None):
            if inplace:
                result = self
            result = result._clip_with_one_bound(upper, method=self.le, axis=axis, inplace=inplace)
        return result

    @doc(**_shared_doc_kwargs)
    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        "\n        Convert time series to specified frequency.\n\n        Returns the original data conformed to a new index with the specified\n        frequency.\n\n        If the index of this {klass} is a :class:`~pandas.PeriodIndex`, the new index\n        is the result of transforming the original index with\n        :meth:`PeriodIndex.asfreq <pandas.PeriodIndex.asfreq>` (so the original index\n        will map one-to-one to the new index).\n\n        Otherwise, the new index will be equivalent to ``pd.date_range(start, end,\n        freq=freq)`` where ``start`` and ``end`` are, respectively, the first and\n        last entries in the original index (see :func:`pandas.date_range`). The\n        values corresponding to any timesteps in the new index which were not present\n        in the original index will be null (``NaN``), unless a method for filling\n        such unknowns is provided (see the ``method`` parameter below).\n\n        The :meth:`resample` method is more appropriate if an operation on each group of\n        timesteps (such as an aggregate) is necessary to represent the data at the new\n        frequency.\n\n        Parameters\n        ----------\n        freq : DateOffset or str\n            Frequency DateOffset or string.\n        method : {{'backfill'/'bfill', 'pad'/'ffill'}}, default None\n            Method to use for filling holes in reindexed Series (note this\n            does not fill NaNs that already were present):\n\n            * 'pad' / 'ffill': propagate last valid observation forward to next\n              valid\n            * 'backfill' / 'bfill': use NEXT valid observation to fill.\n        how : {{'start', 'end'}}, default end\n            For PeriodIndex only (see PeriodIndex.asfreq).\n        normalize : bool, default False\n            Whether to reset output index to midnight.\n        fill_value : scalar, optional\n            Value to use for missing values, applied during upsampling (note\n            this does not fill NaNs that already were present).\n\n        Returns\n        -------\n        {klass}\n            {klass} object reindexed to the specified frequency.\n\n        See Also\n        --------\n        reindex : Conform DataFrame to new index with optional filling logic.\n\n        Notes\n        -----\n        To learn more about the frequency strings, please see `this link\n        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n\n        Examples\n        --------\n        Start by creating a series with 4 one minute timestamps.\n\n        >>> index = pd.date_range('1/1/2000', periods=4, freq='T')\n        >>> series = pd.Series([0.0, None, 2.0, 3.0], index=index)\n        >>> df = pd.DataFrame({{'s': series}})\n        >>> df\n                               s\n        2000-01-01 00:00:00    0.0\n        2000-01-01 00:01:00    NaN\n        2000-01-01 00:02:00    2.0\n        2000-01-01 00:03:00    3.0\n\n        Upsample the series into 30 second bins.\n\n        >>> df.asfreq(freq='30S')\n                               s\n        2000-01-01 00:00:00    0.0\n        2000-01-01 00:00:30    NaN\n        2000-01-01 00:01:00    NaN\n        2000-01-01 00:01:30    NaN\n        2000-01-01 00:02:00    2.0\n        2000-01-01 00:02:30    NaN\n        2000-01-01 00:03:00    3.0\n\n        Upsample again, providing a ``fill value``.\n\n        >>> df.asfreq(freq='30S', fill_value=9.0)\n                               s\n        2000-01-01 00:00:00    0.0\n        2000-01-01 00:00:30    9.0\n        2000-01-01 00:01:00    NaN\n        2000-01-01 00:01:30    9.0\n        2000-01-01 00:02:00    2.0\n        2000-01-01 00:02:30    9.0\n        2000-01-01 00:03:00    3.0\n\n        Upsample again, providing a ``method``.\n\n        >>> df.asfreq(freq='30S', method='bfill')\n                               s\n        2000-01-01 00:00:00    0.0\n        2000-01-01 00:00:30    NaN\n        2000-01-01 00:01:00    NaN\n        2000-01-01 00:01:30    2.0\n        2000-01-01 00:02:00    2.0\n        2000-01-01 00:02:30    3.0\n        2000-01-01 00:03:00    3.0\n        "
        from pandas.core.resample import asfreq
        return asfreq(self, freq, method=method, how=how, normalize=normalize, fill_value=fill_value)

    @final
    def at_time(self, time, asof=False, axis=None):
        "\n        Select values at particular time of day (e.g., 9:30AM).\n\n        Parameters\n        ----------\n        time : datetime.time or str\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        Series or DataFrame\n\n        Raises\n        ------\n        TypeError\n            If the index is not  a :class:`DatetimeIndex`\n\n        See Also\n        --------\n        between_time : Select values between particular times of the day.\n        first : Select initial periods of time series based on a date offset.\n        last : Select final periods of time series based on a date offset.\n        DatetimeIndex.indexer_at_time : Get just the index locations for\n            values at particular time of the day.\n\n        Examples\n        --------\n        >>> i = pd.date_range('2018-04-09', periods=4, freq='12H')\n        >>> ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)\n        >>> ts\n                             A\n        2018-04-09 00:00:00  1\n        2018-04-09 12:00:00  2\n        2018-04-10 00:00:00  3\n        2018-04-10 12:00:00  4\n\n        >>> ts.at_time('12:00')\n                             A\n        2018-04-09 12:00:00  2\n        2018-04-10 12:00:00  4\n        "
        if (axis is None):
            axis = self._stat_axis_number
        axis = self._get_axis_number(axis)
        index = self._get_axis(axis)
        if (not isinstance(index, DatetimeIndex)):
            raise TypeError('Index must be DatetimeIndex')
        indexer = index.indexer_at_time(time, asof=asof)
        return self._take_with_is_copy(indexer, axis=axis)

    @final
    def between_time(self, start_time, end_time, include_start=True, include_end=True, axis=None):
        "\n        Select values between particular times of the day (e.g., 9:00-9:30 AM).\n\n        By setting ``start_time`` to be later than ``end_time``,\n        you can get the times that are *not* between the two times.\n\n        Parameters\n        ----------\n        start_time : datetime.time or str\n            Initial time as a time filter limit.\n        end_time : datetime.time or str\n            End time as a time filter limit.\n        include_start : bool, default True\n            Whether the start time needs to be included in the result.\n        include_end : bool, default True\n            Whether the end time needs to be included in the result.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            Determine range time on index or columns value.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        Series or DataFrame\n            Data from the original object filtered to the specified dates range.\n\n        Raises\n        ------\n        TypeError\n            If the index is not  a :class:`DatetimeIndex`\n\n        See Also\n        --------\n        at_time : Select values at a particular time of the day.\n        first : Select initial periods of time series based on a date offset.\n        last : Select final periods of time series based on a date offset.\n        DatetimeIndex.indexer_between_time : Get just the index locations for\n            values between particular times of the day.\n\n        Examples\n        --------\n        >>> i = pd.date_range('2018-04-09', periods=4, freq='1D20min')\n        >>> ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)\n        >>> ts\n                             A\n        2018-04-09 00:00:00  1\n        2018-04-10 00:20:00  2\n        2018-04-11 00:40:00  3\n        2018-04-12 01:00:00  4\n\n        >>> ts.between_time('0:15', '0:45')\n                             A\n        2018-04-10 00:20:00  2\n        2018-04-11 00:40:00  3\n\n        You get the times that are *not* between two times by setting\n        ``start_time`` later than ``end_time``:\n\n        >>> ts.between_time('0:45', '0:15')\n                             A\n        2018-04-09 00:00:00  1\n        2018-04-12 01:00:00  4\n        "
        if (axis is None):
            axis = self._stat_axis_number
        axis = self._get_axis_number(axis)
        index = self._get_axis(axis)
        if (not isinstance(index, DatetimeIndex)):
            raise TypeError('Index must be DatetimeIndex')
        indexer = index.indexer_between_time(start_time, end_time, include_start=include_start, include_end=include_end)
        return self._take_with_is_copy(indexer, axis=axis)

    @doc(**_shared_doc_kwargs)
    def resample(self, rule, axis=0, closed=None, label=None, convention='start', kind=None, loffset=None, base=None, on=None, level=None, origin='start_day', offset=None):
        '\n        Resample time-series data.\n\n        Convenience method for frequency conversion and resampling of time series.\n        The object must have a datetime-like index (`DatetimeIndex`, `PeriodIndex`,\n        or `TimedeltaIndex`), or the caller must pass the label of a datetime-like\n        series/index to the ``on``/``level`` keyword parameter.\n\n        Parameters\n        ----------\n        rule : DateOffset, Timedelta or str\n            The offset string or object representing target conversion.\n        axis : {{0 or \'index\', 1 or \'columns\'}}, default 0\n            Which axis to use for up- or down-sampling. For `Series` this\n            will default to 0, i.e. along the rows. Must be\n            `DatetimeIndex`, `TimedeltaIndex` or `PeriodIndex`.\n        closed : {{\'right\', \'left\'}}, default None\n            Which side of bin interval is closed. The default is \'left\'\n            for all frequency offsets except for \'M\', \'A\', \'Q\', \'BM\',\n            \'BA\', \'BQ\', and \'W\' which all have a default of \'right\'.\n        label : {{\'right\', \'left\'}}, default None\n            Which bin edge label to label bucket with. The default is \'left\'\n            for all frequency offsets except for \'M\', \'A\', \'Q\', \'BM\',\n            \'BA\', \'BQ\', and \'W\' which all have a default of \'right\'.\n        convention : {{\'start\', \'end\', \'s\', \'e\'}}, default \'start\'\n            For `PeriodIndex` only, controls whether to use the start or\n            end of `rule`.\n        kind : {{\'timestamp\', \'period\'}}, optional, default None\n            Pass \'timestamp\' to convert the resulting index to a\n            `DateTimeIndex` or \'period\' to convert it to a `PeriodIndex`.\n            By default the input representation is retained.\n        loffset : timedelta, default None\n            Adjust the resampled time labels.\n\n            .. deprecated:: 1.1.0\n                You should add the loffset to the `df.index` after the resample.\n                See below.\n\n        base : int, default 0\n            For frequencies that evenly subdivide 1 day, the "origin" of the\n            aggregated intervals. For example, for \'5min\' frequency, base could\n            range from 0 through 4. Defaults to 0.\n\n            .. deprecated:: 1.1.0\n                The new arguments that you should use are \'offset\' or \'origin\'.\n\n        on : str, optional\n            For a DataFrame, column to use instead of index for resampling.\n            Column must be datetime-like.\n        level : str or int, optional\n            For a MultiIndex, level (name or number) to use for\n            resampling. `level` must be datetime-like.\n        origin : {{\'epoch\', \'start\', \'start_day\', \'end\', \'end_day\'}}, Timestamp\n            or str, default \'start_day\'\n            The timestamp on which to adjust the grouping. The timezone of origin\n            must match the timezone of the index.\n            If a timestamp is not used, these values are also supported:\n\n            - \'epoch\': `origin` is 1970-01-01\n            - \'start\': `origin` is the first value of the timeseries\n            - \'start_day\': `origin` is the first day at midnight of the timeseries\n\n            .. versionadded:: 1.1.0\n\n            - \'end\': `origin` is the last value of the timeseries\n            - \'end_day\': `origin` is the ceiling midnight of the last day\n\n            .. versionadded:: 1.3.0\n\n        offset : Timedelta or str, default is None\n            An offset timedelta added to the origin.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        pandas.core.Resampler\n            :class:`~pandas.core.Resampler` object.\n\n        See Also\n        --------\n        Series.resample : Resample a Series.\n        DataFrame.resample : Resample a DataFrame.\n        groupby : Group {klass} by mapping, function, label, or list of labels.\n        asfreq : Reindex a {klass} with the given frequency without grouping.\n\n        Notes\n        -----\n        See the `user guide\n        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling>`_\n        for more.\n\n        To learn more about the offset strings, please see `this link\n        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`__.\n\n        Examples\n        --------\n        Start by creating a series with 9 one minute timestamps.\n\n        >>> index = pd.date_range(\'1/1/2000\', periods=9, freq=\'T\')\n        >>> series = pd.Series(range(9), index=index)\n        >>> series\n        2000-01-01 00:00:00    0\n        2000-01-01 00:01:00    1\n        2000-01-01 00:02:00    2\n        2000-01-01 00:03:00    3\n        2000-01-01 00:04:00    4\n        2000-01-01 00:05:00    5\n        2000-01-01 00:06:00    6\n        2000-01-01 00:07:00    7\n        2000-01-01 00:08:00    8\n        Freq: T, dtype: int64\n\n        Downsample the series into 3 minute bins and sum the values\n        of the timestamps falling into a bin.\n\n        >>> series.resample(\'3T\').sum()\n        2000-01-01 00:00:00     3\n        2000-01-01 00:03:00    12\n        2000-01-01 00:06:00    21\n        Freq: 3T, dtype: int64\n\n        Downsample the series into 3 minute bins as above, but label each\n        bin using the right edge instead of the left. Please note that the\n        value in the bucket used as the label is not included in the bucket,\n        which it labels. For example, in the original series the\n        bucket ``2000-01-01 00:03:00`` contains the value 3, but the summed\n        value in the resampled bucket with the label ``2000-01-01 00:03:00``\n        does not include 3 (if it did, the summed value would be 6, not 3).\n        To include this value close the right side of the bin interval as\n        illustrated in the example below this one.\n\n        >>> series.resample(\'3T\', label=\'right\').sum()\n        2000-01-01 00:03:00     3\n        2000-01-01 00:06:00    12\n        2000-01-01 00:09:00    21\n        Freq: 3T, dtype: int64\n\n        Downsample the series into 3 minute bins as above, but close the right\n        side of the bin interval.\n\n        >>> series.resample(\'3T\', label=\'right\', closed=\'right\').sum()\n        2000-01-01 00:00:00     0\n        2000-01-01 00:03:00     6\n        2000-01-01 00:06:00    15\n        2000-01-01 00:09:00    15\n        Freq: 3T, dtype: int64\n\n        Upsample the series into 30 second bins.\n\n        >>> series.resample(\'30S\').asfreq()[0:5]   # Select first 5 rows\n        2000-01-01 00:00:00   0.0\n        2000-01-01 00:00:30   NaN\n        2000-01-01 00:01:00   1.0\n        2000-01-01 00:01:30   NaN\n        2000-01-01 00:02:00   2.0\n        Freq: 30S, dtype: float64\n\n        Upsample the series into 30 second bins and fill the ``NaN``\n        values using the ``pad`` method.\n\n        >>> series.resample(\'30S\').pad()[0:5]\n        2000-01-01 00:00:00    0\n        2000-01-01 00:00:30    0\n        2000-01-01 00:01:00    1\n        2000-01-01 00:01:30    1\n        2000-01-01 00:02:00    2\n        Freq: 30S, dtype: int64\n\n        Upsample the series into 30 second bins and fill the\n        ``NaN`` values using the ``bfill`` method.\n\n        >>> series.resample(\'30S\').bfill()[0:5]\n        2000-01-01 00:00:00    0\n        2000-01-01 00:00:30    1\n        2000-01-01 00:01:00    1\n        2000-01-01 00:01:30    2\n        2000-01-01 00:02:00    2\n        Freq: 30S, dtype: int64\n\n        Pass a custom function via ``apply``\n\n        >>> def custom_resampler(array_like):\n        ...     return np.sum(array_like) + 5\n        ...\n        >>> series.resample(\'3T\').apply(custom_resampler)\n        2000-01-01 00:00:00     8\n        2000-01-01 00:03:00    17\n        2000-01-01 00:06:00    26\n        Freq: 3T, dtype: int64\n\n        For a Series with a PeriodIndex, the keyword `convention` can be\n        used to control whether to use the start or end of `rule`.\n\n        Resample a year by quarter using \'start\' `convention`. Values are\n        assigned to the first quarter of the period.\n\n        >>> s = pd.Series([1, 2], index=pd.period_range(\'2012-01-01\',\n        ...                                             freq=\'A\',\n        ...                                             periods=2))\n        >>> s\n        2012    1\n        2013    2\n        Freq: A-DEC, dtype: int64\n        >>> s.resample(\'Q\', convention=\'start\').asfreq()\n        2012Q1    1.0\n        2012Q2    NaN\n        2012Q3    NaN\n        2012Q4    NaN\n        2013Q1    2.0\n        2013Q2    NaN\n        2013Q3    NaN\n        2013Q4    NaN\n        Freq: Q-DEC, dtype: float64\n\n        Resample quarters by month using \'end\' `convention`. Values are\n        assigned to the last month of the period.\n\n        >>> q = pd.Series([1, 2, 3, 4], index=pd.period_range(\'2018-01-01\',\n        ...                                                   freq=\'Q\',\n        ...                                                   periods=4))\n        >>> q\n        2018Q1    1\n        2018Q2    2\n        2018Q3    3\n        2018Q4    4\n        Freq: Q-DEC, dtype: int64\n        >>> q.resample(\'M\', convention=\'end\').asfreq()\n        2018-03    1.0\n        2018-04    NaN\n        2018-05    NaN\n        2018-06    2.0\n        2018-07    NaN\n        2018-08    NaN\n        2018-09    3.0\n        2018-10    NaN\n        2018-11    NaN\n        2018-12    4.0\n        Freq: M, dtype: float64\n\n        For DataFrame objects, the keyword `on` can be used to specify the\n        column instead of the index for resampling.\n\n        >>> d = {{\'price\': [10, 11, 9, 13, 14, 18, 17, 19],\n        ...      \'volume\': [50, 60, 40, 100, 50, 100, 40, 50]}}\n        >>> df = pd.DataFrame(d)\n        >>> df[\'week_starting\'] = pd.date_range(\'01/01/2018\',\n        ...                                     periods=8,\n        ...                                     freq=\'W\')\n        >>> df\n           price  volume week_starting\n        0     10      50    2018-01-07\n        1     11      60    2018-01-14\n        2      9      40    2018-01-21\n        3     13     100    2018-01-28\n        4     14      50    2018-02-04\n        5     18     100    2018-02-11\n        6     17      40    2018-02-18\n        7     19      50    2018-02-25\n        >>> df.resample(\'M\', on=\'week_starting\').mean()\n                       price  volume\n        week_starting\n        2018-01-31     10.75    62.5\n        2018-02-28     17.00    60.0\n\n        For a DataFrame with MultiIndex, the keyword `level` can be used to\n        specify on which level the resampling needs to take place.\n\n        >>> days = pd.date_range(\'1/1/2000\', periods=4, freq=\'D\')\n        >>> d2 = {{\'price\': [10, 11, 9, 13, 14, 18, 17, 19],\n        ...       \'volume\': [50, 60, 40, 100, 50, 100, 40, 50]}}\n        >>> df2 = pd.DataFrame(\n        ...     d2,\n        ...     index=pd.MultiIndex.from_product(\n        ...         [days, [\'morning\', \'afternoon\']]\n        ...     )\n        ... )\n        >>> df2\n                              price  volume\n        2000-01-01 morning       10      50\n                   afternoon     11      60\n        2000-01-02 morning        9      40\n                   afternoon     13     100\n        2000-01-03 morning       14      50\n                   afternoon     18     100\n        2000-01-04 morning       17      40\n                   afternoon     19      50\n        >>> df2.resample(\'D\', level=0).sum()\n                    price  volume\n        2000-01-01     21     110\n        2000-01-02     22     140\n        2000-01-03     32     150\n        2000-01-04     36      90\n\n        If you want to adjust the start of the bins based on a fixed timestamp:\n\n        >>> start, end = \'2000-10-01 23:30:00\', \'2000-10-02 00:30:00\'\n        >>> rng = pd.date_range(start, end, freq=\'7min\')\n        >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)\n        >>> ts\n        2000-10-01 23:30:00     0\n        2000-10-01 23:37:00     3\n        2000-10-01 23:44:00     6\n        2000-10-01 23:51:00     9\n        2000-10-01 23:58:00    12\n        2000-10-02 00:05:00    15\n        2000-10-02 00:12:00    18\n        2000-10-02 00:19:00    21\n        2000-10-02 00:26:00    24\n        Freq: 7T, dtype: int64\n\n        >>> ts.resample(\'17min\').sum()\n        2000-10-01 23:14:00     0\n        2000-10-01 23:31:00     9\n        2000-10-01 23:48:00    21\n        2000-10-02 00:05:00    54\n        2000-10-02 00:22:00    24\n        Freq: 17T, dtype: int64\n\n        >>> ts.resample(\'17min\', origin=\'epoch\').sum()\n        2000-10-01 23:18:00     0\n        2000-10-01 23:35:00    18\n        2000-10-01 23:52:00    27\n        2000-10-02 00:09:00    39\n        2000-10-02 00:26:00    24\n        Freq: 17T, dtype: int64\n\n        >>> ts.resample(\'17min\', origin=\'2000-01-01\').sum()\n        2000-10-01 23:24:00     3\n        2000-10-01 23:41:00    15\n        2000-10-01 23:58:00    45\n        2000-10-02 00:15:00    45\n        Freq: 17T, dtype: int64\n\n        If you want to adjust the start of the bins with an `offset` Timedelta, the two\n        following lines are equivalent:\n\n        >>> ts.resample(\'17min\', origin=\'start\').sum()\n        2000-10-01 23:30:00     9\n        2000-10-01 23:47:00    21\n        2000-10-02 00:04:00    54\n        2000-10-02 00:21:00    24\n        Freq: 17T, dtype: int64\n\n        >>> ts.resample(\'17min\', offset=\'23h30min\').sum()\n        2000-10-01 23:30:00     9\n        2000-10-01 23:47:00    21\n        2000-10-02 00:04:00    54\n        2000-10-02 00:21:00    24\n        Freq: 17T, dtype: int64\n\n        If you want to take the largest Timestamp as the end of the bins:\n\n        >>> ts.resample(\'17min\', origin=\'end\').sum()\n        2000-10-01 23:35:00     0\n        2000-10-01 23:52:00    18\n        2000-10-02 00:09:00    27\n        2000-10-02 00:26:00    63\n        Freq: 17T, dtype: int64\n\n        In contrast with the `start_day`, you can use `end_day` to take the ceiling\n        midnight of the largest Timestamp as the end of the bins and drop the bins\n        not containing data:\n\n        >>> ts.resample(\'17min\', origin=\'end_day\').sum()\n        2000-10-01 23:38:00     3\n        2000-10-01 23:55:00    15\n        2000-10-02 00:12:00    45\n        2000-10-02 00:29:00    45\n        Freq: 17T, dtype: int64\n\n        To replace the use of the deprecated `base` argument, you can now use `offset`,\n        in this example it is equivalent to have `base=2`:\n\n        >>> ts.resample(\'17min\', offset=\'2min\').sum()\n        2000-10-01 23:16:00     0\n        2000-10-01 23:33:00     9\n        2000-10-01 23:50:00    36\n        2000-10-02 00:07:00    39\n        2000-10-02 00:24:00    24\n        Freq: 17T, dtype: int64\n\n        To replace the use of the deprecated `loffset` argument:\n\n        >>> from pandas.tseries.frequencies import to_offset\n        >>> loffset = \'19min\'\n        >>> ts_out = ts.resample(\'17min\').sum()\n        >>> ts_out.index = ts_out.index + to_offset(loffset)\n        >>> ts_out\n        2000-10-01 23:33:00     0\n        2000-10-01 23:50:00     9\n        2000-10-02 00:07:00    21\n        2000-10-02 00:24:00    54\n        2000-10-02 00:41:00    24\n        Freq: 17T, dtype: int64\n        '
        from pandas.core.resample import get_resampler
        axis = self._get_axis_number(axis)
        return get_resampler(self, freq=rule, label=label, closed=closed, axis=axis, kind=kind, loffset=loffset, convention=convention, base=base, key=on, level=level, origin=origin, offset=offset)

    @final
    def first(self, offset):
        "\n        Select initial periods of time series data based on a date offset.\n\n        When having a DataFrame with dates as index, this function can\n        select the first few rows based on a date offset.\n\n        Parameters\n        ----------\n        offset : str, DateOffset or dateutil.relativedelta\n            The offset length of the data that will be selected. For instance,\n            '1M' will display all the rows having their index within the first month.\n\n        Returns\n        -------\n        Series or DataFrame\n            A subset of the caller.\n\n        Raises\n        ------\n        TypeError\n            If the index is not  a :class:`DatetimeIndex`\n\n        See Also\n        --------\n        last : Select final periods of time series based on a date offset.\n        at_time : Select values at a particular time of the day.\n        between_time : Select values between particular times of the day.\n\n        Examples\n        --------\n        >>> i = pd.date_range('2018-04-09', periods=4, freq='2D')\n        >>> ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)\n        >>> ts\n                    A\n        2018-04-09  1\n        2018-04-11  2\n        2018-04-13  3\n        2018-04-15  4\n\n        Get the rows for the first 3 days:\n\n        >>> ts.first('3D')\n                    A\n        2018-04-09  1\n        2018-04-11  2\n\n        Notice the data for 3 first calendar days were returned, not the first\n        3 days observed in the dataset, and therefore data for 2018-04-13 was\n        not returned.\n        "
        if (not isinstance(self.index, DatetimeIndex)):
            raise TypeError("'first' only supports a DatetimeIndex index")
        if (len(self.index) == 0):
            return self
        offset = to_offset(offset)
        if ((not isinstance(offset, Tick)) and offset.is_on_offset(self.index[0])):
            end_date = end = ((self.index[0] - offset.base) + offset)
        else:
            end_date = end = (self.index[0] + offset)
        if isinstance(offset, Tick):
            if (end_date in self.index):
                end = self.index.searchsorted(end_date, side='left')
                return self.iloc[:end]
        return self.loc[:end]

    @final
    def last(self, offset):
        "\n        Select final periods of time series data based on a date offset.\n\n        For a DataFrame with a sorted DatetimeIndex, this function\n        selects the last few rows based on a date offset.\n\n        Parameters\n        ----------\n        offset : str, DateOffset, dateutil.relativedelta\n            The offset length of the data that will be selected. For instance,\n            '3D' will display all the rows having their index within the last 3 days.\n\n        Returns\n        -------\n        Series or DataFrame\n            A subset of the caller.\n\n        Raises\n        ------\n        TypeError\n            If the index is not  a :class:`DatetimeIndex`\n\n        See Also\n        --------\n        first : Select initial periods of time series based on a date offset.\n        at_time : Select values at a particular time of the day.\n        between_time : Select values between particular times of the day.\n\n        Examples\n        --------\n        >>> i = pd.date_range('2018-04-09', periods=4, freq='2D')\n        >>> ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)\n        >>> ts\n                    A\n        2018-04-09  1\n        2018-04-11  2\n        2018-04-13  3\n        2018-04-15  4\n\n        Get the rows for the last 3 days:\n\n        >>> ts.last('3D')\n                    A\n        2018-04-13  3\n        2018-04-15  4\n\n        Notice the data for 3 last calendar days were returned, not the last\n        3 observed days in the dataset, and therefore data for 2018-04-11 was\n        not returned.\n        "
        if (not isinstance(self.index, DatetimeIndex)):
            raise TypeError("'last' only supports a DatetimeIndex index")
        if (len(self.index) == 0):
            return self
        offset = to_offset(offset)
        start_date = (self.index[(- 1)] - offset)
        start = self.index.searchsorted(start_date, side='right')
        return self.iloc[start:]

    @final
    def rank(self, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False):
        "\n        Compute numerical data ranks (1 through n) along axis.\n\n        By default, equal values are assigned a rank that is the average of the\n        ranks of those values.\n\n        Parameters\n        ----------\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            Index to direct ranking.\n        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'\n            How to rank the group of records that have the same value (i.e. ties):\n\n            * average: average rank of the group\n            * min: lowest rank in the group\n            * max: highest rank in the group\n            * first: ranks assigned in order they appear in the array\n            * dense: like 'min', but rank always increases by 1 between groups.\n\n        numeric_only : bool, optional\n            For DataFrame objects, rank only numeric columns if set to True.\n        na_option : {'keep', 'top', 'bottom'}, default 'keep'\n            How to rank NaN values:\n\n            * keep: assign NaN rank to NaN values\n            * top: assign smallest rank to NaN values if ascending\n            * bottom: assign highest rank to NaN values if ascending.\n\n        ascending : bool, default True\n            Whether or not the elements should be ranked in ascending order.\n        pct : bool, default False\n            Whether or not to display the returned rankings in percentile\n            form.\n\n        Returns\n        -------\n        same type as caller\n            Return a Series or DataFrame with data ranks as values.\n\n        See Also\n        --------\n        core.groupby.GroupBy.rank : Rank of values within each group.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(data={'Animal': ['cat', 'penguin', 'dog',\n        ...                                    'spider', 'snake'],\n        ...                         'Number_legs': [4, 2, 4, 8, np.nan]})\n        >>> df\n            Animal  Number_legs\n        0      cat          4.0\n        1  penguin          2.0\n        2      dog          4.0\n        3   spider          8.0\n        4    snake          NaN\n\n        The following example shows how the method behaves with the above\n        parameters:\n\n        * default_rank: this is the default behaviour obtained without using\n          any parameter.\n        * max_rank: setting ``method = 'max'`` the records that have the\n          same values are ranked using the highest rank (e.g.: since 'cat'\n          and 'dog' are both in the 2nd and 3rd position, rank 3 is assigned.)\n        * NA_bottom: choosing ``na_option = 'bottom'``, if there are records\n          with NaN values they are placed at the bottom of the ranking.\n        * pct_rank: when setting ``pct = True``, the ranking is expressed as\n          percentile rank.\n\n        >>> df['default_rank'] = df['Number_legs'].rank()\n        >>> df['max_rank'] = df['Number_legs'].rank(method='max')\n        >>> df['NA_bottom'] = df['Number_legs'].rank(na_option='bottom')\n        >>> df['pct_rank'] = df['Number_legs'].rank(pct=True)\n        >>> df\n            Animal  Number_legs  default_rank  max_rank  NA_bottom  pct_rank\n        0      cat          4.0           2.5       3.0        2.5     0.625\n        1  penguin          2.0           1.0       1.0        1.0     0.250\n        2      dog          4.0           2.5       3.0        2.5     0.625\n        3   spider          8.0           4.0       4.0        4.0     1.000\n        4    snake          NaN           NaN       NaN        5.0       NaN\n        "
        axis = self._get_axis_number(axis)
        if (na_option not in {'keep', 'top', 'bottom'}):
            msg = "na_option must be one of 'keep', 'top', or 'bottom'"
            raise ValueError(msg)

        def ranker(data):
            ranks = algos.rank(data.values, axis=axis, method=method, ascending=ascending, na_option=na_option, pct=pct)
            ranks = self._constructor(ranks, **data._construct_axes_dict())
            return ranks.__finalize__(self, method='rank')
        if (numeric_only is None):
            try:
                return ranker(self)
            except TypeError:
                numeric_only = True
        if numeric_only:
            data = self._get_numeric_data()
        else:
            data = self
        return ranker(data)

    @doc(_shared_docs['compare'], klass=_shared_doc_kwargs['klass'])
    def compare(self, other, align_axis=1, keep_shape=False, keep_equal=False):
        from pandas.core.reshape.concat import concat
        if (type(self) is not type(other)):
            (cls_self, cls_other) = (type(self).__name__, type(other).__name__)
            raise TypeError(f"can only compare '{cls_self}' (not '{cls_other}') with '{cls_self}'")
        mask = (~ ((self == other) | (self.isna() & other.isna())))
        keys = ['self', 'other']
        if (not keep_equal):
            self = self.where(mask)
            other = other.where(mask)
        if (not keep_shape):
            if isinstance(self, ABCDataFrame):
                cmask = mask.any()
                rmask = mask.any(axis=1)
                self = self.loc[(rmask, cmask)]
                other = other.loc[(rmask, cmask)]
            else:
                self = self[mask]
                other = other[mask]
        if (align_axis in (1, 'columns')):
            axis = 1
        else:
            axis = self._get_axis_number(align_axis)
        diff = concat([self, other], axis=axis, keys=keys)
        if (axis >= self.ndim):
            return diff
        ax = diff._get_axis(axis)
        ax_names = np.array(ax.names)
        ax.names = np.arange(len(ax_names))
        order = (list(range(1, ax.nlevels)) + [0])
        if isinstance(diff, ABCDataFrame):
            diff = diff.reorder_levels(order, axis=axis)
        else:
            diff = diff.reorder_levels(order)
        diff._get_axis(axis=axis).names = ax_names[order]
        indices = np.arange(diff.shape[axis]).reshape([2, (diff.shape[axis] // 2)]).T.flatten()
        diff = diff.take(indices, axis=axis)
        return diff

    @doc(**_shared_doc_kwargs)
    def align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None, limit=None, fill_axis=0, broadcast_axis=None):
        '\n        Align two objects on their axes with the specified join method.\n\n        Join method is specified for each axis Index.\n\n        Parameters\n        ----------\n        other : DataFrame or Series\n        join : {{\'outer\', \'inner\', \'left\', \'right\'}}, default \'outer\'\n        axis : allowed axis of the other object, default None\n            Align on index (0), columns (1), or both (None).\n        level : int or level name, default None\n            Broadcast across a level, matching Index values on the\n            passed MultiIndex level.\n        copy : bool, default True\n            Always returns new objects. If copy=False and no reindexing is\n            required then original objects are returned.\n        fill_value : scalar, default np.NaN\n            Value to use for missing values. Defaults to NaN, but can be any\n            "compatible" value.\n        method : {{\'backfill\', \'bfill\', \'pad\', \'ffill\', None}}, default None\n            Method to use for filling holes in reindexed Series:\n\n            - pad / ffill: propagate last valid observation forward to next valid.\n            - backfill / bfill: use NEXT valid observation to fill gap.\n\n        limit : int, default None\n            If method is specified, this is the maximum number of consecutive\n            NaN values to forward/backward fill. In other words, if there is\n            a gap with more than this number of consecutive NaNs, it will only\n            be partially filled. If method is not specified, this is the\n            maximum number of entries along the entire axis where NaNs will be\n            filled. Must be greater than 0 if not None.\n        fill_axis : {axes_single_arg}, default 0\n            Filling axis, method and limit.\n        broadcast_axis : {axes_single_arg}, default None\n            Broadcast values along this axis, if aligning two objects of\n            different dimensions.\n\n        Returns\n        -------\n        (left, right) : ({klass}, type of other)\n            Aligned objects.\n        '
        method = missing.clean_fill_method(method)
        if ((broadcast_axis == 1) and (self.ndim != other.ndim)):
            if isinstance(self, ABCSeries):
                cons = self._constructor_expanddim
                df = cons({c: self for c in other.columns}, **other._construct_axes_dict())
                return df._align_frame(other, join=join, axis=axis, level=level, copy=copy, fill_value=fill_value, method=method, limit=limit, fill_axis=fill_axis)
            elif isinstance(other, ABCSeries):
                cons = other._constructor_expanddim
                df = cons({c: other for c in self.columns}, **self._construct_axes_dict())
                return self._align_frame(df, join=join, axis=axis, level=level, copy=copy, fill_value=fill_value, method=method, limit=limit, fill_axis=fill_axis)
        if (axis is not None):
            axis = self._get_axis_number(axis)
        if isinstance(other, ABCDataFrame):
            return self._align_frame(other, join=join, axis=axis, level=level, copy=copy, fill_value=fill_value, method=method, limit=limit, fill_axis=fill_axis)
        elif isinstance(other, ABCSeries):
            return self._align_series(other, join=join, axis=axis, level=level, copy=copy, fill_value=fill_value, method=method, limit=limit, fill_axis=fill_axis)
        else:
            raise TypeError(f'unsupported type: {type(other)}')

    @final
    def _align_frame(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None, limit=None, fill_axis=0):
        (join_index, join_columns) = (None, None)
        (ilidx, iridx) = (None, None)
        (clidx, cridx) = (None, None)
        is_series = isinstance(self, ABCSeries)
        if ((axis is None) or (axis == 0)):
            if (not self.index.equals(other.index)):
                (join_index, ilidx, iridx) = self.index.join(other.index, how=join, level=level, return_indexers=True)
        if ((axis is None) or (axis == 1)):
            if ((not is_series) and (not self.columns.equals(other.columns))):
                (join_columns, clidx, cridx) = self.columns.join(other.columns, how=join, level=level, return_indexers=True)
        if is_series:
            reindexers = {0: [join_index, ilidx]}
        else:
            reindexers = {0: [join_index, ilidx], 1: [join_columns, clidx]}
        left = self._reindex_with_indexers(reindexers, copy=copy, fill_value=fill_value, allow_dups=True)
        right = other._reindex_with_indexers({0: [join_index, iridx], 1: [join_columns, cridx]}, copy=copy, fill_value=fill_value, allow_dups=True)
        if (method is not None):
            _left = left.fillna(method=method, axis=fill_axis, limit=limit)
            assert (_left is not None)
            left = _left
            right = right.fillna(method=method, axis=fill_axis, limit=limit)
        if is_datetime64tz_dtype(left.index.dtype):
            if (left.index.tz != right.index.tz):
                if (join_index is not None):
                    left = left.copy()
                    right = right.copy()
                    left.index = join_index
                    right.index = join_index
        return (left.__finalize__(self), right.__finalize__(other))

    @final
    def _align_series(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None, limit=None, fill_axis=0):
        is_series = isinstance(self, ABCSeries)
        if is_series:
            if axis:
                raise ValueError('cannot align series to a series other than axis 0')
            if self.index.equals(other.index):
                (join_index, lidx, ridx) = (None, None, None)
            else:
                (join_index, lidx, ridx) = self.index.join(other.index, how=join, level=level, return_indexers=True)
            left = self._reindex_indexer(join_index, lidx, copy)
            right = other._reindex_indexer(join_index, ridx, copy)
        else:
            fdata = self._mgr
            if (axis == 0):
                join_index = self.index
                (lidx, ridx) = (None, None)
                if (not self.index.equals(other.index)):
                    (join_index, lidx, ridx) = self.index.join(other.index, how=join, level=level, return_indexers=True)
                if (lidx is not None):
                    fdata = fdata.reindex_indexer(join_index, lidx, axis=1)
            elif (axis == 1):
                join_index = self.columns
                (lidx, ridx) = (None, None)
                if (not self.columns.equals(other.index)):
                    (join_index, lidx, ridx) = self.columns.join(other.index, how=join, level=level, return_indexers=True)
                if (lidx is not None):
                    fdata = fdata.reindex_indexer(join_index, lidx, axis=0)
            else:
                raise ValueError('Must specify axis=0 or 1')
            if (copy and (fdata is self._mgr)):
                fdata = fdata.copy()
            left = self._constructor(fdata)
            if (ridx is None):
                right = other
            else:
                right = other.reindex(join_index, level=level)
        fill_na = (notna(fill_value) or (method is not None))
        if fill_na:
            left = left.fillna(fill_value, method=method, limit=limit, axis=fill_axis)
            right = right.fillna(fill_value, method=method, limit=limit)
        if (is_series or ((not is_series) and (axis == 0))):
            if is_datetime64tz_dtype(left.index.dtype):
                if (left.index.tz != right.index.tz):
                    if (join_index is not None):
                        left = left.copy()
                        right = right.copy()
                        left.index = join_index
                        right.index = join_index
        return (left.__finalize__(self), right.__finalize__(other))

    @final
    def _where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise'):
        '\n        Equivalent to public method `where`, except that `other` is not\n        applied as a function even if callable. Used in __setitem__.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (axis is not None):
            axis = self._get_axis_number(axis)
        cond = com.apply_if_callable(cond, self)
        if isinstance(cond, NDFrame):
            (cond, _) = cond.align(self, join='right', broadcast_axis=1)
        else:
            if (not hasattr(cond, 'shape')):
                cond = np.asanyarray(cond)
            if (cond.shape != self.shape):
                raise ValueError('Array conditional must be same shape as self')
            cond = self._constructor(cond, **self._construct_axes_dict())
        fill_value = bool(inplace)
        cond = cond.fillna(fill_value)
        msg = 'Boolean array expected for the condition, not {dtype}'
        if (not cond.empty):
            if (not isinstance(cond, ABCDataFrame)):
                if (not is_bool_dtype(cond)):
                    raise ValueError(msg.format(dtype=cond.dtype))
            else:
                for dt in cond.dtypes:
                    if (not is_bool_dtype(dt)):
                        raise ValueError(msg.format(dtype=dt))
        else:
            cond = cond.astype(bool)
        cond = ((- cond) if inplace else cond)
        if isinstance(other, NDFrame):
            if (other.ndim <= self.ndim):
                (_, other) = self.align(other, join='left', axis=axis, level=level, fill_value=np.nan, copy=False)
                if ((axis is None) and (not other._indexed_same(self))):
                    raise InvalidIndexError
                elif (other.ndim < self.ndim):
                    other = other._values
                    if (axis == 0):
                        other = np.reshape(other, ((- 1), 1))
                    elif (axis == 1):
                        other = np.reshape(other, (1, (- 1)))
                    other = np.broadcast_to(other, self.shape)
            else:
                raise NotImplementedError('cannot align with a higher dimensional NDFrame')
        if (not isinstance(other, (MultiIndex, NDFrame))):
            other = extract_array(other, extract_numpy=True)
        if isinstance(other, (np.ndarray, ExtensionArray)):
            if (other.shape != self.shape):
                if (self.ndim == 1):
                    icond = cond._values
                    if (len(other) == 1):
                        other = other[0]
                    elif (len(cond[icond]) == len(other)):
                        new_other = self._values
                        new_other = new_other.copy()
                        new_other[icond] = other
                        other = new_other
                    else:
                        raise ValueError('Length of replacements must equal series length')
                else:
                    raise ValueError('other must be the same shape as self when an ndarray')
            else:
                other = self._constructor(other, **self._construct_axes_dict())
        if (axis is None):
            axis = 0
        if (self.ndim == getattr(other, 'ndim', 0)):
            align = True
        else:
            align = (self._get_axis_number(axis) == 1)
        if isinstance(cond, NDFrame):
            cond = cond.reindex(self._info_axis, axis=self._info_axis_number, copy=False)
        block_axis = self._get_block_manager_axis(axis)
        if inplace:
            self._check_inplace_setting(other)
            new_data = self._mgr.putmask(mask=cond, new=other, align=align, axis=block_axis)
            result = self._constructor(new_data)
            return self._update_inplace(result)
        else:
            new_data = self._mgr.where(other=other, cond=cond, align=align, errors=errors, axis=block_axis)
            result = self._constructor(new_data)
            return result.__finalize__(self)

    @final
    @doc(klass=_shared_doc_kwargs['klass'], cond='True', cond_rev='False', name='where', name_other='mask')
    def where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=lib.no_default):
        "\n        Replace values where the condition is {cond_rev}.\n\n        Parameters\n        ----------\n        cond : bool {klass}, array-like, or callable\n            Where `cond` is {cond}, keep the original value. Where\n            {cond_rev}, replace with corresponding value from `other`.\n            If `cond` is callable, it is computed on the {klass} and\n            should return boolean {klass} or array. The callable must\n            not change input {klass} (though pandas doesn't check it).\n        other : scalar, {klass}, or callable\n            Entries where `cond` is {cond_rev} are replaced with\n            corresponding value from `other`.\n            If other is callable, it is computed on the {klass} and\n            should return scalar or {klass}. The callable must not\n            change input {klass} (though pandas doesn't check it).\n        inplace : bool, default False\n            Whether to perform the operation in place on the data.\n        axis : int, default None\n            Alignment axis if needed.\n        level : int, default None\n            Alignment level if needed.\n        errors : str, {{'raise', 'ignore'}}, default 'raise'\n            Note that currently this parameter won't affect\n            the results and will always coerce to a suitable dtype.\n\n            - 'raise' : allow exceptions to be raised.\n            - 'ignore' : suppress exceptions. On error return original object.\n\n        try_cast : bool, default None\n            Try to cast the result back to the input type (if possible).\n\n            .. deprecated:: 1.3.0\n                Manually cast back if necessary.\n\n        Returns\n        -------\n        Same type as caller or None if ``inplace=True``.\n\n        See Also\n        --------\n        :func:`DataFrame.{name_other}` : Return an object of same shape as\n            self.\n\n        Notes\n        -----\n        The {name} method is an application of the if-then idiom. For each\n        element in the calling DataFrame, if ``cond`` is ``{cond}`` the\n        element is used; otherwise the corresponding element from the DataFrame\n        ``other`` is used.\n\n        The signature for :func:`DataFrame.where` differs from\n        :func:`numpy.where`. Roughly ``df1.where(m, df2)`` is equivalent to\n        ``np.where(m, df1, df2)``.\n\n        For further details and examples see the ``{name}`` documentation in\n        :ref:`indexing <indexing.where_mask>`.\n\n        Examples\n        --------\n        >>> s = pd.Series(range(5))\n        >>> s.where(s > 0)\n        0    NaN\n        1    1.0\n        2    2.0\n        3    3.0\n        4    4.0\n        dtype: float64\n        >>> s.mask(s > 0)\n        0    0.0\n        1    NaN\n        2    NaN\n        3    NaN\n        4    NaN\n        dtype: float64\n\n        >>> s.where(s > 1, 10)\n        0    10\n        1    10\n        2    2\n        3    3\n        4    4\n        dtype: int64\n        >>> s.mask(s > 1, 10)\n        0     0\n        1     1\n        2    10\n        3    10\n        4    10\n        dtype: int64\n\n        >>> df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=['A', 'B'])\n        >>> df\n           A  B\n        0  0  1\n        1  2  3\n        2  4  5\n        3  6  7\n        4  8  9\n        >>> m = df % 3 == 0\n        >>> df.where(m, -df)\n           A  B\n        0  0 -1\n        1 -2  3\n        2 -4 -5\n        3  6 -7\n        4 -8  9\n        >>> df.where(m, -df) == np.where(m, df, -df)\n              A     B\n        0  True  True\n        1  True  True\n        2  True  True\n        3  True  True\n        4  True  True\n        >>> df.where(m, -df) == df.mask(~m, -df)\n              A     B\n        0  True  True\n        1  True  True\n        2  True  True\n        3  True  True\n        4  True  True\n        "
        other = com.apply_if_callable(other, self)
        if (try_cast is not lib.no_default):
            warnings.warn('try_cast keyword is deprecated and will be removed in a future version', FutureWarning, stacklevel=2)
        return self._where(cond, other, inplace, axis, level, errors=errors)

    @final
    @doc(where, klass=_shared_doc_kwargs['klass'], cond='False', cond_rev='True', name='mask', name_other='where')
    def mask(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=lib.no_default):
        inplace = validate_bool_kwarg(inplace, 'inplace')
        cond = com.apply_if_callable(cond, self)
        if (try_cast is not lib.no_default):
            warnings.warn('try_cast keyword is deprecated and will be removed in a future version', FutureWarning, stacklevel=2)
        if (not hasattr(cond, '__invert__')):
            cond = np.array(cond)
        return self.where((~ cond), other=other, inplace=inplace, axis=axis, level=level, errors=errors)

    @doc(klass=_shared_doc_kwargs['klass'])
    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        '\n        Shift index by desired number of periods with an optional time `freq`.\n\n        When `freq` is not passed, shift the index without realigning the data.\n        If `freq` is passed (in this case, the index must be date or datetime,\n        or it will raise a `NotImplementedError`), the index will be\n        increased using the periods and the `freq`. `freq` can be inferred\n        when specified as "infer" as long as either freq or inferred_freq\n        attribute is set in the index.\n\n        Parameters\n        ----------\n        periods : int\n            Number of periods to shift. Can be positive or negative.\n        freq : DateOffset, tseries.offsets, timedelta, or str, optional\n            Offset to use from the tseries module or time rule (e.g. \'EOM\').\n            If `freq` is specified then the index values are shifted but the\n            data is not realigned. That is, use `freq` if you would like to\n            extend the index when shifting and preserve the original data.\n            If `freq` is specified as "infer" then it will be inferred from\n            the freq or inferred_freq attributes of the index. If neither of\n            those attributes exist, a ValueError is thrown.\n        axis : {{0 or \'index\', 1 or \'columns\', None}}, default None\n            Shift direction.\n        fill_value : object, optional\n            The scalar value to use for newly introduced missing values.\n            the default depends on the dtype of `self`.\n            For numeric data, ``np.nan`` is used.\n            For datetime, timedelta, or period data, etc. :attr:`NaT` is used.\n            For extension dtypes, ``self.dtype.na_value`` is used.\n\n            .. versionchanged:: 1.1.0\n\n        Returns\n        -------\n        {klass}\n            Copy of input object, shifted.\n\n        See Also\n        --------\n        Index.shift : Shift values of Index.\n        DatetimeIndex.shift : Shift values of DatetimeIndex.\n        PeriodIndex.shift : Shift values of PeriodIndex.\n        tshift : Shift the time index, using the index\'s frequency if\n            available.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({{"Col1": [10, 20, 15, 30, 45],\n        ...                    "Col2": [13, 23, 18, 33, 48],\n        ...                    "Col3": [17, 27, 22, 37, 52]}},\n        ...                   index=pd.date_range("2020-01-01", "2020-01-05"))\n        >>> df\n                    Col1  Col2  Col3\n        2020-01-01    10    13    17\n        2020-01-02    20    23    27\n        2020-01-03    15    18    22\n        2020-01-04    30    33    37\n        2020-01-05    45    48    52\n\n        >>> df.shift(periods=3)\n                    Col1  Col2  Col3\n        2020-01-01   NaN   NaN   NaN\n        2020-01-02   NaN   NaN   NaN\n        2020-01-03   NaN   NaN   NaN\n        2020-01-04  10.0  13.0  17.0\n        2020-01-05  20.0  23.0  27.0\n\n        >>> df.shift(periods=1, axis="columns")\n                    Col1  Col2  Col3\n        2020-01-01   NaN    10    13\n        2020-01-02   NaN    20    23\n        2020-01-03   NaN    15    18\n        2020-01-04   NaN    30    33\n        2020-01-05   NaN    45    48\n\n        >>> df.shift(periods=3, fill_value=0)\n                    Col1  Col2  Col3\n        2020-01-01     0     0     0\n        2020-01-02     0     0     0\n        2020-01-03     0     0     0\n        2020-01-04    10    13    17\n        2020-01-05    20    23    27\n\n        >>> df.shift(periods=3, freq="D")\n                    Col1  Col2  Col3\n        2020-01-04    10    13    17\n        2020-01-05    20    23    27\n        2020-01-06    15    18    22\n        2020-01-07    30    33    37\n        2020-01-08    45    48    52\n\n        >>> df.shift(periods=3, freq="infer")\n                    Col1  Col2  Col3\n        2020-01-04    10    13    17\n        2020-01-05    20    23    27\n        2020-01-06    15    18    22\n        2020-01-07    30    33    37\n        2020-01-08    45    48    52\n        '
        if (periods == 0):
            return self.copy()
        if (freq is None):
            block_axis = self._get_block_manager_axis(axis)
            new_data = self._mgr.shift(periods=periods, axis=block_axis, fill_value=fill_value)
            return self._constructor(new_data).__finalize__(self, method='shift')
        index = self._get_axis(axis)
        if (freq == 'infer'):
            freq = getattr(index, 'freq', None)
            if (freq is None):
                freq = getattr(index, 'inferred_freq', None)
            if (freq is None):
                msg = 'Freq was not set in the index hence cannot be inferred'
                raise ValueError(msg)
        elif isinstance(freq, str):
            freq = to_offset(freq)
        if isinstance(index, PeriodIndex):
            orig_freq = to_offset(index.freq)
            if (freq != orig_freq):
                assert (orig_freq is not None)
                raise ValueError(f'Given freq {freq.rule_code} does not match PeriodIndex freq {orig_freq.rule_code}')
            new_ax = index.shift(periods)
        else:
            new_ax = index.shift(periods, freq)
        result = self.set_axis(new_ax, axis)
        return result.__finalize__(self, method='shift')

    @final
    def slice_shift(self, periods=1, axis=0):
        '\n        Equivalent to `shift` without copying data.\n        The shifted data will not include the dropped periods and the\n        shifted axis will be smaller than the original.\n\n        .. deprecated:: 1.2.0\n            slice_shift is deprecated,\n            use DataFrame/Series.shift instead.\n\n        Parameters\n        ----------\n        periods : int\n            Number of periods to move, can be positive or negative.\n\n        Returns\n        -------\n        shifted : same type as caller\n\n        Notes\n        -----\n        While the `slice_shift` is faster than `shift`, you may pay for it\n        later during alignment.\n        '
        msg = "The 'slice_shift' method is deprecated and will be removed in a future version. You can use DataFrame/Series.shift instead"
        warnings.warn(msg, FutureWarning, stacklevel=2)
        if (periods == 0):
            return self
        if (periods > 0):
            vslicer = slice(None, (- periods))
            islicer = slice(periods, None)
        else:
            vslicer = slice((- periods), None)
            islicer = slice(None, periods)
        new_obj = self._slice(vslicer, axis=axis)
        shifted_axis = self._get_axis(axis)[islicer]
        new_obj.set_axis(shifted_axis, axis=axis, inplace=True)
        return new_obj.__finalize__(self, method='slice_shift')

    @final
    def tshift(self, periods=1, freq=None, axis=0):
        "\n        Shift the time index, using the index's frequency if available.\n\n        .. deprecated:: 1.1.0\n            Use `shift` instead.\n\n        Parameters\n        ----------\n        periods : int\n            Number of periods to move, can be positive or negative.\n        freq : DateOffset, timedelta, or str, default None\n            Increment to use from the tseries module\n            or time rule expressed as a string (e.g. 'EOM').\n        axis : {0 or ‘index’, 1 or ‘columns’, None}, default 0\n            Corresponds to the axis that contains the Index.\n\n        Returns\n        -------\n        shifted : Series/DataFrame\n\n        Notes\n        -----\n        If freq is not specified then tries to use the freq or inferred_freq\n        attributes of the index. If neither of those attributes exist, a\n        ValueError is thrown\n        "
        warnings.warn('tshift is deprecated and will be removed in a future version. Please use shift instead.', FutureWarning, stacklevel=2)
        if (freq is None):
            freq = 'infer'
        return self.shift(periods, freq, axis)

    def truncate(self, before=None, after=None, axis=None, copy=True):
        '\n        Truncate a Series or DataFrame before and after some index value.\n\n        This is a useful shorthand for boolean indexing based on index\n        values above or below certain thresholds.\n\n        Parameters\n        ----------\n        before : date, str, int\n            Truncate all rows before this index value.\n        after : date, str, int\n            Truncate all rows after this index value.\n        axis : {0 or \'index\', 1 or \'columns\'}, optional\n            Axis to truncate. Truncates the index (rows) by default.\n        copy : bool, default is True,\n            Return a copy of the truncated section.\n\n        Returns\n        -------\n        type of caller\n            The truncated Series or DataFrame.\n\n        See Also\n        --------\n        DataFrame.loc : Select a subset of a DataFrame by label.\n        DataFrame.iloc : Select a subset of a DataFrame by position.\n\n        Notes\n        -----\n        If the index being truncated contains only datetime values,\n        `before` and `after` may be specified as strings instead of\n        Timestamps.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'A\': [\'a\', \'b\', \'c\', \'d\', \'e\'],\n        ...                    \'B\': [\'f\', \'g\', \'h\', \'i\', \'j\'],\n        ...                    \'C\': [\'k\', \'l\', \'m\', \'n\', \'o\']},\n        ...                   index=[1, 2, 3, 4, 5])\n        >>> df\n           A  B  C\n        1  a  f  k\n        2  b  g  l\n        3  c  h  m\n        4  d  i  n\n        5  e  j  o\n\n        >>> df.truncate(before=2, after=4)\n           A  B  C\n        2  b  g  l\n        3  c  h  m\n        4  d  i  n\n\n        The columns of a DataFrame can be truncated.\n\n        >>> df.truncate(before="A", after="B", axis="columns")\n           A  B\n        1  a  f\n        2  b  g\n        3  c  h\n        4  d  i\n        5  e  j\n\n        For Series, only rows can be truncated.\n\n        >>> df[\'A\'].truncate(before=2, after=4)\n        2    b\n        3    c\n        4    d\n        Name: A, dtype: object\n\n        The index values in ``truncate`` can be datetimes or string\n        dates.\n\n        >>> dates = pd.date_range(\'2016-01-01\', \'2016-02-01\', freq=\'s\')\n        >>> df = pd.DataFrame(index=dates, data={\'A\': 1})\n        >>> df.tail()\n                             A\n        2016-01-31 23:59:56  1\n        2016-01-31 23:59:57  1\n        2016-01-31 23:59:58  1\n        2016-01-31 23:59:59  1\n        2016-02-01 00:00:00  1\n\n        >>> df.truncate(before=pd.Timestamp(\'2016-01-05\'),\n        ...             after=pd.Timestamp(\'2016-01-10\')).tail()\n                             A\n        2016-01-09 23:59:56  1\n        2016-01-09 23:59:57  1\n        2016-01-09 23:59:58  1\n        2016-01-09 23:59:59  1\n        2016-01-10 00:00:00  1\n\n        Because the index is a DatetimeIndex containing only dates, we can\n        specify `before` and `after` as strings. They will be coerced to\n        Timestamps before truncation.\n\n        >>> df.truncate(\'2016-01-05\', \'2016-01-10\').tail()\n                             A\n        2016-01-09 23:59:56  1\n        2016-01-09 23:59:57  1\n        2016-01-09 23:59:58  1\n        2016-01-09 23:59:59  1\n        2016-01-10 00:00:00  1\n\n        Note that ``truncate`` assumes a 0 value for any unspecified time\n        component (midnight). This differs from partial string slicing, which\n        returns any partially matching dates.\n\n        >>> df.loc[\'2016-01-05\':\'2016-01-10\', :].tail()\n                             A\n        2016-01-10 23:59:55  1\n        2016-01-10 23:59:56  1\n        2016-01-10 23:59:57  1\n        2016-01-10 23:59:58  1\n        2016-01-10 23:59:59  1\n        '
        if (axis is None):
            axis = self._stat_axis_number
        axis = self._get_axis_number(axis)
        ax = self._get_axis(axis)
        if ((not ax.is_monotonic_increasing) and (not ax.is_monotonic_decreasing)):
            raise ValueError('truncate requires a sorted index')
        if ax._is_all_dates:
            from pandas.core.tools.datetimes import to_datetime
            before = to_datetime(before)
            after = to_datetime(after)
        if ((before is not None) and (after is not None)):
            if (before > after):
                raise ValueError(f'Truncate: {after} must be after {before}')
        if ((len(ax) > 1) and ax.is_monotonic_decreasing):
            (before, after) = (after, before)
        slicer = ([slice(None, None)] * self._AXIS_LEN)
        slicer[axis] = slice(before, after)
        result = self.loc[tuple(slicer)]
        if isinstance(ax, MultiIndex):
            setattr(result, self._get_axis_name(axis), ax.truncate(before, after))
        if copy:
            result = result.copy()
        return result

    @final
    def tz_convert(self, tz, axis=0, level=None, copy=True):
        '\n        Convert tz-aware axis to target time zone.\n\n        Parameters\n        ----------\n        tz : str or tzinfo object\n        axis : the axis to convert\n        level : int, str, default None\n            If axis is a MultiIndex, convert a specific level. Otherwise\n            must be None.\n        copy : bool, default True\n            Also make a copy of the underlying data.\n\n        Returns\n        -------\n        {klass}\n            Object with time zone converted axis.\n\n        Raises\n        ------\n        TypeError\n            If the axis is tz-naive.\n        '
        axis = self._get_axis_number(axis)
        ax = self._get_axis(axis)

        def _tz_convert(ax, tz):
            if (not hasattr(ax, 'tz_convert')):
                if (len(ax) > 0):
                    ax_name = self._get_axis_name(axis)
                    raise TypeError(f'{ax_name} is not a valid DatetimeIndex or PeriodIndex')
                else:
                    ax = DatetimeIndex([], tz=tz)
            else:
                ax = ax.tz_convert(tz)
            return ax
        if isinstance(ax, MultiIndex):
            level = ax._get_level_number(level)
            new_level = _tz_convert(ax.levels[level], tz)
            ax = ax.set_levels(new_level, level=level)
        else:
            if (level not in (None, 0, ax.name)):
                raise ValueError(f'The level {level} is not valid')
            ax = _tz_convert(ax, tz)
        result = self.copy(deep=copy)
        result = result.set_axis(ax, axis=axis, inplace=False)
        return result.__finalize__(self, method='tz_convert')

    @final
    def tz_localize(self, tz, axis=0, level=None, copy=True, ambiguous='raise', nonexistent='raise'):
        "\n        Localize tz-naive index of a Series or DataFrame to target time zone.\n\n        This operation localizes the Index. To localize the values in a\n        timezone-naive Series, use :meth:`Series.dt.tz_localize`.\n\n        Parameters\n        ----------\n        tz : str or tzinfo\n        axis : the axis to localize\n        level : int, str, default None\n            If axis ia a MultiIndex, localize a specific level. Otherwise\n            must be None.\n        copy : bool, default True\n            Also make a copy of the underlying data.\n        ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'\n            When clocks moved backward due to DST, ambiguous times may arise.\n            For example in Central European Time (UTC+01), when going from\n            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at\n            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the\n            `ambiguous` parameter dictates how ambiguous times should be\n            handled.\n\n            - 'infer' will attempt to infer fall dst-transition hours based on\n              order\n            - bool-ndarray where True signifies a DST time, False designates\n              a non-DST time (note that this flag is only applicable for\n              ambiguous times)\n            - 'NaT' will return NaT where there are ambiguous times\n            - 'raise' will raise an AmbiguousTimeError if there are ambiguous\n              times.\n        nonexistent : str, default 'raise'\n            A nonexistent time does not exist in a particular timezone\n            where clocks moved forward due to DST. Valid values are:\n\n            - 'shift_forward' will shift the nonexistent time forward to the\n              closest existing time\n            - 'shift_backward' will shift the nonexistent time backward to the\n              closest existing time\n            - 'NaT' will return NaT where there are nonexistent times\n            - timedelta objects will shift nonexistent times by the timedelta\n            - 'raise' will raise an NonExistentTimeError if there are\n              nonexistent times.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        Series or DataFrame\n            Same type as the input.\n\n        Raises\n        ------\n        TypeError\n            If the TimeSeries is tz-aware and tz is not None.\n\n        Examples\n        --------\n        Localize local times:\n\n        >>> s = pd.Series([1],\n        ...               index=pd.DatetimeIndex(['2018-09-15 01:30:00']))\n        >>> s.tz_localize('CET')\n        2018-09-15 01:30:00+02:00    1\n        dtype: int64\n\n        Be careful with DST changes. When there is sequential data, pandas\n        can infer the DST time:\n\n        >>> s = pd.Series(range(7),\n        ...               index=pd.DatetimeIndex(['2018-10-28 01:30:00',\n        ...                                       '2018-10-28 02:00:00',\n        ...                                       '2018-10-28 02:30:00',\n        ...                                       '2018-10-28 02:00:00',\n        ...                                       '2018-10-28 02:30:00',\n        ...                                       '2018-10-28 03:00:00',\n        ...                                       '2018-10-28 03:30:00']))\n        >>> s.tz_localize('CET', ambiguous='infer')\n        2018-10-28 01:30:00+02:00    0\n        2018-10-28 02:00:00+02:00    1\n        2018-10-28 02:30:00+02:00    2\n        2018-10-28 02:00:00+01:00    3\n        2018-10-28 02:30:00+01:00    4\n        2018-10-28 03:00:00+01:00    5\n        2018-10-28 03:30:00+01:00    6\n        dtype: int64\n\n        In some cases, inferring the DST is impossible. In such cases, you can\n        pass an ndarray to the ambiguous parameter to set the DST explicitly\n\n        >>> s = pd.Series(range(3),\n        ...               index=pd.DatetimeIndex(['2018-10-28 01:20:00',\n        ...                                       '2018-10-28 02:36:00',\n        ...                                       '2018-10-28 03:46:00']))\n        >>> s.tz_localize('CET', ambiguous=np.array([True, True, False]))\n        2018-10-28 01:20:00+02:00    0\n        2018-10-28 02:36:00+02:00    1\n        2018-10-28 03:46:00+01:00    2\n        dtype: int64\n\n        If the DST transition causes nonexistent times, you can shift these\n        dates forward or backward with a timedelta object or `'shift_forward'`\n        or `'shift_backward'`.\n\n        >>> s = pd.Series(range(2),\n        ...               index=pd.DatetimeIndex(['2015-03-29 02:30:00',\n        ...                                       '2015-03-29 03:30:00']))\n        >>> s.tz_localize('Europe/Warsaw', nonexistent='shift_forward')\n        2015-03-29 03:00:00+02:00    0\n        2015-03-29 03:30:00+02:00    1\n        dtype: int64\n        >>> s.tz_localize('Europe/Warsaw', nonexistent='shift_backward')\n        2015-03-29 01:59:59.999999999+01:00    0\n        2015-03-29 03:30:00+02:00              1\n        dtype: int64\n        >>> s.tz_localize('Europe/Warsaw', nonexistent=pd.Timedelta('1H'))\n        2015-03-29 03:30:00+02:00    0\n        2015-03-29 03:30:00+02:00    1\n        dtype: int64\n        "
        nonexistent_options = ('raise', 'NaT', 'shift_forward', 'shift_backward')
        if ((nonexistent not in nonexistent_options) and (not isinstance(nonexistent, timedelta))):
            raise ValueError("The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward', 'shift_backward' or a timedelta object")
        axis = self._get_axis_number(axis)
        ax = self._get_axis(axis)

        def _tz_localize(ax, tz, ambiguous, nonexistent):
            if (not hasattr(ax, 'tz_localize')):
                if (len(ax) > 0):
                    ax_name = self._get_axis_name(axis)
                    raise TypeError(f'{ax_name} is not a valid DatetimeIndex or PeriodIndex')
                else:
                    ax = DatetimeIndex([], tz=tz)
            else:
                ax = ax.tz_localize(tz, ambiguous=ambiguous, nonexistent=nonexistent)
            return ax
        if isinstance(ax, MultiIndex):
            level = ax._get_level_number(level)
            new_level = _tz_localize(ax.levels[level], tz, ambiguous, nonexistent)
            ax = ax.set_levels(new_level, level=level)
        else:
            if (level not in (None, 0, ax.name)):
                raise ValueError(f'The level {level} is not valid')
            ax = _tz_localize(ax, tz, ambiguous, nonexistent)
        result = self.copy(deep=copy)
        result = result.set_axis(ax, axis=axis, inplace=False)
        return result.__finalize__(self, method='tz_localize')

    @final
    def abs(self):
        "\n        Return a Series/DataFrame with absolute numeric value of each element.\n\n        This function only applies to elements that are all numeric.\n\n        Returns\n        -------\n        abs\n            Series/DataFrame containing the absolute value of each element.\n\n        See Also\n        --------\n        numpy.absolute : Calculate the absolute value element-wise.\n\n        Notes\n        -----\n        For ``complex`` inputs, ``1.2 + 1j``, the absolute value is\n        :math:`\\sqrt{ a^2 + b^2 }`.\n\n        Examples\n        --------\n        Absolute numeric values in a Series.\n\n        >>> s = pd.Series([-1.10, 2, -3.33, 4])\n        >>> s.abs()\n        0    1.10\n        1    2.00\n        2    3.33\n        3    4.00\n        dtype: float64\n\n        Absolute numeric values in a Series with complex numbers.\n\n        >>> s = pd.Series([1.2 + 1j])\n        >>> s.abs()\n        0    1.56205\n        dtype: float64\n\n        Absolute numeric values in a Series with a Timedelta element.\n\n        >>> s = pd.Series([pd.Timedelta('1 days')])\n        >>> s.abs()\n        0   1 days\n        dtype: timedelta64[ns]\n\n        Select rows with data closest to certain value using argsort (from\n        `StackOverflow <https://stackoverflow.com/a/17758115>`__).\n\n        >>> df = pd.DataFrame({\n        ...     'a': [4, 5, 6, 7],\n        ...     'b': [10, 20, 30, 40],\n        ...     'c': [100, 50, -30, -50]\n        ... })\n        >>> df\n             a    b    c\n        0    4   10  100\n        1    5   20   50\n        2    6   30  -30\n        3    7   40  -50\n        >>> df.loc[(df.c - 43).abs().argsort()]\n             a    b    c\n        1    5   20   50\n        0    4   10  100\n        2    6   30  -30\n        3    7   40  -50\n        "
        return np.abs(self)

    @final
    def describe(self, percentiles=None, include=None, exclude=None, datetime_is_numeric=False):
        '\n        Generate descriptive statistics.\n\n        Descriptive statistics include those that summarize the central\n        tendency, dispersion and shape of a\n        dataset\'s distribution, excluding ``NaN`` values.\n\n        Analyzes both numeric and object series, as well\n        as ``DataFrame`` column sets of mixed data types. The output\n        will vary depending on what is provided. Refer to the notes\n        below for more detail.\n\n        Parameters\n        ----------\n        percentiles : list-like of numbers, optional\n            The percentiles to include in the output. All should\n            fall between 0 and 1. The default is\n            ``[.25, .5, .75]``, which returns the 25th, 50th, and\n            75th percentiles.\n        include : \'all\', list-like of dtypes or None (default), optional\n            A white list of data types to include in the result. Ignored\n            for ``Series``. Here are the options:\n\n            - \'all\' : All columns of the input will be included in the output.\n            - A list-like of dtypes : Limits the results to the\n              provided data types.\n              To limit the result to numeric types submit\n              ``numpy.number``. To limit it instead to object columns submit\n              the ``numpy.object`` data type. Strings\n              can also be used in the style of\n              ``select_dtypes`` (e.g. ``df.describe(include=[\'O\'])``). To\n              select pandas categorical columns, use ``\'category\'``\n            - None (default) : The result will include all numeric columns.\n        exclude : list-like of dtypes or None (default), optional,\n            A black list of data types to omit from the result. Ignored\n            for ``Series``. Here are the options:\n\n            - A list-like of dtypes : Excludes the provided data types\n              from the result. To exclude numeric types submit\n              ``numpy.number``. To exclude object columns submit the data\n              type ``numpy.object``. Strings can also be used in the style of\n              ``select_dtypes`` (e.g. ``df.describe(include=[\'O\'])``). To\n              exclude pandas categorical columns, use ``\'category\'``\n            - None (default) : The result will exclude nothing.\n        datetime_is_numeric : bool, default False\n            Whether to treat datetime dtypes as numeric. This affects statistics\n            calculated for the column. For DataFrame input, this also\n            controls whether datetime columns are included by default.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        Series or DataFrame\n            Summary statistics of the Series or Dataframe provided.\n\n        See Also\n        --------\n        DataFrame.count: Count number of non-NA/null observations.\n        DataFrame.max: Maximum of the values in the object.\n        DataFrame.min: Minimum of the values in the object.\n        DataFrame.mean: Mean of the values.\n        DataFrame.std: Standard deviation of the observations.\n        DataFrame.select_dtypes: Subset of a DataFrame including/excluding\n            columns based on their dtype.\n\n        Notes\n        -----\n        For numeric data, the result\'s index will include ``count``,\n        ``mean``, ``std``, ``min``, ``max`` as well as lower, ``50`` and\n        upper percentiles. By default the lower percentile is ``25`` and the\n        upper percentile is ``75``. The ``50`` percentile is the\n        same as the median.\n\n        For object data (e.g. strings or timestamps), the result\'s index\n        will include ``count``, ``unique``, ``top``, and ``freq``. The ``top``\n        is the most common value. The ``freq`` is the most common value\'s\n        frequency. Timestamps also include the ``first`` and ``last`` items.\n\n        If multiple object values have the highest count, then the\n        ``count`` and ``top`` results will be arbitrarily chosen from\n        among those with the highest count.\n\n        For mixed data types provided via a ``DataFrame``, the default is to\n        return only an analysis of numeric columns. If the dataframe consists\n        only of object and categorical data without any numeric columns, the\n        default is to return an analysis of both the object and categorical\n        columns. If ``include=\'all\'`` is provided as an option, the result\n        will include a union of attributes of each type.\n\n        The `include` and `exclude` parameters can be used to limit\n        which columns in a ``DataFrame`` are analyzed for the output.\n        The parameters are ignored when analyzing a ``Series``.\n\n        Examples\n        --------\n        Describing a numeric ``Series``.\n\n        >>> s = pd.Series([1, 2, 3])\n        >>> s.describe()\n        count    3.0\n        mean     2.0\n        std      1.0\n        min      1.0\n        25%      1.5\n        50%      2.0\n        75%      2.5\n        max      3.0\n        dtype: float64\n\n        Describing a categorical ``Series``.\n\n        >>> s = pd.Series([\'a\', \'a\', \'b\', \'c\'])\n        >>> s.describe()\n        count     4\n        unique    3\n        top       a\n        freq      2\n        dtype: object\n\n        Describing a timestamp ``Series``.\n\n        >>> s = pd.Series([\n        ...   np.datetime64("2000-01-01"),\n        ...   np.datetime64("2010-01-01"),\n        ...   np.datetime64("2010-01-01")\n        ... ])\n        >>> s.describe(datetime_is_numeric=True)\n        count                      3\n        mean     2006-09-01 08:00:00\n        min      2000-01-01 00:00:00\n        25%      2004-12-31 12:00:00\n        50%      2010-01-01 00:00:00\n        75%      2010-01-01 00:00:00\n        max      2010-01-01 00:00:00\n        dtype: object\n\n        Describing a ``DataFrame``. By default only numeric fields\n        are returned.\n\n        >>> df = pd.DataFrame({\'categorical\': pd.Categorical([\'d\',\'e\',\'f\']),\n        ...                    \'numeric\': [1, 2, 3],\n        ...                    \'object\': [\'a\', \'b\', \'c\']\n        ...                   })\n        >>> df.describe()\n               numeric\n        count      3.0\n        mean       2.0\n        std        1.0\n        min        1.0\n        25%        1.5\n        50%        2.0\n        75%        2.5\n        max        3.0\n\n        Describing all columns of a ``DataFrame`` regardless of data type.\n\n        >>> df.describe(include=\'all\')  # doctest: +SKIP\n               categorical  numeric object\n        count            3      3.0      3\n        unique           3      NaN      3\n        top              f      NaN      a\n        freq             1      NaN      1\n        mean           NaN      2.0    NaN\n        std            NaN      1.0    NaN\n        min            NaN      1.0    NaN\n        25%            NaN      1.5    NaN\n        50%            NaN      2.0    NaN\n        75%            NaN      2.5    NaN\n        max            NaN      3.0    NaN\n\n        Describing a column from a ``DataFrame`` by accessing it as\n        an attribute.\n\n        >>> df.numeric.describe()\n        count    3.0\n        mean     2.0\n        std      1.0\n        min      1.0\n        25%      1.5\n        50%      2.0\n        75%      2.5\n        max      3.0\n        Name: numeric, dtype: float64\n\n        Including only numeric columns in a ``DataFrame`` description.\n\n        >>> df.describe(include=[np.number])\n               numeric\n        count      3.0\n        mean       2.0\n        std        1.0\n        min        1.0\n        25%        1.5\n        50%        2.0\n        75%        2.5\n        max        3.0\n\n        Including only string columns in a ``DataFrame`` description.\n\n        >>> df.describe(include=[object])  # doctest: +SKIP\n               object\n        count       3\n        unique      3\n        top         a\n        freq        1\n\n        Including only categorical columns from a ``DataFrame`` description.\n\n        >>> df.describe(include=[\'category\'])\n               categorical\n        count            3\n        unique           3\n        top              d\n        freq             1\n\n        Excluding numeric columns from a ``DataFrame`` description.\n\n        >>> df.describe(exclude=[np.number])  # doctest: +SKIP\n               categorical object\n        count            3      3\n        unique           3      3\n        top              f      a\n        freq             1      1\n\n        Excluding object columns from a ``DataFrame`` description.\n\n        >>> df.describe(exclude=[object])  # doctest: +SKIP\n               categorical  numeric\n        count            3      3.0\n        unique           3      NaN\n        top              f      NaN\n        freq             1      NaN\n        mean           NaN      2.0\n        std            NaN      1.0\n        min            NaN      1.0\n        25%            NaN      1.5\n        50%            NaN      2.0\n        75%            NaN      2.5\n        max            NaN      3.0\n        '
        if ((self.ndim == 2) and (self.columns.size == 0)):
            raise ValueError('Cannot describe a DataFrame without columns')
        if (percentiles is not None):
            percentiles = list(percentiles)
            validate_percentile(percentiles)
            if (0.5 not in percentiles):
                percentiles.append(0.5)
            percentiles = np.asarray(percentiles)
        else:
            percentiles = np.array([0.25, 0.5, 0.75])
        unique_pcts = np.unique(percentiles)
        if (len(unique_pcts) < len(percentiles)):
            raise ValueError('percentiles cannot contain duplicates')
        percentiles = unique_pcts
        formatted_percentiles = format_percentiles(percentiles)

        def describe_numeric_1d(series) -> 'Series':
            stat_index = ((['count', 'mean', 'std', 'min'] + formatted_percentiles) + ['max'])
            d = (([series.count(), series.mean(), series.std(), series.min()] + series.quantile(percentiles).tolist()) + [series.max()])
            return pd.Series(d, index=stat_index, name=series.name)

        def describe_categorical_1d(data) -> 'Series':
            names = ['count', 'unique']
            objcounts = data.value_counts()
            count_unique = len(objcounts[(objcounts != 0)])
            result = [data.count(), count_unique]
            dtype = None
            if (result[1] > 0):
                (top, freq) = (objcounts.index[0], objcounts.iloc[0])
                if is_datetime64_any_dtype(data.dtype):
                    if (self.ndim == 1):
                        stacklevel = 4
                    else:
                        stacklevel = 5
                    warnings.warn('Treating datetime data as categorical rather than numeric in `.describe` is deprecated and will be removed in a future version of pandas. Specify `datetime_is_numeric=True` to silence this warning and adopt the future behavior now.', FutureWarning, stacklevel=stacklevel)
                    tz = data.dt.tz
                    asint = data.dropna().values.view('i8')
                    top = Timestamp(top)
                    if ((top.tzinfo is not None) and (tz is not None)):
                        top = top.tz_convert(tz)
                    else:
                        top = top.tz_localize(tz)
                    names += ['top', 'freq', 'first', 'last']
                    result += [top, freq, Timestamp(asint.min(), tz=tz), Timestamp(asint.max(), tz=tz)]
                else:
                    names += ['top', 'freq']
                    result += [top, freq]
            else:
                names += ['top', 'freq']
                result += [np.nan, np.nan]
                dtype = 'object'
            return pd.Series(result, index=names, name=data.name, dtype=dtype)

        def describe_timestamp_1d(data) -> 'Series':
            stat_index = ((['count', 'mean', 'min'] + formatted_percentiles) + ['max'])
            d = (([data.count(), data.mean(), data.min()] + data.quantile(percentiles).tolist()) + [data.max()])
            return pd.Series(d, index=stat_index, name=data.name)

        def describe_1d(data) -> 'Series':
            if is_bool_dtype(data.dtype):
                return describe_categorical_1d(data)
            elif is_numeric_dtype(data):
                return describe_numeric_1d(data)
            elif (is_datetime64_any_dtype(data.dtype) and datetime_is_numeric):
                return describe_timestamp_1d(data)
            elif is_timedelta64_dtype(data.dtype):
                return describe_numeric_1d(data)
            else:
                return describe_categorical_1d(data)
        if (self.ndim == 1):
            return describe_1d(self)
        elif ((include is None) and (exclude is None)):
            default_include = [np.number]
            if datetime_is_numeric:
                default_include.append('datetime')
            data = self.select_dtypes(include=default_include)
            if (len(data.columns) == 0):
                data = self
        elif (include == 'all'):
            if (exclude is not None):
                msg = "exclude must be None when include is 'all'"
                raise ValueError(msg)
            data = self
        else:
            data = self.select_dtypes(include=include, exclude=exclude)
        ldesc = [describe_1d(s) for (_, s) in data.items()]
        names: List[Label] = []
        ldesc_indexes = sorted((x.index for x in ldesc), key=len)
        for idxnames in ldesc_indexes:
            for name in idxnames:
                if (name not in names):
                    names.append(name)
        d = pd.concat([x.reindex(names, copy=False) for x in ldesc], axis=1, sort=False)
        d.columns = data.columns.copy()
        return d

    @final
    def pct_change(self, periods=1, fill_method='pad', limit=None, freq=None, **kwargs):
        "\n        Percentage change between the current and a prior element.\n\n        Computes the percentage change from the immediately previous row by\n        default. This is useful in comparing the percentage of change in a time\n        series of elements.\n\n        Parameters\n        ----------\n        periods : int, default 1\n            Periods to shift for forming percent change.\n        fill_method : str, default 'pad'\n            How to handle NAs before computing percent changes.\n        limit : int, default None\n            The number of consecutive NAs to fill before stopping.\n        freq : DateOffset, timedelta, or str, optional\n            Increment to use from time series API (e.g. 'M' or BDay()).\n        **kwargs\n            Additional keyword arguments are passed into\n            `DataFrame.shift` or `Series.shift`.\n\n        Returns\n        -------\n        chg : Series or DataFrame\n            The same type as the calling object.\n\n        See Also\n        --------\n        Series.diff : Compute the difference of two elements in a Series.\n        DataFrame.diff : Compute the difference of two elements in a DataFrame.\n        Series.shift : Shift the index by some number of periods.\n        DataFrame.shift : Shift the index by some number of periods.\n\n        Examples\n        --------\n        **Series**\n\n        >>> s = pd.Series([90, 91, 85])\n        >>> s\n        0    90\n        1    91\n        2    85\n        dtype: int64\n\n        >>> s.pct_change()\n        0         NaN\n        1    0.011111\n        2   -0.065934\n        dtype: float64\n\n        >>> s.pct_change(periods=2)\n        0         NaN\n        1         NaN\n        2   -0.055556\n        dtype: float64\n\n        See the percentage change in a Series where filling NAs with last\n        valid observation forward to next valid.\n\n        >>> s = pd.Series([90, 91, None, 85])\n        >>> s\n        0    90.0\n        1    91.0\n        2     NaN\n        3    85.0\n        dtype: float64\n\n        >>> s.pct_change(fill_method='ffill')\n        0         NaN\n        1    0.011111\n        2    0.000000\n        3   -0.065934\n        dtype: float64\n\n        **DataFrame**\n\n        Percentage change in French franc, Deutsche Mark, and Italian lira from\n        1980-01-01 to 1980-03-01.\n\n        >>> df = pd.DataFrame({\n        ...     'FR': [4.0405, 4.0963, 4.3149],\n        ...     'GR': [1.7246, 1.7482, 1.8519],\n        ...     'IT': [804.74, 810.01, 860.13]},\n        ...     index=['1980-01-01', '1980-02-01', '1980-03-01'])\n        >>> df\n                        FR      GR      IT\n        1980-01-01  4.0405  1.7246  804.74\n        1980-02-01  4.0963  1.7482  810.01\n        1980-03-01  4.3149  1.8519  860.13\n\n        >>> df.pct_change()\n                          FR        GR        IT\n        1980-01-01       NaN       NaN       NaN\n        1980-02-01  0.013810  0.013684  0.006549\n        1980-03-01  0.053365  0.059318  0.061876\n\n        Percentage of change in GOOG and APPL stock volume. Shows computing\n        the percentage change between columns.\n\n        >>> df = pd.DataFrame({\n        ...     '2016': [1769950, 30586265],\n        ...     '2015': [1500923, 40912316],\n        ...     '2014': [1371819, 41403351]},\n        ...     index=['GOOG', 'APPL'])\n        >>> df\n                  2016      2015      2014\n        GOOG   1769950   1500923   1371819\n        APPL  30586265  40912316  41403351\n\n        >>> df.pct_change(axis='columns')\n              2016      2015      2014\n        GOOG   NaN -0.151997 -0.086016\n        APPL   NaN  0.337604  0.012002\n        "
        axis = self._get_axis_number(kwargs.pop('axis', self._stat_axis_name))
        if (fill_method is None):
            data = self
        else:
            _data = self.fillna(method=fill_method, axis=axis, limit=limit)
            assert (_data is not None)
            data = _data
        rs = (data.div(data.shift(periods=periods, freq=freq, axis=axis, **kwargs)) - 1)
        if (freq is not None):
            rs = rs.loc[(~ rs.index.duplicated())]
            rs = rs.reindex_like(data)
        return rs

    @final
    def _agg_by_level(self, name, axis=0, level=0, skipna=True, **kwargs):
        if (axis is None):
            raise ValueError("Must specify 'axis' when aggregating by level.")
        grouped = self.groupby(level=level, axis=axis, sort=False)
        if (hasattr(grouped, name) and skipna):
            return getattr(grouped, name)(**kwargs)
        axis = self._get_axis_number(axis)
        method = getattr(type(self), name)
        applyf = (lambda x: method(x, axis=axis, skipna=skipna, **kwargs))
        return grouped.aggregate(applyf)

    @final
    def _logical_func(self, name, func, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        nv.validate_logical_func((), kwargs, fname=name)
        if (level is not None):
            if (bool_only is not None):
                raise NotImplementedError('Option bool_only is not implemented with option level.')
            return self._agg_by_level(name, axis=axis, level=level, skipna=skipna)
        if ((self.ndim > 1) and (axis is None)):
            res = self._logical_func(name, func, axis=0, bool_only=bool_only, skipna=skipna, **kwargs)
            return res._logical_func(name, func, skipna=skipna, **kwargs)
        return self._reduce(func, name=name, axis=axis, skipna=skipna, numeric_only=bool_only, filter_type='bool')

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        return self._logical_func('any', nanops.nanany, axis, bool_only, skipna, level, **kwargs)

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        return self._logical_func('all', nanops.nanall, axis, bool_only, skipna, level, **kwargs)

    @final
    def _accum_func(self, name, func, axis=None, skipna=True, *args, **kwargs):
        skipna = nv.validate_cum_func_with_skipna(skipna, args, kwargs, name)
        if (axis is None):
            axis = self._stat_axis_number
        else:
            axis = self._get_axis_number(axis)
        if (axis == 1):
            return self.T._accum_func(name, func, *args, axis=0, skipna=skipna, **kwargs).T

        def block_accum_func(blk_values):
            values = (blk_values.T if hasattr(blk_values, 'T') else blk_values)
            result = nanops.na_accum_func(values, func, skipna=skipna)
            result = (result.T if hasattr(result, 'T') else result)
            return result
        result = self._mgr.apply(block_accum_func)
        return self._constructor(result).__finalize__(self, method=name)

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        return self._accum_func('cummax', np.maximum.accumulate, axis, skipna, *args, **kwargs)

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        return self._accum_func('cummin', np.minimum.accumulate, axis, skipna, *args, **kwargs)

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        return self._accum_func('cumsum', np.cumsum, axis, skipna, *args, **kwargs)

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        return self._accum_func('cumprod', np.cumprod, axis, skipna, *args, **kwargs)

    @final
    def _stat_function_ddof(self, name, func, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        nv.validate_stat_ddof_func((), kwargs, fname=name)
        if (skipna is None):
            skipna = True
        if (axis is None):
            axis = self._stat_axis_number
        if (level is not None):
            return self._agg_by_level(name, axis=axis, level=level, skipna=skipna, ddof=ddof)
        return self._reduce(func, name, axis=axis, numeric_only=numeric_only, skipna=skipna, ddof=ddof)

    def sem(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        return self._stat_function_ddof('sem', nanops.nansem, axis, skipna, level, ddof, numeric_only, **kwargs)

    def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        return self._stat_function_ddof('var', nanops.nanvar, axis, skipna, level, ddof, numeric_only, **kwargs)

    def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        return self._stat_function_ddof('std', nanops.nanstd, axis, skipna, level, ddof, numeric_only, **kwargs)

    @final
    def _stat_function(self, name, func, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        if (name == 'median'):
            nv.validate_median((), kwargs)
        else:
            nv.validate_stat_func((), kwargs, fname=name)
        if (skipna is None):
            skipna = True
        if (axis is None):
            axis = self._stat_axis_number
        if (level is not None):
            return self._agg_by_level(name, axis=axis, level=level, skipna=skipna)
        return self._reduce(func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only)

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_function('min', nanops.nanmin, axis, skipna, level, numeric_only, **kwargs)

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_function('max', nanops.nanmax, axis, skipna, level, numeric_only, **kwargs)

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_function('mean', nanops.nanmean, axis, skipna, level, numeric_only, **kwargs)

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_function('median', nanops.nanmedian, axis, skipna, level, numeric_only, **kwargs)

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_function('skew', nanops.nanskew, axis, skipna, level, numeric_only, **kwargs)

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_function('kurt', nanops.nankurt, axis, skipna, level, numeric_only, **kwargs)
    kurtosis = kurt

    @final
    def _min_count_stat_function(self, name, func, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        if (name == 'sum'):
            nv.validate_sum((), kwargs)
        elif (name == 'prod'):
            nv.validate_prod((), kwargs)
        else:
            nv.validate_stat_func((), kwargs, fname=name)
        if (skipna is None):
            skipna = True
        if (axis is None):
            axis = self._stat_axis_number
        if (level is not None):
            return self._agg_by_level(name, axis=axis, level=level, skipna=skipna, min_count=min_count)
        return self._reduce(func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        return self._min_count_stat_function('sum', nanops.nansum, axis, skipna, level, numeric_only, min_count, **kwargs)

    def prod(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        return self._min_count_stat_function('prod', nanops.nanprod, axis, skipna, level, numeric_only, min_count, **kwargs)
    product = prod

    def mad(self, axis=None, skipna=None, level=None):
        '\n        {desc}\n\n        Parameters\n        ----------\n        axis : {axis_descr}\n            Axis for the function to be applied on.\n        skipna : bool, default None\n            Exclude NA/null values when computing the result.\n        level : int or level name, default None\n            If the axis is a MultiIndex (hierarchical), count along a\n            particular level, collapsing into a {name1}.\n\n        Returns\n        -------\n        {name1} or {name2} (if level specified)        {see_also}        {examples}\n        '
        if (skipna is None):
            skipna = True
        if (axis is None):
            axis = self._stat_axis_number
        if (level is not None):
            return self._agg_by_level('mad', axis=axis, level=level, skipna=skipna)
        data = self._get_numeric_data()
        if (axis == 0):
            demeaned = (data - data.mean(axis=0))
        else:
            demeaned = data.sub(data.mean(axis=1), axis=0)
        return np.abs(demeaned).mean(axis=axis, skipna=skipna)

    @classmethod
    def _add_numeric_operations(cls):
        '\n        Add the operations to the cls; evaluate the doc strings again\n        '
        (axis_descr, name1, name2) = _doc_params(cls)

        @doc(_bool_doc, desc=_any_desc, name1=name1, name2=name2, axis_descr=axis_descr, see_also=_any_see_also, examples=_any_examples, empty_value=False)
        def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
            return NDFrame.any(self, axis, bool_only, skipna, level, **kwargs)
        cls.any = any

        @doc(_bool_doc, desc=_all_desc, name1=name1, name2=name2, axis_descr=axis_descr, see_also=_all_see_also, examples=_all_examples, empty_value=True)
        def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
            return NDFrame.all(self, axis, bool_only, skipna, level, **kwargs)
        cls.all = all

        @doc(NDFrame.mad, desc='Return the mean absolute deviation of the values over the requested axis.', name1=name1, name2=name2, axis_descr=axis_descr, see_also='', examples='')
        def mad(self, axis=None, skipna=None, level=None):
            return NDFrame.mad(self, axis, skipna, level)
        cls.mad = mad

        @doc(_num_ddof_doc, desc='Return unbiased standard error of the mean over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument', name1=name1, name2=name2, axis_descr=axis_descr)
        def sem(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
            return NDFrame.sem(self, axis, skipna, level, ddof, numeric_only, **kwargs)
        cls.sem = sem

        @doc(_num_ddof_doc, desc='Return unbiased variance over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument', name1=name1, name2=name2, axis_descr=axis_descr)
        def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
            return NDFrame.var(self, axis, skipna, level, ddof, numeric_only, **kwargs)
        cls.var = var

        @doc(_num_ddof_doc, desc='Return sample standard deviation over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument', name1=name1, name2=name2, axis_descr=axis_descr)
        def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
            return NDFrame.std(self, axis, skipna, level, ddof, numeric_only, **kwargs)
        cls.std = std

        @doc(_cnum_doc, desc='minimum', name1=name1, name2=name2, axis_descr=axis_descr, accum_func_name='min', examples=_cummin_examples)
        def cummin(self, axis=None, skipna=True, *args, **kwargs):
            return NDFrame.cummin(self, axis, skipna, *args, **kwargs)
        cls.cummin = cummin

        @doc(_cnum_doc, desc='maximum', name1=name1, name2=name2, axis_descr=axis_descr, accum_func_name='max', examples=_cummax_examples)
        def cummax(self, axis=None, skipna=True, *args, **kwargs):
            return NDFrame.cummax(self, axis, skipna, *args, **kwargs)
        cls.cummax = cummax

        @doc(_cnum_doc, desc='sum', name1=name1, name2=name2, axis_descr=axis_descr, accum_func_name='sum', examples=_cumsum_examples)
        def cumsum(self, axis=None, skipna=True, *args, **kwargs):
            return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)
        cls.cumsum = cumsum

        @doc(_cnum_doc, desc='product', name1=name1, name2=name2, axis_descr=axis_descr, accum_func_name='prod', examples=_cumprod_examples)
        def cumprod(self, axis=None, skipna=True, *args, **kwargs):
            return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)
        cls.cumprod = cumprod

        @doc(_num_doc, desc='Return the sum of the values over the requested axis.\n\nThis is equivalent to the method ``numpy.sum``.', name1=name1, name2=name2, axis_descr=axis_descr, min_count=_min_count_stub, see_also=_stat_func_see_also, examples=_sum_examples)
        def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
            return NDFrame.sum(self, axis, skipna, level, numeric_only, min_count, **kwargs)
        cls.sum = sum

        @doc(_num_doc, desc='Return the product of the values over the requested axis.', name1=name1, name2=name2, axis_descr=axis_descr, min_count=_min_count_stub, see_also=_stat_func_see_also, examples=_prod_examples)
        def prod(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
            return NDFrame.prod(self, axis, skipna, level, numeric_only, min_count, **kwargs)
        cls.prod = prod
        cls.product = prod

        @doc(_num_doc, desc='Return the mean of the values over the requested axis.', name1=name1, name2=name2, axis_descr=axis_descr, min_count='', see_also='', examples='')
        def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
            return NDFrame.mean(self, axis, skipna, level, numeric_only, **kwargs)
        cls.mean = mean

        @doc(_num_doc, desc='Return unbiased skew over requested axis.\n\nNormalized by N-1.', name1=name1, name2=name2, axis_descr=axis_descr, min_count='', see_also='', examples='')
        def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
            return NDFrame.skew(self, axis, skipna, level, numeric_only, **kwargs)
        cls.skew = skew

        @doc(_num_doc, desc="Return unbiased kurtosis over requested axis.\n\nKurtosis obtained using Fisher's definition of\nkurtosis (kurtosis of normal == 0.0). Normalized by N-1.", name1=name1, name2=name2, axis_descr=axis_descr, min_count='', see_also='', examples='')
        def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
            return NDFrame.kurt(self, axis, skipna, level, numeric_only, **kwargs)
        cls.kurt = kurt
        cls.kurtosis = kurt

        @doc(_num_doc, desc='Return the median of the values over the requested axis.', name1=name1, name2=name2, axis_descr=axis_descr, min_count='', see_also='', examples='')
        def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
            return NDFrame.median(self, axis, skipna, level, numeric_only, **kwargs)
        cls.median = median

        @doc(_num_doc, desc='Return the maximum of the values over the requested axis.\n\nIf you want the *index* of the maximum, use ``idxmax``. This isthe equivalent of the ``numpy.ndarray`` method ``argmax``.', name1=name1, name2=name2, axis_descr=axis_descr, min_count='', see_also=_stat_func_see_also, examples=_max_examples)
        def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
            return NDFrame.max(self, axis, skipna, level, numeric_only, **kwargs)
        cls.max = max

        @doc(_num_doc, desc='Return the minimum of the values over the requested axis.\n\nIf you want the *index* of the minimum, use ``idxmin``. This isthe equivalent of the ``numpy.ndarray`` method ``argmin``.', name1=name1, name2=name2, axis_descr=axis_descr, min_count='', see_also=_stat_func_see_also, examples=_min_examples)
        def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
            return NDFrame.min(self, axis, skipna, level, numeric_only, **kwargs)
        cls.min = min

    @final
    @doc(Rolling)
    def rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single'):
        axis = self._get_axis_number(axis)
        if (win_type is not None):
            return Window(self, window=window, min_periods=min_periods, center=center, win_type=win_type, on=on, axis=axis, closed=closed, method=method)
        return Rolling(self, window=window, min_periods=min_periods, center=center, win_type=win_type, on=on, axis=axis, closed=closed, method=method)

    @final
    @doc(Expanding)
    def expanding(self, min_periods=1, center=None, axis=0, method='single'):
        axis = self._get_axis_number(axis)
        if (center is not None):
            warnings.warn('The `center` argument on `expanding` will be removed in the future', FutureWarning, stacklevel=2)
        else:
            center = False
        return Expanding(self, min_periods=min_periods, center=center, axis=axis, method=method)

    @final
    @doc(ExponentialMovingWindow)
    def ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None):
        axis = self._get_axis_number(axis)
        return ExponentialMovingWindow(self, com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis, times=times)

    @final
    def _inplace_method(self, other, op):
        '\n        Wrap arithmetic method to operate inplace.\n        '
        result = op(self, other)
        if ((self.ndim == 1) and result._indexed_same(self) and is_dtype_equal(result.dtype, self.dtype)):
            self._values[:] = result._values
            return self
        self._reset_cacher()
        self._update_inplace(result.reindex_like(self, copy=False), verify_is_copy=False)
        return self

    def __iadd__(self, other):
        return self._inplace_method(other, type(self).__add__)

    def __isub__(self, other):
        return self._inplace_method(other, type(self).__sub__)

    def __imul__(self, other):
        return self._inplace_method(other, type(self).__mul__)

    def __itruediv__(self, other):
        return self._inplace_method(other, type(self).__truediv__)

    def __ifloordiv__(self, other):
        return self._inplace_method(other, type(self).__floordiv__)

    def __imod__(self, other):
        return self._inplace_method(other, type(self).__mod__)

    def __ipow__(self, other):
        return self._inplace_method(other, type(self).__pow__)

    def __iand__(self, other):
        return self._inplace_method(other, type(self).__and__)

    def __ior__(self, other):
        return self._inplace_method(other, type(self).__or__)

    def __ixor__(self, other):
        return self._inplace_method(other, type(self).__xor__)

    @final
    def _find_valid_index(self, how):
        "\n        Retrieves the index of the first valid value.\n\n        Parameters\n        ----------\n        how : {'first', 'last'}\n            Use this parameter to change between the first or last valid index.\n\n        Returns\n        -------\n        idx_first_valid : type of index\n        "
        idxpos = find_valid_index(self._values, how)
        if (idxpos is None):
            return None
        return self.index[idxpos]

    @final
    @doc(position='first', klass=_shared_doc_kwargs['klass'])
    def first_valid_index(self):
        '\n        Return index for {position} non-NA/null value.\n\n        Returns\n        -------\n        scalar : type of index\n\n        Notes\n        -----\n        If all elements are non-NA/null, returns None.\n        Also returns None for empty {klass}.\n        '
        return self._find_valid_index('first')

    @final
    @doc(first_valid_index, position='last', klass=_shared_doc_kwargs['klass'])
    def last_valid_index(self):
        return self._find_valid_index('last')

def _doc_params(cls):
    'Return a tuple of the doc params.'
    axis_descr = f"{{{', '.join((f'{a} ({i})' for (i, a) in enumerate(cls._AXIS_ORDERS)))}}}"
    name = (cls._constructor_sliced.__name__ if (cls._AXIS_LEN > 1) else 'scalar')
    name2 = cls.__name__
    return (axis_descr, name, name2)
_num_doc = '\n{desc}\n\nParameters\n----------\naxis : {axis_descr}\n    Axis for the function to be applied on.\nskipna : bool, default True\n    Exclude NA/null values when computing the result.\nlevel : int or level name, default None\n    If the axis is a MultiIndex (hierarchical), count along a\n    particular level, collapsing into a {name1}.\nnumeric_only : bool, default None\n    Include only float, int, boolean columns. If None, will attempt to use\n    everything, then use only numeric data. Not implemented for Series.\n{min_count}**kwargs\n    Additional keyword arguments to be passed to the function.\n\nReturns\n-------\n{name1} or {name2} (if level specified){see_also}{examples}\n'
_num_ddof_doc = '\n{desc}\n\nParameters\n----------\naxis : {axis_descr}\nskipna : bool, default True\n    Exclude NA/null values. If an entire row/column is NA, the result\n    will be NA.\nlevel : int or level name, default None\n    If the axis is a MultiIndex (hierarchical), count along a\n    particular level, collapsing into a {name1}.\nddof : int, default 1\n    Delta Degrees of Freedom. The divisor used in calculations is N - ddof,\n    where N represents the number of elements.\nnumeric_only : bool, default None\n    Include only float, int, boolean columns. If None, will attempt to use\n    everything, then use only numeric data. Not implemented for Series.\n\nReturns\n-------\n{name1} or {name2} (if level specified)\n\nNotes\n-----\nTo have the same behaviour as `numpy.std`, use `ddof=0` (instead of the\ndefault `ddof=1`)\n'
_bool_doc = "\n{desc}\n\nParameters\n----------\naxis : {{0 or 'index', 1 or 'columns', None}}, default 0\n    Indicate which axis or axes should be reduced.\n\n    * 0 / 'index' : reduce the index, return a Series whose index is the\n      original column labels.\n    * 1 / 'columns' : reduce the columns, return a Series whose index is the\n      original index.\n    * None : reduce all axes, return a scalar.\n\nbool_only : bool, default None\n    Include only boolean columns. If None, will attempt to use everything,\n    then use only boolean data. Not implemented for Series.\nskipna : bool, default True\n    Exclude NA/null values. If the entire row/column is NA and skipna is\n    True, then the result will be {empty_value}, as for an empty row/column.\n    If skipna is False, then NA are treated as True, because these are not\n    equal to zero.\nlevel : int or level name, default None\n    If the axis is a MultiIndex (hierarchical), count along a\n    particular level, collapsing into a {name1}.\n**kwargs : any, default None\n    Additional keywords have no effect but might be accepted for\n    compatibility with NumPy.\n\nReturns\n-------\n{name1} or {name2}\n    If level is specified, then, {name2} is returned; otherwise, {name1}\n    is returned.\n\n{see_also}\n{examples}"
_all_desc = 'Return whether all elements are True, potentially over an axis.\n\nReturns True unless there at least one element within a series or\nalong a Dataframe axis that is False or equivalent (e.g. zero or\nempty).'
_all_examples = "Examples\n--------\n**Series**\n\n>>> pd.Series([True, True]).all()\nTrue\n>>> pd.Series([True, False]).all()\nFalse\n>>> pd.Series([]).all()\nTrue\n>>> pd.Series([np.nan]).all()\nTrue\n>>> pd.Series([np.nan]).all(skipna=False)\nTrue\n\n**DataFrames**\n\nCreate a dataframe from a dictionary.\n\n>>> df = pd.DataFrame({'col1': [True, True], 'col2': [True, False]})\n>>> df\n   col1   col2\n0  True   True\n1  True  False\n\nDefault behaviour checks if column-wise values all return True.\n\n>>> df.all()\ncol1     True\ncol2    False\ndtype: bool\n\nSpecify ``axis='columns'`` to check if row-wise values all return True.\n\n>>> df.all(axis='columns')\n0     True\n1    False\ndtype: bool\n\nOr ``axis=None`` for whether every value is True.\n\n>>> df.all(axis=None)\nFalse\n"
_all_see_also = 'See Also\n--------\nSeries.all : Return True if all elements are True.\nDataFrame.any : Return True if one (or more) elements are True.\n'
_cnum_doc = "\nReturn cumulative {desc} over a DataFrame or Series axis.\n\nReturns a DataFrame or Series of the same size containing the cumulative\n{desc}.\n\nParameters\n----------\naxis : {{0 or 'index', 1 or 'columns'}}, default 0\n    The index or the name of the axis. 0 is equivalent to None or 'index'.\nskipna : bool, default True\n    Exclude NA/null values. If an entire row/column is NA, the result\n    will be NA.\n*args, **kwargs\n    Additional keywords have no effect but might be accepted for\n    compatibility with NumPy.\n\nReturns\n-------\n{name1} or {name2}\n    Return cumulative {desc} of {name1} or {name2}.\n\nSee Also\n--------\ncore.window.Expanding.{accum_func_name} : Similar functionality\n    but ignores ``NaN`` values.\n{name2}.{accum_func_name} : Return the {desc} over\n    {name2} axis.\n{name2}.cummax : Return cumulative maximum over {name2} axis.\n{name2}.cummin : Return cumulative minimum over {name2} axis.\n{name2}.cumsum : Return cumulative sum over {name2} axis.\n{name2}.cumprod : Return cumulative product over {name2} axis.\n\n{examples}"
_cummin_examples = "Examples\n--------\n**Series**\n\n>>> s = pd.Series([2, np.nan, 5, -1, 0])\n>>> s\n0    2.0\n1    NaN\n2    5.0\n3   -1.0\n4    0.0\ndtype: float64\n\nBy default, NA values are ignored.\n\n>>> s.cummin()\n0    2.0\n1    NaN\n2    2.0\n3   -1.0\n4   -1.0\ndtype: float64\n\nTo include NA values in the operation, use ``skipna=False``\n\n>>> s.cummin(skipna=False)\n0    2.0\n1    NaN\n2    NaN\n3    NaN\n4    NaN\ndtype: float64\n\n**DataFrame**\n\n>>> df = pd.DataFrame([[2.0, 1.0],\n...                    [3.0, np.nan],\n...                    [1.0, 0.0]],\n...                    columns=list('AB'))\n>>> df\n     A    B\n0  2.0  1.0\n1  3.0  NaN\n2  1.0  0.0\n\nBy default, iterates over rows and finds the minimum\nin each column. This is equivalent to ``axis=None`` or ``axis='index'``.\n\n>>> df.cummin()\n     A    B\n0  2.0  1.0\n1  2.0  NaN\n2  1.0  0.0\n\nTo iterate over columns and find the minimum in each row,\nuse ``axis=1``\n\n>>> df.cummin(axis=1)\n     A    B\n0  2.0  1.0\n1  3.0  NaN\n2  1.0  0.0\n"
_cumsum_examples = "Examples\n--------\n**Series**\n\n>>> s = pd.Series([2, np.nan, 5, -1, 0])\n>>> s\n0    2.0\n1    NaN\n2    5.0\n3   -1.0\n4    0.0\ndtype: float64\n\nBy default, NA values are ignored.\n\n>>> s.cumsum()\n0    2.0\n1    NaN\n2    7.0\n3    6.0\n4    6.0\ndtype: float64\n\nTo include NA values in the operation, use ``skipna=False``\n\n>>> s.cumsum(skipna=False)\n0    2.0\n1    NaN\n2    NaN\n3    NaN\n4    NaN\ndtype: float64\n\n**DataFrame**\n\n>>> df = pd.DataFrame([[2.0, 1.0],\n...                    [3.0, np.nan],\n...                    [1.0, 0.0]],\n...                    columns=list('AB'))\n>>> df\n     A    B\n0  2.0  1.0\n1  3.0  NaN\n2  1.0  0.0\n\nBy default, iterates over rows and finds the sum\nin each column. This is equivalent to ``axis=None`` or ``axis='index'``.\n\n>>> df.cumsum()\n     A    B\n0  2.0  1.0\n1  5.0  NaN\n2  6.0  1.0\n\nTo iterate over columns and find the sum in each row,\nuse ``axis=1``\n\n>>> df.cumsum(axis=1)\n     A    B\n0  2.0  3.0\n1  3.0  NaN\n2  1.0  1.0\n"
_cumprod_examples = "Examples\n--------\n**Series**\n\n>>> s = pd.Series([2, np.nan, 5, -1, 0])\n>>> s\n0    2.0\n1    NaN\n2    5.0\n3   -1.0\n4    0.0\ndtype: float64\n\nBy default, NA values are ignored.\n\n>>> s.cumprod()\n0     2.0\n1     NaN\n2    10.0\n3   -10.0\n4    -0.0\ndtype: float64\n\nTo include NA values in the operation, use ``skipna=False``\n\n>>> s.cumprod(skipna=False)\n0    2.0\n1    NaN\n2    NaN\n3    NaN\n4    NaN\ndtype: float64\n\n**DataFrame**\n\n>>> df = pd.DataFrame([[2.0, 1.0],\n...                    [3.0, np.nan],\n...                    [1.0, 0.0]],\n...                    columns=list('AB'))\n>>> df\n     A    B\n0  2.0  1.0\n1  3.0  NaN\n2  1.0  0.0\n\nBy default, iterates over rows and finds the product\nin each column. This is equivalent to ``axis=None`` or ``axis='index'``.\n\n>>> df.cumprod()\n     A    B\n0  2.0  1.0\n1  6.0  NaN\n2  6.0  0.0\n\nTo iterate over columns and find the product in each row,\nuse ``axis=1``\n\n>>> df.cumprod(axis=1)\n     A    B\n0  2.0  2.0\n1  3.0  NaN\n2  1.0  0.0\n"
_cummax_examples = "Examples\n--------\n**Series**\n\n>>> s = pd.Series([2, np.nan, 5, -1, 0])\n>>> s\n0    2.0\n1    NaN\n2    5.0\n3   -1.0\n4    0.0\ndtype: float64\n\nBy default, NA values are ignored.\n\n>>> s.cummax()\n0    2.0\n1    NaN\n2    5.0\n3    5.0\n4    5.0\ndtype: float64\n\nTo include NA values in the operation, use ``skipna=False``\n\n>>> s.cummax(skipna=False)\n0    2.0\n1    NaN\n2    NaN\n3    NaN\n4    NaN\ndtype: float64\n\n**DataFrame**\n\n>>> df = pd.DataFrame([[2.0, 1.0],\n...                    [3.0, np.nan],\n...                    [1.0, 0.0]],\n...                    columns=list('AB'))\n>>> df\n     A    B\n0  2.0  1.0\n1  3.0  NaN\n2  1.0  0.0\n\nBy default, iterates over rows and finds the maximum\nin each column. This is equivalent to ``axis=None`` or ``axis='index'``.\n\n>>> df.cummax()\n     A    B\n0  2.0  1.0\n1  3.0  NaN\n2  3.0  1.0\n\nTo iterate over columns and find the maximum in each row,\nuse ``axis=1``\n\n>>> df.cummax(axis=1)\n     A    B\n0  2.0  2.0\n1  3.0  NaN\n2  1.0  1.0\n"
_any_see_also = 'See Also\n--------\nnumpy.any : Numpy version of this method.\nSeries.any : Return whether any element is True.\nSeries.all : Return whether all elements are True.\nDataFrame.any : Return whether any element is True over requested axis.\nDataFrame.all : Return whether all elements are True over requested axis.\n'
_any_desc = 'Return whether any element is True, potentially over an axis.\n\nReturns False unless there is at least one element within a series or\nalong a Dataframe axis that is True or equivalent (e.g. non-zero or\nnon-empty).'
_any_examples = 'Examples\n--------\n**Series**\n\nFor Series input, the output is a scalar indicating whether any element\nis True.\n\n>>> pd.Series([False, False]).any()\nFalse\n>>> pd.Series([True, False]).any()\nTrue\n>>> pd.Series([]).any()\nFalse\n>>> pd.Series([np.nan]).any()\nFalse\n>>> pd.Series([np.nan]).any(skipna=False)\nTrue\n\n**DataFrame**\n\nWhether each column contains at least one True element (the default).\n\n>>> df = pd.DataFrame({"A": [1, 2], "B": [0, 2], "C": [0, 0]})\n>>> df\n   A  B  C\n0  1  0  0\n1  2  2  0\n\n>>> df.any()\nA     True\nB     True\nC    False\ndtype: bool\n\nAggregating over the columns.\n\n>>> df = pd.DataFrame({"A": [True, False], "B": [1, 2]})\n>>> df\n       A  B\n0   True  1\n1  False  2\n\n>>> df.any(axis=\'columns\')\n0    True\n1    True\ndtype: bool\n\n>>> df = pd.DataFrame({"A": [True, False], "B": [1, 0]})\n>>> df\n       A  B\n0   True  1\n1  False  0\n\n>>> df.any(axis=\'columns\')\n0    True\n1    False\ndtype: bool\n\nAggregating over the entire DataFrame with ``axis=None``.\n\n>>> df.any(axis=None)\nTrue\n\n`any` for an empty DataFrame is an empty Series.\n\n>>> pd.DataFrame([]).any()\nSeries([], dtype: bool)\n'
_shared_docs['stat_func_example'] = "\n\nExamples\n--------\n>>> idx = pd.MultiIndex.from_arrays([\n...     ['warm', 'warm', 'cold', 'cold'],\n...     ['dog', 'falcon', 'fish', 'spider']],\n...     names=['blooded', 'animal'])\n>>> s = pd.Series([4, 2, 0, 8], name='legs', index=idx)\n>>> s\nblooded  animal\nwarm     dog       4\n         falcon    2\ncold     fish      0\n         spider    8\nName: legs, dtype: int64\n\n>>> s.{stat_func}()\n{default_output}\n\n{verb} using level names, as well as indices.\n\n>>> s.{stat_func}(level='blooded')\nblooded\nwarm    {level_output_0}\ncold    {level_output_1}\nName: legs, dtype: int64\n\n>>> s.{stat_func}(level=0)\nblooded\nwarm    {level_output_0}\ncold    {level_output_1}\nName: legs, dtype: int64"
_sum_examples = _shared_docs['stat_func_example'].format(stat_func='sum', verb='Sum', default_output=14, level_output_0=6, level_output_1=8)
_sum_examples += "\n\nBy default, the sum of an empty or all-NA Series is ``0``.\n\n>>> pd.Series([]).sum()  # min_count=0 is the default\n0.0\n\nThis can be controlled with the ``min_count`` parameter. For example, if\nyou'd like the sum of an empty series to be NaN, pass ``min_count=1``.\n\n>>> pd.Series([]).sum(min_count=1)\nnan\n\nThanks to the ``skipna`` parameter, ``min_count`` handles all-NA and\nempty series identically.\n\n>>> pd.Series([np.nan]).sum()\n0.0\n\n>>> pd.Series([np.nan]).sum(min_count=1)\nnan"
_max_examples = _shared_docs['stat_func_example'].format(stat_func='max', verb='Max', default_output=8, level_output_0=4, level_output_1=8)
_min_examples = _shared_docs['stat_func_example'].format(stat_func='min', verb='Min', default_output=0, level_output_0=2, level_output_1=0)
_stat_func_see_also = '\n\nSee Also\n--------\nSeries.sum : Return the sum.\nSeries.min : Return the minimum.\nSeries.max : Return the maximum.\nSeries.idxmin : Return the index of the minimum.\nSeries.idxmax : Return the index of the maximum.\nDataFrame.sum : Return the sum over the requested axis.\nDataFrame.min : Return the minimum over the requested axis.\nDataFrame.max : Return the maximum over the requested axis.\nDataFrame.idxmin : Return the index of the minimum over the requested axis.\nDataFrame.idxmax : Return the index of the maximum over the requested axis.'
_prod_examples = '\n\nExamples\n--------\nBy default, the product of an empty or all-NA Series is ``1``\n\n>>> pd.Series([]).prod()\n1.0\n\nThis can be controlled with the ``min_count`` parameter\n\n>>> pd.Series([]).prod(min_count=1)\nnan\n\nThanks to the ``skipna`` parameter, ``min_count`` handles all-NA and\nempty series identically.\n\n>>> pd.Series([np.nan]).prod()\n1.0\n\n>>> pd.Series([np.nan]).prod(min_count=1)\nnan'
_min_count_stub = 'min_count : int, default 0\n    The required number of valid values to perform the operation. If fewer than\n    ``min_count`` non-NA values are present the result will be NA.\n'
