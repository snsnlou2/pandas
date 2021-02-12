
'\nProvide user facing operators for doing the split part of the\nsplit-apply-combine paradigm.\n'
from typing import Dict, Hashable, List, Optional, Set, Tuple
import warnings
import numpy as np
from pandas._typing import FrameOrSeries, Label, final
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_categorical_dtype, is_datetime64_dtype, is_list_like, is_scalar, is_timedelta64_dtype
import pandas.core.algorithms as algorithms
from pandas.core.arrays import Categorical, ExtensionArray
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby, recode_from_groupby
from pandas.core.indexes.api import CategoricalIndex, Index, MultiIndex
from pandas.core.series import Series
from pandas.io.formats.printing import pprint_thing

class Grouper():
    '\n    A Grouper allows the user to specify a groupby instruction for an object.\n\n    This specification will select a column via the key parameter, or if the\n    level and/or axis parameters are given, a level of the index of the target\n    object.\n\n    If `axis` and/or `level` are passed as keywords to both `Grouper` and\n    `groupby`, the values passed to `Grouper` take precedence.\n\n    Parameters\n    ----------\n    key : str, defaults to None\n        Groupby key, which selects the grouping column of the target.\n    level : name/number, defaults to None\n        The level for the target index.\n    freq : str / frequency object, defaults to None\n        This will groupby the specified frequency if the target selection\n        (via key or level) is a datetime-like object. For full specification\n        of available frequencies, please see `here\n        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.\n    axis : str, int, defaults to 0\n        Number/name of the axis.\n    sort : bool, default to False\n        Whether to sort the resulting labels.\n    closed : {\'left\' or \'right\'}\n        Closed end of interval. Only when `freq` parameter is passed.\n    label : {\'left\' or \'right\'}\n        Interval boundary to use for labeling.\n        Only when `freq` parameter is passed.\n    convention : {\'start\', \'end\', \'e\', \'s\'}\n        If grouper is PeriodIndex and `freq` parameter is passed.\n    base : int, default 0\n        Only when `freq` parameter is passed.\n        For frequencies that evenly subdivide 1 day, the "origin" of the\n        aggregated intervals. For example, for \'5min\' frequency, base could\n        range from 0 through 4. Defaults to 0.\n\n        .. deprecated:: 1.1.0\n            The new arguments that you should use are \'offset\' or \'origin\'.\n\n    loffset : str, DateOffset, timedelta object\n        Only when `freq` parameter is passed.\n\n        .. deprecated:: 1.1.0\n            loffset is only working for ``.resample(...)`` and not for\n            Grouper (:issue:`28302`).\n            However, loffset is also deprecated for ``.resample(...)``\n            See: :class:`DataFrame.resample`\n\n    origin : {{\'epoch\', \'start\', \'start_day\', \'end\', \'end_day\'}}, Timestamp\n        or str, default \'start_day\'\n        The timestamp on which to adjust the grouping. The timezone of origin must\n        match the timezone of the index.\n        If a timestamp is not used, these values are also supported:\n\n        - \'epoch\': `origin` is 1970-01-01\n        - \'start\': `origin` is the first value of the timeseries\n        - \'start_day\': `origin` is the first day at midnight of the timeseries\n\n        .. versionadded:: 1.1.0\n\n        - \'end\': `origin` is the last value of the timeseries\n        - \'end_day\': `origin` is the ceiling midnight of the last day\n\n        .. versionadded:: 1.3.0\n\n    offset : Timedelta or str, default is None\n        An offset timedelta added to the origin.\n\n        .. versionadded:: 1.1.0\n\n    dropna : bool, default True\n        If True, and if group keys contain NA values, NA values together with\n        row/column will be dropped. If False, NA values will also be treated as\n        the key in groups.\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    A specification for a groupby instruction\n\n    Examples\n    --------\n    Syntactic sugar for ``df.groupby(\'A\')``\n\n    >>> df = pd.DataFrame(\n    ...     {\n    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],\n    ...         "Speed": [100, 5, 200, 300, 15],\n    ...     }\n    ... )\n    >>> df\n       Animal  Speed\n    0  Falcon    100\n    1  Parrot      5\n    2  Falcon    200\n    3  Falcon    300\n    4  Parrot     15\n    >>> df.groupby(pd.Grouper(key="Animal")).mean()\n            Speed\n    Animal\n    Falcon    200\n    Parrot     10\n\n    Specify a resample operation on the column \'Publish date\'\n\n    >>> df = pd.DataFrame(\n    ...    {\n    ...        "Publish date": [\n    ...             pd.Timestamp("2000-01-02"),\n    ...             pd.Timestamp("2000-01-02"),\n    ...             pd.Timestamp("2000-01-09"),\n    ...             pd.Timestamp("2000-01-16")\n    ...         ],\n    ...         "ID": [0, 1, 2, 3],\n    ...         "Price": [10, 20, 30, 40]\n    ...     }\n    ... )\n    >>> df\n      Publish date  ID  Price\n    0   2000-01-02   0     10\n    1   2000-01-02   1     20\n    2   2000-01-09   2     30\n    3   2000-01-16   3     40\n    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()\n                   ID  Price\n    Publish date\n    2000-01-02    0.5   15.0\n    2000-01-09    2.0   30.0\n    2000-01-16    3.0   40.0\n\n    If you want to adjust the start of the bins based on a fixed timestamp:\n\n    >>> start, end = \'2000-10-01 23:30:00\', \'2000-10-02 00:30:00\'\n    >>> rng = pd.date_range(start, end, freq=\'7min\')\n    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)\n    >>> ts\n    2000-10-01 23:30:00     0\n    2000-10-01 23:37:00     3\n    2000-10-01 23:44:00     6\n    2000-10-01 23:51:00     9\n    2000-10-01 23:58:00    12\n    2000-10-02 00:05:00    15\n    2000-10-02 00:12:00    18\n    2000-10-02 00:19:00    21\n    2000-10-02 00:26:00    24\n    Freq: 7T, dtype: int64\n\n    >>> ts.groupby(pd.Grouper(freq=\'17min\')).sum()\n    2000-10-01 23:14:00     0\n    2000-10-01 23:31:00     9\n    2000-10-01 23:48:00    21\n    2000-10-02 00:05:00    54\n    2000-10-02 00:22:00    24\n    Freq: 17T, dtype: int64\n\n    >>> ts.groupby(pd.Grouper(freq=\'17min\', origin=\'epoch\')).sum()\n    2000-10-01 23:18:00     0\n    2000-10-01 23:35:00    18\n    2000-10-01 23:52:00    27\n    2000-10-02 00:09:00    39\n    2000-10-02 00:26:00    24\n    Freq: 17T, dtype: int64\n\n    >>> ts.groupby(pd.Grouper(freq=\'17min\', origin=\'2000-01-01\')).sum()\n    2000-10-01 23:24:00     3\n    2000-10-01 23:41:00    15\n    2000-10-01 23:58:00    45\n    2000-10-02 00:15:00    45\n    Freq: 17T, dtype: int64\n\n    If you want to adjust the start of the bins with an `offset` Timedelta, the two\n    following lines are equivalent:\n\n    >>> ts.groupby(pd.Grouper(freq=\'17min\', origin=\'start\')).sum()\n    2000-10-01 23:30:00     9\n    2000-10-01 23:47:00    21\n    2000-10-02 00:04:00    54\n    2000-10-02 00:21:00    24\n    Freq: 17T, dtype: int64\n\n    >>> ts.groupby(pd.Grouper(freq=\'17min\', offset=\'23h30min\')).sum()\n    2000-10-01 23:30:00     9\n    2000-10-01 23:47:00    21\n    2000-10-02 00:04:00    54\n    2000-10-02 00:21:00    24\n    Freq: 17T, dtype: int64\n\n    To replace the use of the deprecated `base` argument, you can now use `offset`,\n    in this example it is equivalent to have `base=2`:\n\n    >>> ts.groupby(pd.Grouper(freq=\'17min\', offset=\'2min\')).sum()\n    2000-10-01 23:16:00     0\n    2000-10-01 23:33:00     9\n    2000-10-01 23:50:00    36\n    2000-10-02 00:07:00    39\n    2000-10-02 00:24:00    24\n    Freq: 17T, dtype: int64\n    '
    _attributes = ('key', 'level', 'freq', 'axis', 'sort')

    def __new__(cls, *args, **kwargs):
        if (kwargs.get('freq') is not None):
            from pandas.core.resample import TimeGrouper
            _check_deprecated_resample_kwargs(kwargs, origin=cls)
            cls = TimeGrouper
        return super().__new__(cls)

    def __init__(self, key=None, level=None, freq=None, axis=0, sort=False, dropna=True):
        self.key = key
        self.level = level
        self.freq = freq
        self.axis = axis
        self.sort = sort
        self.grouper = None
        self.obj = None
        self.indexer = None
        self.binner = None
        self._grouper = None
        self._indexer = None
        self.dropna = dropna

    @final
    @property
    def ax(self):
        return self.grouper

    def _get_grouper(self, obj, validate=True):
        '\n        Parameters\n        ----------\n        obj : the subject object\n        validate : boolean, default True\n            if True, validate the grouper\n\n        Returns\n        -------\n        a tuple of binner, grouper, obj (possibly sorted)\n        '
        self._set_grouper(obj)
        (self.grouper, _, self.obj) = get_grouper(self.obj, [self.key], axis=self.axis, level=self.level, sort=self.sort, validate=validate, dropna=self.dropna)
        return (self.binner, self.grouper, self.obj)

    @final
    def _set_grouper(self, obj, sort=False):
        '\n        given an object and the specifications, setup the internal grouper\n        for this particular specification\n\n        Parameters\n        ----------\n        obj : Series or DataFrame\n        sort : bool, default False\n            whether the resulting grouper should be sorted\n        '
        assert (obj is not None)
        if ((self.key is not None) and (self.level is not None)):
            raise ValueError('The Grouper cannot specify both a key and a level!')
        if (self._grouper is None):
            self._grouper = self.grouper
            self._indexer = self.indexer
        if (self.key is not None):
            key = self.key
            if ((getattr(self.grouper, 'name', None) == key) and isinstance(obj, Series)):
                assert (self._grouper is not None)
                if (self._indexer is not None):
                    reverse_indexer = self._indexer.argsort()
                    unsorted_ax = self._grouper.take(reverse_indexer)
                    ax = unsorted_ax.take(obj.index)
                else:
                    ax = self._grouper.take(obj.index)
            else:
                if (key not in obj._info_axis):
                    raise KeyError(f'The grouper name {key} is not found')
                ax = Index(obj[key], name=key)
        else:
            ax = obj._get_axis(self.axis)
            if (self.level is not None):
                level = self.level
                if isinstance(ax, MultiIndex):
                    level = ax._get_level_number(level)
                    ax = Index(ax._get_level_values(level), name=ax.names[level])
                elif (level not in (0, ax.name)):
                    raise ValueError(f'The level {level} is not valid')
        if ((self.sort or sort) and (not ax.is_monotonic)):
            indexer = self.indexer = ax.array.argsort(kind='mergesort', na_position='first')
            ax = ax.take(indexer)
            obj = obj.take(indexer, axis=self.axis)
        self.obj = obj
        self.grouper = ax
        return self.grouper

    @final
    @property
    def groups(self):
        return self.grouper.groups

    @final
    def __repr__(self):
        attrs_list = (f'{attr_name}={repr(getattr(self, attr_name))}' for attr_name in self._attributes if (getattr(self, attr_name) is not None))
        attrs = ', '.join(attrs_list)
        cls_name = type(self).__name__
        return f'{cls_name}({attrs})'

@final
class Grouping():
    '\n    Holds the grouping information for a single key\n\n    Parameters\n    ----------\n    index : Index\n    grouper :\n    obj : DataFrame or Series\n    name : Label\n    level :\n    observed : bool, default False\n        If we are a Categorical, use the observed values\n    in_axis : if the Grouping is a column in self.obj and hence among\n        Groupby.exclusions list\n\n    Returns\n    -------\n    **Attributes**:\n      * indices : dict of {group -> index_list}\n      * codes : ndarray, group codes\n      * group_index : unique groups\n      * groups : dict of {group -> label_list}\n    '

    def __init__(self, index, grouper=None, obj=None, name=None, level=None, sort=True, observed=False, in_axis=False, dropna=True):
        self.name = name
        self.level = level
        self.grouper = _convert_grouper(index, grouper)
        self.all_grouper = None
        self.index = index
        self.sort = sort
        self.obj = obj
        self.observed = observed
        self.in_axis = in_axis
        self.dropna = dropna
        if (isinstance(grouper, (Series, Index)) and (name is None)):
            self.name = grouper.name
        if isinstance(grouper, MultiIndex):
            self.grouper = grouper._values
        if (level is not None):
            if (not isinstance(level, int)):
                if (level not in index.names):
                    raise AssertionError(f'Level {level} not in index')
                level = index.names.index(level)
            if (self.name is None):
                self.name = index.names[level]
            (self.grouper, self._codes, self._group_index) = index._get_grouper_for_level(self.grouper, level)
        elif isinstance(self.grouper, Grouper):
            (_, grouper, _) = self.grouper._get_grouper(self.obj, validate=False)
            if (self.name is None):
                self.name = grouper.result_index.name
            self.obj = self.grouper.obj
            self.grouper = grouper._get_grouper()
        else:
            if ((self.grouper is None) and (self.name is not None) and (self.obj is not None)):
                self.grouper = self.obj[self.name]
            elif isinstance(self.grouper, (list, tuple)):
                self.grouper = com.asarray_tuplesafe(self.grouper)
            elif is_categorical_dtype(self.grouper):
                (self.grouper, self.all_grouper) = recode_for_groupby(self.grouper, self.sort, observed)
                categories = self.grouper.categories
                self._codes = self.grouper.codes
                if observed:
                    codes = algorithms.unique1d(self.grouper.codes)
                    codes = codes[(codes != (- 1))]
                    if (sort or self.grouper.ordered):
                        codes = np.sort(codes)
                else:
                    codes = np.arange(len(categories))
                self._group_index = CategoricalIndex(Categorical.from_codes(codes=codes, categories=categories, ordered=self.grouper.ordered), name=self.name)
            if isinstance(self.grouper, Grouping):
                self.grouper = self.grouper.grouper
            elif (not isinstance(self.grouper, (Series, Index, ExtensionArray, np.ndarray))):
                if (getattr(self.grouper, 'ndim', 1) != 1):
                    t = (self.name or str(type(self.grouper)))
                    raise ValueError(f"Grouper for '{t}' not 1-dimensional")
                self.grouper = self.index.map(self.grouper)
                if (not (hasattr(self.grouper, '__len__') and (len(self.grouper) == len(self.index)))):
                    grper = pprint_thing(self.grouper)
                    errmsg = f'''Grouper result violates len(labels) == len(data)
result: {grper}'''
                    self.grouper = None
                    raise AssertionError(errmsg)
        if (getattr(self.grouper, 'dtype', None) is not None):
            if is_datetime64_dtype(self.grouper):
                self.grouper = self.grouper.astype('datetime64[ns]')
            elif is_timedelta64_dtype(self.grouper):
                self.grouper = self.grouper.astype('timedelta64[ns]')

    def __repr__(self):
        return f'Grouping({self.name})'

    def __iter__(self):
        return iter(self.indices)
    _codes = None
    _group_index = None

    @property
    def ngroups(self):
        return len(self.group_index)

    @cache_readonly
    def indices(self):
        if isinstance(self.grouper, ops.BaseGrouper):
            return self.grouper.indices
        values = Categorical(self.grouper)
        return values._reverse_indexer()

    @property
    def codes(self):
        if (self._codes is None):
            self._make_codes()
        return self._codes

    @cache_readonly
    def result_index(self):
        if (self.all_grouper is not None):
            group_idx = self.group_index
            assert isinstance(group_idx, CategoricalIndex)
            return recode_from_groupby(self.all_grouper, self.sort, group_idx)
        return self.group_index

    @property
    def group_index(self):
        if (self._group_index is None):
            self._make_codes()
        assert (self._group_index is not None)
        return self._group_index

    def _make_codes(self):
        if ((self._codes is not None) and (self._group_index is not None)):
            return
        if isinstance(self.grouper, ops.BaseGrouper):
            codes = self.grouper.codes_info
            uniques = self.grouper.result_index
        else:
            if (not self.dropna):
                na_sentinel = None
            else:
                na_sentinel = (- 1)
            (codes, uniques) = algorithms.factorize(self.grouper, sort=self.sort, na_sentinel=na_sentinel)
            uniques = Index(uniques, name=self.name)
        self._codes = codes
        self._group_index = uniques

    @cache_readonly
    def groups(self):
        return self.index.groupby(Categorical.from_codes(self.codes, self.group_index))

def get_grouper(obj, key=None, axis=0, level=None, sort=True, observed=False, mutated=False, validate=True, dropna=True):
    "\n    Create and return a BaseGrouper, which is an internal\n    mapping of how to create the grouper indexers.\n    This may be composed of multiple Grouping objects, indicating\n    multiple groupers\n\n    Groupers are ultimately index mappings. They can originate as:\n    index mappings, keys to columns, functions, or Groupers\n\n    Groupers enable local references to axis,level,sort, while\n    the passed in axis, level, and sort are 'global'.\n\n    This routine tries to figure out what the passing in references\n    are and then creates a Grouping for each one, combined into\n    a BaseGrouper.\n\n    If observed & we have a categorical grouper, only show the observed\n    values.\n\n    If validate, then check for key/level overlaps.\n\n    "
    group_axis = obj._get_axis(axis)
    if (level is not None):
        if isinstance(group_axis, MultiIndex):
            if (is_list_like(level) and (len(level) == 1)):
                level = level[0]
            if ((key is None) and is_scalar(level)):
                key = group_axis.get_level_values(level)
                level = None
        else:
            if is_list_like(level):
                nlevels = len(level)
                if (nlevels == 1):
                    level = level[0]
                elif (nlevels == 0):
                    raise ValueError('No group keys passed!')
                else:
                    raise ValueError('multiple levels only valid with MultiIndex')
            if isinstance(level, str):
                if (obj._get_axis(axis).name != level):
                    raise ValueError(f'level name {level} is not the name of the {obj._get_axis_name(axis)}')
            elif ((level > 0) or (level < (- 1))):
                raise ValueError('level > 0 or level < -1 only valid with MultiIndex')
            level = None
            key = group_axis
    if isinstance(key, Grouper):
        (binner, grouper, obj) = key._get_grouper(obj, validate=False)
        if (key.key is None):
            return (grouper, set(), obj)
        else:
            return (grouper, {key.key}, obj)
    elif isinstance(key, ops.BaseGrouper):
        return (key, set(), obj)
    if (not isinstance(key, list)):
        keys = [key]
        match_axis_length = False
    else:
        keys = key
        match_axis_length = (len(keys) == len(group_axis))
    any_callable = any(((callable(g) or isinstance(g, dict)) for g in keys))
    any_groupers = any((isinstance(g, Grouper) for g in keys))
    any_arraylike = any((isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys))
    if ((not any_callable) and (not any_arraylike) and (not any_groupers) and match_axis_length and (level is None)):
        if isinstance(obj, DataFrame):
            all_in_columns_index = all((((g in obj.columns) or (g in obj.index.names)) for g in keys))
        else:
            assert isinstance(obj, Series)
            all_in_columns_index = all(((g in obj.index.names) for g in keys))
        if (not all_in_columns_index):
            keys = [com.asarray_tuplesafe(keys)]
    if isinstance(level, (tuple, list)):
        if (key is None):
            keys = ([None] * len(level))
        levels = level
    else:
        levels = ([level] * len(keys))
    groupings: List[Grouping] = []
    exclusions: Set[Label] = set()

    def is_in_axis(key) -> bool:
        if (not _is_label_like(key)):
            items = obj.axes[(- 1)]
            try:
                items.get_loc(key)
            except (KeyError, TypeError, InvalidIndexError):
                return False
        return True

    def is_in_obj(gpr) -> bool:
        if (not hasattr(gpr, 'name')):
            return False
        try:
            return (gpr is obj[gpr.name])
        except (KeyError, IndexError):
            return False
    for (i, (gpr, level)) in enumerate(zip(keys, levels)):
        if is_in_obj(gpr):
            (in_axis, name) = (True, gpr.name)
            exclusions.add(name)
        elif is_in_axis(gpr):
            if (gpr in obj):
                if validate:
                    obj._check_label_or_level_ambiguity(gpr, axis=axis)
                (in_axis, name, gpr) = (True, gpr, obj[gpr])
                exclusions.add(name)
            elif obj._is_level_reference(gpr, axis=axis):
                (in_axis, name, level, gpr) = (False, None, gpr, None)
            else:
                raise KeyError(gpr)
        elif (isinstance(gpr, Grouper) and (gpr.key is not None)):
            exclusions.add(gpr.key)
            (in_axis, name) = (False, None)
        else:
            (in_axis, name) = (False, None)
        if (is_categorical_dtype(gpr) and (len(gpr) != obj.shape[axis])):
            raise ValueError(f'Length of grouper ({len(gpr)}) and axis ({obj.shape[axis]}) must be same length')
        ping = (Grouping(group_axis, gpr, obj=obj, name=name, level=level, sort=sort, observed=observed, in_axis=in_axis, dropna=dropna) if (not isinstance(gpr, Grouping)) else gpr)
        groupings.append(ping)
    if ((len(groupings) == 0) and len(obj)):
        raise ValueError('No group keys passed!')
    elif (len(groupings) == 0):
        groupings.append(Grouping(Index([], dtype='int'), np.array([], dtype=np.intp)))
    grouper = ops.BaseGrouper(group_axis, groupings, sort=sort, mutated=mutated, dropna=dropna)
    return (grouper, exclusions, obj)

def _is_label_like(val):
    return (isinstance(val, (str, tuple)) or ((val is not None) and is_scalar(val)))

def _convert_grouper(axis, grouper):
    if isinstance(grouper, dict):
        return grouper.get
    elif isinstance(grouper, Series):
        if grouper.index.equals(axis):
            return grouper._values
        else:
            return grouper.reindex(axis)._values
    elif isinstance(grouper, (list, Series, Index, np.ndarray)):
        if (len(grouper) != len(axis)):
            raise ValueError('Grouper and axis must be same length')
        return grouper
    else:
        return grouper

def _check_deprecated_resample_kwargs(kwargs, origin):
    '\n    Check for use of deprecated parameters in ``resample`` and related functions.\n\n    Raises the appropriate warnings if these parameters are detected.\n    Only sets an approximate ``stacklevel`` for the warnings (see #37603, #36629).\n\n    Parameters\n    ----------\n    kwargs : dict\n        Dictionary of keyword arguments to check for deprecated parameters.\n    origin : object\n        From where this function is being called; either Grouper or TimeGrouper. Used\n        to determine an approximate stacklevel.\n    '
    from pandas.core.resample import TimeGrouper
    stacklevel = ((5 if (origin is TimeGrouper) else 2) + 1)
    if (kwargs.get('base', None) is not None):
        warnings.warn('\'base\' in .resample() and in Grouper() is deprecated.\nThe new arguments that you should use are \'offset\' or \'origin\'.\n\n>>> df.resample(freq="3s", base=2)\n\nbecomes:\n\n>>> df.resample(freq="3s", offset="2s")\n', FutureWarning, stacklevel=stacklevel)
    if (kwargs.get('loffset', None) is not None):
        warnings.warn('\'loffset\' in .resample() and in Grouper() is deprecated.\n\n>>> df.resample(freq="3s", loffset="8H")\n\nbecomes:\n\n>>> from pandas.tseries.frequencies import to_offset\n>>> df = df.resample(freq="3s").mean()\n>>> df.index = df.index.to_timestamp() + to_offset("8H")\n', FutureWarning, stacklevel=stacklevel)
