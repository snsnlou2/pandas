
import copy
from datetime import timedelta
from textwrap import dedent
from typing import Dict, Optional, Union, no_type_check
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency, NaT, Period, Timedelta, Timestamp, to_offset
from pandas._typing import TimedeltaConvertibleTypes, TimestampConvertibleTypes
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.aggregation import aggregate
import pandas.core.algorithms as algos
from pandas.core.base import DataError
from pandas.core.generic import NDFrame, _shared_docs
from pandas.core.groupby.base import GotItemMixin, ShallowMixin
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy, GroupBy, _pipe_template, get_groupby
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import Index
from pandas.core.indexes.datetimes import DatetimeIndex, date_range
from pandas.core.indexes.period import PeriodIndex, period_range
from pandas.core.indexes.timedeltas import TimedeltaIndex, timedelta_range
from pandas.tseries.frequencies import is_subperiod, is_superperiod
from pandas.tseries.offsets import DateOffset, Day, Nano, Tick
_shared_docs_kwargs = {}

class Resampler(BaseGroupBy, ShallowMixin):
    "\n    Class for resampling datetimelike data, a groupby-like operation.\n    See aggregate, transform, and apply functions on this object.\n\n    It's easiest to use obj.resample(...) to use Resampler.\n\n    Parameters\n    ----------\n    obj : pandas object\n    groupby : a TimeGrouper object\n    axis : int, default 0\n    kind : str or None\n        'period', 'timestamp' to override default index treatment\n\n    Returns\n    -------\n    a Resampler of the appropriate type\n\n    Notes\n    -----\n    After resampling, see aggregate, apply, and transform functions.\n    "
    _attributes = ['freq', 'axis', 'closed', 'label', 'convention', 'loffset', 'kind', 'origin', 'offset']

    def __init__(self, obj, groupby=None, axis=0, kind=None, **kwargs):
        self.groupby = groupby
        self.keys = None
        self.sort = True
        self.axis = axis
        self.kind = kind
        self.squeeze = False
        self.group_keys = True
        self.as_index = True
        self.exclusions = set()
        self.binner = None
        self.grouper = None
        if (self.groupby is not None):
            self.groupby._set_grouper(self._convert_obj(obj), sort=True)

    def __str__(self):
        '\n        Provide a nice str repr of our rolling object.\n        '
        attrs = (f'{k}={getattr(self.groupby, k)}' for k in self._attributes if (getattr(self.groupby, k, None) is not None))
        return f"{type(self).__name__} [{', '.join(attrs)}]"

    def __getattr__(self, attr):
        if (attr in self._internal_names_set):
            return object.__getattribute__(self, attr)
        if (attr in self._attributes):
            return getattr(self.groupby, attr)
        if (attr in self.obj):
            return self[attr]
        return object.__getattribute__(self, attr)

    def __iter__(self):
        '\n        Resampler iterator.\n\n        Returns\n        -------\n        Generator yielding sequence of (name, subsetted object)\n        for each group.\n\n        See Also\n        --------\n        GroupBy.__iter__ : Generator yielding sequence for each group.\n        '
        self._set_binner()
        return super().__iter__()

    @property
    def obj(self):
        return self.groupby.obj

    @property
    def ax(self):
        return self.groupby.ax

    @property
    def _typ(self):
        '\n        Masquerade for compat as a Series or a DataFrame.\n        '
        if isinstance(self._selected_obj, ABCSeries):
            return 'series'
        return 'dataframe'

    @property
    def _from_selection(self):
        '\n        Is the resampling from a DataFrame column or MultiIndex level.\n        '
        return ((self.groupby is not None) and ((self.groupby.key is not None) or (self.groupby.level is not None)))

    def _convert_obj(self, obj):
        '\n        Provide any conversions for the object in order to correctly handle.\n\n        Parameters\n        ----------\n        obj : the object to be resampled\n\n        Returns\n        -------\n        obj : converted object\n        '
        obj = obj._consolidate()
        return obj

    def _get_binner_for_time(self):
        raise AbstractMethodError(self)

    def _set_binner(self):
        '\n        Setup our binners.\n\n        Cache these as we are an immutable object\n        '
        if (self.binner is None):
            (self.binner, self.grouper) = self._get_binner()

    def _get_binner(self):
        '\n        Create the BinGrouper, assume that self.set_grouper(obj)\n        has already been called.\n        '
        (binner, bins, binlabels) = self._get_binner_for_time()
        assert (len(bins) == len(binlabels))
        bin_grouper = BinGrouper(bins, binlabels, indexer=self.groupby.indexer)
        return (binner, bin_grouper)

    def _assure_grouper(self):
        '\n        Make sure that we are creating our binner & grouper.\n        '
        self._set_binner()

    @Substitution(klass='Resampler', examples="\n    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},\n    ...                   index=pd.date_range('2012-08-02', periods=4))\n    >>> df\n                A\n    2012-08-02  1\n    2012-08-03  2\n    2012-08-04  3\n    2012-08-05  4\n\n    To get the difference between each 2-day period's maximum and minimum\n    value in one pass, you can do\n\n    >>> df.resample('2D').pipe(lambda x: x.max() - x.min())\n                A\n    2012-08-02  1\n    2012-08-04  1")
    @Appender(_pipe_template)
    def pipe(self, func, *args, **kwargs):
        return super().pipe(func, *args, **kwargs)
    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    DataFrame.groupby.aggregate : Aggregate using callable, string, dict,\n        or list of string/callables.\n    DataFrame.resample.transform : Transforms the Series on each group\n        based on the given function.\n    DataFrame.aggregate: Aggregate using one or more\n        operations over the specified axis.\n    ')
    _agg_examples_doc = dedent("\n    Examples\n    --------\n    >>> s = pd.Series([1,2,3,4,5],\n                      index=pd.date_range('20130101', periods=5,freq='s'))\n    2013-01-01 00:00:00    1\n    2013-01-01 00:00:01    2\n    2013-01-01 00:00:02    3\n    2013-01-01 00:00:03    4\n    2013-01-01 00:00:04    5\n    Freq: S, dtype: int64\n\n    >>> r = s.resample('2s')\n    DatetimeIndexResampler [freq=<2 * Seconds>, axis=0, closed=left,\n                            label=left, convention=start]\n\n    >>> r.agg(np.sum)\n    2013-01-01 00:00:00    3\n    2013-01-01 00:00:02    7\n    2013-01-01 00:00:04    5\n    Freq: 2S, dtype: int64\n\n    >>> r.agg(['sum','mean','max'])\n                         sum  mean  max\n    2013-01-01 00:00:00    3   1.5    2\n    2013-01-01 00:00:02    7   3.5    4\n    2013-01-01 00:00:04    5   5.0    5\n\n    >>> r.agg({'result' : lambda x: x.mean() / x.std(),\n               'total' : np.sum})\n                         total    result\n    2013-01-01 00:00:00      3  2.121320\n    2013-01-01 00:00:02      7  4.949747\n    2013-01-01 00:00:04      5       NaN\n    ")

    @doc(_shared_docs['aggregate'], see_also=_agg_see_also_doc, examples=_agg_examples_doc, klass='DataFrame', axis='')
    def aggregate(self, func, *args, **kwargs):
        self._set_binner()
        (result, how) = aggregate(self, func, *args, **kwargs)
        if (result is None):
            how = func
            grouper = None
            result = self._groupby_and_aggregate(how, grouper, *args, **kwargs)
        result = self._apply_loffset(result)
        return result
    agg = aggregate
    apply = aggregate

    def transform(self, arg, *args, **kwargs):
        '\n        Call function producing a like-indexed Series on each group and return\n        a Series with the transformed values.\n\n        Parameters\n        ----------\n        arg : function\n            To apply to each group. Should return a Series with the same index.\n\n        Returns\n        -------\n        transformed : Series\n\n        Examples\n        --------\n        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())\n        '
        return self._selected_obj.groupby(self.groupby).transform(arg, *args, **kwargs)

    def _downsample(self, f):
        raise AbstractMethodError(self)

    def _upsample(self, f, limit=None, fill_value=None):
        raise AbstractMethodError(self)

    def _gotitem(self, key, ndim, subset=None):
        '\n        Sub-classes to define. Return a sliced object.\n\n        Parameters\n        ----------\n        key : string / list of selections\n        ndim : {1, 2}\n            requested ndim of result\n        subset : object, default None\n            subset to act on\n        '
        self._set_binner()
        grouper = self.grouper
        if (subset is None):
            subset = self.obj
        grouped = get_groupby(subset, by=None, grouper=grouper, axis=self.axis)
        try:
            return grouped[key]
        except KeyError:
            return grouped

    def _groupby_and_aggregate(self, how, grouper=None, *args, **kwargs):
        '\n        Re-evaluate the obj with a groupby aggregation.\n        '
        if (grouper is None):
            self._set_binner()
            grouper = self.grouper
        obj = self._selected_obj
        grouped = get_groupby(obj, by=None, grouper=grouper, axis=self.axis)
        try:
            if (isinstance(obj, ABCDataFrame) and callable(how)):
                result = grouped._aggregate_item_by_item(how, *args, **kwargs)
            else:
                result = grouped.aggregate(how, *args, **kwargs)
        except (DataError, AttributeError, KeyError):
            result = grouped.apply(how, *args, **kwargs)
        except ValueError as err:
            if ('Must produce aggregated value' in str(err)):
                pass
            elif ('len(index) != len(labels)' in str(err)):
                pass
            elif ('No objects to concatenate' in str(err)):
                pass
            else:
                raise
            result = grouped.apply(how, *args, **kwargs)
        result = self._apply_loffset(result)
        return self._wrap_result(result)

    def _apply_loffset(self, result):
        '\n        If loffset is set, offset the result index.\n\n        This is NOT an idempotent routine, it will be applied\n        exactly once to the result.\n\n        Parameters\n        ----------\n        result : Series or DataFrame\n            the result of resample\n        '
        needs_offset = (isinstance(self.loffset, (DateOffset, timedelta, np.timedelta64)) and isinstance(result.index, DatetimeIndex) and (len(result.index) > 0))
        if needs_offset:
            result.index = (result.index + self.loffset)
        self.loffset = None
        return result

    def _get_resampler_for_grouping(self, groupby, **kwargs):
        '\n        Return the correct class for resampling with groupby.\n        '
        return self._resampler_for_grouping(self, groupby=groupby, **kwargs)

    def _wrap_result(self, result):
        '\n        Potentially wrap any results.\n        '
        if (isinstance(result, ABCSeries) and (self._selection is not None)):
            result.name = self._selection
        if (isinstance(result, ABCSeries) and result.empty):
            obj = self.obj
            result.index = _asfreq_compat(obj.index, freq=self.freq)
            result.name = getattr(obj, 'name', None)
        return result

    def pad(self, limit=None):
        '\n        Forward fill the values.\n\n        Parameters\n        ----------\n        limit : int, optional\n            Limit of how many values to fill.\n\n        Returns\n        -------\n        An upsampled Series.\n\n        See Also\n        --------\n        Series.fillna: Fill NA/NaN values using the specified method.\n        DataFrame.fillna: Fill NA/NaN values using the specified method.\n        '
        return self._upsample('pad', limit=limit)
    ffill = pad

    def nearest(self, limit=None):
        "\n        Resample by using the nearest value.\n\n        When resampling data, missing values may appear (e.g., when the\n        resampling frequency is higher than the original frequency).\n        The `nearest` method will replace ``NaN`` values that appeared in\n        the resampled data with the value from the nearest member of the\n        sequence, based on the index value.\n        Missing values that existed in the original data will not be modified.\n        If `limit` is given, fill only this many values in each direction for\n        each of the original values.\n\n        Parameters\n        ----------\n        limit : int, optional\n            Limit of how many values to fill.\n\n        Returns\n        -------\n        Series or DataFrame\n            An upsampled Series or DataFrame with ``NaN`` values filled with\n            their nearest value.\n\n        See Also\n        --------\n        backfill : Backward fill the new missing values in the resampled data.\n        pad : Forward fill ``NaN`` values.\n\n        Examples\n        --------\n        >>> s = pd.Series([1, 2],\n        ...               index=pd.date_range('20180101',\n        ...                                   periods=2,\n        ...                                   freq='1h'))\n        >>> s\n        2018-01-01 00:00:00    1\n        2018-01-01 01:00:00    2\n        Freq: H, dtype: int64\n\n        >>> s.resample('15min').nearest()\n        2018-01-01 00:00:00    1\n        2018-01-01 00:15:00    1\n        2018-01-01 00:30:00    2\n        2018-01-01 00:45:00    2\n        2018-01-01 01:00:00    2\n        Freq: 15T, dtype: int64\n\n        Limit the number of upsampled values imputed by the nearest:\n\n        >>> s.resample('15min').nearest(limit=1)\n        2018-01-01 00:00:00    1.0\n        2018-01-01 00:15:00    1.0\n        2018-01-01 00:30:00    NaN\n        2018-01-01 00:45:00    2.0\n        2018-01-01 01:00:00    2.0\n        Freq: 15T, dtype: float64\n        "
        return self._upsample('nearest', limit=limit)

    def backfill(self, limit=None):
        "\n        Backward fill the new missing values in the resampled data.\n\n        In statistics, imputation is the process of replacing missing data with\n        substituted values [1]_. When resampling data, missing values may\n        appear (e.g., when the resampling frequency is higher than the original\n        frequency). The backward fill will replace NaN values that appeared in\n        the resampled data with the next value in the original sequence.\n        Missing values that existed in the original data will not be modified.\n\n        Parameters\n        ----------\n        limit : int, optional\n            Limit of how many values to fill.\n\n        Returns\n        -------\n        Series, DataFrame\n            An upsampled Series or DataFrame with backward filled NaN values.\n\n        See Also\n        --------\n        bfill : Alias of backfill.\n        fillna : Fill NaN values using the specified method, which can be\n            'backfill'.\n        nearest : Fill NaN values with nearest neighbor starting from center.\n        pad : Forward fill NaN values.\n        Series.fillna : Fill NaN values in the Series using the\n            specified method, which can be 'backfill'.\n        DataFrame.fillna : Fill NaN values in the DataFrame using the\n            specified method, which can be 'backfill'.\n\n        References\n        ----------\n        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)\n\n        Examples\n        --------\n        Resampling a Series:\n\n        >>> s = pd.Series([1, 2, 3],\n        ...               index=pd.date_range('20180101', periods=3, freq='h'))\n        >>> s\n        2018-01-01 00:00:00    1\n        2018-01-01 01:00:00    2\n        2018-01-01 02:00:00    3\n        Freq: H, dtype: int64\n\n        >>> s.resample('30min').backfill()\n        2018-01-01 00:00:00    1\n        2018-01-01 00:30:00    2\n        2018-01-01 01:00:00    2\n        2018-01-01 01:30:00    3\n        2018-01-01 02:00:00    3\n        Freq: 30T, dtype: int64\n\n        >>> s.resample('15min').backfill(limit=2)\n        2018-01-01 00:00:00    1.0\n        2018-01-01 00:15:00    NaN\n        2018-01-01 00:30:00    2.0\n        2018-01-01 00:45:00    2.0\n        2018-01-01 01:00:00    2.0\n        2018-01-01 01:15:00    NaN\n        2018-01-01 01:30:00    3.0\n        2018-01-01 01:45:00    3.0\n        2018-01-01 02:00:00    3.0\n        Freq: 15T, dtype: float64\n\n        Resampling a DataFrame that has missing values:\n\n        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},\n        ...                   index=pd.date_range('20180101', periods=3,\n        ...                                       freq='h'))\n        >>> df\n                               a  b\n        2018-01-01 00:00:00  2.0  1\n        2018-01-01 01:00:00  NaN  3\n        2018-01-01 02:00:00  6.0  5\n\n        >>> df.resample('30min').backfill()\n                               a  b\n        2018-01-01 00:00:00  2.0  1\n        2018-01-01 00:30:00  NaN  3\n        2018-01-01 01:00:00  NaN  3\n        2018-01-01 01:30:00  6.0  5\n        2018-01-01 02:00:00  6.0  5\n\n        >>> df.resample('15min').backfill(limit=2)\n                               a    b\n        2018-01-01 00:00:00  2.0  1.0\n        2018-01-01 00:15:00  NaN  NaN\n        2018-01-01 00:30:00  NaN  3.0\n        2018-01-01 00:45:00  NaN  3.0\n        2018-01-01 01:00:00  NaN  3.0\n        2018-01-01 01:15:00  NaN  NaN\n        2018-01-01 01:30:00  6.0  5.0\n        2018-01-01 01:45:00  6.0  5.0\n        2018-01-01 02:00:00  6.0  5.0\n        "
        return self._upsample('backfill', limit=limit)
    bfill = backfill

    def fillna(self, method, limit=None):
        '\n        Fill missing values introduced by upsampling.\n\n        In statistics, imputation is the process of replacing missing data with\n        substituted values [1]_. When resampling data, missing values may\n        appear (e.g., when the resampling frequency is higher than the original\n        frequency).\n\n        Missing values that existed in the original data will\n        not be modified.\n\n        Parameters\n        ----------\n        method : {\'pad\', \'backfill\', \'ffill\', \'bfill\', \'nearest\'}\n            Method to use for filling holes in resampled data\n\n            * \'pad\' or \'ffill\': use previous valid observation to fill gap\n              (forward fill).\n            * \'backfill\' or \'bfill\': use next valid observation to fill gap.\n            * \'nearest\': use nearest valid observation to fill gap.\n\n        limit : int, optional\n            Limit of how many consecutive missing values to fill.\n\n        Returns\n        -------\n        Series or DataFrame\n            An upsampled Series or DataFrame with missing values filled.\n\n        See Also\n        --------\n        backfill : Backward fill NaN values in the resampled data.\n        pad : Forward fill NaN values in the resampled data.\n        nearest : Fill NaN values in the resampled data\n            with nearest neighbor starting from center.\n        interpolate : Fill NaN values using interpolation.\n        Series.fillna : Fill NaN values in the Series using the\n            specified method, which can be \'bfill\' and \'ffill\'.\n        DataFrame.fillna : Fill NaN values in the DataFrame using the\n            specified method, which can be \'bfill\' and \'ffill\'.\n\n        References\n        ----------\n        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)\n\n        Examples\n        --------\n        Resampling a Series:\n\n        >>> s = pd.Series([1, 2, 3],\n        ...               index=pd.date_range(\'20180101\', periods=3, freq=\'h\'))\n        >>> s\n        2018-01-01 00:00:00    1\n        2018-01-01 01:00:00    2\n        2018-01-01 02:00:00    3\n        Freq: H, dtype: int64\n\n        Without filling the missing values you get:\n\n        >>> s.resample("30min").asfreq()\n        2018-01-01 00:00:00    1.0\n        2018-01-01 00:30:00    NaN\n        2018-01-01 01:00:00    2.0\n        2018-01-01 01:30:00    NaN\n        2018-01-01 02:00:00    3.0\n        Freq: 30T, dtype: float64\n\n        >>> s.resample(\'30min\').fillna("backfill")\n        2018-01-01 00:00:00    1\n        2018-01-01 00:30:00    2\n        2018-01-01 01:00:00    2\n        2018-01-01 01:30:00    3\n        2018-01-01 02:00:00    3\n        Freq: 30T, dtype: int64\n\n        >>> s.resample(\'15min\').fillna("backfill", limit=2)\n        2018-01-01 00:00:00    1.0\n        2018-01-01 00:15:00    NaN\n        2018-01-01 00:30:00    2.0\n        2018-01-01 00:45:00    2.0\n        2018-01-01 01:00:00    2.0\n        2018-01-01 01:15:00    NaN\n        2018-01-01 01:30:00    3.0\n        2018-01-01 01:45:00    3.0\n        2018-01-01 02:00:00    3.0\n        Freq: 15T, dtype: float64\n\n        >>> s.resample(\'30min\').fillna("pad")\n        2018-01-01 00:00:00    1\n        2018-01-01 00:30:00    1\n        2018-01-01 01:00:00    2\n        2018-01-01 01:30:00    2\n        2018-01-01 02:00:00    3\n        Freq: 30T, dtype: int64\n\n        >>> s.resample(\'30min\').fillna("nearest")\n        2018-01-01 00:00:00    1\n        2018-01-01 00:30:00    2\n        2018-01-01 01:00:00    2\n        2018-01-01 01:30:00    3\n        2018-01-01 02:00:00    3\n        Freq: 30T, dtype: int64\n\n        Missing values present before the upsampling are not affected.\n\n        >>> sm = pd.Series([1, None, 3],\n        ...               index=pd.date_range(\'20180101\', periods=3, freq=\'h\'))\n        >>> sm\n        2018-01-01 00:00:00    1.0\n        2018-01-01 01:00:00    NaN\n        2018-01-01 02:00:00    3.0\n        Freq: H, dtype: float64\n\n        >>> sm.resample(\'30min\').fillna(\'backfill\')\n        2018-01-01 00:00:00    1.0\n        2018-01-01 00:30:00    NaN\n        2018-01-01 01:00:00    NaN\n        2018-01-01 01:30:00    3.0\n        2018-01-01 02:00:00    3.0\n        Freq: 30T, dtype: float64\n\n        >>> sm.resample(\'30min\').fillna(\'pad\')\n        2018-01-01 00:00:00    1.0\n        2018-01-01 00:30:00    1.0\n        2018-01-01 01:00:00    NaN\n        2018-01-01 01:30:00    NaN\n        2018-01-01 02:00:00    3.0\n        Freq: 30T, dtype: float64\n\n        >>> sm.resample(\'30min\').fillna(\'nearest\')\n        2018-01-01 00:00:00    1.0\n        2018-01-01 00:30:00    NaN\n        2018-01-01 01:00:00    NaN\n        2018-01-01 01:30:00    3.0\n        2018-01-01 02:00:00    3.0\n        Freq: 30T, dtype: float64\n\n        DataFrame resampling is done column-wise. All the same options are\n        available.\n\n        >>> df = pd.DataFrame({\'a\': [2, np.nan, 6], \'b\': [1, 3, 5]},\n        ...                   index=pd.date_range(\'20180101\', periods=3,\n        ...                                       freq=\'h\'))\n        >>> df\n                               a  b\n        2018-01-01 00:00:00  2.0  1\n        2018-01-01 01:00:00  NaN  3\n        2018-01-01 02:00:00  6.0  5\n\n        >>> df.resample(\'30min\').fillna("bfill")\n                               a  b\n        2018-01-01 00:00:00  2.0  1\n        2018-01-01 00:30:00  NaN  3\n        2018-01-01 01:00:00  NaN  3\n        2018-01-01 01:30:00  6.0  5\n        2018-01-01 02:00:00  6.0  5\n        '
        return self._upsample(method, limit=limit)

    @doc(NDFrame.interpolate, **_shared_docs_kwargs)
    def interpolate(self, method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None, **kwargs):
        '\n        Interpolate values according to different methods.\n        '
        result = self._upsample('asfreq')
        return result.interpolate(method=method, axis=axis, limit=limit, inplace=inplace, limit_direction=limit_direction, limit_area=limit_area, downcast=downcast, **kwargs)

    def asfreq(self, fill_value=None):
        '\n        Return the values at the new freq, essentially a reindex.\n\n        Parameters\n        ----------\n        fill_value : scalar, optional\n            Value to use for missing values, applied during upsampling (note\n            this does not fill NaNs that already were present).\n\n        Returns\n        -------\n        DataFrame or Series\n            Values at the specified freq.\n\n        See Also\n        --------\n        Series.asfreq: Convert TimeSeries to specified frequency.\n        DataFrame.asfreq: Convert TimeSeries to specified frequency.\n        '
        return self._upsample('asfreq', fill_value=fill_value)

    def std(self, ddof=1, *args, **kwargs):
        '\n        Compute standard deviation of groups, excluding missing values.\n\n        Parameters\n        ----------\n        ddof : int, default 1\n            Degrees of freedom.\n\n        Returns\n        -------\n        DataFrame or Series\n            Standard deviation of values within each group.\n        '
        nv.validate_resampler_func('std', args, kwargs)
        return self._downsample('std', ddof=ddof)

    def var(self, ddof=1, *args, **kwargs):
        '\n        Compute variance of groups, excluding missing values.\n\n        Parameters\n        ----------\n        ddof : int, default 1\n            Degrees of freedom.\n\n        Returns\n        -------\n        DataFrame or Series\n            Variance of values within each group.\n        '
        nv.validate_resampler_func('var', args, kwargs)
        return self._downsample('var', ddof=ddof)

    @doc(GroupBy.size)
    def size(self):
        result = self._downsample('size')
        if (not len(self.ax)):
            from pandas import Series
            if (self._selected_obj.ndim == 1):
                name = self._selected_obj.name
            else:
                name = None
            result = Series([], index=result.index, dtype='int64', name=name)
        return result

    @doc(GroupBy.count)
    def count(self):
        result = self._downsample('count')
        if (not len(self.ax)):
            if (self._selected_obj.ndim == 1):
                result = type(self._selected_obj)([], index=result.index, dtype='int64', name=self._selected_obj.name)
            else:
                from pandas import DataFrame
                result = DataFrame([], index=result.index, columns=result.columns, dtype='int64')
        return result

    def quantile(self, q=0.5, **kwargs):
        '\n        Return value at the given quantile.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        q : float or array-like, default 0.5 (50% quantile)\n\n        Returns\n        -------\n        DataFrame or Series\n            Quantile of values within each group.\n\n        See Also\n        --------\n        Series.quantile\n            Return a series, where the index is q and the values are the quantiles.\n        DataFrame.quantile\n            Return a DataFrame, where the columns are the columns of self,\n            and the values are the quantiles.\n        DataFrameGroupBy.quantile\n            Return a DataFrame, where the coulmns are groupby columns,\n            and the values are its quantiles.\n        '
        return self._downsample('quantile', q=q, **kwargs)
for method in ['sum', 'prod', 'min', 'max', 'first', 'last']:

    def f(self, _method=method, min_count=0, *args, **kwargs):
        nv.validate_resampler_func(_method, args, kwargs)
        return self._downsample(_method, min_count=min_count)
    f.__doc__ = getattr(GroupBy, method).__doc__
    setattr(Resampler, method, f)
for method in ['mean', 'sem', 'median', 'ohlc']:

    def g(self, _method=method, *args, **kwargs):
        nv.validate_resampler_func(_method, args, kwargs)
        return self._downsample(_method)
    g.__doc__ = getattr(GroupBy, method).__doc__
    setattr(Resampler, method, g)
for method in ['nunique']:

    def h(self, _method=method):
        return self._downsample(_method)
    h.__doc__ = getattr(SeriesGroupBy, method).__doc__
    setattr(Resampler, method, h)

class _GroupByMixin(GotItemMixin):
    '\n    Provide the groupby facilities.\n    '

    def __init__(self, obj, *args, **kwargs):
        parent = kwargs.pop('parent', None)
        groupby = kwargs.pop('groupby', None)
        if (parent is None):
            parent = obj
        for attr in self._attributes:
            setattr(self, attr, kwargs.get(attr, getattr(parent, attr)))
        super().__init__(None)
        self._groupby = groupby
        self._groupby.mutated = True
        self._groupby.grouper.mutated = True
        self.groupby = copy.copy(parent.groupby)

    @no_type_check
    def _apply(self, f, grouper=None, *args, **kwargs):
        '\n        Dispatch to _upsample; we are stripping all of the _upsample kwargs and\n        performing the original function call on the grouped object.\n        '

        def func(x):
            x = self._shallow_copy(x, groupby=self.groupby)
            if isinstance(f, str):
                return getattr(x, f)(**kwargs)
            return x.apply(f, *args, **kwargs)
        result = self._groupby.apply(func)
        return self._wrap_result(result)
    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply

class DatetimeIndexResampler(Resampler):

    @property
    def _resampler_for_grouping(self):
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self):
        if (self.kind == 'period'):
            return self.groupby._get_time_period_bins(self.ax)
        return self.groupby._get_time_bins(self.ax)

    def _downsample(self, how, **kwargs):
        '\n        Downsample the cython defined function.\n\n        Parameters\n        ----------\n        how : string / cython mapped function\n        **kwargs : kw args passed to how function\n        '
        self._set_binner()
        how = (self._get_cython_func(how) or how)
        ax = self.ax
        obj = self._selected_obj
        if (not len(ax)):
            obj = obj.copy()
            obj.index = obj.index._with_freq(self.freq)
            assert (obj.index.freq == self.freq), (obj.index.freq, self.freq)
            return obj
        if ((ax.freq is not None) or (ax.inferred_freq is not None)):
            if ((len(self.grouper.binlabels) > len(ax)) and (how is None)):
                return self.asfreq()
        result = obj.groupby(self.grouper, axis=self.axis).aggregate(how, **kwargs)
        result = self._apply_loffset(result)
        return self._wrap_result(result)

    def _adjust_binner_for_upsample(self, binner):
        '\n        Adjust our binner when upsampling.\n\n        The range of a new index should not be outside specified range\n        '
        if (self.closed == 'right'):
            binner = binner[1:]
        else:
            binner = binner[:(- 1)]
        return binner

    def _upsample(self, method, limit=None, fill_value=None):
        "\n        Parameters\n        ----------\n        method : string {'backfill', 'bfill', 'pad',\n            'ffill', 'asfreq'} method for upsampling\n        limit : int, default None\n            Maximum size gap to fill when reindexing\n        fill_value : scalar, default None\n            Value to use for missing values\n\n        See Also\n        --------\n        .fillna: Fill NA/NaN values using the specified method.\n\n        "
        self._set_binner()
        if self.axis:
            raise AssertionError('axis must be 0')
        if self._from_selection:
            raise ValueError('Upsampling from level= or on= selection is not supported, use .set_index(...) to explicitly set index to datetime-like')
        ax = self.ax
        obj = self._selected_obj
        binner = self.binner
        res_index = self._adjust_binner_for_upsample(binner)
        if ((limit is None) and (to_offset(ax.inferred_freq) == self.freq) and (len(obj) == len(res_index))):
            result = obj.copy()
            result.index = res_index
        else:
            result = obj.reindex(res_index, method=method, limit=limit, fill_value=fill_value)
        result = self._apply_loffset(result)
        return self._wrap_result(result)

    def _wrap_result(self, result):
        result = super()._wrap_result(result)
        if ((self.kind == 'period') and (not isinstance(result.index, PeriodIndex))):
            result.index = result.index.to_period(self.freq)
        return result

class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    '\n    Provides a resample of a groupby implementation\n    '

    @property
    def _constructor(self):
        return DatetimeIndexResampler

class PeriodIndexResampler(DatetimeIndexResampler):

    @property
    def _resampler_for_grouping(self):
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self):
        if (self.kind == 'timestamp'):
            return super()._get_binner_for_time()
        return self.groupby._get_period_bins(self.ax)

    def _convert_obj(self, obj):
        obj = super()._convert_obj(obj)
        if self._from_selection:
            msg = 'Resampling from level= or on= selection with a PeriodIndex is not currently supported, use .set_index(...) to explicitly set index'
            raise NotImplementedError(msg)
        if (self.loffset is not None):
            self.kind = 'timestamp'
        if (self.kind == 'timestamp'):
            obj = obj.to_timestamp(how=self.convention)
        return obj

    def _downsample(self, how, **kwargs):
        '\n        Downsample the cython defined function.\n\n        Parameters\n        ----------\n        how : string / cython mapped function\n        **kwargs : kw args passed to how function\n        '
        if (self.kind == 'timestamp'):
            return super()._downsample(how, **kwargs)
        how = (self._get_cython_func(how) or how)
        ax = self.ax
        if is_subperiod(ax.freq, self.freq):
            return self._groupby_and_aggregate(how, grouper=self.grouper, **kwargs)
        elif is_superperiod(ax.freq, self.freq):
            if (how == 'ohlc'):
                return self._groupby_and_aggregate(how, grouper=self.grouper)
            return self.asfreq()
        elif (ax.freq == self.freq):
            return self.asfreq()
        raise IncompatibleFrequency(f'Frequency {ax.freq} cannot be resampled to {self.freq}, as they are not sub or super periods')

    def _upsample(self, method, limit=None, fill_value=None):
        "\n        Parameters\n        ----------\n        method : string {'backfill', 'bfill', 'pad', 'ffill'}\n            Method for upsampling.\n        limit : int, default None\n            Maximum size gap to fill when reindexing.\n        fill_value : scalar, default None\n            Value to use for missing values.\n\n        See Also\n        --------\n        .fillna: Fill NA/NaN values using the specified method.\n\n        "
        if (self.kind == 'timestamp'):
            return super()._upsample(method, limit=limit, fill_value=fill_value)
        self._set_binner()
        ax = self.ax
        obj = self.obj
        new_index = self.binner
        memb = ax.asfreq(self.freq, how=self.convention)
        indexer = memb.get_indexer(new_index, method=method, limit=limit)
        return self._wrap_result(_take_new_index(obj, indexer, new_index, axis=self.axis))

class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    '\n    Provides a resample of a groupby implementation.\n    '

    @property
    def _constructor(self):
        return PeriodIndexResampler

class TimedeltaIndexResampler(DatetimeIndexResampler):

    @property
    def _resampler_for_grouping(self):
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self):
        return self.groupby._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner):
        "\n        Adjust our binner when upsampling.\n\n        The range of a new index is allowed to be greater than original range\n        so we don't need to change the length of a binner, GH 13022\n        "
        return binner

class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    '\n    Provides a resample of a groupby implementation.\n    '

    @property
    def _constructor(self):
        return TimedeltaIndexResampler

def get_resampler(obj, kind=None, **kwds):
    '\n    Create a TimeGrouper and return our resampler.\n    '
    tg = TimeGrouper(**kwds)
    return tg._get_resampler(obj, kind=kind)
get_resampler.__doc__ = Resampler.__doc__

def get_resampler_for_grouping(groupby, rule, how=None, fill_method=None, limit=None, kind=None, **kwargs):
    '\n    Return our appropriate resampler when grouping as well.\n    '
    kwargs['key'] = kwargs.pop('on', None)
    tg = TimeGrouper(freq=rule, **kwargs)
    resampler = tg._get_resampler(groupby.obj, kind=kind)
    return resampler._get_resampler_for_grouping(groupby=groupby)

class TimeGrouper(Grouper):
    "\n    Custom groupby class for time-interval grouping.\n\n    Parameters\n    ----------\n    freq : pandas date offset or offset alias for identifying bin edges\n    closed : closed end of interval; 'left' or 'right'\n    label : interval boundary to use for labeling; 'left' or 'right'\n    convention : {'start', 'end', 'e', 's'}\n        If axis is PeriodIndex\n    "
    _attributes = (Grouper._attributes + ('closed', 'label', 'how', 'loffset', 'kind', 'convention', 'origin', 'offset'))

    def __init__(self, freq='Min', closed=None, label=None, how='mean', axis=0, fill_method=None, limit=None, loffset=None, kind=None, convention=None, base=None, origin='start_day', offset=None, **kwargs):
        if (label not in {None, 'left', 'right'}):
            raise ValueError(f'Unsupported value {label} for `label`')
        if (closed not in {None, 'left', 'right'}):
            raise ValueError(f'Unsupported value {closed} for `closed`')
        if (convention not in {None, 'start', 'end', 'e', 's'}):
            raise ValueError(f'Unsupported value {convention} for `convention`')
        freq = to_offset(freq)
        end_types = {'M', 'A', 'Q', 'BM', 'BA', 'BQ', 'W'}
        rule = freq.rule_code
        if ((rule in end_types) or (('-' in rule) and (rule[:rule.find('-')] in end_types))):
            if (closed is None):
                closed = 'right'
            if (label is None):
                label = 'right'
        elif (origin in ['end', 'end_day']):
            if (closed is None):
                closed = 'right'
            if (label is None):
                label = 'right'
        else:
            if (closed is None):
                closed = 'left'
            if (label is None):
                label = 'left'
        self.closed = closed
        self.label = label
        self.kind = kind
        self.convention = (convention or 'E')
        self.convention = self.convention.lower()
        self.how = how
        self.fill_method = fill_method
        self.limit = limit
        if (origin in ('epoch', 'start', 'start_day', 'end', 'end_day')):
            self.origin = origin
        else:
            try:
                self.origin = Timestamp(origin)
            except Exception as e:
                raise ValueError(f"'origin' should be equal to 'epoch', 'start', 'start_day', 'end', 'end_day' or should be a Timestamp convertible type. Got '{origin}' instead.") from e
        try:
            self.offset = (Timedelta(offset) if (offset is not None) else None)
        except Exception as e:
            raise ValueError(f"'offset' should be a Timedelta convertible type. Got '{offset}' instead.") from e
        kwargs['sort'] = True
        if ((base is not None) and (offset is not None)):
            raise ValueError("'offset' and 'base' cannot be present at the same time")
        if (base and isinstance(freq, Tick)):
            self.offset = Timedelta(((base * freq.nanos) // freq.n))
        if isinstance(loffset, str):
            loffset = to_offset(loffset)
        self.loffset = loffset
        super().__init__(freq=freq, axis=axis, **kwargs)

    def _get_resampler(self, obj, kind=None):
        "\n        Return my resampler or raise if we have an invalid axis.\n\n        Parameters\n        ----------\n        obj : input object\n        kind : string, optional\n            'period','timestamp','timedelta' are valid\n\n        Returns\n        -------\n        a Resampler\n\n        Raises\n        ------\n        TypeError if incompatible axis\n\n        "
        self._set_grouper(obj)
        ax = self.ax
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(obj, groupby=self, kind=kind, axis=self.axis)
        elif (isinstance(ax, PeriodIndex) or (kind == 'period')):
            return PeriodIndexResampler(obj, groupby=self, kind=kind, axis=self.axis)
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(obj, groupby=self, axis=self.axis)
        raise TypeError(f"Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of '{type(ax).__name__}'")

    def _get_grouper(self, obj, validate=True):
        r = self._get_resampler(obj)
        r._set_binner()
        return (r.binner, r.grouper, r.obj)

    def _get_time_bins(self, ax):
        if (not isinstance(ax, DatetimeIndex)):
            raise TypeError(f'axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}')
        if (len(ax) == 0):
            binner = labels = DatetimeIndex(data=[], freq=self.freq, name=ax.name)
            return (binner, [], labels)
        (first, last) = _get_timestamp_range_edges(ax.min(), ax.max(), self.freq, closed=self.closed, origin=self.origin, offset=self.offset)
        binner = labels = date_range(freq=self.freq, start=first, end=last, tz=ax.tz, name=ax.name, ambiguous=True, nonexistent='shift_forward')
        ax_values = ax.asi8
        (binner, bin_edges) = self._adjust_bin_edges(binner, ax_values)
        bins = lib.generate_bins_dt64(ax_values, bin_edges, self.closed, hasnans=ax.hasnans)
        if (self.closed == 'right'):
            labels = binner
            if (self.label == 'right'):
                labels = labels[1:]
        elif (self.label == 'right'):
            labels = labels[1:]
        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)
        if (len(bins) < len(labels)):
            labels = labels[:len(bins)]
        return (binner, bins, labels)

    def _adjust_bin_edges(self, binner, ax_values):
        if ((self.freq != 'D') and is_superperiod(self.freq, 'D')):
            if (self.closed == 'right'):
                bin_edges = binner.tz_localize(None)
                bin_edges = ((bin_edges + timedelta(1)) - Nano(1))
                bin_edges = bin_edges.tz_localize(binner.tz).asi8
            else:
                bin_edges = binner.asi8
            if (bin_edges[(- 2)] > ax_values.max()):
                bin_edges = bin_edges[:(- 1)]
                binner = binner[:(- 1)]
        else:
            bin_edges = binner.asi8
        return (binner, bin_edges)

    def _get_time_delta_bins(self, ax):
        if (not isinstance(ax, TimedeltaIndex)):
            raise TypeError(f'axis must be a TimedeltaIndex, but got an instance of {type(ax).__name__}')
        if (not len(ax)):
            binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
            return (binner, [], labels)
        (start, end) = (ax.min(), ax.max())
        labels = binner = timedelta_range(start=start, end=end, freq=self.freq, name=ax.name)
        end_stamps = (labels + self.freq)
        bins = ax.searchsorted(end_stamps, side='left')
        if self.offset:
            labels += self.offset
        if self.loffset:
            labels += self.loffset
        return (binner, bins, labels)

    def _get_time_period_bins(self, ax):
        if (not isinstance(ax, DatetimeIndex)):
            raise TypeError(f'axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}')
        freq = self.freq
        if (not len(ax)):
            binner = labels = PeriodIndex(data=[], freq=freq, name=ax.name)
            return (binner, [], labels)
        labels = binner = period_range(start=ax[0], end=ax[(- 1)], freq=freq, name=ax.name)
        end_stamps = (labels + freq).asfreq(freq, 's').to_timestamp()
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)
        bins = ax.searchsorted(end_stamps, side='left')
        return (binner, bins, labels)

    def _get_period_bins(self, ax):
        if (not isinstance(ax, PeriodIndex)):
            raise TypeError(f'axis must be a PeriodIndex, but got an instance of {type(ax).__name__}')
        memb = ax.asfreq(self.freq, how=self.convention)
        nat_count = 0
        if memb.hasnans:
            nat_count = np.sum(memb._isnan)
            memb = memb[(~ memb._isnan)]
        if (not len(memb)):
            binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
            return (binner, [], labels)
        freq_mult = self.freq.n
        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how='end')
        bin_shift = 0
        if isinstance(self.freq, Tick):
            (p_start, end) = _get_period_range_edges(start, end, self.freq, closed=self.closed, origin=self.origin, offset=self.offset)
            start_offset = (Period(start, self.freq) - Period(p_start, self.freq))
            bin_shift = (start_offset.n % freq_mult)
            start = p_start
        labels = binner = period_range(start=start, end=end, freq=self.freq, name=ax.name)
        i8 = memb.asi8
        expected_bins_count = (len(binner) * freq_mult)
        i8_extend = (expected_bins_count - (i8[(- 1)] - i8[0]))
        rng = np.arange(i8[0], (i8[(- 1)] + i8_extend), freq_mult)
        rng += freq_mult
        rng -= bin_shift
        prng = type(memb._data)(rng, dtype=memb.dtype)
        bins = memb.searchsorted(prng, side='left')
        if (nat_count > 0):
            bins += nat_count
            bins = np.insert(bins, 0, nat_count)
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)
        return (binner, bins, labels)

def _take_new_index(obj, indexer, new_index, axis=0):
    if isinstance(obj, ABCSeries):
        new_values = algos.take_1d(obj._values, indexer)
        return obj._constructor(new_values, index=new_index, name=obj.name)
    elif isinstance(obj, ABCDataFrame):
        if (axis == 1):
            raise NotImplementedError('axis 1 is not supported')
        return obj._constructor(obj._mgr.reindex_indexer(new_axis=new_index, indexer=indexer, axis=1))
    else:
        raise ValueError("'obj' should be either a Series or a DataFrame")

def _get_timestamp_range_edges(first, last, freq, closed='left', origin='start_day', offset=None):
    "\n    Adjust the `first` Timestamp to the preceding Timestamp that resides on\n    the provided offset. Adjust the `last` Timestamp to the following\n    Timestamp that resides on the provided offset. Input Timestamps that\n    already reside on the offset will be adjusted depending on the type of\n    offset and the `closed` parameter.\n\n    Parameters\n    ----------\n    first : pd.Timestamp\n        The beginning Timestamp of the range to be adjusted.\n    last : pd.Timestamp\n        The ending Timestamp of the range to be adjusted.\n    freq : pd.DateOffset\n        The dateoffset to which the Timestamps will be adjusted.\n    closed : {'right', 'left'}, default None\n        Which side of bin interval is closed.\n    origin : {'epoch', 'start', 'start_day'} or Timestamp, default 'start_day'\n        The timestamp on which to adjust the grouping. The timezone of origin must\n        match the timezone of the index.\n        If a timestamp is not used, these values are also supported:\n\n        - 'epoch': `origin` is 1970-01-01\n        - 'start': `origin` is the first value of the timeseries\n        - 'start_day': `origin` is the first day at midnight of the timeseries\n    offset : pd.Timedelta, default is None\n        An offset timedelta added to the origin.\n\n    Returns\n    -------\n    A tuple of length 2, containing the adjusted pd.Timestamp objects.\n    "
    if isinstance(freq, Tick):
        index_tz = first.tz
        if (isinstance(origin, Timestamp) and ((origin.tz is None) != (index_tz is None))):
            raise ValueError('The origin must have the same timezone as the index.')
        elif (origin == 'epoch'):
            origin = Timestamp('1970-01-01', tz=index_tz)
        if isinstance(freq, Day):
            first = first.tz_localize(None)
            last = last.tz_localize(None)
            if isinstance(origin, Timestamp):
                origin = origin.tz_localize(None)
        (first, last) = _adjust_dates_anchored(first, last, freq, closed=closed, origin=origin, offset=offset)
        if isinstance(freq, Day):
            first = first.tz_localize(index_tz)
            last = last.tz_localize(index_tz)
    else:
        first = first.normalize()
        last = last.normalize()
        if (closed == 'left'):
            first = Timestamp(freq.rollback(first))
        else:
            first = Timestamp((first - freq))
        last = Timestamp((last + freq))
    return (first, last)

def _get_period_range_edges(first, last, freq, closed='left', origin='start_day', offset=None):
    "\n    Adjust the provided `first` and `last` Periods to the respective Period of\n    the given offset that encompasses them.\n\n    Parameters\n    ----------\n    first : pd.Period\n        The beginning Period of the range to be adjusted.\n    last : pd.Period\n        The ending Period of the range to be adjusted.\n    freq : pd.DateOffset\n        The freq to which the Periods will be adjusted.\n    closed : {'right', 'left'}, default None\n        Which side of bin interval is closed.\n    origin : {'epoch', 'start', 'start_day'}, Timestamp, default 'start_day'\n        The timestamp on which to adjust the grouping. The timezone of origin must\n        match the timezone of the index.\n\n        If a timestamp is not used, these values are also supported:\n\n        - 'epoch': `origin` is 1970-01-01\n        - 'start': `origin` is the first value of the timeseries\n        - 'start_day': `origin` is the first day at midnight of the timeseries\n    offset : pd.Timedelta, default is None\n        An offset timedelta added to the origin.\n\n    Returns\n    -------\n    A tuple of length 2, containing the adjusted pd.Period objects.\n    "
    if (not all((isinstance(obj, Period) for obj in [first, last]))):
        raise TypeError("'first' and 'last' must be instances of type Period")
    first = first.to_timestamp()
    last = last.to_timestamp()
    adjust_first = (not freq.is_on_offset(first))
    adjust_last = freq.is_on_offset(last)
    (first, last) = _get_timestamp_range_edges(first, last, freq, closed=closed, origin=origin, offset=offset)
    first = (first + (int(adjust_first) * freq)).to_period(freq)
    last = (last - (int(adjust_last) * freq)).to_period(freq)
    return (first, last)

def _adjust_dates_anchored(first, last, freq, closed='right', origin='start_day', offset=None):
    origin_nanos = 0
    if (origin == 'start_day'):
        origin_nanos = first.normalize().value
    elif (origin == 'start'):
        origin_nanos = first.value
    elif isinstance(origin, Timestamp):
        origin_nanos = origin.value
    elif (origin in ['end', 'end_day']):
        origin = (last if (origin == 'end') else last.ceil('D'))
        sub_freq_times = ((origin.value - first.value) // freq.nanos)
        if (closed == 'left'):
            sub_freq_times += 1
        first = (origin - (sub_freq_times * freq))
        origin_nanos = first.value
    origin_nanos += (offset.value if offset else 0)
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if (first_tzinfo is not None):
        first = first.tz_convert('UTC')
    if (last_tzinfo is not None):
        last = last.tz_convert('UTC')
    foffset = ((first.value - origin_nanos) % freq.nanos)
    loffset = ((last.value - origin_nanos) % freq.nanos)
    if (closed == 'right'):
        if (foffset > 0):
            fresult = (first.value - foffset)
        else:
            fresult = (first.value - freq.nanos)
        if (loffset > 0):
            lresult = (last.value + (freq.nanos - loffset))
        else:
            lresult = last.value
    else:
        if (foffset > 0):
            fresult = (first.value - foffset)
        else:
            fresult = first.value
        if (loffset > 0):
            lresult = (last.value + (freq.nanos - loffset))
        else:
            lresult = (last.value + freq.nanos)
    fresult = Timestamp(fresult)
    lresult = Timestamp(lresult)
    if (first_tzinfo is not None):
        fresult = fresult.tz_localize('UTC').tz_convert(first_tzinfo)
    if (last_tzinfo is not None):
        lresult = lresult.tz_localize('UTC').tz_convert(last_tzinfo)
    return (fresult, lresult)

def asfreq(obj, freq, method=None, how=None, normalize=False, fill_value=None):
    '\n    Utility frequency conversion method for Series/DataFrame.\n\n    See :meth:`pandas.NDFrame.asfreq` for full documentation.\n    '
    if isinstance(obj.index, PeriodIndex):
        if (method is not None):
            raise NotImplementedError("'method' argument is not supported")
        if (how is None):
            how = 'E'
        new_obj = obj.copy()
        new_obj.index = obj.index.asfreq(freq, how=how)
    elif (len(obj.index) == 0):
        new_obj = obj.copy()
        new_obj.index = _asfreq_compat(obj.index, freq)
    else:
        dti = date_range(obj.index[0], obj.index[(- 1)], freq=freq)
        dti.name = obj.index.name
        new_obj = obj.reindex(dti, method=method, fill_value=fill_value)
        if normalize:
            new_obj.index = new_obj.index.normalize()
    return new_obj

def _asfreq_compat(index, freq):
    '\n    Helper to mimic asfreq on (empty) DatetimeIndex and TimedeltaIndex.\n\n    Parameters\n    ----------\n    index : PeriodIndex, DatetimeIndex, or TimedeltaIndex\n    freq : DateOffset\n\n    Returns\n    -------\n    same type as index\n    '
    if (len(index) != 0):
        raise ValueError('Can only set arbitrary freq for empty DatetimeIndex or TimedeltaIndex')
    new_index: Index
    if isinstance(index, PeriodIndex):
        new_index = index.asfreq(freq=freq)
    elif isinstance(index, DatetimeIndex):
        new_index = DatetimeIndex([], dtype=index.dtype, freq=freq, name=index.name)
    elif isinstance(index, TimedeltaIndex):
        new_index = TimedeltaIndex([], dtype=index.dtype, freq=freq, name=index.name)
    else:
        raise TypeError(type(index))
    return new_index
