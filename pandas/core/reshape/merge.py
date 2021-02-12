
'\nSQL-style merge routines\n'
import copy
import datetime
from functools import partial
import hashlib
import string
from typing import TYPE_CHECKING, Optional, Tuple, cast
import warnings
import numpy as np
from pandas._libs import Timedelta, hashtable as libhashtable, join as libjoin, lib
from pandas._typing import ArrayLike, FrameOrSeries, FrameOrSeriesUnion, IndexLabel, Suffixes
from pandas.errors import MergeError
from pandas.util._decorators import Appender, Substitution
from pandas.core.dtypes.common import ensure_float64, ensure_int64, ensure_object, is_array_like, is_bool, is_bool_dtype, is_categorical_dtype, is_datetime64tz_dtype, is_dtype_equal, is_extension_array_dtype, is_float_dtype, is_integer, is_integer_dtype, is_list_like, is_number, is_numeric_dtype, is_object_dtype, needs_i8_conversion
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna, na_value_for_dtype
from pandas import Categorical, Index, MultiIndex
from pandas.core import groupby
import pandas.core.algorithms as algos
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.frame import _merge_doc
from pandas.core.internals import concatenate_block_managers
from pandas.core.sorting import is_int64_overflow_possible
if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.core.arrays import DatetimeArray

@Substitution('\nleft : DataFrame')
@Appender(_merge_doc, indents=0)
def merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
    op = _MergeOperation(left, right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)
    return op.get_result()
if __debug__:
    merge.__doc__ = (_merge_doc % '\nleft : DataFrame')

def _groupby_and_merge(by, on, left, right, merge_pieces):
    '\n    groupby & merge; we are always performing a left-by type operation\n\n    Parameters\n    ----------\n    by: field to group\n    on: duplicates field\n    left: DataFrame\n    right: DataFrame\n    merge_pieces: function for merging\n    '
    pieces = []
    if (not isinstance(by, (list, tuple))):
        by = [by]
    lby = left.groupby(by, sort=False)
    rby: Optional[groupby.DataFrameGroupBy] = None
    if all(((item in right.columns) for item in by)):
        rby = right.groupby(by, sort=False)
    for (key, lhs) in lby:
        if (rby is None):
            rhs = right
        else:
            try:
                rhs = right.take(rby.indices[key])
            except KeyError:
                lcols = lhs.columns.tolist()
                cols = (lcols + [r for r in right.columns if (r not in set(lcols))])
                merged = lhs.reindex(columns=cols)
                merged.index = range(len(merged))
                pieces.append(merged)
                continue
        merged = merge_pieces(lhs, rhs)
        merged[by] = key
        pieces.append(merged)
    from pandas.core.reshape.concat import concat
    result = concat(pieces, ignore_index=True)
    result = result.reindex(columns=pieces[0].columns, copy=False)
    return (result, lby)

def merge_ordered(left, right, on=None, left_on=None, right_on=None, left_by=None, right_by=None, fill_method=None, suffixes=('_x', '_y'), how='outer'):
    '\n    Perform merge with optional filling/interpolation.\n\n    Designed for ordered data like time series data. Optionally\n    perform group-wise merge (see examples).\n\n    Parameters\n    ----------\n    left : DataFrame\n    right : DataFrame\n    on : label or list\n        Field names to join on. Must be found in both DataFrames.\n    left_on : label or list, or array-like\n        Field names to join on in left DataFrame. Can be a vector or list of\n        vectors of the length of the DataFrame to use a particular vector as\n        the join key instead of columns.\n    right_on : label or list, or array-like\n        Field names to join on in right DataFrame or vector/list of vectors per\n        left_on docs.\n    left_by : column name or list of column names\n        Group left DataFrame by group columns and merge piece by piece with\n        right DataFrame.\n    right_by : column name or list of column names\n        Group right DataFrame by group columns and merge piece by piece with\n        left DataFrame.\n    fill_method : {\'ffill\', None}, default None\n        Interpolation method for data.\n    suffixes : list-like, default is ("_x", "_y")\n        A length-2 sequence where each element is optionally a string\n        indicating the suffix to add to overlapping column names in\n        `left` and `right` respectively. Pass a value of `None` instead\n        of a string to indicate that the column name from `left` or\n        `right` should be left as-is, with no suffix. At least one of the\n        values must not be None.\n\n        .. versionchanged:: 0.25.0\n    how : {\'left\', \'right\', \'outer\', \'inner\'}, default \'outer\'\n        * left: use only keys from left frame (SQL: left outer join)\n        * right: use only keys from right frame (SQL: right outer join)\n        * outer: use union of keys from both frames (SQL: full outer join)\n        * inner: use intersection of keys from both frames (SQL: inner join).\n\n    Returns\n    -------\n    DataFrame\n        The merged DataFrame output type will the be same as\n        \'left\', if it is a subclass of DataFrame.\n\n    See Also\n    --------\n    merge : Merge with a database-style join.\n    merge_asof : Merge on nearest keys.\n\n    Examples\n    --------\n    >>> df1 = pd.DataFrame(\n    ...     {\n    ...         "key": ["a", "c", "e", "a", "c", "e"],\n    ...         "lvalue": [1, 2, 3, 1, 2, 3],\n    ...         "group": ["a", "a", "a", "b", "b", "b"]\n    ...     }\n    ... )\n    >>> df1\n          key  lvalue group\n    0   a       1     a\n    1   c       2     a\n    2   e       3     a\n    3   a       1     b\n    4   c       2     b\n    5   e       3     b\n\n    >>> df2 = pd.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})\n    >>> df2\n          key  rvalue\n    0   b       1\n    1   c       2\n    2   d       3\n\n    >>> merge_ordered(df1, df2, fill_method="ffill", left_by="group")\n      key  lvalue group  rvalue\n    0   a       1     a     NaN\n    1   b       1     a     1.0\n    2   c       2     a     2.0\n    3   d       2     a     3.0\n    4   e       3     a     3.0\n    5   a       1     b     NaN\n    6   b       1     b     1.0\n    7   c       2     b     2.0\n    8   d       2     b     3.0\n    9   e       3     b     3.0\n    '

    def _merger(x, y):
        op = _OrderedMerge(x, y, on=on, left_on=left_on, right_on=right_on, suffixes=suffixes, fill_method=fill_method, how=how)
        return op.get_result()
    if ((left_by is not None) and (right_by is not None)):
        raise ValueError('Can only group either left or right frames')
    elif (left_by is not None):
        if isinstance(left_by, str):
            left_by = [left_by]
        check = set(left_by).difference(left.columns)
        if (len(check) != 0):
            raise KeyError(f'{check} not found in left columns')
        (result, _) = _groupby_and_merge(left_by, on, left, right, (lambda x, y: _merger(x, y)))
    elif (right_by is not None):
        if isinstance(right_by, str):
            right_by = [right_by]
        check = set(right_by).difference(right.columns)
        if (len(check) != 0):
            raise KeyError(f'{check} not found in right columns')
        (result, _) = _groupby_and_merge(right_by, on, right, left, (lambda x, y: _merger(y, x)))
    else:
        result = _merger(left, right)
    return result

def merge_asof(left, right, on=None, left_on=None, right_on=None, left_index=False, right_index=False, by=None, left_by=None, right_by=None, suffixes=('_x', '_y'), tolerance=None, allow_exact_matches=True, direction='backward'):
    '\n    Perform an asof merge.\n\n    This is similar to a left-join except that we match on nearest\n    key rather than equal keys. Both DataFrames must be sorted by the key.\n\n    For each row in the left DataFrame:\n\n      - A "backward" search selects the last row in the right DataFrame whose\n        \'on\' key is less than or equal to the left\'s key.\n\n      - A "forward" search selects the first row in the right DataFrame whose\n        \'on\' key is greater than or equal to the left\'s key.\n\n      - A "nearest" search selects the row in the right DataFrame whose \'on\'\n        key is closest in absolute distance to the left\'s key.\n\n    The default is "backward" and is compatible in versions below 0.20.0.\n    The direction parameter was added in version 0.20.0 and introduces\n    "forward" and "nearest".\n\n    Optionally match on equivalent keys with \'by\' before searching with \'on\'.\n\n    Parameters\n    ----------\n    left : DataFrame\n    right : DataFrame\n    on : label\n        Field name to join on. Must be found in both DataFrames.\n        The data MUST be ordered. Furthermore this must be a numeric column,\n        such as datetimelike, integer, or float. On or left_on/right_on\n        must be given.\n    left_on : label\n        Field name to join on in left DataFrame.\n    right_on : label\n        Field name to join on in right DataFrame.\n    left_index : bool\n        Use the index of the left DataFrame as the join key.\n    right_index : bool\n        Use the index of the right DataFrame as the join key.\n    by : column name or list of column names\n        Match on these columns before performing merge operation.\n    left_by : column name\n        Field names to match on in the left DataFrame.\n    right_by : column name\n        Field names to match on in the right DataFrame.\n    suffixes : 2-length sequence (tuple, list, ...)\n        Suffix to apply to overlapping column names in the left and right\n        side, respectively.\n    tolerance : int or Timedelta, optional, default None\n        Select asof tolerance within this range; must be compatible\n        with the merge index.\n    allow_exact_matches : bool, default True\n\n        - If True, allow matching with the same \'on\' value\n          (i.e. less-than-or-equal-to / greater-than-or-equal-to)\n        - If False, don\'t match the same \'on\' value\n          (i.e., strictly less-than / strictly greater-than).\n\n    direction : \'backward\' (default), \'forward\', or \'nearest\'\n        Whether to search for prior, subsequent, or closest matches.\n\n    Returns\n    -------\n    merged : DataFrame\n\n    See Also\n    --------\n    merge : Merge with a database-style join.\n    merge_ordered : Merge with optional filling/interpolation.\n\n    Examples\n    --------\n    >>> left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})\n    >>> left\n        a left_val\n    0   1        a\n    1   5        b\n    2  10        c\n\n    >>> right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})\n    >>> right\n       a  right_val\n    0  1          1\n    1  2          2\n    2  3          3\n    3  6          6\n    4  7          7\n\n    >>> pd.merge_asof(left, right, on="a")\n        a left_val  right_val\n    0   1        a          1\n    1   5        b          3\n    2  10        c          7\n\n    >>> pd.merge_asof(left, right, on="a", allow_exact_matches=False)\n        a left_val  right_val\n    0   1        a        NaN\n    1   5        b        3.0\n    2  10        c        7.0\n\n    >>> pd.merge_asof(left, right, on="a", direction="forward")\n        a left_val  right_val\n    0   1        a        1.0\n    1   5        b        6.0\n    2  10        c        NaN\n\n    >>> pd.merge_asof(left, right, on="a", direction="nearest")\n        a left_val  right_val\n    0   1        a          1\n    1   5        b          6\n    2  10        c          7\n\n    We can use indexed DataFrames as well.\n\n    >>> left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])\n    >>> left\n       left_val\n    1         a\n    5         b\n    10        c\n\n    >>> right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])\n    >>> right\n       right_val\n    1          1\n    2          2\n    3          3\n    6          6\n    7          7\n\n    >>> pd.merge_asof(left, right, left_index=True, right_index=True)\n       left_val  right_val\n    1         a          1\n    5         b          3\n    10        c          7\n\n    Here is a real-world times-series example\n\n    >>> quotes = pd.DataFrame(\n    ...     {\n    ...         "time": [\n    ...             pd.Timestamp("2016-05-25 13:30:00.023"),\n    ...             pd.Timestamp("2016-05-25 13:30:00.023"),\n    ...             pd.Timestamp("2016-05-25 13:30:00.030"),\n    ...             pd.Timestamp("2016-05-25 13:30:00.041"),\n    ...             pd.Timestamp("2016-05-25 13:30:00.048"),\n    ...             pd.Timestamp("2016-05-25 13:30:00.049"),\n    ...             pd.Timestamp("2016-05-25 13:30:00.072"),\n    ...             pd.Timestamp("2016-05-25 13:30:00.075")\n    ...         ],\n    ...         "ticker": [\n    ...                "GOOG",\n    ...                "MSFT",\n    ...                "MSFT",\n    ...                "MSFT",\n    ...                "GOOG",\n    ...                "AAPL",\n    ...                "GOOG",\n    ...                "MSFT"\n    ...            ],\n    ...            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],\n    ...            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03]\n    ...     }\n    ... )\n    >>> quotes\n                         time ticker     bid     ask\n    0 2016-05-25 13:30:00.023   GOOG  720.50  720.93\n    1 2016-05-25 13:30:00.023   MSFT   51.95   51.96\n    2 2016-05-25 13:30:00.030   MSFT   51.97   51.98\n    3 2016-05-25 13:30:00.041   MSFT   51.99   52.00\n    4 2016-05-25 13:30:00.048   GOOG  720.50  720.93\n    5 2016-05-25 13:30:00.049   AAPL   97.99   98.01\n    6 2016-05-25 13:30:00.072   GOOG  720.50  720.88\n    7 2016-05-25 13:30:00.075   MSFT   52.01   52.03\n\n    >>> trades = pd.DataFrame(\n    ...        {\n    ...            "time": [\n    ...                pd.Timestamp("2016-05-25 13:30:00.023"),\n    ...                pd.Timestamp("2016-05-25 13:30:00.038"),\n    ...                pd.Timestamp("2016-05-25 13:30:00.048"),\n    ...                pd.Timestamp("2016-05-25 13:30:00.048"),\n    ...                pd.Timestamp("2016-05-25 13:30:00.048")\n    ...            ],\n    ...            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],\n    ...            "price": [51.95, 51.95, 720.77, 720.92, 98.0],\n    ...            "quantity": [75, 155, 100, 100, 100]\n    ...        }\n    ...    )\n    >>> trades\n                         time ticker   price  quantity\n    0 2016-05-25 13:30:00.023   MSFT   51.95        75\n    1 2016-05-25 13:30:00.038   MSFT   51.95       155\n    2 2016-05-25 13:30:00.048   GOOG  720.77       100\n    3 2016-05-25 13:30:00.048   GOOG  720.92       100\n    4 2016-05-25 13:30:00.048   AAPL   98.00       100\n\n    By default we are taking the asof of the quotes\n\n    >>> pd.merge_asof(trades, quotes, on="time", by="ticker")\n                         time ticker   price  quantity     bid     ask\n    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96\n    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98\n    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93\n    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93\n    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN\n\n    We only asof within 2ms between the quote time and the trade time\n\n    >>> pd.merge_asof(\n    ...     trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("2ms")\n    ... )\n                         time ticker   price  quantity     bid     ask\n    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96\n    1 2016-05-25 13:30:00.038   MSFT   51.95       155     NaN     NaN\n    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93\n    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93\n    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN\n\n    We only asof within 10ms between the quote time and the trade time\n    and we exclude exact matches on time. However *prior* data will\n    propagate forward\n\n    >>> pd.merge_asof(\n    ...     trades,\n    ...     quotes,\n    ...     on="time",\n    ...     by="ticker",\n    ...     tolerance=pd.Timedelta("10ms"),\n    ...     allow_exact_matches=False\n    ... )\n                         time ticker   price  quantity     bid     ask\n    0 2016-05-25 13:30:00.023   MSFT   51.95        75     NaN     NaN\n    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98\n    2 2016-05-25 13:30:00.048   GOOG  720.77       100     NaN     NaN\n    3 2016-05-25 13:30:00.048   GOOG  720.92       100     NaN     NaN\n    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN\n    '
    op = _AsOfMerge(left, right, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, by=by, left_by=left_by, right_by=right_by, suffixes=suffixes, how='asof', tolerance=tolerance, allow_exact_matches=allow_exact_matches, direction=direction)
    return op.get_result()

class _MergeOperation():
    '\n    Perform a database (SQL) merge operation between two DataFrame or Series\n    objects using either columns as keys or their row indexes\n    '
    _merge_type = 'merge'

    def __init__(self, left, right, how='inner', on=None, left_on=None, right_on=None, axis=1, left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        _left = _validate_operand(left)
        _right = _validate_operand(right)
        self.left = self.orig_left = _left
        self.right = self.orig_right = _right
        self.how = how
        self.bm_axis = axis
        self.axis = ((1 - axis) if (self.left.ndim == 2) else 0)
        self.on = com.maybe_make_list(on)
        self.left_on = com.maybe_make_list(left_on)
        self.right_on = com.maybe_make_list(right_on)
        self.copy = copy
        self.suffixes = suffixes
        self.sort = sort
        self.left_index = left_index
        self.right_index = right_index
        self.indicator = indicator
        self.indicator_name: Optional[str]
        if isinstance(self.indicator, str):
            self.indicator_name = self.indicator
        elif isinstance(self.indicator, bool):
            self.indicator_name = ('_merge' if self.indicator else None)
        else:
            raise ValueError('indicator option can only accept boolean or string arguments')
        if (not is_bool(left_index)):
            raise ValueError(f'left_index parameter must be of type bool, not {type(left_index)}')
        if (not is_bool(right_index)):
            raise ValueError(f'right_index parameter must be of type bool, not {type(right_index)}')
        if (_left.columns.nlevels != _right.columns.nlevels):
            msg = f'merging between different levels can give an unintended result ({left.columns.nlevels} levels on the left,{right.columns.nlevels} on the right)'
            warnings.warn(msg, UserWarning)
        self._validate_specification()
        cross_col = None
        if (self.how == 'cross'):
            (self.left, self.right, self.how, cross_col) = self._create_cross_configuration(self.left, self.right)
            self.left_on = self.right_on = [cross_col]
        self._cross = cross_col
        (self.left_join_keys, self.right_join_keys, self.join_names) = self._get_merge_keys()
        self._maybe_coerce_merge_keys()
        if (validate is not None):
            self._validate(validate)

    def get_result(self):
        if self.indicator:
            (self.left, self.right) = self._indicator_pre_merge(self.left, self.right)
        (join_index, left_indexer, right_indexer) = self._get_join_info()
        (llabels, rlabels) = _items_overlap_with_suffix(self.left._info_axis, self.right._info_axis, self.suffixes)
        lindexers = ({1: left_indexer} if (left_indexer is not None) else {})
        rindexers = ({1: right_indexer} if (right_indexer is not None) else {})
        result_data = concatenate_block_managers([(self.left._mgr, lindexers), (self.right._mgr, rindexers)], axes=[llabels.append(rlabels), join_index], concat_axis=0, copy=self.copy)
        typ = self.left._constructor
        result = typ(result_data).__finalize__(self, method=self._merge_type)
        if self.indicator:
            result = self._indicator_post_merge(result)
        self._maybe_add_join_keys(result, left_indexer, right_indexer)
        self._maybe_restore_index_levels(result)
        self._maybe_drop_cross_column(result, self._cross)
        return result.__finalize__(self, method='merge')

    def _maybe_drop_cross_column(self, result, cross_col):
        if (cross_col is not None):
            result.drop(columns=cross_col, inplace=True)

    def _indicator_pre_merge(self, left, right):
        columns = left.columns.union(right.columns)
        for i in ['_left_indicator', '_right_indicator']:
            if (i in columns):
                raise ValueError(f'Cannot use `indicator=True` option when data contains a column named {i}')
        if (self.indicator_name in columns):
            raise ValueError('Cannot use name of an existing column for indicator column')
        left = left.copy()
        right = right.copy()
        left['_left_indicator'] = 1
        left['_left_indicator'] = left['_left_indicator'].astype('int8')
        right['_right_indicator'] = 2
        right['_right_indicator'] = right['_right_indicator'].astype('int8')
        return (left, right)

    def _indicator_post_merge(self, result):
        result['_left_indicator'] = result['_left_indicator'].fillna(0)
        result['_right_indicator'] = result['_right_indicator'].fillna(0)
        result[self.indicator_name] = Categorical((result['_left_indicator'] + result['_right_indicator']), categories=[1, 2, 3])
        result[self.indicator_name] = result[self.indicator_name].cat.rename_categories(['left_only', 'right_only', 'both'])
        result = result.drop(labels=['_left_indicator', '_right_indicator'], axis=1)
        return result

    def _maybe_restore_index_levels(self, result):
        '\n        Restore index levels specified as `on` parameters\n\n        Here we check for cases where `self.left_on` and `self.right_on` pairs\n        each reference an index level in their respective DataFrames. The\n        joined columns corresponding to these pairs are then restored to the\n        index of `result`.\n\n        **Note:** This method has side effects. It modifies `result` in-place\n\n        Parameters\n        ----------\n        result: DataFrame\n            merge result\n\n        Returns\n        -------\n        None\n        '
        names_to_restore = []
        for (name, left_key, right_key) in zip(self.join_names, self.left_on, self.right_on):
            if (self.orig_left._is_level_reference(left_key) and self.orig_right._is_level_reference(right_key) and (name not in result.index.names)):
                names_to_restore.append(name)
        if names_to_restore:
            result.set_index(names_to_restore, inplace=True)

    def _maybe_add_join_keys(self, result, left_indexer, right_indexer):
        left_has_missing = None
        right_has_missing = None
        keys = zip(self.join_names, self.left_on, self.right_on)
        for (i, (name, lname, rname)) in enumerate(keys):
            if (not _should_fill(lname, rname)):
                continue
            (take_left, take_right) = (None, None)
            if (name in result):
                if ((left_indexer is not None) and (right_indexer is not None)):
                    if (name in self.left):
                        if (left_has_missing is None):
                            left_has_missing = (left_indexer == (- 1)).any()
                        if left_has_missing:
                            take_right = self.right_join_keys[i]
                            if (not is_dtype_equal(result[name].dtype, self.left[name].dtype)):
                                take_left = self.left[name]._values
                    elif (name in self.right):
                        if (right_has_missing is None):
                            right_has_missing = (right_indexer == (- 1)).any()
                        if right_has_missing:
                            take_left = self.left_join_keys[i]
                            if (not is_dtype_equal(result[name].dtype, self.right[name].dtype)):
                                take_right = self.right[name]._values
            elif ((left_indexer is not None) and is_array_like(self.left_join_keys[i])):
                take_left = self.left_join_keys[i]
                take_right = self.right_join_keys[i]
            if ((take_left is not None) or (take_right is not None)):
                if (take_left is None):
                    lvals = result[name]._values
                else:
                    lfill = na_value_for_dtype(take_left.dtype)
                    lvals = algos.take_1d(take_left, left_indexer, fill_value=lfill)
                if (take_right is None):
                    rvals = result[name]._values
                else:
                    rfill = na_value_for_dtype(take_right.dtype)
                    rvals = algos.take_1d(take_right, right_indexer, fill_value=rfill)
                mask_left = (left_indexer == (- 1))
                mask_right = (right_indexer == (- 1))
                if mask_left.all():
                    key_col = rvals
                elif ((right_indexer is not None) and mask_right.all()):
                    key_col = lvals
                else:
                    key_col = Index(lvals).where((~ mask_left), rvals)
                if result._is_label_reference(name):
                    result[name] = key_col
                elif result._is_level_reference(name):
                    if isinstance(result.index, MultiIndex):
                        key_col.name = name
                        idx_list = [(result.index.get_level_values(level_name) if (level_name != name) else key_col) for level_name in result.index.names]
                        result.set_index(idx_list, inplace=True)
                    else:
                        result.index = Index(key_col, name=name)
                else:
                    result.insert(i, (name or f'key_{i}'), key_col)

    def _get_join_indexers(self):
        ' return the join indexers '
        return get_join_indexers(self.left_join_keys, self.right_join_keys, sort=self.sort, how=self.how)

    def _get_join_info(self):
        left_ax = self.left.axes[self.axis]
        right_ax = self.right.axes[self.axis]
        if (self.left_index and self.right_index and (self.how != 'asof')):
            (join_index, left_indexer, right_indexer) = left_ax.join(right_ax, how=self.how, return_indexers=True, sort=self.sort)
        elif (self.right_index and (self.how == 'left')):
            (join_index, left_indexer, right_indexer) = _left_join_on_index(left_ax, right_ax, self.left_join_keys, sort=self.sort)
        elif (self.left_index and (self.how == 'right')):
            (join_index, right_indexer, left_indexer) = _left_join_on_index(right_ax, left_ax, self.right_join_keys, sort=self.sort)
        else:
            (left_indexer, right_indexer) = self._get_join_indexers()
            if self.right_index:
                if (len(self.left) > 0):
                    join_index = self._create_join_index(self.left.index, self.right.index, left_indexer, right_indexer, how='right')
                else:
                    join_index = self.right.index.take(right_indexer)
                    left_indexer = np.array(([(- 1)] * len(join_index)))
            elif self.left_index:
                if (len(self.right) > 0):
                    join_index = self._create_join_index(self.right.index, self.left.index, right_indexer, left_indexer, how='left')
                else:
                    join_index = self.left.index.take(left_indexer)
                    right_indexer = np.array(([(- 1)] * len(join_index)))
            else:
                join_index = Index(np.arange(len(left_indexer)))
        if (len(join_index) == 0):
            join_index = join_index.astype(object)
        return (join_index, left_indexer, right_indexer)

    def _create_join_index(self, index, other_index, indexer, other_indexer, how='left'):
        '\n        Create a join index by rearranging one index to match another\n\n        Parameters\n        ----------\n        index: Index being rearranged\n        other_index: Index used to supply values not found in index\n        indexer: how to rearrange index\n        how: replacement is only necessary if indexer based on other_index\n\n        Returns\n        -------\n        join_index\n        '
        if ((self.how in (how, 'outer')) and (not isinstance(other_index, MultiIndex))):
            mask = (indexer == (- 1))
            if np.any(mask):
                fill_value = na_value_for_dtype(index.dtype, compat=False)
                index = index.append(Index([fill_value]))
        return index.take(indexer)

    def _get_merge_keys(self):
        '\n        Note: has side effects (copy/delete key columns)\n\n        Parameters\n        ----------\n        left\n        right\n        on\n\n        Returns\n        -------\n        left_keys, right_keys\n        '
        left_keys = []
        right_keys = []
        join_names = []
        right_drop = []
        left_drop = []
        (left, right) = (self.left, self.right)
        is_lkey = (lambda x: (is_array_like(x) and (len(x) == len(left))))
        is_rkey = (lambda x: (is_array_like(x) and (len(x) == len(right))))
        if (_any(self.left_on) and _any(self.right_on)):
            for (lk, rk) in zip(self.left_on, self.right_on):
                if is_lkey(lk):
                    left_keys.append(lk)
                    if is_rkey(rk):
                        right_keys.append(rk)
                        join_names.append(None)
                    elif (rk is not None):
                        right_keys.append(right._get_label_or_level_values(rk))
                        join_names.append(rk)
                    else:
                        right_keys.append(right.index)
                        join_names.append(right.index.name)
                else:
                    if (not is_rkey(rk)):
                        if (rk is not None):
                            right_keys.append(right._get_label_or_level_values(rk))
                        else:
                            right_keys.append(right.index)
                        if ((lk is not None) and (lk == rk)):
                            if (len(left) > 0):
                                right_drop.append(rk)
                            else:
                                left_drop.append(lk)
                    else:
                        right_keys.append(rk)
                    if (lk is not None):
                        left_keys.append(left._get_label_or_level_values(lk))
                        join_names.append(lk)
                    else:
                        left_keys.append(left.index)
                        join_names.append(left.index.name)
        elif _any(self.left_on):
            for k in self.left_on:
                if is_lkey(k):
                    left_keys.append(k)
                    join_names.append(None)
                else:
                    left_keys.append(left._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.right.index, MultiIndex):
                right_keys = [lev._values.take(lev_codes) for (lev, lev_codes) in zip(self.right.index.levels, self.right.index.codes)]
            else:
                right_keys = [self.right.index._values]
        elif _any(self.right_on):
            for k in self.right_on:
                if is_rkey(k):
                    right_keys.append(k)
                    join_names.append(None)
                else:
                    right_keys.append(right._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.left.index, MultiIndex):
                left_keys = [lev._values.take(lev_codes) for (lev, lev_codes) in zip(self.left.index.levels, self.left.index.codes)]
            else:
                left_keys = [self.left.index._values]
        if left_drop:
            self.left = self.left._drop_labels_or_levels(left_drop)
        if right_drop:
            self.right = self.right._drop_labels_or_levels(right_drop)
        return (left_keys, right_keys, join_names)

    def _maybe_coerce_merge_keys(self):
        for (lk, rk, name) in zip(self.left_join_keys, self.right_join_keys, self.join_names):
            if ((len(lk) and (not len(rk))) or ((not len(lk)) and len(rk))):
                continue
            lk_is_cat = is_categorical_dtype(lk.dtype)
            rk_is_cat = is_categorical_dtype(rk.dtype)
            lk_is_object = is_object_dtype(lk.dtype)
            rk_is_object = is_object_dtype(rk.dtype)
            if (lk_is_cat and rk_is_cat):
                if lk._categories_match_up_to_permutation(rk):
                    continue
            elif (lk_is_cat or rk_is_cat):
                pass
            elif is_dtype_equal(lk.dtype, rk.dtype):
                continue
            msg = f'You are trying to merge on {lk.dtype} and {rk.dtype} columns. If you wish to proceed you should use pd.concat'
            if (is_numeric_dtype(lk.dtype) and is_numeric_dtype(rk.dtype)):
                if (lk.dtype.kind == rk.dtype.kind):
                    continue
                elif (is_integer_dtype(rk.dtype) and is_float_dtype(lk.dtype)):
                    if (not (lk == lk.astype(rk.dtype))[(~ np.isnan(lk))].all()):
                        warnings.warn('You are merging on int and float columns where the float values are not equal to their int representation', UserWarning)
                    continue
                elif (is_float_dtype(rk.dtype) and is_integer_dtype(lk.dtype)):
                    if (not (rk == rk.astype(lk.dtype))[(~ np.isnan(rk))].all()):
                        warnings.warn('You are merging on int and float columns where the float values are not equal to their int representation', UserWarning)
                    continue
                elif (lib.infer_dtype(lk, skipna=False) == lib.infer_dtype(rk, skipna=False)):
                    continue
            elif ((lk_is_object and is_bool_dtype(rk.dtype)) or (is_bool_dtype(lk.dtype) and rk_is_object)):
                pass
            elif ((lk_is_object and is_numeric_dtype(rk.dtype)) or (is_numeric_dtype(lk.dtype) and rk_is_object)):
                inferred_left = lib.infer_dtype(lk, skipna=False)
                inferred_right = lib.infer_dtype(rk, skipna=False)
                bool_types = ['integer', 'mixed-integer', 'boolean', 'empty']
                string_types = ['string', 'unicode', 'mixed', 'bytes', 'empty']
                if ((inferred_left in bool_types) and (inferred_right in bool_types)):
                    pass
                elif (((inferred_left in string_types) and (inferred_right not in string_types)) or ((inferred_right in string_types) and (inferred_left not in string_types))):
                    raise ValueError(msg)
            elif (needs_i8_conversion(lk.dtype) and (not needs_i8_conversion(rk.dtype))):
                raise ValueError(msg)
            elif ((not needs_i8_conversion(lk.dtype)) and needs_i8_conversion(rk.dtype)):
                raise ValueError(msg)
            elif (is_datetime64tz_dtype(lk.dtype) and (not is_datetime64tz_dtype(rk.dtype))):
                raise ValueError(msg)
            elif ((not is_datetime64tz_dtype(lk.dtype)) and is_datetime64tz_dtype(rk.dtype)):
                raise ValueError(msg)
            elif (lk_is_object and rk_is_object):
                continue
            if (name in self.left.columns):
                typ = (lk.categories.dtype if lk_is_cat else object)
                self.left = self.left.assign(**{name: self.left[name].astype(typ)})
            if (name in self.right.columns):
                typ = (rk.categories.dtype if rk_is_cat else object)
                self.right = self.right.assign(**{name: self.right[name].astype(typ)})

    def _create_cross_configuration(self, left, right):
        '\n        Creates the configuration to dispatch the cross operation to inner join,\n        e.g. adding a join column and resetting parameters. Join column is added\n        to a new object, no inplace modification\n\n        Parameters\n        ----------\n        left: DataFrame\n        right DataFrame\n\n        Returns\n        -------\n            a tuple (left, right, how, cross_col) representing the adjusted\n            DataFrames with cross_col, the merge operation set to inner and the column\n            to join over.\n        '
        cross_col = f'_cross_{hashlib.md5().hexdigest()}'
        how = 'inner'
        return (left.assign(**{cross_col: 1}), right.assign(**{cross_col: 1}), how, cross_col)

    def _validate_specification(self):
        if (self.how == 'cross'):
            if (self.left_index or self.right_index or (self.right_on is not None) or (self.left_on is not None) or (self.on is not None)):
                raise MergeError('Can not pass on, right_on, left_on or set right_index=True or left_index=True')
            return
        elif ((self.on is None) and (self.left_on is None) and (self.right_on is None)):
            if (self.left_index and self.right_index):
                (self.left_on, self.right_on) = ((), ())
            elif self.left_index:
                raise MergeError('Must pass right_on or right_index=True')
            elif self.right_index:
                raise MergeError('Must pass left_on or left_index=True')
            else:
                left_cols = self.left.columns
                right_cols = self.right.columns
                common_cols = left_cols.intersection(right_cols)
                if (len(common_cols) == 0):
                    raise MergeError(f'No common columns to perform merge on. Merge options: left_on={self.left_on}, right_on={self.right_on}, left_index={self.left_index}, right_index={self.right_index}')
                if ((not left_cols.join(common_cols, how='inner').is_unique) or (not right_cols.join(common_cols, how='inner').is_unique)):
                    raise MergeError(f'Data columns not unique: {repr(common_cols)}')
                self.left_on = self.right_on = common_cols
        elif (self.on is not None):
            if ((self.left_on is not None) or (self.right_on is not None)):
                raise MergeError('Can only pass argument "on" OR "left_on" and "right_on", not a combination of both.')
            if (self.left_index or self.right_index):
                raise MergeError('Can only pass argument "on" OR "left_index" and "right_index", not a combination of both.')
            self.left_on = self.right_on = self.on
        elif (self.left_on is not None):
            if self.left_index:
                raise MergeError('Can only pass argument "left_on" OR "left_index" not both.')
            if ((not self.right_index) and (self.right_on is None)):
                raise MergeError('Must pass "right_on" OR "right_index".')
            n = len(self.left_on)
            if self.right_index:
                if (len(self.left_on) != self.right.index.nlevels):
                    raise ValueError('len(left_on) must equal the number of levels in the index of "right"')
                self.right_on = ([None] * n)
        elif (self.right_on is not None):
            if self.right_index:
                raise MergeError('Can only pass argument "right_on" OR "right_index" not both.')
            if ((not self.left_index) and (self.left_on is None)):
                raise MergeError('Must pass "left_on" OR "left_index".')
            n = len(self.right_on)
            if self.left_index:
                if (len(self.right_on) != self.left.index.nlevels):
                    raise ValueError('len(right_on) must equal the number of levels in the index of "left"')
                self.left_on = ([None] * n)
        if ((self.how != 'cross') and (len(self.right_on) != len(self.left_on))):
            raise ValueError('len(right_on) must equal len(left_on)')

    def _validate(self, validate):
        if self.left_index:
            left_unique = self.orig_left.index.is_unique
        else:
            left_unique = MultiIndex.from_arrays(self.left_join_keys).is_unique
        if self.right_index:
            right_unique = self.orig_right.index.is_unique
        else:
            right_unique = MultiIndex.from_arrays(self.right_join_keys).is_unique
        if (validate in ['one_to_one', '1:1']):
            if ((not left_unique) and (not right_unique)):
                raise MergeError('Merge keys are not unique in either left or right dataset; not a one-to-one merge')
            elif (not left_unique):
                raise MergeError('Merge keys are not unique in left dataset; not a one-to-one merge')
            elif (not right_unique):
                raise MergeError('Merge keys are not unique in right dataset; not a one-to-one merge')
        elif (validate in ['one_to_many', '1:m']):
            if (not left_unique):
                raise MergeError('Merge keys are not unique in left dataset; not a one-to-many merge')
        elif (validate in ['many_to_one', 'm:1']):
            if (not right_unique):
                raise MergeError('Merge keys are not unique in right dataset; not a many-to-one merge')
        elif (validate in ['many_to_many', 'm:m']):
            pass
        else:
            raise ValueError('Not a valid argument for validate')

def get_join_indexers(left_keys, right_keys, sort=False, how='inner', **kwargs):
    "\n\n    Parameters\n    ----------\n    left_keys: ndarray, Index, Series\n    right_keys: ndarray, Index, Series\n    sort: bool, default False\n    how: string {'inner', 'outer', 'left', 'right'}, default 'inner'\n\n    Returns\n    -------\n    tuple of (left_indexer, right_indexer)\n        indexers into the left_keys, right_keys\n\n    "
    assert (len(left_keys) == len(right_keys)), 'left_key and right_keys must be the same length'
    mapped = (_factorize_keys(left_keys[n], right_keys[n], sort=sort, how=how) for n in range(len(left_keys)))
    zipped = zip(*mapped)
    (llab, rlab, shape) = [list(x) for x in zipped]
    (lkey, rkey) = _get_join_keys(llab, rlab, shape, sort)
    (lkey, rkey, count) = _factorize_keys(lkey, rkey, sort=sort, how=how)
    kwargs = copy.copy(kwargs)
    if (how in ('left', 'right')):
        kwargs['sort'] = sort
    join_func = {'inner': libjoin.inner_join, 'left': libjoin.left_outer_join, 'right': (lambda x, y, count, **kwargs: libjoin.left_outer_join(y, x, count, **kwargs)[::(- 1)]), 'outer': libjoin.full_outer_join}[how]
    return join_func(lkey, rkey, count, **kwargs)

def restore_dropped_levels_multijoin(left, right, dropped_level_names, join_index, lindexer, rindexer):
    '\n    *this is an internal non-public method*\n\n    Returns the levels, labels and names of a multi-index to multi-index join.\n    Depending on the type of join, this method restores the appropriate\n    dropped levels of the joined multi-index.\n    The method relies on lidx, rindexer which hold the index positions of\n    left and right, where a join was feasible\n\n    Parameters\n    ----------\n    left : MultiIndex\n        left index\n    right : MultiIndex\n        right index\n    dropped_level_names : str array\n        list of non-common level names\n    join_index : MultiIndex\n        the index of the join between the\n        common levels of left and right\n    lindexer : intp array\n        left indexer\n    rindexer : intp array\n        right indexer\n\n    Returns\n    -------\n    levels : list of Index\n        levels of combined multiindexes\n    labels : intp array\n        labels of combined multiindexes\n    names : str array\n        names of combined multiindexes\n\n    '

    def _convert_to_multiindex(index) -> MultiIndex:
        if isinstance(index, MultiIndex):
            return index
        else:
            return MultiIndex.from_arrays([index._values], names=[index.name])
    join_index = _convert_to_multiindex(join_index)
    join_levels = join_index.levels
    join_codes = join_index.codes
    join_names = join_index.names
    if (lindexer is None):
        lindexer = range(left.size)
    if (rindexer is None):
        rindexer = range(right.size)
    for dropped_level_name in dropped_level_names:
        if (dropped_level_name in left.names):
            idx = left
            indexer = lindexer
        else:
            idx = right
            indexer = rindexer
        name_idx = idx.names.index(dropped_level_name)
        restore_levels = idx.levels[name_idx]
        codes = idx.codes[name_idx]
        restore_codes = algos.take_nd(codes, indexer, fill_value=(- 1))
        join_levels = (join_levels + [restore_levels])
        join_codes = (join_codes + [restore_codes])
        join_names = (join_names + [dropped_level_name])
    return (join_levels, join_codes, join_names)

class _OrderedMerge(_MergeOperation):
    _merge_type = 'ordered_merge'

    def __init__(self, left, right, on=None, left_on=None, right_on=None, left_index=False, right_index=False, axis=1, suffixes=('_x', '_y'), copy=True, fill_method=None, how='outer'):
        self.fill_method = fill_method
        _MergeOperation.__init__(self, left, right, on=on, left_on=left_on, left_index=left_index, right_index=right_index, right_on=right_on, axis=axis, how=how, suffixes=suffixes, sort=True)

    def get_result(self):
        (join_index, left_indexer, right_indexer) = self._get_join_info()
        (llabels, rlabels) = _items_overlap_with_suffix(self.left._info_axis, self.right._info_axis, self.suffixes)
        if (self.fill_method == 'ffill'):
            left_join_indexer = libjoin.ffill_indexer(left_indexer)
            right_join_indexer = libjoin.ffill_indexer(right_indexer)
        else:
            left_join_indexer = left_indexer
            right_join_indexer = right_indexer
        lindexers = ({1: left_join_indexer} if (left_join_indexer is not None) else {})
        rindexers = ({1: right_join_indexer} if (right_join_indexer is not None) else {})
        result_data = concatenate_block_managers([(self.left._mgr, lindexers), (self.right._mgr, rindexers)], axes=[llabels.append(rlabels), join_index], concat_axis=0, copy=self.copy)
        typ = self.left._constructor
        result = typ(result_data)
        self._maybe_add_join_keys(result, left_indexer, right_indexer)
        return result

def _asof_function(direction):
    name = f'asof_join_{direction}'
    return getattr(libjoin, name, None)

def _asof_by_function(direction):
    name = f'asof_join_{direction}_on_X_by_Y'
    return getattr(libjoin, name, None)
_type_casters = {'int64_t': ensure_int64, 'double': ensure_float64, 'object': ensure_object}

def _get_cython_type_upcast(dtype):
    " Upcast a dtype to 'int64_t', 'double', or 'object' "
    if is_integer_dtype(dtype):
        return 'int64_t'
    elif is_float_dtype(dtype):
        return 'double'
    else:
        return 'object'

class _AsOfMerge(_OrderedMerge):
    _merge_type = 'asof_merge'

    def __init__(self, left, right, on=None, left_on=None, right_on=None, left_index=False, right_index=False, by=None, left_by=None, right_by=None, axis=1, suffixes=('_x', '_y'), copy=True, fill_method=None, how='asof', tolerance=None, allow_exact_matches=True, direction='backward'):
        self.by = by
        self.left_by = left_by
        self.right_by = right_by
        self.tolerance = tolerance
        self.allow_exact_matches = allow_exact_matches
        self.direction = direction
        _OrderedMerge.__init__(self, left, right, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, axis=axis, how=how, suffixes=suffixes, fill_method=fill_method)

    def _validate_specification(self):
        super()._validate_specification()
        if ((len(self.left_on) != 1) and (not self.left_index)):
            raise MergeError('can only asof on a key for left')
        if ((len(self.right_on) != 1) and (not self.right_index)):
            raise MergeError('can only asof on a key for right')
        if (self.left_index and isinstance(self.left.index, MultiIndex)):
            raise MergeError('left can only have one index')
        if (self.right_index and isinstance(self.right.index, MultiIndex)):
            raise MergeError('right can only have one index')
        if (self.by is not None):
            if ((self.left_by is not None) or (self.right_by is not None)):
                raise MergeError('Can only pass by OR left_by and right_by')
            self.left_by = self.right_by = self.by
        if ((self.left_by is None) and (self.right_by is not None)):
            raise MergeError('missing left_by')
        if ((self.left_by is not None) and (self.right_by is None)):
            raise MergeError('missing right_by')
        if (self.left_by is not None):
            if (not is_list_like(self.left_by)):
                self.left_by = [self.left_by]
            if (not is_list_like(self.right_by)):
                self.right_by = [self.right_by]
            if (len(self.left_by) != len(self.right_by)):
                raise MergeError('left_by and right_by must be same length')
            self.left_on = (self.left_by + list(self.left_on))
            self.right_on = (self.right_by + list(self.right_on))
        if (self.direction not in ['backward', 'forward', 'nearest']):
            raise MergeError(f'direction invalid: {self.direction}')

    def _get_merge_keys(self):
        (left_join_keys, right_join_keys, join_names) = super()._get_merge_keys()
        for (i, (lk, rk)) in enumerate(zip(left_join_keys, right_join_keys)):
            if (not is_dtype_equal(lk.dtype, rk.dtype)):
                if (is_categorical_dtype(lk.dtype) and is_categorical_dtype(rk.dtype)):
                    msg = f'incompatible merge keys [{i}] {repr(lk.dtype)} and {repr(rk.dtype)}, both sides category, but not equal ones'
                else:
                    msg = f'incompatible merge keys [{i}] {repr(lk.dtype)} and {repr(rk.dtype)}, must be the same type'
                raise MergeError(msg)
        if (self.tolerance is not None):
            if self.left_index:
                lt = self.left.index
            else:
                lt = left_join_keys[(- 1)]
            msg = f'incompatible tolerance {self.tolerance}, must be compat with type {repr(lt.dtype)}'
            if needs_i8_conversion(lt):
                if (not isinstance(self.tolerance, datetime.timedelta)):
                    raise MergeError(msg)
                if (self.tolerance < Timedelta(0)):
                    raise MergeError('tolerance must be positive')
            elif is_integer_dtype(lt):
                if (not is_integer(self.tolerance)):
                    raise MergeError(msg)
                if (self.tolerance < 0):
                    raise MergeError('tolerance must be positive')
            elif is_float_dtype(lt):
                if (not is_number(self.tolerance)):
                    raise MergeError(msg)
                if (self.tolerance < 0):
                    raise MergeError('tolerance must be positive')
            else:
                raise MergeError('key must be integer, timestamp or float')
        if (not is_bool(self.allow_exact_matches)):
            msg = f'allow_exact_matches must be boolean, passed {self.allow_exact_matches}'
            raise MergeError(msg)
        return (left_join_keys, right_join_keys, join_names)

    def _get_join_indexers(self):
        ' return the join indexers '

        def flip(xs) -> np.ndarray:
            ' unlike np.transpose, this returns an array of tuples '
            xs = [(x if (not is_extension_array_dtype(x)) else extract_array(x)._values_for_argsort()) for x in xs]
            labels = list(string.ascii_lowercase[:len(xs)])
            dtypes = [x.dtype for x in xs]
            labeled_dtypes = list(zip(labels, dtypes))
            return np.array(list(zip(*xs)), labeled_dtypes)
        left_values = (self.left.index._values if self.left_index else self.left_join_keys[(- 1)])
        right_values = (self.right.index._values if self.right_index else self.right_join_keys[(- 1)])
        tolerance = self.tolerance
        if (not Index(left_values).is_monotonic):
            side = 'left'
            if isna(left_values).any():
                raise ValueError(f'Merge keys contain null values on {side} side')
            else:
                raise ValueError(f'{side} keys must be sorted')
        if (not Index(right_values).is_monotonic):
            side = 'right'
            if isna(right_values).any():
                raise ValueError(f'Merge keys contain null values on {side} side')
            else:
                raise ValueError(f'{side} keys must be sorted')
        if needs_i8_conversion(left_values):
            left_values = left_values.view('i8')
            right_values = right_values.view('i8')
            if (tolerance is not None):
                tolerance = Timedelta(tolerance)
                tolerance = tolerance.value
        if (self.left_by is not None):
            if (self.left_index and self.right_index):
                left_by_values = self.left_join_keys
                right_by_values = self.right_join_keys
            else:
                left_by_values = self.left_join_keys[0:(- 1)]
                right_by_values = self.right_join_keys[0:(- 1)]
            if (len(left_by_values) == 1):
                left_by_values = left_by_values[0]
                right_by_values = right_by_values[0]
            else:
                left_by_values = flip(left_by_values)
                right_by_values = flip(right_by_values)
            by_type = _get_cython_type_upcast(left_by_values.dtype)
            by_type_caster = _type_casters[by_type]
            left_by_values = by_type_caster(left_by_values)
            right_by_values = by_type_caster(right_by_values)
            func = _asof_by_function(self.direction)
            return func(left_values, right_values, left_by_values, right_by_values, self.allow_exact_matches, tolerance)
        else:
            func = _asof_function(self.direction)
            return func(left_values, right_values, self.allow_exact_matches, tolerance)

def _get_multiindex_indexer(join_keys, index, sort):
    mapped = (_factorize_keys(index.levels[n], join_keys[n], sort=sort) for n in range(index.nlevels))
    zipped = zip(*mapped)
    (rcodes, lcodes, shape) = [list(x) for x in zipped]
    if sort:
        rcodes = list(map(np.take, rcodes, index.codes))
    else:
        i8copy = (lambda a: a.astype('i8', subok=False, copy=True))
        rcodes = list(map(i8copy, index.codes))
    for i in range(len(join_keys)):
        mask = (index.codes[i] == (- 1))
        if mask.any():
            a = join_keys[i][(lcodes[i] == (shape[i] - 1))]
            if ((a.size == 0) or (not (a[0] != a[0]))):
                shape[i] += 1
            rcodes[i][mask] = (shape[i] - 1)
    (lkey, rkey) = _get_join_keys(lcodes, rcodes, shape, sort)
    (lkey, rkey, count) = _factorize_keys(lkey, rkey, sort=sort)
    return libjoin.left_outer_join(lkey, rkey, count, sort=sort)

def _get_single_indexer(join_key, index, sort=False):
    (left_key, right_key, count) = _factorize_keys(join_key, index, sort=sort)
    (left_indexer, right_indexer) = libjoin.left_outer_join(ensure_int64(left_key), ensure_int64(right_key), count, sort=sort)
    return (left_indexer, right_indexer)

def _left_join_on_index(left_ax, right_ax, join_keys, sort=False):
    if (len(join_keys) > 1):
        if (not (isinstance(right_ax, MultiIndex) and (len(join_keys) == right_ax.nlevels))):
            raise AssertionError("If more than one join key is given then 'right_ax' must be a MultiIndex and the number of join keys must be the number of levels in right_ax")
        (left_indexer, right_indexer) = _get_multiindex_indexer(join_keys, right_ax, sort=sort)
    else:
        jkey = join_keys[0]
        (left_indexer, right_indexer) = _get_single_indexer(jkey, right_ax, sort=sort)
    if (sort or (len(left_ax) != len(left_indexer))):
        join_index = left_ax.take(left_indexer)
        return (join_index, left_indexer, right_indexer)
    return (left_ax, None, right_indexer)

def _factorize_keys(lk, rk, sort=True, how='inner'):
    '\n    Encode left and right keys as enumerated types.\n\n    This is used to get the join indexers to be used when merging DataFrames.\n\n    Parameters\n    ----------\n    lk : array-like\n        Left key.\n    rk : array-like\n        Right key.\n    sort : bool, defaults to True\n        If True, the encoding is done such that the unique elements in the\n        keys are sorted.\n    how : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’\n        Type of merge.\n\n    Returns\n    -------\n    array\n        Left (resp. right if called with `key=\'right\'`) labels, as enumerated type.\n    array\n        Right (resp. left if called with `key=\'right\'`) labels, as enumerated type.\n    int\n        Number of unique elements in union of left and right labels.\n\n    See Also\n    --------\n    merge : Merge DataFrame or named Series objects\n        with a database-style join.\n    algorithms.factorize : Encode the object as an enumerated type\n        or categorical variable.\n\n    Examples\n    --------\n    >>> lk = np.array(["a", "c", "b"])\n    >>> rk = np.array(["a", "c"])\n\n    Here, the unique values are `\'a\', \'b\', \'c\'`. With the default\n    `sort=True`, the encoding will be `{0: \'a\', 1: \'b\', 2: \'c\'}`:\n\n    >>> pd.core.reshape.merge._factorize_keys(lk, rk)\n    (array([0, 2, 1]), array([0, 2]), 3)\n\n    With the `sort=False`, the encoding will correspond to the order\n    in which the unique elements first appear: `{0: \'a\', 1: \'c\', 2: \'b\'}`:\n\n    >>> pd.core.reshape.merge._factorize_keys(lk, rk, sort=False)\n    (array([0, 1, 2]), array([0, 1]), 3)\n    '
    lk = extract_array(lk, extract_numpy=True)
    rk = extract_array(rk, extract_numpy=True)
    if (is_datetime64tz_dtype(lk.dtype) and is_datetime64tz_dtype(rk.dtype)):
        lk = cast('DatetimeArray', lk)._ndarray
        rk = cast('DatetimeArray', rk)._ndarray
    elif (is_categorical_dtype(lk.dtype) and is_categorical_dtype(rk.dtype) and is_dtype_equal(lk.dtype, rk.dtype)):
        assert isinstance(lk, Categorical)
        assert isinstance(rk, Categorical)
        rk = lk._encode_with_my_categories(rk)
        lk = ensure_int64(lk.codes)
        rk = ensure_int64(rk.codes)
    elif (is_extension_array_dtype(lk.dtype) and is_dtype_equal(lk.dtype, rk.dtype)):
        (lk, _) = lk._values_for_factorize()
        (rk, _) = rk._values_for_factorize()
    if (is_integer_dtype(lk.dtype) and is_integer_dtype(rk.dtype)):
        klass = libhashtable.Int64Factorizer
        lk = ensure_int64(np.asarray(lk))
        rk = ensure_int64(np.asarray(rk))
    elif (needs_i8_conversion(lk.dtype) and is_dtype_equal(lk.dtype, rk.dtype)):
        klass = libhashtable.Int64Factorizer
        lk = ensure_int64(np.asarray(lk, dtype=np.int64))
        rk = ensure_int64(np.asarray(rk, dtype=np.int64))
    else:
        klass = libhashtable.Factorizer
        lk = ensure_object(lk)
        rk = ensure_object(rk)
    rizer = klass(max(len(lk), len(rk)))
    llab = rizer.factorize(lk)
    rlab = rizer.factorize(rk)
    count = rizer.get_count()
    if sort:
        uniques = rizer.uniques.to_array()
        (llab, rlab) = _sort_labels(uniques, llab, rlab)
    lmask = (llab == (- 1))
    lany = lmask.any()
    rmask = (rlab == (- 1))
    rany = rmask.any()
    if (lany or rany):
        if lany:
            np.putmask(llab, lmask, count)
        if rany:
            np.putmask(rlab, rmask, count)
        count += 1
    if (how == 'right'):
        return (rlab, llab, count)
    return (llab, rlab, count)

def _sort_labels(uniques, left, right):
    llength = len(left)
    labels = np.concatenate([left, right])
    (_, new_labels) = algos.safe_sort(uniques, labels, na_sentinel=(- 1))
    new_labels = ensure_int64(new_labels)
    (new_left, new_right) = (new_labels[:llength], new_labels[llength:])
    return (new_left, new_right)

def _get_join_keys(llab, rlab, shape, sort):
    nlev = next((lev for lev in range(len(shape), 0, (- 1)) if (not is_int64_overflow_possible(shape[:lev]))))
    stride = np.prod(shape[1:nlev], dtype='i8')
    lkey = (stride * llab[0].astype('i8', subok=False, copy=False))
    rkey = (stride * rlab[0].astype('i8', subok=False, copy=False))
    for i in range(1, nlev):
        with np.errstate(divide='ignore'):
            stride //= shape[i]
        lkey += (llab[i] * stride)
        rkey += (rlab[i] * stride)
    if (nlev == len(shape)):
        return (lkey, rkey)
    (lkey, rkey, count) = _factorize_keys(lkey, rkey, sort=sort)
    llab = ([lkey] + llab[nlev:])
    rlab = ([rkey] + rlab[nlev:])
    shape = ([count] + shape[nlev:])
    return _get_join_keys(llab, rlab, shape, sort)

def _should_fill(lname, rname):
    if ((not isinstance(lname, str)) or (not isinstance(rname, str))):
        return True
    return (lname == rname)

def _any(x):
    return ((x is not None) and com.any_not_none(*x))

def _validate_operand(obj):
    if isinstance(obj, ABCDataFrame):
        return obj
    elif isinstance(obj, ABCSeries):
        if (obj.name is None):
            raise ValueError('Cannot merge a Series without a name')
        else:
            return obj.to_frame()
    else:
        raise TypeError(f'Can only merge Series or DataFrame objects, a {type(obj)} was passed')

def _items_overlap_with_suffix(left, right, suffixes):
    '\n    Suffixes type validation.\n\n    If two indices overlap, add suffixes to overlapping entries.\n\n    If corresponding suffix is empty, the entry is simply converted to string.\n\n    '
    if (not is_list_like(suffixes, allow_sets=False)):
        warnings.warn(f"Passing 'suffixes' as a {type(suffixes)}, is not supported and may give unexpected results. Provide 'suffixes' as a tuple instead. In the future a 'TypeError' will be raised.", FutureWarning, stacklevel=4)
    to_rename = left.intersection(right)
    if (len(to_rename) == 0):
        return (left, right)
    (lsuffix, rsuffix) = suffixes
    if ((not lsuffix) and (not rsuffix)):
        raise ValueError(f'columns overlap but no suffix specified: {to_rename}')

    def renamer(x, suffix):
        '\n        Rename the left and right indices.\n\n        If there is overlap, and suffix is not None, add\n        suffix, otherwise, leave it as-is.\n\n        Parameters\n        ----------\n        x : original column name\n        suffix : str or None\n\n        Returns\n        -------\n        x : renamed column name\n        '
        if ((x in to_rename) and (suffix is not None)):
            return f'{x}{suffix}'
        return x
    lrenamer = partial(renamer, suffix=lsuffix)
    rrenamer = partial(renamer, suffix=rsuffix)
    return (left._transform_index(lrenamer), right._transform_index(rrenamer))
