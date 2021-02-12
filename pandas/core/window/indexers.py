
'Indexer objects for computing start/end window bounds for rolling operations'
from datetime import timedelta
from typing import Dict, Optional, Tuple, Type
import numpy as np
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.tseries.offsets import Nano
get_window_bounds_doc = '\nComputes the bounds of a window.\n\nParameters\n----------\nnum_values : int, default 0\n    number of values that will be aggregated over\nwindow_size : int, default 0\n    the number of rows in a window\nmin_periods : int, default None\n    min_periods passed from the top level rolling API\ncenter : bool, default None\n    center passed from the top level rolling API\nclosed : str, default None\n    closed passed from the top level rolling API\nwin_type : str, default None\n    win_type passed from the top level rolling API\n\nReturns\n-------\nA tuple of ndarray[int64]s, indicating the boundaries of each\nwindow\n'

class BaseIndexer():
    'Base class for window bounds calculations.'

    def __init__(self, index_array=None, window_size=0, **kwargs):
        '\n        Parameters\n        ----------\n        **kwargs :\n            keyword arguments that will be available when get_window_bounds is called\n        '
        self.index_array = index_array
        self.window_size = window_size
        for (key, value) in kwargs.items():
            setattr(self, key, value)

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        raise NotImplementedError

class FixedWindowIndexer(BaseIndexer):
    'Creates window boundaries that are of fixed length.'

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        if center:
            offset = ((self.window_size - 1) // 2)
        else:
            offset = 0
        end = np.arange((1 + offset), ((num_values + 1) + offset), dtype='int64')
        start = (end - self.window_size)
        if (closed in ['left', 'both']):
            start -= 1
        if (closed in ['left', 'neither']):
            end -= 1
        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)
        return (start, end)

class VariableWindowIndexer(BaseIndexer):
    'Creates window boundaries that are of variable length, namely for time series.'

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        return calculate_variable_window_bounds(num_values, self.window_size, min_periods, center, closed, self.index_array)

class VariableOffsetWindowIndexer(BaseIndexer):
    'Calculate window boundaries based on a non-fixed offset such as a BusinessDay'

    def __init__(self, index_array=None, window_size=0, index=None, offset=None, **kwargs):
        super().__init__(index_array, window_size, **kwargs)
        self.index = index
        self.offset = offset

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        if (closed is None):
            closed = ('right' if (self.index is not None) else 'both')
        right_closed = (closed in ['right', 'both'])
        left_closed = (closed in ['left', 'both'])
        if (self.index[(num_values - 1)] < self.index[0]):
            index_growth_sign = (- 1)
        else:
            index_growth_sign = 1
        start = np.empty(num_values, dtype='int64')
        start.fill((- 1))
        end = np.empty(num_values, dtype='int64')
        end.fill((- 1))
        start[0] = 0
        if right_closed:
            end[0] = 1
        else:
            end[0] = 0
        for i in range(1, num_values):
            end_bound = self.index[i]
            start_bound = (self.index[i] - (index_growth_sign * self.offset))
            if left_closed:
                start_bound -= Nano(1)
            start[i] = i
            for j in range(start[(i - 1)], i):
                if (((self.index[j] - start_bound) * index_growth_sign) > timedelta(0)):
                    start[i] = j
                    break
            if (((self.index[end[(i - 1)]] - end_bound) * index_growth_sign) <= timedelta(0)):
                end[i] = (i + 1)
            else:
                end[i] = end[(i - 1)]
            if (not right_closed):
                end[i] -= 1
        return (start, end)

class ExpandingIndexer(BaseIndexer):
    'Calculate expanding window bounds, mimicking df.expanding()'

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        return (np.zeros(num_values, dtype=np.int64), np.arange(1, (num_values + 1), dtype=np.int64))

class FixedForwardWindowIndexer(BaseIndexer):
    "\n    Creates window boundaries for fixed-length windows that include the\n    current row.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})\n    >>> df\n         B\n    0  0.0\n    1  1.0\n    2  2.0\n    3  NaN\n    4  4.0\n\n    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)\n    >>> df.rolling(window=indexer, min_periods=1).sum()\n         B\n    0  1.0\n    1  3.0\n    2  2.0\n    3  4.0\n    4  4.0\n    "

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        if center:
            raise ValueError("Forward-looking windows can't have center=True")
        if (closed is not None):
            raise ValueError("Forward-looking windows don't support setting the closed argument")
        start = np.arange(num_values, dtype='int64')
        end_s = (start[:(- self.window_size)] + self.window_size)
        end_e = np.full(self.window_size, num_values, dtype='int64')
        end = np.concatenate([end_s, end_e])
        return (start, end)

class GroupbyIndexer(BaseIndexer):
    'Calculate bounds to compute groupby rolling, mimicking df.groupby().rolling()'

    def __init__(self, index_array=None, window_size=0, groupby_indicies=None, window_indexer=BaseIndexer, indexer_kwargs=None, **kwargs):
        '\n        Parameters\n        ----------\n        index_array : np.ndarray or None\n            np.ndarray of the index of the original object that we are performing\n            a chained groupby operation over. This index has been pre-sorted relative to\n            the groups\n        window_size : int\n            window size during the windowing operation\n        groupby_indicies : dict or None\n            dict of {group label: [positional index of rows belonging to the group]}\n        window_indexer : BaseIndexer\n            BaseIndexer class determining the start and end bounds of each group\n        indexer_kwargs : dict or None\n            Custom kwargs to be passed to window_indexer\n        **kwargs :\n            keyword arguments that will be available when get_window_bounds is called\n        '
        self.groupby_indicies = (groupby_indicies or {})
        self.window_indexer = window_indexer
        self.indexer_kwargs = (indexer_kwargs or {})
        super().__init__(index_array, self.indexer_kwargs.pop('window_size', window_size), **kwargs)

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        start_arrays = []
        end_arrays = []
        window_indicies_start = 0
        for (key, indices) in self.groupby_indicies.items():
            if (self.index_array is not None):
                index_array = self.index_array.take(ensure_platform_int(indices))
            else:
                index_array = self.index_array
            indexer = self.window_indexer(index_array=index_array, window_size=self.window_size, **self.indexer_kwargs)
            (start, end) = indexer.get_window_bounds(len(indices), min_periods, center, closed)
            start = start.astype(np.int64)
            end = end.astype(np.int64)
            window_indicies = np.arange(window_indicies_start, (window_indicies_start + len(indices)))
            window_indicies_start += len(indices)
            window_indicies = np.append(window_indicies, [(window_indicies[(- 1)] + 1)]).astype(np.int64)
            start_arrays.append(window_indicies.take(ensure_platform_int(start)))
            end_arrays.append(window_indicies.take(ensure_platform_int(end)))
        start = np.concatenate(start_arrays)
        end = np.concatenate(end_arrays)
        return (start, end)

class ExponentialMovingWindowIndexer(BaseIndexer):
    'Calculate ewm window bounds (the entire window)'

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        return (np.array([0], dtype=np.int64), np.array([num_values], dtype=np.int64))
