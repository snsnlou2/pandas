
from textwrap import dedent
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
from pandas._typing import FrameOrSeries
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.window.common import _doc_template, _shared_docs
from pandas.core.window.indexers import BaseIndexer, ExpandingIndexer, GroupbyIndexer
from pandas.core.window.rolling import BaseWindowGroupby, RollingAndExpandingMixin

class Expanding(RollingAndExpandingMixin):
    '\n    Provide expanding transformations.\n\n    Parameters\n    ----------\n    min_periods : int, default 1\n        Minimum number of observations in window required to have a value\n        (otherwise result is NA).\n    center : bool, default False\n        Set the labels at the center of the window.\n    axis : int or str, default 0\n    method : str {\'single\', \'table\'}, default \'single\'\n        Execute the rolling operation per single column or row (``\'single\'``)\n        or over the entire object (``\'table\'``).\n\n        This argument is only implemented when specifying ``engine=\'numba\'``\n        in the method call.\n\n        .. versionadded:: 1.3.0\n\n    Returns\n    -------\n    a Window sub-classed for the particular operation\n\n    See Also\n    --------\n    rolling : Provides rolling window calculations.\n    ewm : Provides exponential weighted functions.\n\n    Notes\n    -----\n    By default, the result is set to the right edge of the window. This can be\n    changed to the center of the window by setting ``center=True``.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})\n    >>> df\n         B\n    0  0.0\n    1  1.0\n    2  2.0\n    3  NaN\n    4  4.0\n\n    >>> df.expanding(2).sum()\n         B\n    0  NaN\n    1  1.0\n    2  3.0\n    3  3.0\n    4  7.0\n    '
    _attributes = ['min_periods', 'center', 'axis', 'method']

    def __init__(self, obj, min_periods=1, center=None, axis=0, method='single', **kwargs):
        super().__init__(obj=obj, min_periods=min_periods, center=center, axis=axis, method=method)

    def _get_window_indexer(self):
        '\n        Return an indexer class that will compute the window start and end bounds\n        '
        return ExpandingIndexer()

    def _get_cov_corr_window(self, other=None, **kwargs):
        '\n        Get the window length over which to perform cov and corr operations.\n\n        Parameters\n        ----------\n        other : object, default None\n            The other object that is involved in the operation.\n            Such an object is involved for operations like covariance.\n\n        Returns\n        -------\n        window : int\n            The window length.\n        '
        axis = self.obj._get_axis(self.axis)
        length = (len(axis) + ((other is not None) * len(axis)))
        other = (self.min_periods or (- 1))
        return max(length, other)
    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    pandas.DataFrame.aggregate : Similar DataFrame method.\n    pandas.Series.aggregate : Similar Series method.\n    ')
    _agg_examples_doc = dedent('\n    Examples\n    --------\n    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n    >>> df\n       A  B  C\n    0  1  4  7\n    1  2  5  8\n    2  3  6  9\n\n    >>> df.ewm(alpha=0.5).mean()\n              A         B         C\n    0  1.000000  4.000000  7.000000\n    1  1.666667  4.666667  7.666667\n    2  2.428571  5.428571  8.428571\n    ')

    @doc(_shared_docs['aggregate'], see_also=_agg_see_also_doc, examples=_agg_examples_doc, klass='Series/Dataframe', axis='')
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)
    agg = aggregate

    @Substitution(name='expanding')
    @Appender(_shared_docs['count'])
    def count(self):
        return super().count()

    @Substitution(name='expanding')
    @Appender(_shared_docs['apply'])
    def apply(self, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None):
        return super().apply(func, raw=raw, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    @Substitution(name='expanding')
    @Appender(_shared_docs['sum'])
    def sum(self, *args, **kwargs):
        nv.validate_expanding_func('sum', args, kwargs)
        return super().sum(*args, **kwargs)

    @Substitution(name='expanding', func_name='max')
    @Appender(_doc_template)
    @Appender(_shared_docs['max'])
    def max(self, *args, **kwargs):
        nv.validate_expanding_func('max', args, kwargs)
        return super().max(*args, **kwargs)

    @Substitution(name='expanding')
    @Appender(_shared_docs['min'])
    def min(self, *args, **kwargs):
        nv.validate_expanding_func('min', args, kwargs)
        return super().min(*args, **kwargs)

    @Substitution(name='expanding')
    @Appender(_shared_docs['mean'])
    def mean(self, *args, **kwargs):
        nv.validate_expanding_func('mean', args, kwargs)
        return super().mean(*args, **kwargs)

    @Substitution(name='expanding')
    @Appender(_shared_docs['median'])
    def median(self, **kwargs):
        return super().median(**kwargs)

    @Substitution(name='expanding', versionadded='')
    @Appender(_shared_docs['std'])
    def std(self, ddof=1, *args, **kwargs):
        nv.validate_expanding_func('std', args, kwargs)
        return super().std(ddof=ddof, **kwargs)

    @Substitution(name='expanding', versionadded='')
    @Appender(_shared_docs['var'])
    def var(self, ddof=1, *args, **kwargs):
        nv.validate_expanding_func('var', args, kwargs)
        return super().var(ddof=ddof, **kwargs)

    @Substitution(name='expanding')
    @Appender(_shared_docs['sem'])
    def sem(self, ddof=1, *args, **kwargs):
        return super().sem(ddof=ddof, **kwargs)

    @Substitution(name='expanding', func_name='skew')
    @Appender(_doc_template)
    @Appender(_shared_docs['skew'])
    def skew(self, **kwargs):
        return super().skew(**kwargs)
    _agg_doc = dedent('\n    Examples\n    --------\n\n    The example below will show an expanding calculation with a window size of\n    four matching the equivalent function call using `scipy.stats`.\n\n    >>> arr = [1, 2, 3, 4, 999]\n    >>> import scipy.stats\n    >>> print(f"{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}")\n    -1.200000\n    >>> print(f"{scipy.stats.kurtosis(arr, bias=False):.6f}")\n    4.999874\n    >>> s = pd.Series(arr)\n    >>> s.expanding(4).kurt()\n    0         NaN\n    1         NaN\n    2         NaN\n    3   -1.200000\n    4    4.999874\n    dtype: float64\n    ')

    @Appender(_agg_doc)
    @Substitution(name='expanding')
    @Appender(_shared_docs['kurt'])
    def kurt(self, **kwargs):
        return super().kurt(**kwargs)

    @Substitution(name='expanding')
    @Appender(_shared_docs['quantile'])
    def quantile(self, quantile, interpolation='linear', **kwargs):
        return super().quantile(quantile=quantile, interpolation=interpolation, **kwargs)

    @Substitution(name='expanding', func_name='cov')
    @Appender(_doc_template)
    @Appender(_shared_docs['cov'])
    def cov(self, other=None, pairwise=None, ddof=1, **kwargs):
        return super().cov(other=other, pairwise=pairwise, ddof=ddof, **kwargs)

    @Substitution(name='expanding')
    @Appender(_shared_docs['corr'])
    def corr(self, other=None, pairwise=None, **kwargs):
        return super().corr(other=other, pairwise=pairwise, **kwargs)

class ExpandingGroupby(BaseWindowGroupby, Expanding):
    '\n    Provide a expanding groupby implementation.\n    '

    @property
    def _constructor(self):
        return Expanding

    def _get_window_indexer(self):
        '\n        Return an indexer class that will compute the window start and end bounds\n\n        Returns\n        -------\n        GroupbyIndexer\n        '
        window_indexer = GroupbyIndexer(groupby_indicies=self._groupby.indices, window_indexer=ExpandingIndexer)
        return window_indexer
