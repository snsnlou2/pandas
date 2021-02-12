
import datetime
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas._typing import FrameOrSeries, TimedeltaConvertibleTypes
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.dtypes.common import is_datetime64_ns_dtype
from pandas.core.dtypes.missing import isna
import pandas.core.common as common
from pandas.core.util.numba_ import maybe_use_numba
from pandas.core.window.common import _doc_template, _shared_docs, flex_binary_moment, zsqrt
from pandas.core.window.indexers import BaseIndexer, ExponentialMovingWindowIndexer, GroupbyIndexer
from pandas.core.window.numba_ import generate_numba_groupby_ewma_func
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby, dispatch
if TYPE_CHECKING:
    from pandas import Series
_bias_template = '\n        Parameters\n        ----------\n        bias : bool, default False\n            Use a standard estimation bias correction.\n        *args, **kwargs\n            Arguments and keyword arguments to be passed into func.\n'

def get_center_of_mass(comass, span, halflife, alpha):
    valid_count = common.count_not_none(comass, span, halflife, alpha)
    if (valid_count > 1):
        raise ValueError('comass, span, halflife, and alpha are mutually exclusive')
    if (comass is not None):
        if (comass < 0):
            raise ValueError('comass must satisfy: comass >= 0')
    elif (span is not None):
        if (span < 1):
            raise ValueError('span must satisfy: span >= 1')
        comass = ((span - 1) / 2.0)
    elif (halflife is not None):
        if (halflife <= 0):
            raise ValueError('halflife must satisfy: halflife > 0')
        decay = (1 - np.exp((np.log(0.5) / halflife)))
        comass = ((1 / decay) - 1)
    elif (alpha is not None):
        if ((alpha <= 0) or (alpha > 1)):
            raise ValueError('alpha must satisfy: 0 < alpha <= 1')
        comass = ((1.0 - alpha) / alpha)
    else:
        raise ValueError('Must pass one of comass, span, halflife, or alpha')
    return float(comass)

def wrap_result(obj, result):
    '\n    Wrap a single 1D result.\n    '
    obj = obj._selected_obj
    return obj._constructor(result, obj.index, name=obj.name)

class ExponentialMovingWindow(BaseWindow):
    "\n    Provide exponential weighted (EW) functions.\n\n    Available EW functions: ``mean()``, ``var()``, ``std()``, ``corr()``, ``cov()``.\n\n    Exactly one parameter: ``com``, ``span``, ``halflife``, or ``alpha`` must be\n    provided.\n\n    Parameters\n    ----------\n    com : float, optional\n        Specify decay in terms of center of mass,\n        :math:`\\alpha = 1 / (1 + com)`, for :math:`com \\geq 0`.\n    span : float, optional\n        Specify decay in terms of span,\n        :math:`\\alpha = 2 / (span + 1)`, for :math:`span \\geq 1`.\n    halflife : float, str, timedelta, optional\n        Specify decay in terms of half-life,\n        :math:`\\alpha = 1 - \\exp\\left(-\\ln(2) / halflife\\right)`, for\n        :math:`halflife > 0`.\n\n        If ``times`` is specified, the time unit (str or timedelta) over which an\n        observation decays to half its value. Only applicable to ``mean()``\n        and halflife value will not apply to the other functions.\n\n        .. versionadded:: 1.1.0\n\n    alpha : float, optional\n        Specify smoothing factor :math:`\\alpha` directly,\n        :math:`0 < \\alpha \\leq 1`.\n    min_periods : int, default 0\n        Minimum number of observations in window required to have a value\n        (otherwise result is NA).\n    adjust : bool, default True\n        Divide by decaying adjustment factor in beginning periods to account\n        for imbalance in relative weightings (viewing EWMA as a moving average).\n\n        - When ``adjust=True`` (default), the EW function is calculated using weights\n          :math:`w_i = (1 - \\alpha)^i`. For example, the EW moving average of the series\n          [:math:`x_0, x_1, ..., x_t`] would be:\n\n        .. math::\n            y_t = \\frac{x_t + (1 - \\alpha)x_{t-1} + (1 - \\alpha)^2 x_{t-2} + ... + (1 -\n            \\alpha)^t x_0}{1 + (1 - \\alpha) + (1 - \\alpha)^2 + ... + (1 - \\alpha)^t}\n\n        - When ``adjust=False``, the exponentially weighted function is calculated\n          recursively:\n\n        .. math::\n            \\begin{split}\n                y_0 &= x_0\\\\\n                y_t &= (1 - \\alpha) y_{t-1} + \\alpha x_t,\n            \\end{split}\n    ignore_na : bool, default False\n        Ignore missing values when calculating weights; specify ``True`` to reproduce\n        pre-0.15.0 behavior.\n\n        - When ``ignore_na=False`` (default), weights are based on absolute positions.\n          For example, the weights of :math:`x_0` and :math:`x_2` used in calculating\n          the final weighted average of [:math:`x_0`, None, :math:`x_2`] are\n          :math:`(1-\\alpha)^2` and :math:`1` if ``adjust=True``, and\n          :math:`(1-\\alpha)^2` and :math:`\\alpha` if ``adjust=False``.\n\n        - When ``ignore_na=True`` (reproducing pre-0.15.0 behavior), weights are based\n          on relative positions. For example, the weights of :math:`x_0` and :math:`x_2`\n          used in calculating the final weighted average of\n          [:math:`x_0`, None, :math:`x_2`] are :math:`1-\\alpha` and :math:`1` if\n          ``adjust=True``, and :math:`1-\\alpha` and :math:`\\alpha` if ``adjust=False``.\n    axis : {0, 1}, default 0\n        The axis to use. The value 0 identifies the rows, and 1\n        identifies the columns.\n    times : str, np.ndarray, Series, default None\n\n        .. versionadded:: 1.1.0\n\n        Times corresponding to the observations. Must be monotonically increasing and\n        ``datetime64[ns]`` dtype.\n\n        If str, the name of the column in the DataFrame representing the times.\n\n        If 1-D array like, a sequence with the same shape as the observations.\n\n        Only applicable to ``mean()``.\n\n    Returns\n    -------\n    DataFrame\n        A Window sub-classed for the particular operation.\n\n    See Also\n    --------\n    rolling : Provides rolling window calculations.\n    expanding : Provides expanding transformations.\n\n    Notes\n    -----\n\n    More details can be found at:\n    :ref:`Exponentially weighted windows <window.exponentially_weighted>`.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})\n    >>> df\n         B\n    0  0.0\n    1  1.0\n    2  2.0\n    3  NaN\n    4  4.0\n\n    >>> df.ewm(com=0.5).mean()\n              B\n    0  0.000000\n    1  0.750000\n    2  1.615385\n    3  1.615385\n    4  3.670213\n\n    Specifying ``times`` with a timedelta ``halflife`` when computing mean.\n\n    >>> times = ['2020-01-01', '2020-01-03', '2020-01-10', '2020-01-15', '2020-01-17']\n    >>> df.ewm(halflife='4 days', times=pd.DatetimeIndex(times)).mean()\n              B\n    0  0.000000\n    1  0.585786\n    2  1.523889\n    3  1.523889\n    4  3.233686\n    "
    _attributes = ['com', 'min_periods', 'adjust', 'ignore_na', 'axis']

    def __init__(self, obj, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None, **kwargs):
        self.obj = obj
        self.min_periods = max(int(min_periods), 1)
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.axis = axis
        self.on = None
        self.center = False
        self.closed = None
        self.method = 'single'
        if (times is not None):
            if isinstance(times, str):
                times = self._selected_obj[times]
            if (not is_datetime64_ns_dtype(times)):
                raise ValueError('times must be datetime64[ns] dtype.')
            if (len(times) != len(obj)):
                raise ValueError('times must be the same length as the object.')
            if (not isinstance(halflife, (str, datetime.timedelta))):
                raise ValueError('halflife must be a string or datetime.timedelta object')
            if isna(times).any():
                raise ValueError('Cannot convert NaT values to integer')
            self.times = np.asarray(times.view(np.int64))
            self.halflife = Timedelta(halflife).value
            if (common.count_not_none(com, span, alpha) > 0):
                self.com = get_center_of_mass(com, span, None, alpha)
            else:
                self.com = 0.0
        else:
            if ((halflife is not None) and isinstance(halflife, (str, datetime.timedelta))):
                raise ValueError('halflife can only be a timedelta convertible argument if times is not None.')
            self.times = None
            self.halflife = None
            self.com = get_center_of_mass(com, span, halflife, alpha)

    def _get_window_indexer(self):
        '\n        Return an indexer class that will compute the window start and end bounds\n        '
        return ExponentialMovingWindowIndexer()
    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    pandas.DataFrame.rolling.aggregate\n    ')
    _agg_examples_doc = dedent('\n    Examples\n    --------\n    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n    >>> df\n       A  B  C\n    0  1  4  7\n    1  2  5  8\n    2  3  6  9\n\n    >>> df.ewm(alpha=0.5).mean()\n              A         B         C\n    0  1.000000  4.000000  7.000000\n    1  1.666667  4.666667  7.666667\n    2  2.428571  5.428571  8.428571\n    ')

    @doc(_shared_docs['aggregate'], see_also=_agg_see_also_doc, examples=_agg_examples_doc, klass='Series/Dataframe', axis='')
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)
    agg = aggregate

    @Substitution(name='ewm', func_name='mean')
    @Appender(_doc_template)
    def mean(self, *args, **kwargs):
        '\n        Exponential weighted moving average.\n\n        Parameters\n        ----------\n        *args, **kwargs\n            Arguments and keyword arguments to be passed into func.\n        '
        nv.validate_window_func('mean', args, kwargs)
        if (self.times is not None):
            window_func = window_aggregations.ewma_time
            window_func = partial(window_func, times=self.times, halflife=self.halflife)
        else:
            window_func = window_aggregations.ewma
            window_func = partial(window_func, com=self.com, adjust=self.adjust, ignore_na=self.ignore_na)
        return self._apply(window_func)

    @Substitution(name='ewm', func_name='std')
    @Appender(_doc_template)
    @Appender(_bias_template)
    def std(self, bias=False, *args, **kwargs):
        '\n        Exponential weighted moving stddev.\n        '
        nv.validate_window_func('std', args, kwargs)
        return zsqrt(self.var(bias=bias, **kwargs))
    vol = std

    @Substitution(name='ewm', func_name='var')
    @Appender(_doc_template)
    @Appender(_bias_template)
    def var(self, bias=False, *args, **kwargs):
        '\n        Exponential weighted moving variance.\n        '
        nv.validate_window_func('var', args, kwargs)
        window_func = window_aggregations.ewmcov
        window_func = partial(window_func, com=self.com, adjust=self.adjust, ignore_na=self.ignore_na, bias=bias)

        def var_func(values, begin, end, min_periods):
            return window_func(values, begin, end, min_periods, values)
        return self._apply(var_func)

    @Substitution(name='ewm', func_name='cov')
    @Appender(_doc_template)
    def cov(self, other=None, pairwise=None, bias=False, **kwargs):
        '\n        Exponential weighted sample covariance.\n\n        Parameters\n        ----------\n        other : Series, DataFrame, or ndarray, optional\n            If not supplied then will default to self and produce pairwise\n            output.\n        pairwise : bool, default None\n            If False then only matching columns between self and other will be\n            used and the output will be a DataFrame.\n            If True then all pairwise combinations will be calculated and the\n            output will be a MultiIndex DataFrame in the case of DataFrame\n            inputs. In the case of missing elements, only complete pairwise\n            observations will be used.\n        bias : bool, default False\n            Use a standard estimation bias correction.\n        **kwargs\n           Keyword arguments to be passed into func.\n        '
        if (other is None):
            other = self._selected_obj
            pairwise = (True if (pairwise is None) else pairwise)
        other = self._shallow_copy(other)

        def _get_cov(X, Y):
            X = self._shallow_copy(X)
            Y = self._shallow_copy(Y)
            cov = window_aggregations.ewmcov(X._prep_values(), np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), self.min_periods, Y._prep_values(), self.com, self.adjust, self.ignore_na, bias)
            return wrap_result(X, cov)
        return flex_binary_moment(self._selected_obj, other._selected_obj, _get_cov, pairwise=bool(pairwise))

    @Substitution(name='ewm', func_name='corr')
    @Appender(_doc_template)
    def corr(self, other=None, pairwise=None, **kwargs):
        '\n        Exponential weighted sample correlation.\n\n        Parameters\n        ----------\n        other : Series, DataFrame, or ndarray, optional\n            If not supplied then will default to self and produce pairwise\n            output.\n        pairwise : bool, default None\n            If False then only matching columns between self and other will be\n            used and the output will be a DataFrame.\n            If True then all pairwise combinations will be calculated and the\n            output will be a MultiIndex DataFrame in the case of DataFrame\n            inputs. In the case of missing elements, only complete pairwise\n            observations will be used.\n        **kwargs\n           Keyword arguments to be passed into func.\n        '
        if (other is None):
            other = self._selected_obj
            pairwise = (True if (pairwise is None) else pairwise)
        other = self._shallow_copy(other)

        def _get_corr(X, Y):
            X = self._shallow_copy(X)
            Y = self._shallow_copy(Y)

            def _cov(x, y):
                return window_aggregations.ewmcov(x, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), self.min_periods, y, self.com, self.adjust, self.ignore_na, 1)
            x_values = X._prep_values()
            y_values = Y._prep_values()
            with np.errstate(all='ignore'):
                cov = _cov(x_values, y_values)
                x_var = _cov(x_values, x_values)
                y_var = _cov(y_values, y_values)
                corr = (cov / zsqrt((x_var * y_var)))
            return wrap_result(X, corr)
        return flex_binary_moment(self._selected_obj, other._selected_obj, _get_corr, pairwise=bool(pairwise))

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    '\n    Provide an exponential moving window groupby implementation.\n    '

    @property
    def _constructor(self):
        return ExponentialMovingWindow

    def _get_window_indexer(self):
        '\n        Return an indexer class that will compute the window start and end bounds\n\n        Returns\n        -------\n        GroupbyIndexer\n        '
        window_indexer = GroupbyIndexer(groupby_indicies=self._groupby.indices, window_indexer=ExponentialMovingWindowIndexer)
        return window_indexer
    var = dispatch('var', bias=False)
    std = dispatch('std', bias=False)
    cov = dispatch('cov', other=None, pairwise=None, bias=False)
    corr = dispatch('corr', other=None, pairwise=None)

    def mean(self, engine=None, engine_kwargs=None):
        "\n        Parameters\n        ----------\n        engine : str, default None\n            * ``'cython'`` : Runs mean through C-extensions from cython.\n            * ``'numba'`` : Runs mean through JIT compiled code from numba.\n              Only available when ``raw`` is set to ``True``.\n            * ``None`` : Defaults to ``'cython'`` or globally setting\n              ``compute.use_numba``\n\n              .. versionadded:: 1.2.0\n\n        engine_kwargs : dict, default None\n            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``\n            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``\n              and ``parallel`` dictionary keys. The values must either be ``True`` or\n              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is\n              ``{'nopython': True, 'nogil': False, 'parallel': False}``.\n\n              .. versionadded:: 1.2.0\n\n        Returns\n        -------\n        Series or DataFrame\n            Return type is determined by the caller.\n        "
        if maybe_use_numba(engine):
            groupby_ewma_func = generate_numba_groupby_ewma_func(engine_kwargs, self.com, self.adjust, self.ignore_na)
            return self._apply(groupby_ewma_func, numba_cache_key=((lambda x: x), 'groupby_ewma'))
        elif (engine in ('cython', None)):
            if (engine_kwargs is not None):
                raise ValueError('cython engine does not accept engine_kwargs')

            def f(x):
                x = self._shallow_copy(x, groupby=self._groupby)
                return x.mean()
            return self._groupby.apply(f)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")
