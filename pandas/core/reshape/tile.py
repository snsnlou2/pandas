
'\nQuantilization functions and related stuff\n'
import numpy as np
from pandas._libs import Timedelta, Timestamp
from pandas._libs.lib import infer_dtype
from pandas.core.dtypes.common import DT64NS_DTYPE, ensure_int64, is_bool_dtype, is_categorical_dtype, is_datetime64_dtype, is_datetime64tz_dtype, is_datetime_or_timedelta_dtype, is_extension_array_dtype, is_integer, is_integer_dtype, is_list_like, is_scalar, is_timedelta64_dtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas import Categorical, Index, IntervalIndex, to_datetime, to_timedelta
import pandas.core.algorithms as algos
import pandas.core.nanops as nanops

def cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True):
    '\n    Bin values into discrete intervals.\n\n    Use `cut` when you need to segment and sort data values into bins. This\n    function is also useful for going from a continuous variable to a\n    categorical variable. For example, `cut` could convert ages to groups of\n    age ranges. Supports binning into an equal number of bins, or a\n    pre-specified array of bins.\n\n    Parameters\n    ----------\n    x : array-like\n        The input array to be binned. Must be 1-dimensional.\n    bins : int, sequence of scalars, or IntervalIndex\n        The criteria to bin by.\n\n        * int : Defines the number of equal-width bins in the range of `x`. The\n          range of `x` is extended by .1% on each side to include the minimum\n          and maximum values of `x`.\n        * sequence of scalars : Defines the bin edges allowing for non-uniform\n          width. No extension of the range of `x` is done.\n        * IntervalIndex : Defines the exact bins to be used. Note that\n          IntervalIndex for `bins` must be non-overlapping.\n\n    right : bool, default True\n        Indicates whether `bins` includes the rightmost edge or not. If\n        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``\n        indicate (1,2], (2,3], (3,4]. This argument is ignored when\n        `bins` is an IntervalIndex.\n    labels : array or False, default None\n        Specifies the labels for the returned bins. Must be the same length as\n        the resulting bins. If False, returns only integer indicators of the\n        bins. This affects the type of the output container (see below).\n        This argument is ignored when `bins` is an IntervalIndex. If True,\n        raises an error. When `ordered=False`, labels must be provided.\n    retbins : bool, default False\n        Whether to return the bins or not. Useful when bins is provided\n        as a scalar.\n    precision : int, default 3\n        The precision at which to store and display the bins labels.\n    include_lowest : bool, default False\n        Whether the first interval should be left-inclusive or not.\n    duplicates : {default \'raise\', \'drop\'}, optional\n        If bin edges are not unique, raise ValueError or drop non-uniques.\n    ordered : bool, default True\n        Whether the labels are ordered or not. Applies to returned types\n        Categorical and Series (with Categorical dtype). If True,\n        the resulting categorical will be ordered. If False, the resulting\n        categorical will be unordered (labels must be provided).\n\n        .. versionadded:: 1.1.0\n\n    Returns\n    -------\n    out : Categorical, Series, or ndarray\n        An array-like object representing the respective bin for each value\n        of `x`. The type depends on the value of `labels`.\n\n        * True (default) : returns a Series for Series `x` or a\n          Categorical for all other inputs. The values stored within\n          are Interval dtype.\n\n        * sequence of scalars : returns a Series for Series `x` or a\n          Categorical for all other inputs. The values stored within\n          are whatever the type in the sequence is.\n\n        * False : returns an ndarray of integers.\n\n    bins : numpy.ndarray or IntervalIndex.\n        The computed or specified bins. Only returned when `retbins=True`.\n        For scalar or sequence `bins`, this is an ndarray with the computed\n        bins. If set `duplicates=drop`, `bins` will drop non-unique bin. For\n        an IntervalIndex `bins`, this is equal to `bins`.\n\n    See Also\n    --------\n    qcut : Discretize variable into equal-sized buckets based on rank\n        or based on sample quantiles.\n    Categorical : Array type for storing data that come from a\n        fixed set of values.\n    Series : One-dimensional array with axis labels (including time series).\n    IntervalIndex : Immutable Index implementing an ordered, sliceable set.\n\n    Notes\n    -----\n    Any NA values will be NA in the result. Out of bounds values will be NA in\n    the resulting Series or Categorical object.\n\n    Examples\n    --------\n    Discretize into three equal-sized bins.\n\n    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)\n    ... # doctest: +ELLIPSIS\n    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...\n    Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...\n\n    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)\n    ... # doctest: +ELLIPSIS\n    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...\n    Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...\n    array([0.994, 3.   , 5.   , 7.   ]))\n\n    Discovers the same bins, but assign them specific labels. Notice that\n    the returned Categorical\'s categories are `labels` and is ordered.\n\n    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]),\n    ...        3, labels=["bad", "medium", "good"])\n    [\'bad\', \'good\', \'medium\', \'medium\', \'good\', \'bad\']\n    Categories (3, object): [\'bad\' < \'medium\' < \'good\']\n\n    ``ordered=False`` will result in unordered categories when labels are passed.\n    This parameter can be used to allow non-unique labels:\n\n    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,\n    ...        labels=["B", "A", "B"], ordered=False)\n    [\'B\', \'B\', \'A\', \'A\', \'B\', \'B\']\n    Categories (2, object): [\'A\', \'B\']\n\n    ``labels=False`` implies you just want the bins back.\n\n    >>> pd.cut([0, 1, 1, 2], bins=4, labels=False)\n    array([0, 1, 1, 3])\n\n    Passing a Series as an input returns a Series with categorical dtype:\n\n    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),\n    ...               index=[\'a\', \'b\', \'c\', \'d\', \'e\'])\n    >>> pd.cut(s, 3)\n    ... # doctest: +ELLIPSIS\n    a    (1.992, 4.667]\n    b    (1.992, 4.667]\n    c    (4.667, 7.333]\n    d     (7.333, 10.0]\n    e     (7.333, 10.0]\n    dtype: category\n    Categories (3, interval[float64]): [(1.992, 4.667] < (4.667, ...\n\n    Passing a Series as an input returns a Series with mapping value.\n    It is used to map numerically to intervals based on bins.\n\n    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),\n    ...               index=[\'a\', \'b\', \'c\', \'d\', \'e\'])\n    >>> pd.cut(s, [0, 2, 4, 6, 8, 10], labels=False, retbins=True, right=False)\n    ... # doctest: +ELLIPSIS\n    (a    1.0\n     b    2.0\n     c    3.0\n     d    4.0\n     e    NaN\n     dtype: float64,\n     array([ 0,  2,  4,  6,  8, 10]))\n\n    Use `drop` optional when bins is not unique\n\n    >>> pd.cut(s, [0, 2, 4, 6, 10, 10], labels=False, retbins=True,\n    ...        right=False, duplicates=\'drop\')\n    ... # doctest: +ELLIPSIS\n    (a    1.0\n     b    2.0\n     c    3.0\n     d    3.0\n     e    NaN\n     dtype: float64,\n     array([ 0,  2,  4,  6, 10]))\n\n    Passing an IntervalIndex for `bins` results in those categories exactly.\n    Notice that values not covered by the IntervalIndex are set to NaN. 0\n    is to the left of the first bin (which is closed on the right), and 1.5\n    falls between two bins.\n\n    >>> bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])\n    >>> pd.cut([0, 0.5, 1.5, 2.5, 4.5], bins)\n    [NaN, (0.0, 1.0], NaN, (2.0, 3.0], (4.0, 5.0]]\n    Categories (3, interval[int64]): [(0, 1] < (2, 3] < (4, 5]]\n    '
    original = x
    x = _preprocess_for_cut(x)
    (x, dtype) = _coerce_to_type(x)
    if (not np.iterable(bins)):
        if (is_scalar(bins) and (bins < 1)):
            raise ValueError('`bins` should be a positive integer.')
        try:
            sz = x.size
        except AttributeError:
            x = np.asarray(x)
            sz = x.size
        if (sz == 0):
            raise ValueError('Cannot cut empty array')
        rng = (nanops.nanmin(x), nanops.nanmax(x))
        (mn, mx) = [(mi + 0.0) for mi in rng]
        if (np.isinf(mn) or np.isinf(mx)):
            raise ValueError('cannot specify integer `bins` when input data contains infinity')
        elif (mn == mx):
            mn -= ((0.001 * abs(mn)) if (mn != 0) else 0.001)
            mx += ((0.001 * abs(mx)) if (mx != 0) else 0.001)
            bins = np.linspace(mn, mx, (bins + 1), endpoint=True)
        else:
            bins = np.linspace(mn, mx, (bins + 1), endpoint=True)
            adj = ((mx - mn) * 0.001)
            if right:
                bins[0] -= adj
            else:
                bins[(- 1)] += adj
    elif isinstance(bins, IntervalIndex):
        if bins.is_overlapping:
            raise ValueError('Overlapping IntervalIndex is not accepted.')
    else:
        if is_datetime64tz_dtype(bins):
            bins = np.asarray(bins, dtype=DT64NS_DTYPE)
        else:
            bins = np.asarray(bins)
        bins = _convert_bin_to_numeric_type(bins, dtype)
        if (np.diff(bins.astype('float64')) < 0).any():
            raise ValueError('bins must increase monotonically.')
    (fac, bins) = _bins_to_cuts(x, bins, right=right, labels=labels, precision=precision, include_lowest=include_lowest, dtype=dtype, duplicates=duplicates, ordered=ordered)
    return _postprocess_for_cut(fac, bins, retbins, dtype, original)

def qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise'):
    '\n    Quantile-based discretization function.\n\n    Discretize variable into equal-sized buckets based on rank or based\n    on sample quantiles. For example 1000 values for 10 quantiles would\n    produce a Categorical object indicating quantile membership for each data point.\n\n    Parameters\n    ----------\n    x : 1d ndarray or Series\n    q : int or list-like of float\n        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately\n        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.\n    labels : array or False, default None\n        Used as labels for the resulting bins. Must be of the same length as\n        the resulting bins. If False, return only integer indicators of the\n        bins. If True, raises an error.\n    retbins : bool, optional\n        Whether to return the (bins, labels) or not. Can be useful if bins\n        is given as a scalar.\n    precision : int, optional\n        The precision at which to store and display the bins labels.\n    duplicates : {default \'raise\', \'drop\'}, optional\n        If bin edges are not unique, raise ValueError or drop non-uniques.\n\n    Returns\n    -------\n    out : Categorical or Series or array of integers if labels is False\n        The return type (Categorical or Series) depends on the input: a Series\n        of type category if input is a Series else Categorical. Bins are\n        represented as categories when categorical data is returned.\n    bins : ndarray of floats\n        Returned only if `retbins` is True.\n\n    Notes\n    -----\n    Out of bounds values will be NA in the resulting Categorical object\n\n    Examples\n    --------\n    >>> pd.qcut(range(5), 4)\n    ... # doctest: +ELLIPSIS\n    [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]\n    Categories (4, interval[float64]): [(-0.001, 1.0] < (1.0, 2.0] ...\n\n    >>> pd.qcut(range(5), 3, labels=["good", "medium", "bad"])\n    ... # doctest: +SKIP\n    [good, good, medium, bad, bad]\n    Categories (3, object): [good < medium < bad]\n\n    >>> pd.qcut(range(5), 4, labels=False)\n    array([0, 0, 1, 2, 3])\n    '
    original = x
    x = _preprocess_for_cut(x)
    (x, dtype) = _coerce_to_type(x)
    if is_integer(q):
        quantiles = np.linspace(0, 1, (q + 1))
    else:
        quantiles = q
    bins = algos.quantile(x, quantiles)
    (fac, bins) = _bins_to_cuts(x, bins, labels=labels, precision=precision, include_lowest=True, dtype=dtype, duplicates=duplicates)
    return _postprocess_for_cut(fac, bins, retbins, dtype, original)

def _bins_to_cuts(x, bins, right=True, labels=None, precision=3, include_lowest=False, dtype=None, duplicates='raise', ordered=True):
    if ((not ordered) and (labels is None)):
        raise ValueError("'labels' must be provided if 'ordered = False'")
    if (duplicates not in ['raise', 'drop']):
        raise ValueError("invalid value for 'duplicates' parameter, valid options are: raise, drop")
    if isinstance(bins, IntervalIndex):
        ids = bins.get_indexer(x)
        result = Categorical.from_codes(ids, categories=bins, ordered=True)
        return (result, bins)
    unique_bins = algos.unique(bins)
    if ((len(unique_bins) < len(bins)) and (len(bins) != 2)):
        if (duplicates == 'raise'):
            raise ValueError(f'''Bin edges must be unique: {repr(bins)}.
You can drop duplicate edges by setting the 'duplicates' kwarg''')
        else:
            bins = unique_bins
    side = ('left' if right else 'right')
    ids = ensure_int64(bins.searchsorted(x, side=side))
    if include_lowest:
        ids[(x == bins[0])] = 1
    na_mask = ((isna(x) | (ids == len(bins))) | (ids == 0))
    has_nas = na_mask.any()
    if (labels is not False):
        if (not ((labels is None) or is_list_like(labels))):
            raise ValueError('Bin labels must either be False, None or passed in as a list-like argument')
        elif (labels is None):
            labels = _format_labels(bins, precision, right=right, include_lowest=include_lowest, dtype=dtype)
        elif (ordered and (len(set(labels)) != len(labels))):
            raise ValueError('labels must be unique if ordered=True; pass ordered=False for duplicate labels')
        elif (len(labels) != (len(bins) - 1)):
            raise ValueError('Bin labels must be one fewer than the number of bin edges')
        if (not is_categorical_dtype(labels)):
            labels = Categorical(labels, categories=(labels if (len(set(labels)) == len(labels)) else None), ordered=ordered)
        np.putmask(ids, na_mask, 0)
        result = algos.take_nd(labels, (ids - 1))
    else:
        result = (ids - 1)
        if has_nas:
            result = result.astype(np.float64)
            np.putmask(result, na_mask, np.nan)
    return (result, bins)

def _coerce_to_type(x):
    '\n    if the passed data is of datetime/timedelta, bool or nullable int type,\n    this method converts it to numeric so that cut or qcut method can\n    handle it\n    '
    dtype = None
    if is_datetime64tz_dtype(x.dtype):
        dtype = x.dtype
    elif is_datetime64_dtype(x.dtype):
        x = to_datetime(x)
        dtype = np.dtype('datetime64[ns]')
    elif is_timedelta64_dtype(x.dtype):
        x = to_timedelta(x)
        dtype = np.dtype('timedelta64[ns]')
    elif is_bool_dtype(x.dtype):
        x = x.astype(np.int64)
    elif (is_extension_array_dtype(x.dtype) and is_integer_dtype(x.dtype)):
        x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    if (dtype is not None):
        x = np.where(x.notna(), x.view(np.int64), np.nan)
    return (x, dtype)

def _convert_bin_to_numeric_type(bins, dtype):
    '\n    if the passed bin is of datetime/timedelta type,\n    this method converts it to integer\n\n    Parameters\n    ----------\n    bins : list-like of bins\n    dtype : dtype of data\n\n    Raises\n    ------\n    ValueError if bins are not of a compat dtype to dtype\n    '
    bins_dtype = infer_dtype(bins, skipna=False)
    if is_timedelta64_dtype(dtype):
        if (bins_dtype in ['timedelta', 'timedelta64']):
            bins = to_timedelta(bins).view(np.int64)
        else:
            raise ValueError('bins must be of timedelta64 dtype')
    elif (is_datetime64_dtype(dtype) or is_datetime64tz_dtype(dtype)):
        if (bins_dtype in ['datetime', 'datetime64']):
            bins = to_datetime(bins).view(np.int64)
        else:
            raise ValueError('bins must be of datetime64 dtype')
    return bins

def _convert_bin_to_datelike_type(bins, dtype):
    '\n    Convert bins to a DatetimeIndex or TimedeltaIndex if the original dtype is\n    datelike\n\n    Parameters\n    ----------\n    bins : list-like of bins\n    dtype : dtype of data\n\n    Returns\n    -------\n    bins : Array-like of bins, DatetimeIndex or TimedeltaIndex if dtype is\n           datelike\n    '
    if is_datetime64tz_dtype(dtype):
        bins = to_datetime(bins.astype(np.int64), utc=True).tz_convert(dtype.tz)
    elif is_datetime_or_timedelta_dtype(dtype):
        bins = Index(bins.astype(np.int64), dtype=dtype)
    return bins

def _format_labels(bins, precision, right=True, include_lowest=False, dtype=None):
    ' based on the dtype, return our labels '
    closed = ('right' if right else 'left')
    if is_datetime64tz_dtype(dtype):
        formatter = (lambda x: Timestamp(x, tz=dtype.tz))
        adjust = (lambda x: (x - Timedelta('1ns')))
    elif is_datetime64_dtype(dtype):
        formatter = Timestamp
        adjust = (lambda x: (x - Timedelta('1ns')))
    elif is_timedelta64_dtype(dtype):
        formatter = Timedelta
        adjust = (lambda x: (x - Timedelta('1ns')))
    else:
        precision = _infer_precision(precision, bins)
        formatter = (lambda x: _round_frac(x, precision))
        adjust = (lambda x: (x - (10 ** (- precision))))
    breaks = [formatter(b) for b in bins]
    if (right and include_lowest):
        breaks[0] = adjust(breaks[0])
    return IntervalIndex.from_breaks(breaks, closed=closed)

def _preprocess_for_cut(x):
    '\n    handles preprocessing for cut where we convert passed\n    input to array, strip the index information and store it\n    separately\n    '
    ndim = getattr(x, 'ndim', None)
    if (ndim is None):
        x = np.asarray(x)
    if (x.ndim != 1):
        raise ValueError('Input array must be 1 dimensional')
    return x

def _postprocess_for_cut(fac, bins, retbins, dtype, original):
    '\n    handles post processing for the cut method where\n    we combine the index information if the originally passed\n    datatype was a series\n    '
    if isinstance(original, ABCSeries):
        fac = original._constructor(fac, index=original.index, name=original.name)
    if (not retbins):
        return fac
    bins = _convert_bin_to_datelike_type(bins, dtype)
    return (fac, bins)

def _round_frac(x, precision):
    '\n    Round the fractional part of the given number\n    '
    if ((not np.isfinite(x)) or (x == 0)):
        return x
    else:
        (frac, whole) = np.modf(x)
        if (whole == 0):
            digits = (((- int(np.floor(np.log10(abs(frac))))) - 1) + precision)
        else:
            digits = precision
        return np.around(x, digits)

def _infer_precision(base_precision, bins):
    '\n    Infer an appropriate precision for _round_frac\n    '
    for precision in range(base_precision, 20):
        levels = [_round_frac(b, precision) for b in bins]
        if (algos.unique(levels).size == bins.size):
            return precision
    return base_precision
