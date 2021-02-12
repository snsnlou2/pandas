
'\nProvide basic components for groupby. These definitions\nhold the allowlist of methods that are exposed on the\nSeriesGroupBy and the DataFrameGroupBy objects.\n'
import collections
from typing import List
from pandas._typing import final
from pandas.core.dtypes.common import is_list_like, is_scalar
from pandas.core.base import PandasObject
OutputKey = collections.namedtuple('OutputKey', ['label', 'position'])

class ShallowMixin(PandasObject):
    _attributes = []

    @final
    def _shallow_copy(self, obj, **kwargs):
        '\n        return a new object with the replacement attributes\n        '
        if isinstance(obj, self._constructor):
            obj = obj.obj
        for attr in self._attributes:
            if (attr not in kwargs):
                kwargs[attr] = getattr(self, attr)
        return self._constructor(obj, **kwargs)

class GotItemMixin(PandasObject):
    '\n    Provide the groupby facilities to the mixed object.\n    '

    @final
    def _gotitem(self, key, ndim, subset=None):
        '\n        Sub-classes to define. Return a sliced object.\n\n        Parameters\n        ----------\n        key : string / list of selections\n        ndim : {1, 2}\n            requested ndim of result\n        subset : object, default None\n            subset to act on\n        '
        if (subset is None):
            subset = self.obj
        kwargs = {attr: getattr(self, attr) for attr in self._attributes}
        try:
            groupby = self._groupby[key]
        except IndexError:
            groupby = self._groupby
        self = type(self)(subset, groupby=groupby, parent=self, **kwargs)
        self._reset_cache()
        if ((subset.ndim == 2) and ((is_scalar(key) and (key in subset)) or is_list_like(key))):
            self._selection = key
        return self
plotting_methods = frozenset(['plot', 'hist'])
common_apply_allowlist = (frozenset(['quantile', 'fillna', 'mad', 'take', 'idxmax', 'idxmin', 'tshift', 'skew', 'corr', 'cov', 'diff']) | plotting_methods)
series_apply_allowlist = ((common_apply_allowlist | {'nlargest', 'nsmallest', 'is_monotonic_increasing', 'is_monotonic_decreasing'}) | frozenset(['dtype', 'unique']))
dataframe_apply_allowlist = (common_apply_allowlist | frozenset(['dtypes', 'corrwith']))
cythonized_kernels = frozenset(['cumprod', 'cumsum', 'shift', 'cummin', 'cummax'])
cython_cast_blocklist = frozenset(['rank', 'count', 'size', 'idxmin', 'idxmax'])
reduction_kernels = frozenset(['all', 'any', 'corrwith', 'count', 'first', 'idxmax', 'idxmin', 'last', 'mad', 'max', 'mean', 'median', 'min', 'ngroup', 'nth', 'nunique', 'prod', 'quantile', 'sem', 'size', 'skew', 'std', 'sum', 'var'])
transformation_kernels = frozenset(['backfill', 'bfill', 'cumcount', 'cummax', 'cummin', 'cumprod', 'cumsum', 'diff', 'ffill', 'fillna', 'pad', 'pct_change', 'rank', 'shift', 'tshift'])
groupby_other_methods = frozenset(['agg', 'aggregate', 'apply', 'boxplot', 'corr', 'cov', 'describe', 'dtypes', 'expanding', 'ewm', 'filter', 'get_group', 'groups', 'head', 'hist', 'indices', 'ndim', 'ngroups', 'ohlc', 'pipe', 'plot', 'resample', 'rolling', 'tail', 'take', 'transform', 'sample'])
transform_kernel_allowlist = (reduction_kernels | transformation_kernels)
