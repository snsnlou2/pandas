
import abc
import inspect
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple, Type
import numpy as np
from pandas._config import option_context
from pandas._typing import AggFuncType, Axis, FrameOrSeriesUnion
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_dict_like, is_extension_array_dtype, is_list_like, is_sequence
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.construction import create_series_with_explicit_dtype
if TYPE_CHECKING:
    from pandas import DataFrame, Index, Series
ResType = Dict[(int, Any)]

def frame_apply(obj, func, axis=0, raw=False, result_type=None, args=None, kwds=None):
    ' construct and return a row or column based frame apply object '
    axis = obj._get_axis_number(axis)
    klass: Type[FrameApply]
    if (axis == 0):
        klass = FrameRowApply
    elif (axis == 1):
        klass = FrameColumnApply
    return klass(obj, func, raw=raw, result_type=result_type, args=args, kwds=kwds)

class FrameApply(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def result_index(self):
        pass

    @property
    @abc.abstractmethod
    def result_columns(self):
        pass

    @property
    @abc.abstractmethod
    def series_generator(self):
        pass

    @abc.abstractmethod
    def wrap_results_for_axis(self, results, res_index):
        pass

    def __init__(self, obj, func, raw, result_type, args, kwds):
        self.obj = obj
        self.raw = raw
        self.args = (args or ())
        self.kwds = (kwds or {})
        if (result_type not in [None, 'reduce', 'broadcast', 'expand']):
            raise ValueError("invalid value for result_type, must be one of {None, 'reduce', 'broadcast', 'expand'}")
        self.result_type = result_type
        if ((kwds or args) and (not isinstance(func, (np.ufunc, str)))):

            def f(x):
                return func(x, *args, **kwds)
        else:
            f = func
        self.f = f

    @property
    def res_columns(self):
        return self.result_columns

    @property
    def columns(self):
        return self.obj.columns

    @property
    def index(self):
        return self.obj.index

    @cache_readonly
    def values(self):
        return self.obj.values

    @cache_readonly
    def dtypes(self):
        return self.obj.dtypes

    @property
    def agg_axis(self):
        return self.obj._get_agg_axis(self.axis)

    def get_result(self):
        ' compute the results '
        if (is_list_like(self.f) or is_dict_like(self.f)):
            return self.obj.aggregate(self.f, *self.args, axis=self.axis, **self.kwds)
        if ((len(self.columns) == 0) and (len(self.index) == 0)):
            return self.apply_empty_result()
        if isinstance(self.f, str):
            func = getattr(self.obj, self.f)
            sig = inspect.getfullargspec(func)
            if ('axis' in sig.args):
                self.kwds['axis'] = self.axis
            return func(*self.args, **self.kwds)
        elif isinstance(self.f, np.ufunc):
            with np.errstate(all='ignore'):
                results = self.obj._mgr.apply('apply', func=self.f)
            return self.obj._constructor(data=results)
        if (self.result_type == 'broadcast'):
            return self.apply_broadcast(self.obj)
        elif (not all(self.obj.shape)):
            return self.apply_empty_result()
        elif self.raw:
            return self.apply_raw()
        return self.apply_standard()

    def apply_empty_result(self):
        '\n        we have an empty result; at least 1 axis is 0\n\n        we will try to apply the function to an empty\n        series in order to see if this is a reduction function\n        '
        if (self.result_type not in ['reduce', None]):
            return self.obj.copy()
        should_reduce = (self.result_type == 'reduce')
        from pandas import Series
        if (not should_reduce):
            try:
                r = self.f(Series([], dtype=np.float64))
            except Exception:
                pass
            else:
                should_reduce = (not isinstance(r, Series))
        if should_reduce:
            if len(self.agg_axis):
                r = self.f(Series([], dtype=np.float64))
            else:
                r = np.nan
            return self.obj._constructor_sliced(r, index=self.agg_axis)
        else:
            return self.obj.copy()

    def apply_raw(self):
        ' apply to the values as a numpy array '

        def wrap_function(func):
            '\n            Wrap user supplied function to work around numpy issue.\n\n            see https://github.com/numpy/numpy/issues/8352\n            '

            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    result = np.array(result, dtype=object)
                return result
            return wrapper
        result = np.apply_along_axis(wrap_function(self.f), self.axis, self.values)
        if (result.ndim == 2):
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target):
        result_values = np.empty_like(target.values)
        result_compare = target.shape[0]
        for (i, col) in enumerate(target.columns):
            res = self.f(target[col])
            ares = np.asarray(res).ndim
            if (ares > 1):
                raise ValueError('too many dims to broadcast')
            elif (ares == 1):
                if (result_compare != len(res)):
                    raise ValueError('cannot broadcast result')
            result_values[:, i] = res
        result = self.obj._constructor(result_values, index=target.index, columns=target.columns)
        return result

    def apply_standard(self):
        (results, res_index) = self.apply_series_generator()
        return self.wrap_results(results, res_index)

    def apply_series_generator(self):
        series_gen = self.series_generator
        res_index = self.result_index
        results = {}
        with option_context('mode.chained_assignment', None):
            for (i, v) in enumerate(series_gen):
                results[i] = self.f(v)
                if isinstance(results[i], ABCSeries):
                    results[i] = results[i].copy(deep=False)
        return (results, res_index)

    def wrap_results(self, results, res_index):
        from pandas import Series
        if ((len(results) > 0) and (0 in results) and is_sequence(results[0])):
            return self.wrap_results_for_axis(results, res_index)
        constructor_sliced = self.obj._constructor_sliced
        if (constructor_sliced is Series):
            result = create_series_with_explicit_dtype(results, dtype_if_empty=np.float64)
        else:
            result = constructor_sliced(results)
        result.index = res_index
        return result

class FrameRowApply(FrameApply):
    axis = 0

    def apply_broadcast(self, target):
        return super().apply_broadcast(target)

    @property
    def series_generator(self):
        return (self.obj._ixs(i, axis=1) for i in range(len(self.columns)))

    @property
    def result_index(self):
        return self.columns

    @property
    def result_columns(self):
        return self.index

    def wrap_results_for_axis(self, results, res_index):
        ' return the results for the rows '
        if (self.result_type == 'reduce'):
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res
        elif ((self.result_type is None) and all((isinstance(x, dict) for x in results.values()))):
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res
        try:
            result = self.obj._constructor(data=results)
        except ValueError as err:
            if ('All arrays must be of the same length' in str(err)):
                res = self.obj._constructor_sliced(results)
                res.index = res_index
                return res
            else:
                raise
        if (not isinstance(results[0], ABCSeries)):
            if (len(result.index) == len(self.res_columns)):
                result.index = self.res_columns
        if (len(result.columns) == len(res_index)):
            result.columns = res_index
        return result

class FrameColumnApply(FrameApply):
    axis = 1

    def apply_broadcast(self, target):
        result = super().apply_broadcast(target.T)
        return result.T

    @property
    def series_generator(self):
        values = self.values
        assert (len(values) > 0)
        ser = self.obj._ixs(0, axis=0)
        mgr = ser._mgr
        blk = mgr.blocks[0]
        if is_extension_array_dtype(blk.dtype):
            obj = self.obj
            for i in range(len(obj)):
                (yield obj._ixs(i, axis=0))
        else:
            for (arr, name) in zip(values, self.index):
                ser._mgr = mgr
                blk.values = arr
                ser.name = name
                (yield ser)

    @property
    def result_index(self):
        return self.index

    @property
    def result_columns(self):
        return self.columns

    def wrap_results_for_axis(self, results, res_index):
        ' return the results for the columns '
        result: FrameOrSeriesUnion
        if (self.result_type == 'expand'):
            result = self.infer_to_same_shape(results, res_index)
        elif (not isinstance(results[0], ABCSeries)):
            result = self.obj._constructor_sliced(results)
            result.index = res_index
        else:
            result = self.infer_to_same_shape(results, res_index)
        return result

    def infer_to_same_shape(self, results, res_index):
        ' infer the results to the same shape as the input object '
        result = self.obj._constructor(data=results)
        result = result.T
        result.index = res_index
        result = result.infer_objects()
        return result
