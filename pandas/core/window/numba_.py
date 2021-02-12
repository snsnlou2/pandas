
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from pandas._typing import Scalar
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import NUMBA_FUNC_CACHE, get_jit_arguments, jit_user_function

def generate_numba_apply_func(args, kwargs, func, engine_kwargs, name):
    "\n    Generate a numba jitted apply function specified by values from engine_kwargs.\n\n    1. jit the user's function\n    2. Return a rolling apply function with the jitted function inline\n\n    Configurations specified in engine_kwargs apply to both the user's\n    function _AND_ the rolling apply function.\n\n    Parameters\n    ----------\n    args : tuple\n        *args to be passed into the function\n    kwargs : dict\n        **kwargs to be passed into the function\n    func : function\n        function to be applied to each window and will be JITed\n    engine_kwargs : dict\n        dictionary of arguments to be passed into numba.jit\n    name: str\n        name of the caller (Rolling/Expanding)\n\n    Returns\n    -------\n    Numba function\n    "
    (nopython, nogil, parallel) = get_jit_arguments(engine_kwargs, kwargs)
    cache_key = (func, f'{name}_apply_single')
    if (cache_key in NUMBA_FUNC_CACHE):
        return NUMBA_FUNC_CACHE[cache_key]
    numba_func = jit_user_function(func, nopython, nogil, parallel)
    numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_apply(values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int) -> np.ndarray:
        result = np.empty(len(begin))
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window))
            if ((len(window) - count_nan) >= minimum_periods):
                result[i] = numba_func(window, *args)
            else:
                result[i] = np.nan
        return result
    return roll_apply

def generate_numba_groupby_ewma_func(engine_kwargs, com, adjust, ignore_na):
    '\n    Generate a numba jitted groupby ewma function specified by values\n    from engine_kwargs.\n\n    Parameters\n    ----------\n    engine_kwargs : dict\n        dictionary of arguments to be passed into numba.jit\n    com : float\n    adjust : bool\n    ignore_na : bool\n\n    Returns\n    -------\n    Numba function\n    '
    (nopython, nogil, parallel) = get_jit_arguments(engine_kwargs)
    cache_key = ((lambda x: x), 'groupby_ewma')
    if (cache_key in NUMBA_FUNC_CACHE):
        return NUMBA_FUNC_CACHE[cache_key]
    numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def groupby_ewma(values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int) -> np.ndarray:
        result = np.empty(len(values))
        alpha = (1.0 / (1.0 + com))
        for i in numba.prange(len(begin)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            sub_result = np.empty(len(window))
            old_wt_factor = (1.0 - alpha)
            new_wt = (1.0 if adjust else alpha)
            weighted_avg = window[0]
            nobs = int((not np.isnan(weighted_avg)))
            sub_result[0] = (weighted_avg if (nobs >= minimum_periods) else np.nan)
            old_wt = 1.0
            for j in range(1, len(window)):
                cur = window[j]
                is_observation = (not np.isnan(cur))
                nobs += is_observation
                if (not np.isnan(weighted_avg)):
                    if (is_observation or (not ignore_na)):
                        old_wt *= old_wt_factor
                        if is_observation:
                            if (weighted_avg != cur):
                                weighted_avg = (((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt))
                            if adjust:
                                old_wt += new_wt
                            else:
                                old_wt = 1.0
                elif is_observation:
                    weighted_avg = cur
                sub_result[j] = (weighted_avg if (nobs >= minimum_periods) else np.nan)
            result[start:stop] = sub_result
        return result
    return groupby_ewma

def generate_numba_table_func(args, kwargs, func, engine_kwargs, name):
    "\n    Generate a numba jitted function to apply window calculations table-wise.\n\n    Func will be passed a M window size x N number of columns array, and\n    must return a 1 x N number of columns array. Func is intended to operate\n    row-wise, but the result will be transposed for axis=1.\n\n    1. jit the user's function\n    2. Return a rolling apply function with the jitted function inline\n\n    Parameters\n    ----------\n    args : tuple\n        *args to be passed into the function\n    kwargs : dict\n        **kwargs to be passed into the function\n    func : function\n        function to be applied to each window and will be JITed\n    engine_kwargs : dict\n        dictionary of arguments to be passed into numba.jit\n    name : str\n        caller (Rolling/Expanding) and original method name for numba cache key\n\n    Returns\n    -------\n    Numba function\n    "
    (nopython, nogil, parallel) = get_jit_arguments(engine_kwargs, kwargs)
    cache_key = (func, f'{name}_table')
    if (cache_key in NUMBA_FUNC_CACHE):
        return NUMBA_FUNC_CACHE[cache_key]
    numba_func = jit_user_function(func, nopython, nogil, parallel)
    numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_table(values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int):
        result = np.empty(values.shape)
        min_periods_mask = np.empty(values.shape)
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window), axis=0)
            sub_result = numba_func(window, *args)
            nan_mask = ((len(window) - count_nan) >= minimum_periods)
            min_periods_mask[i, :] = nan_mask
            result[i, :] = sub_result
        result = np.where(min_periods_mask, result, np.nan)
        return result
    return roll_table
