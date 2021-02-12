
'Common utilities for Numba operations with groupby ops'
import inspect
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from pandas._typing import Scalar
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import NUMBA_FUNC_CACHE, NumbaUtilError, get_jit_arguments, jit_user_function

def validate_udf(func):
    '\n    Validate user defined function for ops when using Numba with groupby ops.\n\n    The first signature arguments should include:\n\n    def f(values, index, ...):\n        ...\n\n    Parameters\n    ----------\n    func : function, default False\n        user defined function\n\n    Returns\n    -------\n    None\n\n    Raises\n    ------\n    NumbaUtilError\n    '
    udf_signature = list(inspect.signature(func).parameters.keys())
    expected_args = ['values', 'index']
    min_number_args = len(expected_args)
    if ((len(udf_signature) < min_number_args) or (udf_signature[:min_number_args] != expected_args)):
        raise NumbaUtilError(f'The first {min_number_args} arguments to {func.__name__} must be {expected_args}')

def generate_numba_agg_func(args, kwargs, func, engine_kwargs):
    "\n    Generate a numba jitted agg function specified by values from engine_kwargs.\n\n    1. jit the user's function\n    2. Return a groupby agg function with the jitted function inline\n\n    Configurations specified in engine_kwargs apply to both the user's\n    function _AND_ the groupby evaluation loop.\n\n    Parameters\n    ----------\n    args : tuple\n        *args to be passed into the function\n    kwargs : dict\n        **kwargs to be passed into the function\n    func : function\n        function to be applied to each window and will be JITed\n    engine_kwargs : dict\n        dictionary of arguments to be passed into numba.jit\n\n    Returns\n    -------\n    Numba function\n    "
    (nopython, nogil, parallel) = get_jit_arguments(engine_kwargs, kwargs)
    validate_udf(func)
    cache_key = (func, 'groupby_agg')
    if (cache_key in NUMBA_FUNC_CACHE):
        return NUMBA_FUNC_CACHE[cache_key]
    numba_func = jit_user_function(func, nopython, nogil, parallel)
    numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_agg(values: np.ndarray, index: np.ndarray, begin: np.ndarray, end: np.ndarray, num_groups: int, num_columns: int) -> np.ndarray:
        result = np.empty((num_groups, num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i]:end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i]:end[i], j]
                result[(i, j)] = numba_func(group, group_index, *args)
        return result
    return group_agg

def generate_numba_transform_func(args, kwargs, func, engine_kwargs):
    "\n    Generate a numba jitted transform function specified by values from engine_kwargs.\n\n    1. jit the user's function\n    2. Return a groupby transform function with the jitted function inline\n\n    Configurations specified in engine_kwargs apply to both the user's\n    function _AND_ the groupby evaluation loop.\n\n    Parameters\n    ----------\n    args : tuple\n        *args to be passed into the function\n    kwargs : dict\n        **kwargs to be passed into the function\n    func : function\n        function to be applied to each window and will be JITed\n    engine_kwargs : dict\n        dictionary of arguments to be passed into numba.jit\n\n    Returns\n    -------\n    Numba function\n    "
    (nopython, nogil, parallel) = get_jit_arguments(engine_kwargs, kwargs)
    validate_udf(func)
    cache_key = (func, 'groupby_transform')
    if (cache_key in NUMBA_FUNC_CACHE):
        return NUMBA_FUNC_CACHE[cache_key]
    numba_func = jit_user_function(func, nopython, nogil, parallel)
    numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_transform(values: np.ndarray, index: np.ndarray, begin: np.ndarray, end: np.ndarray, num_groups: int, num_columns: int) -> np.ndarray:
        result = np.empty((len(values), num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i]:end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i]:end[i], j]
                result[begin[i]:end[i], j] = numba_func(group, group_index, *args)
        return result
    return group_transform
