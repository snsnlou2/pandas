
'Common utilities for Numba operations'
from distutils.version import LooseVersion
import types
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError
GLOBAL_USE_NUMBA = False
NUMBA_FUNC_CACHE = {}

def maybe_use_numba(engine):
    'Signal whether to use numba routines.'
    return ((engine == 'numba') or ((engine is None) and GLOBAL_USE_NUMBA))

def set_use_numba(enable=False):
    global GLOBAL_USE_NUMBA
    if enable:
        import_optional_dependency('numba')
    GLOBAL_USE_NUMBA = enable

def get_jit_arguments(engine_kwargs=None, kwargs=None):
    '\n    Return arguments to pass to numba.JIT, falling back on pandas default JIT settings.\n\n    Parameters\n    ----------\n    engine_kwargs : dict, default None\n        user passed keyword arguments for numba.JIT\n    kwargs : dict, default None\n        user passed keyword arguments to pass into the JITed function\n\n    Returns\n    -------\n    (bool, bool, bool)\n        nopython, nogil, parallel\n\n    Raises\n    ------\n    NumbaUtilError\n    '
    if (engine_kwargs is None):
        engine_kwargs = {}
    nopython = engine_kwargs.get('nopython', True)
    if (kwargs and nopython):
        raise NumbaUtilError('numba does not support kwargs with nopython=True: https://github.com/numba/numba/issues/2916')
    nogil = engine_kwargs.get('nogil', False)
    parallel = engine_kwargs.get('parallel', False)
    return (nopython, nogil, parallel)

def jit_user_function(func, nopython, nogil, parallel):
    "\n    JIT the user's function given the configurable arguments.\n\n    Parameters\n    ----------\n    func : function\n        user defined function\n    nopython : bool\n        nopython parameter for numba.JIT\n    nogil : bool\n        nogil parameter for numba.JIT\n    parallel : bool\n        parallel parameter for numba.JIT\n\n    Returns\n    -------\n    function\n        Numba JITed function\n    "
    numba = import_optional_dependency('numba')
    if (LooseVersion(numba.__version__) >= LooseVersion('0.49.0')):
        is_jitted = numba.extending.is_jitted(func)
    else:
        is_jitted = isinstance(func, numba.targets.registry.CPUDispatcher)
    if is_jitted:
        numba_func = func
    else:

        @numba.generated_jit(nopython=nopython, nogil=nogil, parallel=parallel)
        def numba_func(data, *_args):
            if ((getattr(np, func.__name__, False) is func) or isinstance(func, types.BuiltinFunctionType)):
                jf = func
            else:
                jf = numba.jit(func, nopython=nopython, nogil=nogil)

            def impl(data, *_args):
                return jf(data, *_args)
            return impl
    return numba_func
