
'\nBoilerplate functions used in defining binary operations.\n'
from functools import wraps
from typing import Callable
from pandas._libs.lib import item_from_zerodim
from pandas._typing import F
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries

def unpack_zerodim_and_defer(name):
    '\n    Boilerplate for pandas conventions in arithmetic and comparison methods.\n\n    Parameters\n    ----------\n    name : str\n\n    Returns\n    -------\n    decorator\n    '

    def wrapper(method: F) -> F:
        return _unpack_zerodim_and_defer(method, name)
    return wrapper

def _unpack_zerodim_and_defer(method, name):
    '\n    Boilerplate for pandas conventions in arithmetic and comparison methods.\n\n    Ensure method returns NotImplemented when operating against "senior"\n    classes.  Ensure zero-dimensional ndarrays are always unpacked.\n\n    Parameters\n    ----------\n    method : binary method\n    name : str\n\n    Returns\n    -------\n    method\n    '
    is_cmp = (name.strip('__') in {'eq', 'ne', 'lt', 'le', 'gt', 'ge'})

    @wraps(method)
    def new_method(self, other):
        if (is_cmp and isinstance(self, ABCIndex) and isinstance(other, ABCSeries)):
            pass
        else:
            for cls in [ABCDataFrame, ABCSeries, ABCIndex]:
                if isinstance(self, cls):
                    break
                if isinstance(other, cls):
                    return NotImplemented
        other = item_from_zerodim(other)
        return method(self, other)
    return new_method

def get_op_result_name(left, right):
    '\n    Find the appropriate name to pin to an operation result.  This result\n    should always be either an Index or a Series.\n\n    Parameters\n    ----------\n    left : {Series, Index}\n    right : object\n\n    Returns\n    -------\n    name : object\n        Usually a string\n    '
    if isinstance(right, (ABCSeries, ABCIndex)):
        name = _maybe_match_name(left, right)
    else:
        name = left.name
    return name

def _maybe_match_name(a, b):
    '\n    Try to find a name to attach to the result of an operation between\n    a and b.  If only one of these has a `name` attribute, return that\n    name.  Otherwise return a consensus name if they match or None if\n    they have different names.\n\n    Parameters\n    ----------\n    a : object\n    b : object\n\n    Returns\n    -------\n    name : str or None\n\n    See Also\n    --------\n    pandas.core.common.consensus_name_attr\n    '
    a_has = hasattr(a, 'name')
    b_has = hasattr(b, 'name')
    if (a_has and b_has):
        if (a.name == b.name):
            return a.name
        else:
            return None
    elif a_has:
        return a.name
    elif b_has:
        return b.name
    return None
