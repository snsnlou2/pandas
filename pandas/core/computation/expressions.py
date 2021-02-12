
'\nExpressions\n-----------\n\nOffer fast expression evaluation through numexpr\n\n'
import operator
from typing import List, Set
import warnings
import numpy as np
from pandas._config import get_option
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.computation.check import NUMEXPR_INSTALLED
from pandas.core.ops import roperator
if NUMEXPR_INSTALLED:
    import numexpr as ne
_TEST_MODE = None
_TEST_RESULT = []
USE_NUMEXPR = NUMEXPR_INSTALLED
_evaluate = None
_where = None
_ALLOWED_DTYPES = {'evaluate': {'int64', 'int32', 'float64', 'float32', 'bool'}, 'where': {'int64', 'float64', 'bool'}}
_MIN_ELEMENTS = 10000

def set_use_numexpr(v=True):
    global USE_NUMEXPR
    if NUMEXPR_INSTALLED:
        USE_NUMEXPR = v
    global _evaluate, _where
    _evaluate = (_evaluate_numexpr if USE_NUMEXPR else _evaluate_standard)
    _where = (_where_numexpr if USE_NUMEXPR else _where_standard)

def set_numexpr_threads(n=None):
    if (NUMEXPR_INSTALLED and USE_NUMEXPR):
        if (n is None):
            n = ne.detect_number_of_cores()
        ne.set_num_threads(n)

def _evaluate_standard(op, op_str, a, b):
    '\n    Standard evaluation.\n    '
    if _TEST_MODE:
        _store_test_result(False)
    with np.errstate(all='ignore'):
        return op(a, b)

def _can_use_numexpr(op, op_str, a, b, dtype_check):
    ' return a boolean if we WILL be using numexpr '
    if (op_str is not None):
        if (np.prod(a.shape) > _MIN_ELEMENTS):
            dtypes: Set[str] = set()
            for o in [a, b]:
                if (hasattr(o, 'dtypes') and (o.ndim > 1)):
                    s = o.dtypes.value_counts()
                    if (len(s) > 1):
                        return False
                    dtypes |= set(s.index.astype(str))
                elif hasattr(o, 'dtype'):
                    dtypes |= {o.dtype.name}
            if ((not len(dtypes)) or (_ALLOWED_DTYPES[dtype_check] >= dtypes)):
                return True
    return False

def _evaluate_numexpr(op, op_str, a, b):
    result = None
    if _can_use_numexpr(op, op_str, a, b, 'evaluate'):
        is_reversed = op.__name__.strip('_').startswith('r')
        if is_reversed:
            (a, b) = (b, a)
        a_value = a
        b_value = b
        result = ne.evaluate(f'a_value {op_str} b_value', local_dict={'a_value': a_value, 'b_value': b_value}, casting='safe')
    if _TEST_MODE:
        _store_test_result((result is not None))
    if (result is None):
        result = _evaluate_standard(op, op_str, a, b)
    return result
_op_str_mapping = {operator.add: '+', roperator.radd: '+', operator.mul: '*', roperator.rmul: '*', operator.sub: '-', roperator.rsub: '-', operator.truediv: '/', roperator.rtruediv: '/', operator.floordiv: '//', roperator.rfloordiv: '//', operator.mod: None, roperator.rmod: '%', operator.pow: '**', roperator.rpow: '**', operator.eq: '==', operator.ne: '!=', operator.le: '<=', operator.lt: '<', operator.ge: '>=', operator.gt: '>', operator.and_: '&', roperator.rand_: '&', operator.or_: '|', roperator.ror_: '|', operator.xor: '^', roperator.rxor: '^', divmod: None, roperator.rdivmod: None}

def _where_standard(cond, a, b):
    return np.where(cond, a, b)

def _where_numexpr(cond, a, b):
    result = None
    if _can_use_numexpr(None, 'where', a, b, 'where'):
        result = ne.evaluate('where(cond_value, a_value, b_value)', local_dict={'cond_value': cond, 'a_value': a, 'b_value': b}, casting='safe')
    if (result is None):
        result = _where_standard(cond, a, b)
    return result
set_use_numexpr(get_option('compute.use_numexpr'))

def _has_bool_dtype(x):
    if isinstance(x, ABCDataFrame):
        return ('bool' in x.dtypes)
    try:
        return (x.dtype == bool)
    except AttributeError:
        return isinstance(x, (bool, np.bool_))

def _bool_arith_check(op_str, a, b, not_allowed=frozenset(('/', '//', '**')), unsupported=None):
    if (unsupported is None):
        unsupported = {'+': '|', '*': '&', '-': '^'}
    if (_has_bool_dtype(a) and _has_bool_dtype(b)):
        if (op_str in unsupported):
            warnings.warn(f'evaluating in Python space because the {repr(op_str)} operator is not supported by numexpr for the bool dtype, use {repr(unsupported[op_str])} instead')
            return False
        if (op_str in not_allowed):
            raise NotImplementedError(f'operator {repr(op_str)} not implemented for bool dtypes')
    return True

def evaluate(op, a, b, use_numexpr=True):
    '\n    Evaluate and return the expression of the op on a and b.\n\n    Parameters\n    ----------\n    op : the actual operand\n    a : left operand\n    b : right operand\n    use_numexpr : bool, default True\n        Whether to try to use numexpr.\n    '
    op_str = _op_str_mapping[op]
    if (op_str is not None):
        use_numexpr = (use_numexpr and _bool_arith_check(op_str, a, b))
        if use_numexpr:
            return _evaluate(op, op_str, a, b)
    return _evaluate_standard(op, op_str, a, b)

def where(cond, a, b, use_numexpr=True):
    '\n    Evaluate the where condition cond on a and b.\n\n    Parameters\n    ----------\n    cond : np.ndarray[bool]\n    a : return if cond is True\n    b : return if cond is False\n    use_numexpr : bool, default True\n        Whether to try to use numexpr.\n    '
    assert (_where is not None)
    return (_where(cond, a, b) if use_numexpr else _where_standard(cond, a, b))

def set_test_mode(v=True):
    '\n    Keeps track of whether numexpr was used.\n\n    Stores an additional ``True`` for every successful use of evaluate with\n    numexpr since the last ``get_test_result``.\n    '
    global _TEST_MODE, _TEST_RESULT
    _TEST_MODE = v
    _TEST_RESULT = []

def _store_test_result(used_numexpr):
    global _TEST_RESULT
    if used_numexpr:
        _TEST_RESULT.append(used_numexpr)

def get_test_result():
    '\n    Get test result and reset test_results.\n    '
    global _TEST_RESULT
    res = _TEST_RESULT
    _TEST_RESULT = []
    return res
