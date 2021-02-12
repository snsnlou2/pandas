
from textwrap import dedent
import pytest
from pandas.util._decorators import deprecate
import pandas._testing as tm

def new_func():
    '\n    This is the summary. The deprecate directive goes next.\n\n    This is the extended summary. The deprecate directive goes before this.\n    '
    return 'new_func called'

def new_func_no_docstring():
    return 'new_func_no_docstring called'

def new_func_wrong_docstring():
    'Summary should be in the next line.'
    return 'new_func_wrong_docstring called'

def new_func_with_deprecation():
    '\n    This is the summary. The deprecate directive goes next.\n\n    .. deprecated:: 1.0\n        Use new_func instead.\n\n    This is the extended summary. The deprecate directive goes before this.\n    '
    pass

def test_deprecate_ok():
    depr_func = deprecate('depr_func', new_func, '1.0', msg='Use new_func instead.')
    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()
    assert (result == 'new_func called')
    assert (depr_func.__doc__ == dedent(new_func_with_deprecation.__doc__))

def test_deprecate_no_docstring():
    depr_func = deprecate('depr_func', new_func_no_docstring, '1.0', msg='Use new_func instead.')
    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()
    assert (result == 'new_func_no_docstring called')

def test_deprecate_wrong_docstring():
    msg = 'deprecate needs a correctly formatted docstring'
    with pytest.raises(AssertionError, match=msg):
        deprecate('depr_func', new_func_wrong_docstring, '1.0', msg='Use new_func instead.')
