
from textwrap import dedent
from pandas.util._decorators import doc

@doc(method='cumsum', operation='sum')
def cumsum(whatever):
    '\n    This is the {method} method.\n\n    It computes the cumulative {operation}.\n    '

@doc(cumsum, dedent('\n        Examples\n        --------\n\n        >>> cumavg([1, 2, 3])\n        2\n        '), method='cumavg', operation='average')
def cumavg(whatever):
    pass

@doc(cumsum, method='cummax', operation='maximum')
def cummax(whatever):
    pass

@doc(cummax, method='cummin', operation='minimum')
def cummin(whatever):
    pass

def test_docstring_formatting():
    docstr = dedent('\n        This is the cumsum method.\n\n        It computes the cumulative sum.\n        ')
    assert (cumsum.__doc__ == docstr)

def test_docstring_appending():
    docstr = dedent('\n        This is the cumavg method.\n\n        It computes the cumulative average.\n\n        Examples\n        --------\n\n        >>> cumavg([1, 2, 3])\n        2\n        ')
    assert (cumavg.__doc__ == docstr)

def test_doc_template_from_func():
    docstr = dedent('\n        This is the cummax method.\n\n        It computes the cumulative maximum.\n        ')
    assert (cummax.__doc__ == docstr)

def test_inherit_doc_template():
    docstr = dedent('\n        This is the cummin method.\n\n        It computes the cumulative minimum.\n        ')
    assert (cummin.__doc__ == docstr)
