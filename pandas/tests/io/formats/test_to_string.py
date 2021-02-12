
from datetime import datetime
from io import StringIO
from textwrap import dedent
import numpy as np
import pytest
from pandas import DataFrame, Series, option_context, to_datetime

def test_repr_embedded_ndarray():
    arr = np.empty(10, dtype=[('err', object)])
    for i in range(len(arr)):
        arr['err'][i] = np.random.randn(i)
    df = DataFrame(arr)
    repr(df['err'])
    repr(df)
    df.to_string()

def test_repr_tuples():
    buf = StringIO()
    df = DataFrame({'tups': list(zip(range(10), range(10)))})
    repr(df)
    df.to_string(col_space=10, buf=buf)

def test_to_string_truncate():
    df = DataFrame([{'a': 'foo', 'b': 'bar', 'c': "let's make this a very VERY long line that is longer than the default 50 character limit", 'd': 1}, {'a': 'foo', 'b': 'bar', 'c': 'stuff', 'd': 1}])
    df.set_index(['a', 'b', 'c'])
    assert (df.to_string() == "     a    b                                                                                         c  d\n0  foo  bar  let's make this a very VERY long line that is longer than the default 50 character limit  1\n1  foo  bar                                                                                     stuff  1")
    with option_context('max_colwidth', 20):
        assert (df.to_string() == "     a    b                                                                                         c  d\n0  foo  bar  let's make this a very VERY long line that is longer than the default 50 character limit  1\n1  foo  bar                                                                                     stuff  1")
    assert (df.to_string(max_colwidth=20) == "     a    b                    c  d\n0  foo  bar  let's make this ...  1\n1  foo  bar                stuff  1")

@pytest.mark.parametrize('input_array, expected', [('a', 'a'), (['a', 'b'], 'a\nb'), ([1, 'a'], '1\na'), (1, '1'), ([0, (- 1)], ' 0\n-1'), (1.0, '1.0'), ([' a', ' b'], ' a\n b'), (['.1', '1'], '.1\n 1'), (['10', '-10'], ' 10\n-10')])
def test_format_remove_leading_space_series(input_array, expected):
    s = Series(input_array).to_string(index=False)
    assert (s == expected)

@pytest.mark.parametrize('input_array, expected', [({'A': ['a']}, 'A\na'), ({'A': ['a', 'b'], 'B': ['c', 'dd']}, 'A  B\na  c\nb dd'), ({'A': ['a', 1], 'B': ['aa', 1]}, 'A  B\na aa\n1  1')])
def test_format_remove_leading_space_dataframe(input_array, expected):
    df = DataFrame(input_array).to_string(index=False)
    assert (df == expected)

def test_to_string_unicode_columns(float_frame):
    df = DataFrame({'σ': np.arange(10.0)})
    buf = StringIO()
    df.to_string(buf=buf)
    buf.getvalue()
    buf = StringIO()
    df.info(buf=buf)
    buf.getvalue()
    result = float_frame.to_string()
    assert isinstance(result, str)

def test_to_string_utf8_columns():
    n = 'א'.encode()
    with option_context('display.max_rows', 1):
        df = DataFrame([1, 2], columns=[n])
        repr(df)

def test_to_string_unicode_two():
    dm = DataFrame({'c/σ': []})
    buf = StringIO()
    dm.to_string(buf)

def test_to_string_unicode_three():
    dm = DataFrame(['Â'])
    buf = StringIO()
    dm.to_string(buf)

def test_to_string_with_formatters():
    df = DataFrame({'int': [1, 2, 3], 'float': [1.0, 2.0, 3.0], 'object': [(1, 2), True, False]}, columns=['int', 'float', 'object'])
    formatters = [('int', (lambda x: f'0x{x:x}')), ('float', (lambda x: f'[{x: 4.1f}]')), ('object', (lambda x: f'-{x!s}-'))]
    result = df.to_string(formatters=dict(formatters))
    result2 = df.to_string(formatters=list(zip(*formatters))[1])
    assert (result == '  int  float    object\n0 0x1 [ 1.0]  -(1, 2)-\n1 0x2 [ 2.0]    -True-\n2 0x3 [ 3.0]   -False-')
    assert (result == result2)

def test_to_string_with_datetime64_monthformatter():
    months = [datetime(2016, 1, 1), datetime(2016, 2, 2)]
    x = DataFrame({'months': months})

    def format_func(x):
        return x.strftime('%Y-%m')
    result = x.to_string(formatters={'months': format_func})
    expected = dedent('        months\n        0 2016-01\n        1 2016-02')
    assert (result.strip() == expected)

def test_to_string_with_datetime64_hourformatter():
    x = DataFrame({'hod': to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f')})

    def format_func(x):
        return x.strftime('%H:%M')
    result = x.to_string(formatters={'hod': format_func})
    expected = dedent('        hod\n        0 10:10\n        1 12:12')
    assert (result.strip() == expected)

def test_to_string_with_formatters_unicode():
    df = DataFrame({'c/σ': [1, 2, 3]})
    result = df.to_string(formatters={'c/σ': str})
    expected = dedent('          c/σ\n        0   1\n        1   2\n        2   3')
    assert (result == expected)

def test_to_string_complex_number_trims_zeros():
    s = Series([(1.0 + 1j), (1.0 + 1j), (1.05 + 1j)])
    result = s.to_string()
    expected = dedent('        0    1.00+1.00j\n        1    1.00+1.00j\n        2    1.05+1.00j')
    assert (result == expected)

def test_nullable_float_to_string(float_ea_dtype):
    dtype = float_ea_dtype
    s = Series([0.0, 1.0, None], dtype=dtype)
    result = s.to_string()
    expected = dedent('        0     0.0\n        1     1.0\n        2    <NA>')
    assert (result == expected)

def test_nullable_int_to_string(any_nullable_int_dtype):
    dtype = any_nullable_int_dtype
    s = Series([0, 1, None], dtype=dtype)
    result = s.to_string()
    expected = dedent('        0       0\n        1       1\n        2    <NA>')
    assert (result == expected)

@pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
def test_to_string_na_rep_and_float_format(na_rep):
    df = DataFrame([['A', 1.2225], ['A', None]], columns=['Group', 'Data'])
    result = df.to_string(na_rep=na_rep, float_format='{:.2f}'.format)
    expected = dedent(f'''           Group  Data
         0     A  1.22
         1     A   {na_rep}''')
    assert (result == expected)
