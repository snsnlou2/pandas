
from datetime import datetime, timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, DataFrame, Index, MultiIndex, Series, date_range, option_context, period_range, timedelta_range
import pandas._testing as tm

class TestSeriesRepr():

    def test_multilevel_name_print(self):
        index = MultiIndex(levels=[['foo', 'bar', 'baz', 'qux'], ['one', 'two', 'three']], codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]], names=['first', 'second'])
        s = Series(range(len(index)), index=index, name='sth')
        expected = ['first  second', 'foo    one       0', '       two       1', '       three     2', 'bar    one       3', '       two       4', 'baz    two       5', '       three     6', 'qux    one       7', '       two       8', '       three     9', 'Name: sth, dtype: int64']
        expected = '\n'.join(expected)
        assert (repr(s) == expected)

    def test_name_printing(self):
        s = Series([0, 1, 2])
        s.name = 'test'
        assert ('Name: test' in repr(s))
        s.name = None
        assert ('Name:' not in repr(s))
        s = Series(range(1000))
        s.name = 'test'
        assert ('Name: test' in repr(s))
        s.name = None
        assert ('Name:' not in repr(s))
        s = Series(index=date_range('20010101', '20020101'), name='test', dtype=object)
        assert ('Name: test' in repr(s))

    def test_repr(self, datetime_series, string_series, object_series):
        str(datetime_series)
        str(string_series)
        str(string_series.astype(int))
        str(object_series)
        str(Series(np.random.randn(1000), index=np.arange(1000)))
        str(Series(np.random.randn(1000), index=np.arange(1000, 0, step=(- 1))))
        str(Series(dtype=object))
        string_series[5:7] = np.NaN
        str(string_series)
        ots = datetime_series.astype('O')
        ots[::2] = None
        repr(ots)
        for name in ['', 1, 1.2, 'foo', 'αβγ', 'loooooooooooooooooooooooooooooooooooooooooooooooooooong', ('foo', 'bar', 'baz'), (1, 2), ('foo', 1, 2.3), ('α', 'β', 'γ'), ('α', 'bar')]:
            string_series.name = name
            repr(string_series)
        biggie = Series(np.random.randn(1000), index=np.arange(1000), name=('foo', 'bar', 'baz'))
        repr(biggie)
        ser = Series(np.random.randn(100), name=0)
        rep_str = repr(ser)
        assert ('Name: 0' in rep_str)
        ser = Series(np.random.randn(1001), name=0)
        rep_str = repr(ser)
        assert ('Name: 0' in rep_str)
        ser = Series(['a\n\r\tb'], name='a\n\r\td', index=['a\n\r\tf'])
        assert ('\t' not in repr(ser))
        assert ('\r' not in repr(ser))
        assert ('a\n' not in repr(ser))
        s = Series([], dtype=np.int64, name='foo')
        assert (repr(s) == 'Series([], Name: foo, dtype: int64)')
        s = Series([], dtype=np.int64, name=None)
        assert (repr(s) == 'Series([], dtype: int64)')

    def test_tidy_repr(self):
        a = Series((['א'] * 1000))
        a.name = 'title1'
        repr(a)

    def test_repr_bool_fails(self, capsys):
        s = Series([DataFrame(np.random.randn(2, 2)) for i in range(5)])
        repr(s)
        captured = capsys.readouterr()
        assert (captured.err == '')

    def test_repr_name_iterable_indexable(self):
        s = Series([1, 2, 3], name=np.int64(3))
        repr(s)
        s.name = (('א',) * 2)
        repr(s)

    def test_repr_should_return_str(self):
        data = [8, 5, 3, 5]
        index1 = ['σ', 'τ', 'υ', 'φ']
        df = Series(data, index=index1)
        assert type((df.__repr__() == str))

    def test_repr_max_rows(self):
        with pd.option_context('max_rows', None):
            str(Series(range(1001)))

    def test_unicode_string_with_unicode(self):
        df = Series(['א'], name='ב')
        str(df)

    def test_str_to_bytes_raises(self):
        df = Series(['abc'], name='abc')
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_timeseries_repr_object_dtype(self):
        index = Index([(datetime(2000, 1, 1) + timedelta(i)) for i in range(1000)], dtype=object)
        ts = Series(np.random.randn(len(index)), index)
        repr(ts)
        ts = tm.makeTimeSeries(1000)
        assert repr(ts).splitlines()[(- 1)].startswith('Freq:')
        ts2 = ts.iloc[np.random.randint(0, (len(ts) - 1), 400)]
        repr(ts2).splitlines()[(- 1)]

    def test_latex_repr(self):
        result = '\\begin{tabular}{ll}\n\\toprule\n{} &         0 \\\\\n\\midrule\n0 &  $\\alpha$ \\\\\n1 &         b \\\\\n2 &         c \\\\\n\\bottomrule\n\\end{tabular}\n'
        with option_context('display.latex.escape', False, 'display.latex.repr', True):
            s = Series(['$\\alpha$', 'b', 'c'])
            assert (result == s._repr_latex_())
        assert (s._repr_latex_() is None)

    def test_index_repr_in_frame_with_nan(self):
        i = Index([1, np.nan])
        s = Series([1, 2], index=i)
        exp = '1.0    1\nNaN    2\ndtype: int64'
        assert (repr(s) == exp)

    def test_format_pre_1900_dates(self):
        rng = date_range('1/1/1850', '1/1/1950', freq='A-DEC')
        rng.format()
        ts = Series(1, index=rng)
        repr(ts)

    def test_series_repr_nat(self):
        series = Series([0, 1000, 2000, pd.NaT.value], dtype='M8[ns]')
        result = repr(series)
        expected = '0   1970-01-01 00:00:00.000000\n1   1970-01-01 00:00:00.000001\n2   1970-01-01 00:00:00.000002\n3                          NaT\ndtype: datetime64[ns]'
        assert (result == expected)

class TestCategoricalRepr():

    def test_categorical_repr_unicode(self):

        class County():
            name = 'San Sebastián'
            state = 'PR'

            def __repr__(self) -> str:
                return ((self.name + ', ') + self.state)
        cat = Categorical([County() for _ in range(61)])
        idx = Index(cat)
        ser = idx.to_series()
        repr(ser)
        str(ser)

    def test_categorical_repr(self):
        a = Series(Categorical([1, 2, 3, 4]))
        exp = ('0    1\n1    2\n2    3\n3    4\n' + 'dtype: category\nCategories (4, int64): [1, 2, 3, 4]')
        assert (exp == a.__str__())
        a = Series(Categorical((['a', 'b'] * 25)))
        exp = ((('0     a\n1     b\n' + '     ..\n') + '48    a\n49    b\n') + "Length: 50, dtype: category\nCategories (2, object): ['a', 'b']")
        with option_context('display.max_rows', 5):
            assert (exp == repr(a))
        levs = list('abcdefghijklmnopqrstuvwxyz')
        a = Series(Categorical(['a', 'b'], categories=levs, ordered=True))
        exp = ('0    a\n1    b\n' + "dtype: category\nCategories (26, object): ['a' < 'b' < 'c' < 'd' ... 'w' < 'x' < 'y' < 'z']")
        assert (exp == a.__str__())

    def test_categorical_series_repr(self):
        s = Series(Categorical([1, 2, 3]))
        exp = '0    1\n1    2\n2    3\ndtype: category\nCategories (3, int64): [1, 2, 3]'
        assert (repr(s) == exp)
        s = Series(Categorical(np.arange(10)))
        exp = '0    0\n1    1\n2    2\n3    3\n4    4\n5    5\n6    6\n7    7\n8    8\n9    9\ndtype: category\nCategories (10, int64): [0, 1, 2, 3, ..., 6, 7, 8, 9]'
        assert (repr(s) == exp)

    def test_categorical_series_repr_ordered(self):
        s = Series(Categorical([1, 2, 3], ordered=True))
        exp = '0    1\n1    2\n2    3\ndtype: category\nCategories (3, int64): [1 < 2 < 3]'
        assert (repr(s) == exp)
        s = Series(Categorical(np.arange(10), ordered=True))
        exp = '0    0\n1    1\n2    2\n3    3\n4    4\n5    5\n6    6\n7    7\n8    8\n9    9\ndtype: category\nCategories (10, int64): [0 < 1 < 2 < 3 ... 6 < 7 < 8 < 9]'
        assert (repr(s) == exp)

    def test_categorical_series_repr_datetime(self):
        idx = date_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx))
        exp = '0   2011-01-01 09:00:00\n1   2011-01-01 10:00:00\n2   2011-01-01 11:00:00\n3   2011-01-01 12:00:00\n4   2011-01-01 13:00:00\ndtype: category\nCategories (5, datetime64[ns]): [2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00,\n                                 2011-01-01 12:00:00, 2011-01-01 13:00:00]'
        assert (repr(s) == exp)
        idx = date_range('2011-01-01 09:00', freq='H', periods=5, tz='US/Eastern')
        s = Series(Categorical(idx))
        exp = '0   2011-01-01 09:00:00-05:00\n1   2011-01-01 10:00:00-05:00\n2   2011-01-01 11:00:00-05:00\n3   2011-01-01 12:00:00-05:00\n4   2011-01-01 13:00:00-05:00\ndtype: category\nCategories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n                                             2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n                                             2011-01-01 13:00:00-05:00]'
        assert (repr(s) == exp)

    def test_categorical_series_repr_datetime_ordered(self):
        idx = date_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0   2011-01-01 09:00:00\n1   2011-01-01 10:00:00\n2   2011-01-01 11:00:00\n3   2011-01-01 12:00:00\n4   2011-01-01 13:00:00\ndtype: category\nCategories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <\n                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]'
        assert (repr(s) == exp)
        idx = date_range('2011-01-01 09:00', freq='H', periods=5, tz='US/Eastern')
        s = Series(Categorical(idx, ordered=True))
        exp = '0   2011-01-01 09:00:00-05:00\n1   2011-01-01 10:00:00-05:00\n2   2011-01-01 11:00:00-05:00\n3   2011-01-01 12:00:00-05:00\n4   2011-01-01 13:00:00-05:00\ndtype: category\nCategories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <\n                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <\n                                             2011-01-01 13:00:00-05:00]'
        assert (repr(s) == exp)

    def test_categorical_series_repr_period(self):
        idx = period_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx))
        exp = '0    2011-01-01 09:00\n1    2011-01-01 10:00\n2    2011-01-01 11:00\n3    2011-01-01 12:00\n4    2011-01-01 13:00\ndtype: category\nCategories (5, period[H]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,\n                            2011-01-01 13:00]'
        assert (repr(s) == exp)
        idx = period_range('2011-01', freq='M', periods=5)
        s = Series(Categorical(idx))
        exp = '0    2011-01\n1    2011-02\n2    2011-03\n3    2011-04\n4    2011-05\ndtype: category\nCategories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]'
        assert (repr(s) == exp)

    def test_categorical_series_repr_period_ordered(self):
        idx = period_range('2011-01-01 09:00', freq='H', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0    2011-01-01 09:00\n1    2011-01-01 10:00\n2    2011-01-01 11:00\n3    2011-01-01 12:00\n4    2011-01-01 13:00\ndtype: category\nCategories (5, period[H]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <\n                            2011-01-01 13:00]'
        assert (repr(s) == exp)
        idx = period_range('2011-01', freq='M', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0    2011-01\n1    2011-02\n2    2011-03\n3    2011-04\n4    2011-05\ndtype: category\nCategories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]'
        assert (repr(s) == exp)

    def test_categorical_series_repr_timedelta(self):
        idx = timedelta_range('1 days', periods=5)
        s = Series(Categorical(idx))
        exp = '0   1 days\n1   2 days\n2   3 days\n3   4 days\n4   5 days\ndtype: category\nCategories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]'
        assert (repr(s) == exp)
        idx = timedelta_range('1 hours', periods=10)
        s = Series(Categorical(idx))
        exp = '0   0 days 01:00:00\n1   1 days 01:00:00\n2   2 days 01:00:00\n3   3 days 01:00:00\n4   4 days 01:00:00\n5   5 days 01:00:00\n6   6 days 01:00:00\n7   7 days 01:00:00\n8   8 days 01:00:00\n9   9 days 01:00:00\ndtype: category\nCategories (10, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,\n                                   3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00,\n                                   8 days 01:00:00, 9 days 01:00:00]'
        assert (repr(s) == exp)

    def test_categorical_series_repr_timedelta_ordered(self):
        idx = timedelta_range('1 days', periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = '0   1 days\n1   2 days\n2   3 days\n3   4 days\n4   5 days\ndtype: category\nCategories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]'
        assert (repr(s) == exp)
        idx = timedelta_range('1 hours', periods=10)
        s = Series(Categorical(idx, ordered=True))
        exp = '0   0 days 01:00:00\n1   1 days 01:00:00\n2   2 days 01:00:00\n3   3 days 01:00:00\n4   4 days 01:00:00\n5   5 days 01:00:00\n6   6 days 01:00:00\n7   7 days 01:00:00\n8   8 days 01:00:00\n9   9 days 01:00:00\ndtype: category\nCategories (10, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <\n                                   3 days 01:00:00 ... 6 days 01:00:00 < 7 days 01:00:00 <\n                                   8 days 01:00:00 < 9 days 01:00:00]'
        assert (repr(s) == exp)
