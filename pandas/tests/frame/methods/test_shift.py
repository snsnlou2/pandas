
import numpy as np
import pytest
import pandas as pd
from pandas import CategoricalIndex, DataFrame, Index, Series, date_range, offsets
import pandas._testing as tm

class TestDataFrameShift():

    def test_shift(self, datetime_frame, int_frame):
        shiftedFrame = datetime_frame.shift(5)
        tm.assert_index_equal(shiftedFrame.index, datetime_frame.index)
        shiftedSeries = datetime_frame['A'].shift(5)
        tm.assert_series_equal(shiftedFrame['A'], shiftedSeries)
        shiftedFrame = datetime_frame.shift((- 5))
        tm.assert_index_equal(shiftedFrame.index, datetime_frame.index)
        shiftedSeries = datetime_frame['A'].shift((- 5))
        tm.assert_series_equal(shiftedFrame['A'], shiftedSeries)
        unshifted = datetime_frame.shift(0)
        tm.assert_frame_equal(unshifted, datetime_frame)
        shiftedFrame = datetime_frame.shift(5, freq=offsets.BDay())
        assert (len(shiftedFrame) == len(datetime_frame))
        shiftedFrame2 = datetime_frame.shift(5, freq='B')
        tm.assert_frame_equal(shiftedFrame, shiftedFrame2)
        d = datetime_frame.index[0]
        shifted_d = (d + offsets.BDay(5))
        tm.assert_series_equal(datetime_frame.xs(d), shiftedFrame.xs(shifted_d), check_names=False)
        int_shifted = int_frame.shift(1)
        ps = tm.makePeriodFrame()
        shifted = ps.shift(1)
        unshifted = shifted.shift((- 1))
        tm.assert_index_equal(shifted.index, ps.index)
        tm.assert_index_equal(unshifted.index, ps.index)
        tm.assert_numpy_array_equal(unshifted.iloc[:, 0].dropna().values, ps.iloc[:(- 1), 0].values)
        shifted2 = ps.shift(1, 'B')
        shifted3 = ps.shift(1, offsets.BDay())
        tm.assert_frame_equal(shifted2, shifted3)
        tm.assert_frame_equal(ps, shifted2.shift((- 1), 'B'))
        msg = 'does not match PeriodIndex freq'
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq='D')
        df = DataFrame(np.random.rand(10, 5))
        expected = pd.concat([DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:(- 1)]], ignore_index=True, axis=1)
        result = df.shift(1, axis=1)
        tm.assert_frame_equal(result, expected)
        df = DataFrame(np.random.rand(10, 5))
        expected = pd.concat([DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:(- 1)]], ignore_index=True, axis=1)
        result = df.shift(1, axis='columns')
        tm.assert_frame_equal(result, expected)

    def test_shift_bool(self):
        df = DataFrame({'high': [True, False], 'low': [False, False]})
        rs = df.shift(1)
        xp = DataFrame(np.array([[np.nan, np.nan], [True, False]], dtype=object), columns=['high', 'low'])
        tm.assert_frame_equal(rs, xp)

    def test_shift_categorical(self):
        s1 = Series(['a', 'b', 'c'], dtype='category')
        s2 = Series(['A', 'B', 'C'], dtype='category')
        df = DataFrame({'one': s1, 'two': s2})
        rs = df.shift(1)
        xp = DataFrame({'one': s1.shift(1), 'two': s2.shift(1)})
        tm.assert_frame_equal(rs, xp)

    def test_shift_fill_value(self):
        df = DataFrame([1, 2, 3, 4, 5], index=date_range('1/1/2000', periods=5, freq='H'))
        exp = DataFrame([0, 1, 2, 3, 4], index=date_range('1/1/2000', periods=5, freq='H'))
        result = df.shift(1, fill_value=0)
        tm.assert_frame_equal(result, exp)
        exp = DataFrame([0, 0, 1, 2, 3], index=date_range('1/1/2000', periods=5, freq='H'))
        result = df.shift(2, fill_value=0)
        tm.assert_frame_equal(result, exp)

    def test_shift_empty(self):
        df = DataFrame({'foo': []})
        rs = df.shift((- 1))
        tm.assert_frame_equal(df, rs)

    def test_shift_duplicate_columns(self):
        column_lists = [list(range(5)), ([1] * 5), [1, 1, 2, 2, 1]]
        data = np.random.randn(20, 5)
        shifted = []
        for columns in column_lists:
            df = DataFrame(data.copy(), columns=columns)
            for s in range(5):
                df.iloc[:, s] = df.iloc[:, s].shift((s + 1))
            df.columns = range(5)
            shifted.append(df)
        nulls = shifted[0].isna().sum()
        tm.assert_series_equal(nulls, Series(range(1, 6), dtype='int64'))
        tm.assert_frame_equal(shifted[0], shifted[1])
        tm.assert_frame_equal(shifted[0], shifted[2])

    def test_shift_axis1_multiple_blocks(self):
        df1 = DataFrame(np.random.randint(1000, size=(5, 3)))
        df2 = DataFrame(np.random.randint(1000, size=(5, 2)))
        df3 = pd.concat([df1, df2], axis=1)
        assert (len(df3._mgr.blocks) == 2)
        result = df3.shift(2, axis=1)
        expected = df3.take([(- 1), (- 1), 0, 1, 2], axis=1)
        expected.iloc[:, :2] = np.nan
        expected.columns = df3.columns
        tm.assert_frame_equal(result, expected)
        df3 = pd.concat([df1, df2], axis=1)
        assert (len(df3._mgr.blocks) == 2)
        result = df3.shift((- 2), axis=1)
        expected = df3.take([2, 3, 4, (- 1), (- 1)], axis=1)
        expected.iloc[:, (- 2):] = np.nan
        expected.columns = df3.columns
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:tshift is deprecated:FutureWarning')
    def test_tshift(self, datetime_frame):
        ps = tm.makePeriodFrame()
        shifted = ps.tshift(1)
        unshifted = shifted.tshift((- 1))
        tm.assert_frame_equal(unshifted, ps)
        shifted2 = ps.tshift(freq='B')
        tm.assert_frame_equal(shifted, shifted2)
        shifted3 = ps.tshift(freq=offsets.BDay())
        tm.assert_frame_equal(shifted, shifted3)
        msg = 'Given freq M does not match PeriodIndex freq B'
        with pytest.raises(ValueError, match=msg):
            ps.tshift(freq='M')
        shifted = datetime_frame.tshift(1)
        unshifted = shifted.tshift((- 1))
        tm.assert_frame_equal(datetime_frame, unshifted)
        shifted2 = datetime_frame.tshift(freq=datetime_frame.index.freq)
        tm.assert_frame_equal(shifted, shifted2)
        inferred_ts = DataFrame(datetime_frame.values, Index(np.asarray(datetime_frame.index)), columns=datetime_frame.columns)
        shifted = inferred_ts.tshift(1)
        expected = datetime_frame.tshift(1)
        expected.index = expected.index._with_freq(None)
        tm.assert_frame_equal(shifted, expected)
        unshifted = shifted.tshift((- 1))
        tm.assert_frame_equal(unshifted, inferred_ts)
        no_freq = datetime_frame.iloc[[0, 5, 7], :]
        msg = 'Freq was not set in the index hence cannot be inferred'
        with pytest.raises(ValueError, match=msg):
            no_freq.tshift()

    def test_tshift_deprecated(self, datetime_frame):
        with tm.assert_produces_warning(FutureWarning):
            datetime_frame.tshift()

    def test_period_index_frame_shift_with_freq(self):
        ps = tm.makePeriodFrame()
        shifted = ps.shift(1, freq='infer')
        unshifted = shifted.shift((- 1), freq='infer')
        tm.assert_frame_equal(unshifted, ps)
        shifted2 = ps.shift(freq='B')
        tm.assert_frame_equal(shifted, shifted2)
        shifted3 = ps.shift(freq=offsets.BDay())
        tm.assert_frame_equal(shifted, shifted3)

    def test_datetime_frame_shift_with_freq(self, datetime_frame):
        shifted = datetime_frame.shift(1, freq='infer')
        unshifted = shifted.shift((- 1), freq='infer')
        tm.assert_frame_equal(datetime_frame, unshifted)
        shifted2 = datetime_frame.shift(freq=datetime_frame.index.freq)
        tm.assert_frame_equal(shifted, shifted2)
        inferred_ts = DataFrame(datetime_frame.values, Index(np.asarray(datetime_frame.index)), columns=datetime_frame.columns)
        shifted = inferred_ts.shift(1, freq='infer')
        expected = datetime_frame.shift(1, freq='infer')
        expected.index = expected.index._with_freq(None)
        tm.assert_frame_equal(shifted, expected)
        unshifted = shifted.shift((- 1), freq='infer')
        tm.assert_frame_equal(unshifted, inferred_ts)

    def test_period_index_frame_shift_with_freq_error(self):
        ps = tm.makePeriodFrame()
        msg = 'Given freq M does not match PeriodIndex freq B'
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq='M')

    def test_datetime_frame_shift_with_freq_error(self, datetime_frame):
        no_freq = datetime_frame.iloc[[0, 5, 7], :]
        msg = 'Freq was not set in the index hence cannot be inferred'
        with pytest.raises(ValueError, match=msg):
            no_freq.shift(freq='infer')

    def test_shift_dt64values_int_fill_deprecated(self):
        ser = Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')])
        df = ser.to_frame()
        with tm.assert_produces_warning(FutureWarning):
            result = df.shift(1, fill_value=0)
        expected = Series([pd.Timestamp(0), ser[0]]).to_frame()
        tm.assert_frame_equal(result, expected)
        df2 = DataFrame({'A': ser, 'B': ser})
        df2._consolidate_inplace()
        with tm.assert_produces_warning(FutureWarning):
            result = df2.shift(1, axis=1, fill_value=0)
        expected = DataFrame({'A': [pd.Timestamp(0), pd.Timestamp(0)], 'B': df2['A']})
        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_categorical_columns(self):
        ci = CategoricalIndex(['a', 'b', 'c'])
        df = DataFrame({'a': [1, 3], 'b': [2, 4], 'c': [5, 6]}, index=ci[:(- 1)], columns=ci)
        result = df.shift(axis=1)
        expected = DataFrame({'a': [np.nan, np.nan], 'b': [1, 3], 'c': [2, 4]}, index=ci[:(- 1)], columns=ci)
        tm.assert_frame_equal(result, expected)
        result = df.shift(2, axis=1)
        expected = DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan], 'c': [1, 3]}, index=ci[:(- 1)], columns=ci)
        tm.assert_frame_equal(result, expected)
