
import numpy as np
import pytest
from pandas.compat.numpy import np_version_under1p17
from pandas import DataFrame, Series
import pandas._testing as tm
import pandas.core.common as com

class TestSample():

    @pytest.fixture(params=[Series, DataFrame])
    def obj(self, request):
        klass = request.param
        if (klass is Series):
            arr = np.random.randn(10)
        else:
            arr = np.random.randn(10, 10)
        return klass(arr, dtype=None)

    @pytest.mark.parametrize('test', list(range(10)))
    def test_sample(self, test, obj):
        seed = np.random.randint(0, 100)
        tm.assert_equal(obj.sample(n=4, random_state=seed), obj.sample(n=4, random_state=seed))
        tm.assert_equal(obj.sample(frac=0.7, random_state=seed), obj.sample(frac=0.7, random_state=seed))
        tm.assert_equal(obj.sample(n=4, random_state=np.random.RandomState(test)), obj.sample(n=4, random_state=np.random.RandomState(test)))
        tm.assert_equal(obj.sample(frac=0.7, random_state=np.random.RandomState(test)), obj.sample(frac=0.7, random_state=np.random.RandomState(test)))
        tm.assert_equal(obj.sample(frac=2, replace=True, random_state=np.random.RandomState(test)), obj.sample(frac=2, replace=True, random_state=np.random.RandomState(test)))
        (os1, os2) = ([], [])
        for _ in range(2):
            np.random.seed(test)
            os1.append(obj.sample(n=4))
            os2.append(obj.sample(frac=0.7))
        tm.assert_equal(*os1)
        tm.assert_equal(*os2)

    def test_sample_lengths(self, obj):
        assert len((obj.sample(n=4) == 4))
        assert len((obj.sample(frac=0.34) == 3))
        assert len((obj.sample(frac=0.36) == 4))

    def test_sample_invalid_random_state(self, obj):
        msg = 'random_state must be an integer, array-like, a BitGenerator, a numpy RandomState, or None'
        with pytest.raises(ValueError, match=msg):
            obj.sample(random_state='a_string')

    def test_sample_wont_accept_n_and_frac(self, obj):
        msg = 'Please enter a value for `frac` OR `n`, not both'
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, frac=0.3)

    def test_sample_requires_positive_n_frac(self, obj):
        msg = 'A negative number of rows requested. Please provide positive value.'
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=(- 3))
        with pytest.raises(ValueError, match=msg):
            obj.sample(frac=(- 0.3))

    def test_sample_requires_integer_n(self, obj):
        with pytest.raises(ValueError, match='Only integers accepted as `n` values'):
            obj.sample(n=3.2)

    def test_sample_invalid_weight_lengths(self, obj):
        msg = 'Weights and axis to be sampled must be of same length'
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=[0, 1])
        with pytest.raises(ValueError, match=msg):
            bad_weights = ([0.5] * 11)
            obj.sample(n=3, weights=bad_weights)
        with pytest.raises(ValueError, match='Fewer non-zero entries in p than size'):
            bad_weight_series = Series([0, 0, 0.2])
            obj.sample(n=4, weights=bad_weight_series)

    def test_sample_negative_weights(self, obj):
        bad_weights = ([(- 0.1)] * 10)
        msg = 'weight vector many not include negative values'
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=bad_weights)

    def test_sample_inf_weights(self, obj):
        weights_with_inf = ([0.1] * 10)
        weights_with_inf[0] = np.inf
        msg = 'weight vector may not include `inf` values'
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=weights_with_inf)
        weights_with_ninf = ([0.1] * 10)
        weights_with_ninf[0] = (- np.inf)
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=weights_with_ninf)

    def test_sample_zero_weights(self, obj):
        zero_weights = ([0] * 10)
        with pytest.raises(ValueError, match='Invalid weights: weights sum to zero'):
            obj.sample(n=3, weights=zero_weights)

    def test_sample_missing_weights(self, obj):
        nan_weights = ([np.nan] * 10)
        with pytest.raises(ValueError, match='Invalid weights: weights sum to zero'):
            obj.sample(n=3, weights=nan_weights)

    def test_sample_none_weights(self, obj):
        weights_with_None = ([None] * 10)
        weights_with_None[5] = 0.5
        tm.assert_equal(obj.sample(n=1, axis=0, weights=weights_with_None), obj.iloc[5:6])

    @pytest.mark.parametrize('func_str,arg', [('np.array', [2, 3, 1, 0]), pytest.param('np.random.MT19937', 3, marks=pytest.mark.skipif(np_version_under1p17, reason='NumPy<1.17')), pytest.param('np.random.PCG64', 11, marks=pytest.mark.skipif(np_version_under1p17, reason='NumPy<1.17'))])
    def test_sample_random_state(self, func_str, arg, frame_or_series):
        obj = DataFrame({'col1': range(10, 20), 'col2': range(20, 30)})
        if (frame_or_series is Series):
            obj = obj['col1']
        result = obj.sample(n=3, random_state=eval(func_str)(arg))
        expected = obj.sample(n=3, random_state=com.random_state(eval(func_str)(arg)))
        tm.assert_equal(result, expected)

    def test_sample_upsampling_without_replacement(self, frame_or_series):
        obj = DataFrame({'A': list('abc')})
        if (frame_or_series is Series):
            obj = obj['A']
        msg = 'Replace has to be set to `True` when upsampling the population `frac` > 1.'
        with pytest.raises(ValueError, match=msg):
            obj.sample(frac=2, replace=False)

class TestSampleDataFrame():

    def test_sample(self):
        easy_weight_list = ([0] * 10)
        easy_weight_list[5] = 1
        df = DataFrame({'col1': range(10, 20), 'col2': range(20, 30), 'colString': (['a'] * 10), 'easyweights': easy_weight_list})
        sample1 = df.sample(n=1, weights='easyweights')
        tm.assert_frame_equal(sample1, df.iloc[5:6])
        ser = Series(range(10))
        msg = 'Strings cannot be passed as weights when sampling from a Series.'
        with pytest.raises(ValueError, match=msg):
            ser.sample(n=3, weights='weight_column')
        msg = 'Strings can only be passed to weights when sampling from rows on a DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, weights='weight_column', axis=1)
        with pytest.raises(KeyError, match="'String passed to weights not a valid column'"):
            df.sample(n=3, weights='not_a_real_column_name')
        weights_less_than_1 = ([0] * 10)
        weights_less_than_1[0] = 0.5
        tm.assert_frame_equal(df.sample(n=1, weights=weights_less_than_1), df.iloc[:1])
        df = DataFrame({'col1': range(10), 'col2': (['a'] * 10)})
        second_column_weight = [0, 1]
        tm.assert_frame_equal(df.sample(n=1, axis=1, weights=second_column_weight), df[['col2']])
        tm.assert_frame_equal(df.sample(n=1, axis='columns', weights=second_column_weight), df[['col2']])
        weight = ([0] * 10)
        weight[5] = 0.5
        tm.assert_frame_equal(df.sample(n=1, axis='rows', weights=weight), df.iloc[5:6])
        tm.assert_frame_equal(df.sample(n=1, axis='index', weights=weight), df.iloc[5:6])
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis=2)
        msg = 'No axis named not_a_name for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis='not_a_name')
        ser = Series(range(10))
        with pytest.raises(ValueError, match='No axis named 1 for object type Series'):
            ser.sample(n=1, axis=1)
        msg = 'Weights and axis to be sampled must be of same length'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis=1, weights=([0.5] * 10))

    def test_sample_axis1(self):
        easy_weight_list = ([0] * 3)
        easy_weight_list[2] = 1
        df = DataFrame({'col1': range(10, 20), 'col2': range(20, 30), 'colString': (['a'] * 10)})
        sample1 = df.sample(n=1, axis=1, weights=easy_weight_list)
        tm.assert_frame_equal(sample1, df[['colString']])
        tm.assert_frame_equal(df.sample(n=3, random_state=42), df.sample(n=3, axis=0, random_state=42))

    def test_sample_aligns_weights_with_frame(self):
        df = DataFrame({'col1': [5, 6, 7], 'col2': ['a', 'b', 'c']}, index=[9, 5, 3])
        ser = Series([1, 0, 0], index=[3, 5, 9])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser))
        ser2 = Series([0.001, 0, 10000], index=[3, 5, 10])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser2))
        ser3 = Series([0.01, 0], index=[3, 5])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser3))
        ser4 = Series([1, 0], index=[1, 2])
        with pytest.raises(ValueError, match='Invalid weights: weights sum to zero'):
            df.sample(1, weights=ser4)

    def test_sample_is_copy(self):
        df = DataFrame(np.random.randn(10, 3), columns=['a', 'b', 'c'])
        df2 = df.sample(3)
        with tm.assert_produces_warning(None):
            df2['d'] = 1
