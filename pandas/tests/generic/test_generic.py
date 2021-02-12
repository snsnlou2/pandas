
from copy import copy, deepcopy
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import DataFrame, Series
import pandas._testing as tm

class Generic():

    @property
    def _ndim(self):
        return self._typ._AXIS_LEN

    def _axes(self):
        ' return the axes for my object typ '
        return self._typ._AXIS_ORDERS

    def _construct(self, shape, value=None, dtype=None, **kwargs):
        '\n        construct an object for the given shape\n        if value is specified use that if its a scalar\n        if value is an array, repeat it as needed\n        '
        if isinstance(shape, int):
            shape = tuple(([shape] * self._ndim))
        if (value is not None):
            if is_scalar(value):
                if (value == 'empty'):
                    arr = None
                    dtype = np.float64
                    kwargs.pop(self._typ._info_axis_name, None)
                else:
                    arr = np.empty(shape, dtype=dtype)
                    arr.fill(value)
            else:
                fshape = np.prod(shape)
                arr = value.ravel()
                new_shape = (fshape / arr.shape[0])
                if ((fshape % arr.shape[0]) != 0):
                    raise Exception('invalid value passed in _construct')
                arr = np.repeat(arr, new_shape).reshape(shape)
        else:
            arr = np.random.randn(*shape)
        return self._typ(arr, dtype=dtype, **kwargs)

    def _compare(self, result, expected):
        self._comparator(result, expected)

    def test_rename(self):
        idx = list('ABCD')
        args = [str.lower, {x: x.lower() for x in idx}, Series({x: x.lower() for x in idx})]
        for axis in self._axes():
            kwargs = {axis: idx}
            obj = self._construct(4, **kwargs)
            for arg in args:
                result = obj.rename(**{axis: arg})
                expected = obj.copy()
                setattr(expected, axis, list('abcd'))
                self._compare(result, expected)

    def test_get_numeric_data(self):
        n = 4
        kwargs = {self._typ._get_axis_name(i): list(range(n)) for i in range(self._ndim)}
        o = self._construct(n, **kwargs)
        result = o._get_numeric_data()
        self._compare(result, o)
        result = o._get_bool_data()
        expected = self._construct(n, value='empty', **kwargs)
        self._compare(result, expected)
        arr = np.array([True, True, False, True])
        o = self._construct(n, value=arr, **kwargs)
        result = o._get_numeric_data()
        self._compare(result, o)

    def test_nonzero(self):
        obj = self._construct(shape=4)
        msg = f'The truth value of a {self._typ.__name__} is ambiguous'
        with pytest.raises(ValueError, match=msg):
            bool((obj == 0))
        with pytest.raises(ValueError, match=msg):
            bool((obj == 1))
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj = self._construct(shape=4, value=1)
        with pytest.raises(ValueError, match=msg):
            bool((obj == 0))
        with pytest.raises(ValueError, match=msg):
            bool((obj == 1))
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj = self._construct(shape=4, value=np.nan)
        with pytest.raises(ValueError, match=msg):
            bool((obj == 0))
        with pytest.raises(ValueError, match=msg):
            bool((obj == 1))
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj = self._construct(shape=0)
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj1 = self._construct(shape=4, value=1)
        obj2 = self._construct(shape=4, value=1)
        with pytest.raises(ValueError, match=msg):
            if obj1:
                pass
        with pytest.raises(ValueError, match=msg):
            (obj1 and obj2)
        with pytest.raises(ValueError, match=msg):
            (obj1 or obj2)
        with pytest.raises(ValueError, match=msg):
            (not obj1)

    def test_downcast(self):
        o = self._construct(shape=4, value=9, dtype=np.int64)
        result = o.copy()
        result._mgr = o._mgr.downcast()
        self._compare(result, o)
        o = self._construct(shape=4, value=9.5)
        result = o.copy()
        result._mgr = o._mgr.downcast()
        self._compare(result, o)

    def test_constructor_compound_dtypes(self):

        def f(dtype):
            return self._construct(shape=3, value=1, dtype=dtype)
        msg = f'compound dtypes are not implemented in the {self._typ.__name__} constructor'
        with pytest.raises(NotImplementedError, match=msg):
            f([('A', 'datetime64[h]'), ('B', 'str'), ('C', 'int32')])
        f('int64')
        f('float64')
        f('M8[ns]')

    def check_metadata(self, x, y=None):
        for m in x._metadata:
            v = getattr(x, m, None)
            if (y is None):
                assert (v is None)
            else:
                assert (v == getattr(y, m, None))

    def test_metadata_propagation(self):
        o = self._construct(shape=3)
        o.name = 'foo'
        o2 = self._construct(shape=3)
        o2.name = 'bar'
        for op in ['__add__', '__sub__', '__truediv__', '__mul__']:
            result = getattr(o, op)(1)
            self.check_metadata(o, result)
        for op in ['__add__', '__sub__', '__truediv__', '__mul__']:
            result = getattr(o, op)(o)
            self.check_metadata(o, result)
        for op in ['__eq__', '__le__', '__ge__']:
            v1 = getattr(o, op)(o)
            self.check_metadata(o, v1)
            self.check_metadata(o, (v1 & v1))
            self.check_metadata(o, (v1 | v1))
        result = o.combine_first(o2)
        self.check_metadata(o, result)
        result = (o + o2)
        self.check_metadata(result)
        for op in ['__eq__', '__le__', '__ge__']:
            v1 = getattr(o, op)(o)
            v2 = getattr(o, op)(o2)
            self.check_metadata(v2)
            self.check_metadata((v1 & v2))
            self.check_metadata((v1 | v2))

    def test_size_compat(self):
        o = self._construct(shape=10)
        assert (o.size == np.prod(o.shape))
        assert (o.size == (10 ** len(o.axes)))

    def test_split_compat(self):
        o = self._construct(shape=10)
        assert (len(np.array_split(o, 5)) == 5)
        assert (len(np.array_split(o, 2)) == 2)

    def test_stat_unexpected_keyword(self):
        obj = self._construct(5)
        starwars = 'Star Wars'
        errmsg = 'unexpected keyword'
        with pytest.raises(TypeError, match=errmsg):
            obj.max(epic=starwars)
        with pytest.raises(TypeError, match=errmsg):
            obj.var(epic=starwars)
        with pytest.raises(TypeError, match=errmsg):
            obj.sum(epic=starwars)
        with pytest.raises(TypeError, match=errmsg):
            obj.any(epic=starwars)

    @pytest.mark.parametrize('func', ['sum', 'cumsum', 'any', 'var'])
    def test_api_compat(self, func):
        obj = self._construct(5)
        f = getattr(obj, func)
        assert (f.__name__ == func)
        assert f.__qualname__.endswith(func)

    def test_stat_non_defaults_args(self):
        obj = self._construct(5)
        out = np.array([0])
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            obj.max(out=out)
        with pytest.raises(ValueError, match=errmsg):
            obj.var(out=out)
        with pytest.raises(ValueError, match=errmsg):
            obj.sum(out=out)
        with pytest.raises(ValueError, match=errmsg):
            obj.any(out=out)

    def test_truncate_out_of_bounds(self):
        shape = ([int(2000.0)] + ([1] * (self._ndim - 1)))
        small = self._construct(shape, dtype='int8', value=1)
        self._compare(small.truncate(), small)
        self._compare(small.truncate(before=0, after=3000.0), small)
        self._compare(small.truncate(before=(- 1), after=2000.0), small)
        shape = ([int(2000000.0)] + ([1] * (self._ndim - 1)))
        big = self._construct(shape, dtype='int8', value=1)
        self._compare(big.truncate(), big)
        self._compare(big.truncate(before=0, after=3000000.0), big)
        self._compare(big.truncate(before=(- 1), after=2000000.0), big)

    @pytest.mark.parametrize('func', [copy, deepcopy, (lambda x: x.copy(deep=False)), (lambda x: x.copy(deep=True))])
    @pytest.mark.parametrize('shape', [0, 1, 2])
    def test_copy_and_deepcopy(self, shape, func):
        obj = self._construct(shape)
        obj_copy = func(obj)
        assert (obj_copy is not obj)
        self._compare(obj_copy, obj)

class TestNDFrame():

    def test_squeeze(self):
        for s in [tm.makeFloatSeries(), tm.makeStringSeries(), tm.makeObjectSeries()]:
            tm.assert_series_equal(s.squeeze(), s)
        for df in [tm.makeTimeDataFrame()]:
            tm.assert_frame_equal(df.squeeze(), df)
        df = tm.makeTimeDataFrame().reindex(columns=['A'])
        tm.assert_series_equal(df.squeeze(), df['A'])
        empty_series = Series([], name='five', dtype=np.float64)
        empty_frame = DataFrame([empty_series])
        tm.assert_series_equal(empty_series, empty_series.squeeze())
        tm.assert_series_equal(empty_series, empty_frame.squeeze())
        df = tm.makeTimeDataFrame(nper=1).iloc[:, :1]
        assert (df.shape == (1, 1))
        tm.assert_series_equal(df.squeeze(axis=0), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis='index'), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis=1), df.iloc[:, 0])
        tm.assert_series_equal(df.squeeze(axis='columns'), df.iloc[:, 0])
        assert (df.squeeze() == df.iloc[(0, 0)])
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis=2)
        msg = 'No axis named x for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis='x')
        df = tm.makeTimeDataFrame(3)
        tm.assert_frame_equal(df.squeeze(axis=0), df)

    def test_numpy_squeeze(self):
        s = tm.makeFloatSeries()
        tm.assert_series_equal(np.squeeze(s), s)
        df = tm.makeTimeDataFrame().reindex(columns=['A'])
        tm.assert_series_equal(np.squeeze(df), df['A'])

    def test_transpose(self):
        for s in [tm.makeFloatSeries(), tm.makeStringSeries(), tm.makeObjectSeries()]:
            tm.assert_series_equal(s.transpose(), s)
        for df in [tm.makeTimeDataFrame()]:
            tm.assert_frame_equal(df.transpose().transpose(), df)

    def test_numpy_transpose(self, frame_or_series):
        obj = tm.makeTimeDataFrame()
        if (frame_or_series is Series):
            obj = obj['A']
        if (frame_or_series is Series):
            tm.assert_series_equal(np.transpose(obj), obj)
        tm.assert_equal(np.transpose(np.transpose(obj)), obj)
        msg = "the 'axes' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.transpose(obj, axes=1)

    def test_take(self):
        indices = [1, 5, (- 2), 6, 3, (- 1)]
        for s in [tm.makeFloatSeries(), tm.makeStringSeries(), tm.makeObjectSeries()]:
            out = s.take(indices)
            expected = Series(data=s.values.take(indices), index=s.index.take(indices), dtype=s.dtype)
            tm.assert_series_equal(out, expected)
        for df in [tm.makeTimeDataFrame()]:
            out = df.take(indices)
            expected = DataFrame(data=df.values.take(indices, axis=0), index=df.index.take(indices), columns=df.columns)
            tm.assert_frame_equal(out, expected)

    def test_take_invalid_kwargs(self, frame_or_series):
        indices = [(- 3), 2, 0, 1]
        obj = tm.makeTimeDataFrame()
        if (frame_or_series is Series):
            obj = obj['A']
        msg = "take\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            obj.take(indices, foo=2)
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, out=indices)
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, mode='clip')

    @pytest.mark.parametrize('is_copy', [True, False])
    def test_depr_take_kwarg_is_copy(self, is_copy, frame_or_series):
        obj = DataFrame({'A': [1, 2, 3]})
        if (frame_or_series is Series):
            obj = obj['A']
        msg = "is_copy is deprecated and will be removed in a future version. 'take' always returns a copy, so there is no need to specify this."
        with tm.assert_produces_warning(FutureWarning) as w:
            obj.take([0, 1], is_copy=is_copy)
        assert (w[0].message.args[0] == msg)

    def test_axis_classmethods(self, frame_or_series):
        box = frame_or_series
        obj = box(dtype=object)
        values = box._AXIS_TO_AXIS_NUMBER.keys()
        for v in values:
            assert (obj._get_axis_number(v) == box._get_axis_number(v))
            assert (obj._get_axis_name(v) == box._get_axis_name(v))
            assert (obj._get_block_manager_axis(v) == box._get_block_manager_axis(v))

    def test_axis_names_deprecated(self, frame_or_series):
        box = frame_or_series
        obj = box(dtype=object)
        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            obj._AXIS_NAMES

    def test_axis_numbers_deprecated(self, frame_or_series):
        box = frame_or_series
        obj = box(dtype=object)
        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            obj._AXIS_NUMBERS

    def test_flags_identity(self, frame_or_series):
        obj = Series([1, 2])
        if (frame_or_series is DataFrame):
            obj = obj.to_frame()
        assert (obj.flags is obj.flags)
        obj2 = obj.copy()
        assert (obj2.flags is not obj.flags)

    def test_slice_shift_deprecated(self, frame_or_series):
        obj = DataFrame({'A': [1, 2, 3, 4]})
        if (frame_or_series is DataFrame):
            obj = obj['A']
        with tm.assert_produces_warning(FutureWarning):
            obj.slice_shift()
