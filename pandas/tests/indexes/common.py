
import gc
from typing import Type
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import is_datetime64tz_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import CategoricalIndex, DatetimeIndex, Index, Int64Index, IntervalIndex, MultiIndex, PeriodIndex, RangeIndex, Series, TimedeltaIndex, UInt64Index, isna
import pandas._testing as tm
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin

class Base():
    ' base class for index sub-class tests '
    _compat_props = ['shape', 'ndim', 'size', 'nbytes']

    def create_index(self):
        raise NotImplementedError('Method not implemented')

    def test_pickle_compat_construction(self):
        msg = "Index\\(\\.\\.\\.\\) must be called with a collection of some kind, None was passed|__new__\\(\\) missing 1 required positional argument: 'data'|__new__\\(\\) takes at least 2 arguments \\(1 given\\)"
        with pytest.raises(TypeError, match=msg):
            self._holder()

    @pytest.mark.parametrize('name', [None, 'new_name'])
    def test_to_frame(self, name):
        idx = self.create_index()
        if name:
            idx_name = name
        else:
            idx_name = (idx.name or 0)
        df = idx.to_frame(name=idx_name)
        assert (df.index is idx)
        assert (len(df.columns) == 1)
        assert (df.columns[0] == idx_name)
        assert (df[idx_name].values is not idx.values)
        df = idx.to_frame(index=False, name=idx_name)
        assert (df.index is not idx)

    def test_shift(self):
        idx = self.create_index()
        msg = f'This method is only implemented for DatetimeIndex, PeriodIndex and TimedeltaIndex; Got type {type(idx).__name__}'
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1)
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1, 2)

    def test_constructor_name_unhashable(self):
        idx = self.create_index()
        with pytest.raises(TypeError, match='Index.name must be a hashable type'):
            type(idx)(idx, name=[])

    def test_create_index_existing_name(self):
        expected = self.create_index()
        if (not isinstance(expected, MultiIndex)):
            expected.name = 'foo'
            result = Index(expected)
            tm.assert_index_equal(result, expected)
            result = Index(expected, name='bar')
            expected.name = 'bar'
            tm.assert_index_equal(result, expected)
        else:
            expected.names = ['foo', 'bar']
            result = Index(expected)
            tm.assert_index_equal(result, Index(Index([('foo', 'one'), ('foo', 'two'), ('bar', 'one'), ('baz', 'two'), ('qux', 'one'), ('qux', 'two')], dtype='object'), names=['foo', 'bar']))
            result = Index(expected, names=['A', 'B'])
            tm.assert_index_equal(result, Index(Index([('foo', 'one'), ('foo', 'two'), ('bar', 'one'), ('baz', 'two'), ('qux', 'one'), ('qux', 'two')], dtype='object'), names=['A', 'B']))

    def test_numeric_compat(self):
        idx = self.create_index()
        assert (not isinstance(idx, MultiIndex))
        if (type(idx) is Index):
            return
        typ = type(idx._data).__name__
        lmsg = '|'.join([f"unsupported operand type\(s\) for \*: '{typ}' and 'int'", f'cannot perform (__mul__|__truediv__|__floordiv__) with this index type: {typ}'])
        with pytest.raises(TypeError, match=lmsg):
            (idx * 1)
        rmsg = '|'.join([f"unsupported operand type\(s\) for \*: 'int' and '{typ}'", f'cannot perform (__rmul__|__rtruediv__|__rfloordiv__) with this index type: {typ}'])
        with pytest.raises(TypeError, match=rmsg):
            (1 * idx)
        div_err = lmsg.replace('*', '/')
        with pytest.raises(TypeError, match=div_err):
            (idx / 1)
        div_err = rmsg.replace('*', '/')
        with pytest.raises(TypeError, match=div_err):
            (1 / idx)
        floordiv_err = lmsg.replace('*', '//')
        with pytest.raises(TypeError, match=floordiv_err):
            (idx // 1)
        floordiv_err = rmsg.replace('*', '//')
        with pytest.raises(TypeError, match=floordiv_err):
            (1 // idx)

    def test_logical_compat(self):
        idx = self.create_index()
        with pytest.raises(TypeError, match='cannot perform all'):
            idx.all()
        with pytest.raises(TypeError, match='cannot perform any'):
            idx.any()

    def test_reindex_base(self):
        idx = self.create_index()
        expected = np.arange(idx.size, dtype=np.intp)
        actual = idx.get_indexer(idx)
        tm.assert_numpy_array_equal(expected, actual)
        with pytest.raises(ValueError, match='Invalid fill method'):
            idx.get_indexer(idx, method='invalid')

    def test_get_indexer_consistency(self, index):
        if isinstance(index, IntervalIndex):
            return
        if index.is_unique:
            indexer = index.get_indexer(index[0:2])
            assert isinstance(indexer, np.ndarray)
            assert (indexer.dtype == np.intp)
        else:
            e = 'Reindexing only valid with uniquely valued Index objects'
            with pytest.raises(InvalidIndexError, match=e):
                index.get_indexer(index[0:2])
        (indexer, _) = index.get_indexer_non_unique(index[0:2])
        assert isinstance(indexer, np.ndarray)
        assert (indexer.dtype == np.intp)

    def test_ndarray_compat_properties(self):
        idx = self.create_index()
        assert idx.T.equals(idx)
        assert idx.transpose().equals(idx)
        values = idx.values
        for prop in self._compat_props:
            assert (getattr(idx, prop) == getattr(values, prop))
        idx.nbytes
        idx.values.nbytes

    def test_repr_roundtrip(self):
        idx = self.create_index()
        tm.assert_index_equal(eval(repr(idx)), idx)

    def test_repr_max_seq_item_setting(self):
        idx = self.create_index()
        idx = idx.repeat(50)
        with pd.option_context('display.max_seq_items', None):
            repr(idx)
            assert ('...' not in str(idx))

    def test_copy_name(self, index):
        if isinstance(index, MultiIndex):
            return
        first = type(index)(index, copy=True, name='mario')
        second = type(first)(first, copy=False)
        assert (first is not second)
        assert index.equals(first)
        assert (first.name == 'mario')
        assert (second.name == 'mario')
        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:(- 1)])
        if (not isinstance(index, CategoricalIndex)):
            s3 = (s1 * s2)
            assert (s3.index.name == 'mario')

    def test_copy_name2(self, index):
        if isinstance(index, MultiIndex):
            return
        assert (index.copy(name='mario').name == 'mario')
        with pytest.raises(ValueError, match='Length of new names must be 1, got 2'):
            index.copy(name=['mario', 'luigi'])
        msg = f'{type(index).__name__}.name must be a hashable type'
        with pytest.raises(TypeError, match=msg):
            index.copy(name=[['mario']])

    def test_copy_dtype_deprecated(self, index):
        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            index.copy(dtype=object)

    def test_ensure_copied_data(self, index):
        init_kwargs = {}
        if isinstance(index, PeriodIndex):
            init_kwargs['freq'] = index.freq
        elif isinstance(index, (RangeIndex, MultiIndex, CategoricalIndex)):
            return
        index_type = type(index)
        result = index_type(index.values, copy=True, **init_kwargs)
        if is_datetime64tz_dtype(index.dtype):
            result = result.tz_localize('UTC').tz_convert(index.tz)
        if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
            index = index._with_freq(None)
        tm.assert_index_equal(index, result)
        if isinstance(index, PeriodIndex):
            result = index_type(ordinal=index.asi8, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(index.asi8, result.asi8, check_same='same')
        elif isinstance(index, IntervalIndex):
            pass
        else:
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(index.values, result.values, check_same='same')

    def test_memory_usage(self, index):
        index._engine.clear_mapping()
        result = index.memory_usage()
        if index.empty:
            assert (result == 0)
            return
        index.get_loc(index[0])
        result2 = index.memory_usage()
        result3 = index.memory_usage(deep=True)
        if (not isinstance(index, (RangeIndex, IntervalIndex))):
            assert (result2 > result)
        if (index.inferred_type == 'object'):
            assert (result3 > result2)

    def test_argsort(self, request, index):
        if isinstance(index, CategoricalIndex):
            return
        result = index.argsort()
        expected = np.array(index).argsort()
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_numpy_argsort(self, index):
        result = np.argsort(index)
        expected = index.argsort()
        tm.assert_numpy_array_equal(result, expected)
        if isinstance(type(index), (CategoricalIndex, RangeIndex)):
            msg = "the 'axis' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, axis=1)
            msg = "the 'kind' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, kind='mergesort')
            msg = "the 'order' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, order=('a', 'b'))

    def test_repeat(self):
        rep = 2
        i = self.create_index()
        expected = Index(i.values.repeat(rep), name=i.name)
        tm.assert_index_equal(i.repeat(rep), expected)
        i = self.create_index()
        rep = np.arange(len(i))
        expected = Index(i.values.repeat(rep), name=i.name)
        tm.assert_index_equal(i.repeat(rep), expected)

    def test_numpy_repeat(self):
        rep = 2
        i = self.create_index()
        expected = i.repeat(rep)
        tm.assert_index_equal(np.repeat(i, rep), expected)
        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(i, rep, axis=0)

    @pytest.mark.parametrize('klass', [list, tuple, np.array, Series])
    def test_where(self, klass):
        i = self.create_index()
        if isinstance(i, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            i = i._with_freq(None)
        cond = ([True] * len(i))
        result = i.where(klass(cond))
        expected = i
        tm.assert_index_equal(result, expected)
        cond = ([False] + ([True] * len(i[1:])))
        expected = Index(([i._na_value] + i[1:].tolist()), dtype=i.dtype)
        result = i.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_insert_base(self, index):
        result = index[1:4]
        if (not len(index)):
            return
        assert index[0:4].equals(result.insert(0, index[0]))

    def test_delete_base(self, index):
        if (not len(index)):
            return
        if isinstance(index, RangeIndex):
            return
        expected = index[1:]
        result = index.delete(0)
        assert result.equals(expected)
        assert (result.name == expected.name)
        expected = index[:(- 1)]
        result = index.delete((- 1))
        assert result.equals(expected)
        assert (result.name == expected.name)
        length = len(index)
        msg = f'index {length} is out of bounds for axis 0 with size {length}'
        with pytest.raises(IndexError, match=msg):
            index.delete(length)

    def test_equals(self, index):
        if isinstance(index, IntervalIndex):
            return
        assert index.equals(index)
        assert index.equals(index.copy())
        assert index.equals(index.astype(object))
        assert (not index.equals(list(index)))
        assert (not index.equals(np.array(index)))
        if (not isinstance(index, RangeIndex)):
            same_values = Index(index, dtype=object)
            assert index.equals(same_values)
            assert same_values.equals(index)
        if (index.nlevels == 1):
            assert (not index.equals(Series(index)))

    def test_equals_op(self):
        index_a = self.create_index()
        n = len(index_a)
        index_b = index_a[0:(- 1)]
        index_c = index_a[0:(- 1)].append(index_a[(- 2):(- 1)])
        index_d = index_a[0:1]
        msg = 'Lengths must match|could not be broadcast'
        with pytest.raises(ValueError, match=msg):
            (index_a == index_b)
        expected1 = np.array(([True] * n))
        expected2 = np.array((([True] * (n - 1)) + [False]))
        tm.assert_numpy_array_equal((index_a == index_a), expected1)
        tm.assert_numpy_array_equal((index_a == index_c), expected2)
        array_a = np.array(index_a)
        array_b = np.array(index_a[0:(- 1)])
        array_c = np.array(index_a[0:(- 1)].append(index_a[(- 2):(- 1)]))
        array_d = np.array(index_a[0:1])
        with pytest.raises(ValueError, match=msg):
            (index_a == array_b)
        tm.assert_numpy_array_equal((index_a == array_a), expected1)
        tm.assert_numpy_array_equal((index_a == array_c), expected2)
        series_a = Series(array_a)
        series_b = Series(array_b)
        series_c = Series(array_c)
        series_d = Series(array_d)
        with pytest.raises(ValueError, match=msg):
            (index_a == series_b)
        tm.assert_numpy_array_equal((index_a == series_a), expected1)
        tm.assert_numpy_array_equal((index_a == series_c), expected2)
        with pytest.raises(ValueError, match='Lengths must match'):
            (index_a == index_d)
        with pytest.raises(ValueError, match='Lengths must match'):
            (index_a == series_d)
        with pytest.raises(ValueError, match='Lengths must match'):
            (index_a == array_d)
        msg = 'Can only compare identically-labeled Series objects'
        with pytest.raises(ValueError, match=msg):
            (series_a == series_d)
        with pytest.raises(ValueError, match='Lengths must match'):
            (series_a == array_d)
        if (not isinstance(index_a, MultiIndex)):
            expected3 = np.array((([False] * (len(index_a) - 2)) + [True, False]))
            item = index_a[(- 2)]
            tm.assert_numpy_array_equal((index_a == item), expected3)
            tm.assert_series_equal((series_a == item), Series(expected3))

    def test_format(self):
        idx = self.create_index()
        expected = [str(x) for x in idx]
        assert (idx.format() == expected)

    def test_format_empty(self):
        empty_idx = self._holder([])
        assert (empty_idx.format() == [])
        assert (empty_idx.format(name=True) == [''])

    def test_hasnans_isnans(self, index):
        if isinstance(index, MultiIndex):
            return
        idx = index.copy(deep=True)
        expected = np.array(([False] * len(idx)), dtype=bool)
        tm.assert_numpy_array_equal(idx._isnan, expected)
        assert (idx.hasnans is False)
        idx = index.copy(deep=True)
        values = np.asarray(idx.values)
        if (len(index) == 0):
            return
        elif isinstance(index, DatetimeIndexOpsMixin):
            values[1] = iNaT
        elif isinstance(index, (Int64Index, UInt64Index)):
            return
        else:
            values[1] = np.nan
        if isinstance(index, PeriodIndex):
            idx = type(index)(values, freq=index.freq)
        else:
            idx = type(index)(values)
            expected = np.array(([False] * len(idx)), dtype=bool)
            expected[1] = True
            tm.assert_numpy_array_equal(idx._isnan, expected)
            assert (idx.hasnans is True)

    def test_fillna(self, index):
        if (len(index) == 0):
            pass
        elif isinstance(index, MultiIndex):
            idx = index.copy(deep=True)
            msg = 'isna is not defined for MultiIndex'
            with pytest.raises(NotImplementedError, match=msg):
                idx.fillna(idx[0])
        else:
            idx = index.copy(deep=True)
            result = idx.fillna(idx[0])
            tm.assert_index_equal(result, idx)
            assert (result is not idx)
            msg = "'value' must be a scalar, passed: "
            with pytest.raises(TypeError, match=msg):
                idx.fillna([idx[0]])
            idx = index.copy(deep=True)
            values = np.asarray(idx.values)
            if isinstance(index, DatetimeIndexOpsMixin):
                values[1] = iNaT
            elif isinstance(index, (Int64Index, UInt64Index)):
                return
            else:
                values[1] = np.nan
            if isinstance(index, PeriodIndex):
                idx = type(index)(values, freq=index.freq)
            else:
                idx = type(index)(values)
            expected = np.array(([False] * len(idx)), dtype=bool)
            expected[1] = True
            tm.assert_numpy_array_equal(idx._isnan, expected)
            assert (idx.hasnans is True)

    def test_nulls(self, index):
        if (len(index) == 0):
            tm.assert_numpy_array_equal(index.isna(), np.array([], dtype=bool))
        elif isinstance(index, MultiIndex):
            idx = index.copy()
            msg = 'isna is not defined for MultiIndex'
            with pytest.raises(NotImplementedError, match=msg):
                idx.isna()
        elif (not index.hasnans):
            tm.assert_numpy_array_equal(index.isna(), np.zeros(len(index), dtype=bool))
            tm.assert_numpy_array_equal(index.notna(), np.ones(len(index), dtype=bool))
        else:
            result = isna(index)
            tm.assert_numpy_array_equal(index.isna(), result)
            tm.assert_numpy_array_equal(index.notna(), (~ result))

    def test_empty(self):
        index = self.create_index()
        assert (not index.empty)
        assert index[:0].empty

    def test_join_self_unique(self, join_type):
        index = self.create_index()
        if index.is_unique:
            joined = index.join(index, how=join_type)
            assert (index == joined).all()

    def test_map(self):
        index = self.create_index()
        if isinstance(index, pd.UInt64Index):
            expected = index.astype('int64')
        else:
            expected = index
        result = index.map((lambda x: x))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('mapper', [(lambda values, index: {i: e for (e, i) in zip(values, index)}), (lambda values, index: Series(values, index))])
    def test_map_dictlike(self, mapper):
        index = self.create_index()
        if isinstance(index, (pd.CategoricalIndex, pd.IntervalIndex)):
            pytest.skip(f'skipping tests for {type(index)}')
        identity = mapper(index.values, index)
        if (isinstance(index, pd.UInt64Index) and isinstance(identity, dict)):
            expected = index.astype('int64')
        else:
            expected = index
        result = index.map(identity)
        tm.assert_index_equal(result, expected)
        expected = Index(([np.nan] * len(index)))
        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

    def test_map_str(self):
        index = self.create_index()
        result = index.map(str)
        expected = Index([str(x) for x in index], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_putmask_with_wrong_mask(self):
        index = self.create_index()
        fill = index[0]
        msg = 'putmask: mask and data must be the same size'
        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones((len(index) + 1), np.bool_), fill)
        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones((len(index) - 1), np.bool_), fill)
        with pytest.raises(ValueError, match=msg):
            index.putmask('foo', fill)

    @pytest.mark.parametrize('copy', [True, False])
    @pytest.mark.parametrize('name', [None, 'foo'])
    @pytest.mark.parametrize('ordered', [True, False])
    def test_astype_category(self, copy, name, ordered):
        index = self.create_index()
        if name:
            index = index.rename(name)
        dtype = CategoricalDtype(ordered=ordered)
        result = index.astype(dtype, copy=copy)
        expected = CategoricalIndex(index.values, name=name, ordered=ordered)
        tm.assert_index_equal(result, expected)
        dtype = CategoricalDtype(index.unique().tolist()[:(- 1)], ordered)
        result = index.astype(dtype, copy=copy)
        expected = CategoricalIndex(index.values, name=name, dtype=dtype)
        tm.assert_index_equal(result, expected)
        if (ordered is False):
            result = index.astype('category', copy=copy)
            expected = CategoricalIndex(index.values, name=name)
            tm.assert_index_equal(result, expected)

    def test_is_unique(self):
        index = self.create_index().drop_duplicates()
        assert (index.is_unique is True)
        index_empty = index[:0]
        assert (index_empty.is_unique is True)
        index_dup = index.insert(0, index[0])
        assert (index_dup.is_unique is False)
        index_na = index.insert(0, np.nan)
        assert (index_na.is_unique is True)
        index_na_dup = index_na.insert(0, np.nan)
        assert (index_na_dup.is_unique is False)

    @pytest.mark.arm_slow
    def test_engine_reference_cycle(self):
        index = self.create_index()
        nrefs_pre = len(gc.get_referrers(index))
        index._engine
        assert (len(gc.get_referrers(index)) == nrefs_pre)

    def test_getitem_2d_deprecated(self):
        idx = self.create_index()
        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            res = idx[:, None]
        assert isinstance(res, np.ndarray), type(res)

    def test_contains_requires_hashable_raises(self):
        idx = self.create_index()
        msg = "unhashable type: 'list'"
        with pytest.raises(TypeError, match=msg):
            ([] in idx)
        msg = '|'.join(["unhashable type: 'dict'", 'must be real number, not dict', 'an integer is required', '\\{\\}', "pandas\\._libs\\.interval\\.IntervalTree' is not iterable"])
        with pytest.raises(TypeError, match=msg):
            ({} in idx._engine)

    def test_copy_shares_cache(self):
        idx = self.create_index()
        idx.get_loc(idx[0])
        copy = idx.copy()
        assert (copy._cache is idx._cache)

    def test_shallow_copy_shares_cache(self):
        idx = self.create_index()
        idx.get_loc(idx[0])
        shallow_copy = idx._shallow_copy()
        assert (shallow_copy._cache is idx._cache)
        shallow_copy = idx._shallow_copy(idx._data)
        assert (shallow_copy._cache is not idx._cache)
        assert (shallow_copy._cache == {})
