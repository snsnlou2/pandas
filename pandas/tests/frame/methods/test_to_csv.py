
import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, Timestamp, date_range, read_csv, to_datetime
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
MIXED_FLOAT_DTYPES = ['float16', 'float32', 'float64']
MIXED_INT_DTYPES = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64']

class TestDataFrameToCSV():

    def read_csv(self, path, **kwargs):
        params = {'index_col': 0, 'parse_dates': True}
        params.update(**kwargs)
        return pd.read_csv(path, **params)

    def test_to_csv_from_csv1(self, float_frame, datetime_frame):
        with tm.ensure_clean('__tmp_to_csv_from_csv1__') as path:
            float_frame['A'][:5] = np.nan
            float_frame.to_csv(path)
            float_frame.to_csv(path, columns=['A', 'B'])
            float_frame.to_csv(path, header=False)
            float_frame.to_csv(path, index=False)
            datetime_frame.index = datetime_frame.index._with_freq(None)
            datetime_frame.to_csv(path)
            recons = self.read_csv(path)
            tm.assert_frame_equal(datetime_frame, recons)
            datetime_frame.to_csv(path, index_label='index')
            recons = self.read_csv(path, index_col=None)
            assert (len(recons.columns) == (len(datetime_frame.columns) + 1))
            datetime_frame.to_csv(path, index=False)
            recons = self.read_csv(path, index_col=None)
            tm.assert_almost_equal(datetime_frame.values, recons.values)
            dm = DataFrame({'s1': Series(range(3), index=np.arange(3)), 's2': Series(range(2), index=np.arange(2))})
            dm.to_csv(path)
            recons = self.read_csv(path)
            tm.assert_frame_equal(dm, recons)

    def test_to_csv_from_csv2(self, float_frame):
        with tm.ensure_clean('__tmp_to_csv_from_csv2__') as path:
            df = DataFrame(np.random.randn(3, 3), index=['a', 'a', 'b'], columns=['x', 'y', 'z'])
            df.to_csv(path)
            result = self.read_csv(path)
            tm.assert_frame_equal(result, df)
            midx = MultiIndex.from_tuples([('A', 1, 2), ('A', 1, 2), ('B', 1, 2)])
            df = DataFrame(np.random.randn(3, 3), index=midx, columns=['x', 'y', 'z'])
            df.to_csv(path)
            result = self.read_csv(path, index_col=[0, 1, 2], parse_dates=False)
            tm.assert_frame_equal(result, df, check_names=False)
            col_aliases = Index(['AA', 'X', 'Y', 'Z'])
            float_frame.to_csv(path, header=col_aliases)
            rs = self.read_csv(path)
            xp = float_frame.copy()
            xp.columns = col_aliases
            tm.assert_frame_equal(xp, rs)
            msg = 'Writing 4 cols but got 2 aliases'
            with pytest.raises(ValueError, match=msg):
                float_frame.to_csv(path, header=['AA', 'X'])

    def test_to_csv_from_csv3(self):
        with tm.ensure_clean('__tmp_to_csv_from_csv3__') as path:
            df1 = DataFrame(np.random.randn(3, 1))
            df2 = DataFrame(np.random.randn(3, 1))
            df1.to_csv(path)
            df2.to_csv(path, mode='a', header=False)
            xp = pd.concat([df1, df2])
            rs = pd.read_csv(path, index_col=0)
            rs.columns = [int(label) for label in rs.columns]
            xp.columns = [int(label) for label in xp.columns]
            tm.assert_frame_equal(xp, rs)

    def test_to_csv_from_csv4(self):
        with tm.ensure_clean('__tmp_to_csv_from_csv4__') as path:
            dt = pd.Timedelta(seconds=1)
            df = DataFrame({'dt_data': [(i * dt) for i in range(3)]}, index=Index([(i * dt) for i in range(3)], name='dt_index'))
            df.to_csv(path)
            result = pd.read_csv(path, index_col='dt_index')
            result.index = pd.to_timedelta(result.index)
            result.index = result.index.rename('dt_index')
            result['dt_data'] = pd.to_timedelta(result['dt_data'])
            tm.assert_frame_equal(df, result, check_index_type=True)

    def test_to_csv_from_csv5(self, timezone_frame):
        with tm.ensure_clean('__tmp_to_csv_from_csv5__') as path:
            timezone_frame.to_csv(path)
            result = pd.read_csv(path, index_col=0, parse_dates=['A'])
            converter = (lambda c: to_datetime(result[c]).dt.tz_convert('UTC').dt.tz_convert(timezone_frame[c].dt.tz))
            result['B'] = converter('B')
            result['C'] = converter('C')
            tm.assert_frame_equal(result, timezone_frame)

    def test_to_csv_cols_reordering(self):
        import pandas as pd
        chunksize = 5
        N = int((chunksize * 2.5))
        df = tm.makeCustomDataframe(N, 3)
        cs = df.columns
        cols = [cs[2], cs[0]]
        with tm.ensure_clean() as path:
            df.to_csv(path, columns=cols, chunksize=chunksize)
            rs_c = pd.read_csv(path, index_col=0)
        tm.assert_frame_equal(df[cols], rs_c, check_names=False)

    def test_to_csv_new_dupe_cols(self):
        import pandas as pd

        def _check_df(df, cols=None):
            with tm.ensure_clean() as path:
                df.to_csv(path, columns=cols, chunksize=chunksize)
                rs_c = pd.read_csv(path, index_col=0)
                if (cols is not None):
                    if df.columns.is_unique:
                        rs_c.columns = cols
                    else:
                        (indexer, missing) = df.columns.get_indexer_non_unique(cols)
                        rs_c.columns = df.columns.take(indexer)
                    for c in cols:
                        obj_df = df[c]
                        obj_rs = rs_c[c]
                        if isinstance(obj_df, Series):
                            tm.assert_series_equal(obj_df, obj_rs)
                        else:
                            tm.assert_frame_equal(obj_df, obj_rs, check_names=False)
                else:
                    rs_c.columns = df.columns
                    tm.assert_frame_equal(df, rs_c, check_names=False)
        chunksize = 5
        N = int((chunksize * 2.5))
        df = tm.makeCustomDataframe(N, 3)
        df.columns = ['a', 'a', 'b']
        _check_df(df, None)
        cols = ['b', 'a']
        _check_df(df, cols)

    @pytest.mark.slow
    def test_to_csv_dtnat(self):
        from pandas import NaT

        def make_dtnat_arr(n, nnat=None):
            if (nnat is None):
                nnat = int((n * 0.1))
            s = list(date_range('2000', freq='5min', periods=n))
            if nnat:
                for i in np.random.randint(0, len(s), nnat):
                    s[i] = NaT
                i = np.random.randint(100)
                s[(- i)] = NaT
                s[i] = NaT
            return s
        chunksize = 1000
        s1 = make_dtnat_arr((chunksize + 5))
        s2 = make_dtnat_arr((chunksize + 5), 0)
        with tm.ensure_clean('1.csv') as pth:
            df = DataFrame({'a': s1, 'b': s2})
            df.to_csv(pth, chunksize=chunksize)
            recons = self.read_csv(pth).apply(to_datetime)
            tm.assert_frame_equal(df, recons, check_names=False)

    @pytest.mark.slow
    def test_to_csv_moar(self):

        def _do_test(df, r_dtype=None, c_dtype=None, rnlvl=None, cnlvl=None, dupe_col=False):
            kwargs = {'parse_dates': False}
            if cnlvl:
                if (rnlvl is not None):
                    kwargs['index_col'] = list(range(rnlvl))
                kwargs['header'] = list(range(cnlvl))
                with tm.ensure_clean('__tmp_to_csv_moar__') as path:
                    df.to_csv(path, encoding='utf8', chunksize=chunksize)
                    recons = self.read_csv(path, **kwargs)
            else:
                kwargs['header'] = 0
                with tm.ensure_clean('__tmp_to_csv_moar__') as path:
                    df.to_csv(path, encoding='utf8', chunksize=chunksize)
                    recons = self.read_csv(path, **kwargs)

            def _to_uni(x):
                if (not isinstance(x, str)):
                    return x.decode('utf8')
                return x
            if dupe_col:
                recons.columns = df.columns
            if (rnlvl and (not cnlvl)):
                delta_lvl = [recons.iloc[:, i].values for i in range((rnlvl - 1))]
                ix = MultiIndex.from_arrays(([list(recons.index)] + delta_lvl))
                recons.index = ix
                recons = recons.iloc[:, (rnlvl - 1):]
            type_map = {'i': 'i', 'f': 'f', 's': 'O', 'u': 'O', 'dt': 'O', 'p': 'O'}
            if r_dtype:
                if (r_dtype == 'u'):
                    r_dtype = 'O'
                    recons.index = np.array([_to_uni(label) for label in recons.index], dtype=r_dtype)
                    df.index = np.array([_to_uni(label) for label in df.index], dtype=r_dtype)
                elif (r_dtype == 'dt'):
                    r_dtype = 'O'
                    recons.index = np.array([Timestamp(label) for label in recons.index], dtype=r_dtype)
                    df.index = np.array([Timestamp(label) for label in df.index], dtype=r_dtype)
                elif (r_dtype == 'p'):
                    r_dtype = 'O'
                    idx_list = to_datetime(recons.index)
                    recons.index = np.array([Timestamp(label) for label in idx_list], dtype=r_dtype)
                    df.index = np.array(list(map(Timestamp, df.index.to_timestamp())), dtype=r_dtype)
                else:
                    r_dtype = type_map.get(r_dtype)
                    recons.index = np.array(recons.index, dtype=r_dtype)
                    df.index = np.array(df.index, dtype=r_dtype)
            if c_dtype:
                if (c_dtype == 'u'):
                    c_dtype = 'O'
                    recons.columns = np.array([_to_uni(label) for label in recons.columns], dtype=c_dtype)
                    df.columns = np.array([_to_uni(label) for label in df.columns], dtype=c_dtype)
                elif (c_dtype == 'dt'):
                    c_dtype = 'O'
                    recons.columns = np.array([Timestamp(label) for label in recons.columns], dtype=c_dtype)
                    df.columns = np.array([Timestamp(label) for label in df.columns], dtype=c_dtype)
                elif (c_dtype == 'p'):
                    c_dtype = 'O'
                    col_list = to_datetime(recons.columns)
                    recons.columns = np.array([Timestamp(label) for label in col_list], dtype=c_dtype)
                    col_list = df.columns.to_timestamp()
                    df.columns = np.array([Timestamp(label) for label in col_list], dtype=c_dtype)
                else:
                    c_dtype = type_map.get(c_dtype)
                    recons.columns = np.array(recons.columns, dtype=c_dtype)
                    df.columns = np.array(df.columns, dtype=c_dtype)
            tm.assert_frame_equal(df, recons, check_names=False)
        N = 100
        chunksize = 1000
        ncols = 4
        base = (chunksize // ncols)
        for nrows in [2, 10, (N - 1), N, (N + 1), (N + 2), ((2 * N) - 2), ((2 * N) - 1), (2 * N), ((2 * N) + 1), ((2 * N) + 2), (base - 1), base, (base + 1)]:
            _do_test(tm.makeCustomDataframe(nrows, ncols, r_idx_type='dt', c_idx_type='s'), 'dt', 's')
        for (r_idx_type, c_idx_type) in [('i', 'i'), ('s', 's'), ('u', 'dt'), ('p', 'p')]:
            for ncols in [1, 2, 3, 4]:
                base = (chunksize // ncols)
                for nrows in [2, 10, (N - 1), N, (N + 1), (N + 2), ((2 * N) - 2), ((2 * N) - 1), (2 * N), ((2 * N) + 1), ((2 * N) + 2), (base - 1), base, (base + 1)]:
                    _do_test(tm.makeCustomDataframe(nrows, ncols, r_idx_type=r_idx_type, c_idx_type=c_idx_type), r_idx_type, c_idx_type)
        for ncols in [1, 2, 3, 4]:
            base = (chunksize // ncols)
            for nrows in [10, (N - 2), (N - 1), N, (N + 1), (N + 2), ((2 * N) - 2), ((2 * N) - 1), (2 * N), ((2 * N) + 1), ((2 * N) + 2), (base - 1), base, (base + 1)]:
                _do_test(tm.makeCustomDataframe(nrows, ncols))
        for nrows in [10, (N - 2), (N - 1), N, (N + 1), (N + 2)]:
            df = tm.makeCustomDataframe(nrows, 3)
            cols = list(df.columns)
            cols[:2] = ['dupe', 'dupe']
            cols[(- 2):] = ['dupe', 'dupe']
            ix = list(df.index)
            ix[:2] = ['rdupe', 'rdupe']
            ix[(- 2):] = ['rdupe', 'rdupe']
            df.index = ix
            df.columns = cols
            _do_test(df, dupe_col=True)
        _do_test(DataFrame(index=np.arange(10)))
        _do_test(tm.makeCustomDataframe(((chunksize // 2) + 1), 2, r_idx_nlevels=2), rnlvl=2)
        for ncols in [2, 3, 4]:
            base = int((chunksize // ncols))
            for nrows in [10, (N - 2), (N - 1), N, (N + 1), (N + 2), ((2 * N) - 2), ((2 * N) - 1), (2 * N), ((2 * N) + 1), ((2 * N) + 2), (base - 1), base, (base + 1)]:
                _do_test(tm.makeCustomDataframe(nrows, ncols, r_idx_nlevels=2), rnlvl=2)
                _do_test(tm.makeCustomDataframe(nrows, ncols, c_idx_nlevels=2), cnlvl=2)
                _do_test(tm.makeCustomDataframe(nrows, ncols, r_idx_nlevels=2, c_idx_nlevels=2), rnlvl=2, cnlvl=2)

    def test_to_csv_from_csv_w_some_infs(self, float_frame):
        float_frame['G'] = np.nan
        f = (lambda x: [np.inf, np.nan][(np.random.rand() < 0.5)])
        float_frame['H'] = float_frame.index.map(f)
        with tm.ensure_clean() as path:
            float_frame.to_csv(path)
            recons = self.read_csv(path)
            tm.assert_frame_equal(float_frame, recons, check_names=False)
            tm.assert_frame_equal(np.isinf(float_frame), np.isinf(recons), check_names=False)

    def test_to_csv_from_csv_w_all_infs(self, float_frame):
        float_frame['E'] = np.inf
        float_frame['F'] = (- np.inf)
        with tm.ensure_clean() as path:
            float_frame.to_csv(path)
            recons = self.read_csv(path)
            tm.assert_frame_equal(float_frame, recons, check_names=False)
            tm.assert_frame_equal(np.isinf(float_frame), np.isinf(recons), check_names=False)

    def test_to_csv_no_index(self):
        with tm.ensure_clean('__tmp_to_csv_no_index__') as path:
            df = DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6]})
            df.to_csv(path, index=False)
            result = read_csv(path)
            tm.assert_frame_equal(df, result)
            df['c3'] = Series([7, 8, 9], dtype='int64')
            df.to_csv(path, index=False)
            result = read_csv(path)
            tm.assert_frame_equal(df, result)

    def test_to_csv_with_mix_columns(self):
        df = DataFrame({0: ['a', 'b', 'c'], 1: ['aa', 'bb', 'cc']})
        df['test'] = 'txt'
        assert (df.to_csv() == df.to_csv(columns=[0, 1, 'test']))

    def test_to_csv_headers(self):
        from_df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        to_df = DataFrame([[1, 2], [3, 4]], columns=['X', 'Y'])
        with tm.ensure_clean('__tmp_to_csv_headers__') as path:
            from_df.to_csv(path, header=['X', 'Y'])
            recons = self.read_csv(path)
            tm.assert_frame_equal(to_df, recons)
            from_df.to_csv(path, index=False, header=['X', 'Y'])
            recons = self.read_csv(path)
            return_value = recons.reset_index(inplace=True)
            assert (return_value is None)
            tm.assert_frame_equal(to_df, recons)

    def test_to_csv_multiindex(self, float_frame, datetime_frame):
        frame = float_frame
        old_index = frame.index
        arrays = np.arange((len(old_index) * 2)).reshape(2, (- 1))
        new_index = MultiIndex.from_arrays(arrays, names=['first', 'second'])
        frame.index = new_index
        with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:
            frame.to_csv(path, header=False)
            frame.to_csv(path, columns=['A', 'B'])
            frame.to_csv(path)
            df = self.read_csv(path, index_col=[0, 1], parse_dates=False)
            tm.assert_frame_equal(frame, df, check_names=False)
            assert (frame.index.names == df.index.names)
            float_frame.index = old_index
            tsframe = datetime_frame
            old_index = tsframe.index
            new_index = [old_index, np.arange(len(old_index))]
            tsframe.index = MultiIndex.from_arrays(new_index)
            tsframe.to_csv(path, index_label=['time', 'foo'])
            recons = self.read_csv(path, index_col=[0, 1])
            tm.assert_frame_equal(tsframe, recons, check_names=False)
            tsframe.to_csv(path)
            recons = self.read_csv(path, index_col=None)
            assert (len(recons.columns) == (len(tsframe.columns) + 2))
            tsframe.to_csv(path, index=False)
            recons = self.read_csv(path, index_col=None)
            tm.assert_almost_equal(recons.values, datetime_frame.values)
            datetime_frame.index = old_index
        with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:

            def _make_frame(names=None):
                if (names is True):
                    names = ['first', 'second']
                return DataFrame(np.random.randint(0, 10, size=(3, 3)), columns=MultiIndex.from_tuples([('bah', 'foo'), ('bah', 'bar'), ('ban', 'baz')], names=names), dtype='int64')
            df = tm.makeCustomDataframe(5, 3, r_idx_nlevels=2, c_idx_nlevels=4)
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1])
            tm.assert_frame_equal(df, result)
            df = tm.makeCustomDataframe(5, 3, r_idx_nlevels=1, c_idx_nlevels=4)
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=0)
            tm.assert_frame_equal(df, result)
            df = tm.makeCustomDataframe(5, 3, r_idx_nlevels=3, c_idx_nlevels=4)
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1, 2])
            tm.assert_frame_equal(df, result)
            df = _make_frame()
            df.to_csv(path, index=False)
            result = read_csv(path, header=[0, 1])
            tm.assert_frame_equal(df, result)
            df = _make_frame(True)
            df.to_csv(path, index=False)
            result = read_csv(path, header=[0, 1])
            assert com.all_none(*result.columns.names)
            result.columns.names = df.columns.names
            tm.assert_frame_equal(df, result)
            df = _make_frame()
            df.to_csv(path)
            result = read_csv(path, header=[0, 1], index_col=[0])
            tm.assert_frame_equal(df, result)
            df = _make_frame(True)
            df.to_csv(path)
            result = read_csv(path, header=[0, 1], index_col=[0])
            tm.assert_frame_equal(df, result)
            df = _make_frame(True)
            df.to_csv(path)
            for i in [6, 7]:
                msg = f'len of {i}, but only 5 lines in file'
                with pytest.raises(ParserError, match=msg):
                    read_csv(path, header=list(range(i)), index_col=0)
            msg = 'cannot specify cols with a MultiIndex'
            with pytest.raises(TypeError, match=msg):
                df.to_csv(path, columns=['foo', 'bar'])
        with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:
            tsframe[:0].to_csv(path)
            recons = self.read_csv(path)
            exp = tsframe[:0]
            exp.index = []
            tm.assert_index_equal(recons.columns, exp.columns)
            assert (len(recons) == 0)

    def test_to_csv_interval_index(self):
        df = DataFrame({'A': list('abc'), 'B': range(3)}, index=pd.interval_range(0, 3))
        with tm.ensure_clean('__tmp_to_csv_interval_index__.csv') as path:
            df.to_csv(path)
            result = self.read_csv(path, index_col=0)
            expected = df.copy()
            expected.index = expected.index.astype(str)
            tm.assert_frame_equal(result, expected)

    def test_to_csv_float32_nanrep(self):
        df = DataFrame(np.random.randn(1, 4).astype(np.float32))
        df[1] = np.nan
        with tm.ensure_clean('__tmp_to_csv_float32_nanrep__.csv') as path:
            df.to_csv(path, na_rep=999)
            with open(path) as f:
                lines = f.readlines()
                assert (lines[1].split(',')[2] == '999')

    def test_to_csv_withcommas(self):
        df = DataFrame({'A': [1, 2, 3], 'B': ['5,6', '7,8', '9,0']})
        with tm.ensure_clean('__tmp_to_csv_withcommas__.csv') as path:
            df.to_csv(path)
            df2 = self.read_csv(path)
            tm.assert_frame_equal(df2, df)

    def test_to_csv_mixed(self):

        def create_cols(name):
            return [f'{name}{i:03d}' for i in range(5)]
        df_float = DataFrame(np.random.randn(100, 5), dtype='float64', columns=create_cols('float'))
        df_int = DataFrame(np.random.randn(100, 5), dtype='int64', columns=create_cols('int'))
        df_bool = DataFrame(True, index=df_float.index, columns=create_cols('bool'))
        df_object = DataFrame('foo', index=df_float.index, columns=create_cols('object'))
        df_dt = DataFrame(Timestamp('20010101'), index=df_float.index, columns=create_cols('date'))
        df_float.iloc[30:50, 1:3] = np.nan
        df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1)
        dtypes = {}
        for (n, dtype) in [('float', np.float64), ('int', np.int64), ('bool', np.bool_), ('object', object)]:
            for c in create_cols(n):
                dtypes[c] = dtype
        with tm.ensure_clean() as filename:
            df.to_csv(filename)
            rs = read_csv(filename, index_col=0, dtype=dtypes, parse_dates=create_cols('date'))
            tm.assert_frame_equal(rs, df)

    def test_to_csv_dups_cols(self):
        df = DataFrame(np.random.randn(1000, 30), columns=(list(range(15)) + list(range(15))), dtype='float64')
        with tm.ensure_clean() as filename:
            df.to_csv(filename)
            result = read_csv(filename, index_col=0)
            result.columns = df.columns
            tm.assert_frame_equal(result, df)
        df_float = DataFrame(np.random.randn(1000, 3), dtype='float64')
        df_int = DataFrame(np.random.randn(1000, 3), dtype='int64')
        df_bool = DataFrame(True, index=df_float.index, columns=range(3))
        df_object = DataFrame('foo', index=df_float.index, columns=range(3))
        df_dt = DataFrame(Timestamp('20010101'), index=df_float.index, columns=range(3))
        df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1, ignore_index=True)
        cols = []
        for i in range(5):
            cols.extend([0, 1, 2])
        df.columns = cols
        with tm.ensure_clean() as filename:
            df.to_csv(filename)
            result = read_csv(filename, index_col=0)
            for i in ['0.4', '1.4', '2.4']:
                result[i] = to_datetime(result[i])
            result.columns = df.columns
            tm.assert_frame_equal(result, df)
        N = 10
        df = tm.makeCustomDataframe(N, 3)
        df.columns = ['a', 'a', 'b']
        with tm.ensure_clean() as filename:
            df.to_csv(filename)
            result = read_csv(filename, index_col=0)
            result = result.rename(columns={'a.1': 'a'})
            tm.assert_frame_equal(result, df)

    def test_to_csv_chunking(self):
        aa = DataFrame({'A': range(100000)})
        aa['B'] = (aa.A + 1.0)
        aa['C'] = (aa.A + 2.0)
        aa['D'] = (aa.A + 3.0)
        for chunksize in [10000, 50000, 100000]:
            with tm.ensure_clean() as filename:
                aa.to_csv(filename, chunksize=chunksize)
                rs = read_csv(filename, index_col=0)
                tm.assert_frame_equal(rs, aa)

    @pytest.mark.slow
    def test_to_csv_wide_frame_formatting(self):
        df = DataFrame(np.random.randn(1, 100010), columns=None, index=None)
        with tm.ensure_clean() as filename:
            df.to_csv(filename, header=False, index=False)
            rs = read_csv(filename, header=None)
            tm.assert_frame_equal(rs, df)

    def test_to_csv_bug(self):
        f1 = StringIO('a,1.0\nb,2.0')
        df = self.read_csv(f1, header=None)
        newdf = DataFrame({'t': df[df.columns[0]]})
        with tm.ensure_clean() as path:
            newdf.to_csv(path)
            recons = read_csv(path, index_col=0)
            tm.assert_frame_equal(recons, newdf, check_names=False)

    def test_to_csv_unicode(self):
        df = DataFrame({'c/σ': [1, 2, 3]})
        with tm.ensure_clean() as path:
            df.to_csv(path, encoding='UTF-8')
            df2 = read_csv(path, index_col=0, encoding='UTF-8')
            tm.assert_frame_equal(df, df2)
            df.to_csv(path, encoding='UTF-8', index=False)
            df2 = read_csv(path, index_col=None, encoding='UTF-8')
            tm.assert_frame_equal(df, df2)

    def test_to_csv_unicode_index_col(self):
        buf = StringIO('')
        df = DataFrame([['א', 'd2', 'd3', 'd4'], ['a1', 'a2', 'a3', 'a4']], columns=['א', 'ב', 'ג', 'ד'], index=['א', 'ב'])
        df.to_csv(buf, encoding='UTF-8')
        buf.seek(0)
        df2 = read_csv(buf, index_col=0, encoding='UTF-8')
        tm.assert_frame_equal(df, df2)

    def test_to_csv_stringio(self, float_frame):
        buf = StringIO()
        float_frame.to_csv(buf)
        buf.seek(0)
        recons = read_csv(buf, index_col=0)
        tm.assert_frame_equal(recons, float_frame, check_names=False)

    def test_to_csv_float_format(self):
        df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
        with tm.ensure_clean() as filename:
            df.to_csv(filename, float_format='%.2f')
            rs = read_csv(filename, index_col=0)
            xp = DataFrame([[0.12, 0.23, 0.57], [12.32, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
            tm.assert_frame_equal(rs, xp)

    def test_to_csv_unicodewriter_quoting(self):
        df = DataFrame({'A': [1, 2, 3], 'B': ['foo', 'bar', 'baz']})
        buf = StringIO()
        df.to_csv(buf, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
        result = buf.getvalue()
        expected_rows = ['"A","B"', '1,"foo"', '2,"bar"', '3,"baz"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (result == expected)

    def test_to_csv_quote_none(self):
        df = DataFrame({'A': ['hello', '{"hello"}']})
        for encoding in (None, 'utf-8'):
            buf = StringIO()
            df.to_csv(buf, quoting=csv.QUOTE_NONE, encoding=encoding, index=False)
            result = buf.getvalue()
            expected_rows = ['A', 'hello', '{"hello"}']
            expected = tm.convert_rows_list_to_csv_str(expected_rows)
            assert (result == expected)

    def test_to_csv_index_no_leading_comma(self):
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['one', 'two', 'three'])
        buf = StringIO()
        df.to_csv(buf, index_label=False)
        expected_rows = ['A,B', 'one,1,4', 'two,2,5', 'three,3,6']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (buf.getvalue() == expected)

    def test_to_csv_line_terminators(self):
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['one', 'two', 'three'])
        with tm.ensure_clean() as path:
            df.to_csv(path, line_terminator='\r\n')
            expected = b',A,B\r\none,1,4\r\ntwo,2,5\r\nthree,3,6\r\n'
            with open(path, mode='rb') as f:
                assert (f.read() == expected)
        with tm.ensure_clean() as path:
            df.to_csv(path, line_terminator='\n')
            expected = b',A,B\none,1,4\ntwo,2,5\nthree,3,6\n'
            with open(path, mode='rb') as f:
                assert (f.read() == expected)
        with tm.ensure_clean() as path:
            df.to_csv(path)
            os_linesep = os.linesep.encode('utf-8')
            expected = (((((((b',A,B' + os_linesep) + b'one,1,4') + os_linesep) + b'two,2,5') + os_linesep) + b'three,3,6') + os_linesep)
            with open(path, mode='rb') as f:
                assert (f.read() == expected)

    def test_to_csv_from_csv_categorical(self):
        s = Series(pd.Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']))
        s2 = Series(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'])
        res = StringIO()
        s.to_csv(res, header=False)
        exp = StringIO()
        s2.to_csv(exp, header=False)
        assert (res.getvalue() == exp.getvalue())
        df = DataFrame({'s': s})
        df2 = DataFrame({'s': s2})
        res = StringIO()
        df.to_csv(res)
        exp = StringIO()
        df2.to_csv(exp)
        assert (res.getvalue() == exp.getvalue())

    def test_to_csv_path_is_none(self, float_frame):
        csv_str = float_frame.to_csv(path_or_buf=None)
        assert isinstance(csv_str, str)
        recons = pd.read_csv(StringIO(csv_str), index_col=0)
        tm.assert_frame_equal(float_frame, recons)

    @pytest.mark.parametrize('df,encoding', [(DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z']), None), (DataFrame([['abc', 'def', 'ghi']], columns=['X', 'Y', 'Z']), 'ascii'), (DataFrame((5 * [[123, '你好', '世界']]), columns=['X', 'Y', 'Z']), 'gb2312'), (DataFrame((5 * [[123, 'Γειά σου', 'Κόσμε']]), columns=['X', 'Y', 'Z']), 'cp737')])
    def test_to_csv_compression(self, df, encoding, compression):
        with tm.ensure_clean() as filename:
            df.to_csv(filename, compression=compression, encoding=encoding)
            result = read_csv(filename, compression=compression, index_col=0, encoding=encoding)
            tm.assert_frame_equal(df, result)
            with get_handle(filename, 'w', compression=compression, encoding=encoding) as handles:
                df.to_csv(handles.handle, encoding=encoding)
                assert (not handles.handle.closed)
            result = pd.read_csv(filename, compression=compression, encoding=encoding, index_col=0, squeeze=True)
            tm.assert_frame_equal(df, result)
            with tm.decompress_file(filename, compression) as fh:
                text = fh.read().decode((encoding or 'utf8'))
                for col in df.columns:
                    assert (col in text)
            with tm.decompress_file(filename, compression) as fh:
                tm.assert_frame_equal(df, read_csv(fh, index_col=0, encoding=encoding))

    def test_to_csv_date_format(self, datetime_frame):
        with tm.ensure_clean('__tmp_to_csv_date_format__') as path:
            dt_index = datetime_frame.index
            datetime_frame = DataFrame({'A': dt_index, 'B': dt_index.shift(1)}, index=dt_index)
            datetime_frame.to_csv(path, date_format='%Y%m%d')
            test = read_csv(path, index_col=0)
            datetime_frame_int = datetime_frame.applymap((lambda x: int(x.strftime('%Y%m%d'))))
            datetime_frame_int.index = datetime_frame_int.index.map((lambda x: int(x.strftime('%Y%m%d'))))
            tm.assert_frame_equal(test, datetime_frame_int)
            datetime_frame.to_csv(path, date_format='%Y-%m-%d')
            test = read_csv(path, index_col=0)
            datetime_frame_str = datetime_frame.applymap((lambda x: x.strftime('%Y-%m-%d')))
            datetime_frame_str.index = datetime_frame_str.index.map((lambda x: x.strftime('%Y-%m-%d')))
            tm.assert_frame_equal(test, datetime_frame_str)
            datetime_frame_columns = datetime_frame.T
            datetime_frame_columns.to_csv(path, date_format='%Y%m%d')
            test = read_csv(path, index_col=0)
            datetime_frame_columns = datetime_frame_columns.applymap((lambda x: int(x.strftime('%Y%m%d'))))
            datetime_frame_columns.columns = datetime_frame_columns.columns.map((lambda x: x.strftime('%Y%m%d')))
            tm.assert_frame_equal(test, datetime_frame_columns)
            nat_index = to_datetime(((['NaT'] * 10) + ['2000-01-01', '1/1/2000', '1-1-2000']))
            nat_frame = DataFrame({'A': nat_index}, index=nat_index)
            nat_frame.to_csv(path, date_format='%Y-%m-%d')
            test = read_csv(path, parse_dates=[0, 1], index_col=0)
            tm.assert_frame_equal(test, nat_frame)

    def test_to_csv_with_dst_transitions(self):
        with tm.ensure_clean('csv_date_format_with_dst') as path:
            times = pd.date_range('2013-10-26 23:00', '2013-10-27 01:00', tz='Europe/London', freq='H', ambiguous='infer')
            for i in [times, (times + pd.Timedelta('10s'))]:
                i = i._with_freq(None)
                time_range = np.array(range(len(i)), dtype='int64')
                df = DataFrame({'A': time_range}, index=i)
                df.to_csv(path, index=True)
                result = read_csv(path, index_col=0)
                result.index = to_datetime(result.index, utc=True).tz_convert('Europe/London')
                tm.assert_frame_equal(result, df)
        idx = pd.date_range('2015-01-01', '2015-12-31', freq='H', tz='Europe/Paris')
        idx = idx._with_freq(None)
        idx._data._freq = None
        df = DataFrame({'values': 1, 'idx': idx}, index=idx)
        with tm.ensure_clean('csv_date_format_with_dst') as path:
            df.to_csv(path, index=True)
            result = read_csv(path, index_col=0)
            result.index = to_datetime(result.index, utc=True).tz_convert('Europe/Paris')
            result['idx'] = to_datetime(result['idx'], utc=True).astype('datetime64[ns, Europe/Paris]')
            tm.assert_frame_equal(result, df)
        df.astype(str)
        with tm.ensure_clean('csv_date_format_with_dst') as path:
            df.to_pickle(path)
            result = pd.read_pickle(path)
            tm.assert_frame_equal(result, df)

    def test_to_csv_quoting(self):
        df = DataFrame({'c_bool': [True, False], 'c_float': [1.0, 3.2], 'c_int': [42, np.nan], 'c_string': ['a', 'b,c']})
        expected_rows = [',c_bool,c_float,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,"b,c"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv()
        assert (result == expected)
        result = df.to_csv(quoting=None)
        assert (result == expected)
        expected_rows = [',c_bool,c_float,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,"b,c"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_MINIMAL)
        assert (result == expected)
        expected_rows = ['"","c_bool","c_float","c_int","c_string"', '"0","True","1.0","42.0","a"', '"1","False","3.2","","b,c"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_ALL)
        assert (result == expected)
        expected_rows = ['"","c_bool","c_float","c_int","c_string"', '0,True,1.0,42.0,"a"', '1,False,3.2,"","b,c"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONNUMERIC)
        assert (result == expected)
        msg = 'need to escape, but no escapechar set'
        with pytest.raises(csv.Error, match=msg):
            df.to_csv(quoting=csv.QUOTE_NONE)
        with pytest.raises(csv.Error, match=msg):
            df.to_csv(quoting=csv.QUOTE_NONE, escapechar=None)
        expected_rows = [',c_bool,c_float,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,b!,c']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONE, escapechar='!')
        assert (result == expected)
        expected_rows = [',c_bool,c_ffloat,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,bf,c']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONE, escapechar='f')
        assert (result == expected)
        text_rows = ['a,b,c', '1,"test \r\n",3']
        text = tm.convert_rows_list_to_csv_str(text_rows)
        df = pd.read_csv(StringIO(text))
        buf = StringIO()
        df.to_csv(buf, encoding='utf-8', index=False)
        assert (buf.getvalue() == text)
        df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        df = df.set_index(['a', 'b'])
        expected_rows = ['"a","b","c"', '"1","3","5"', '"2","4","6"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (df.to_csv(quoting=csv.QUOTE_ALL) == expected)

    def test_period_index_date_overflow(self):
        dates = ['1990-01-01', '2000-01-01', '3005-01-01']
        index = pd.PeriodIndex(dates, freq='D')
        df = DataFrame([4, 5, 6], index=index)
        result = df.to_csv()
        expected_rows = [',0', '1990-01-01,4', '2000-01-01,5', '3005-01-01,6']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (result == expected)
        date_format = '%m-%d-%Y'
        result = df.to_csv(date_format=date_format)
        expected_rows = [',0', '01-01-1990,4', '01-01-2000,5', '01-01-3005,6']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (result == expected)
        dates = ['1990-01-01', pd.NaT, '3005-01-01']
        index = pd.PeriodIndex(dates, freq='D')
        df = DataFrame([4, 5, 6], index=index)
        result = df.to_csv()
        expected_rows = [',0', '1990-01-01,4', ',5', '3005-01-01,6']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (result == expected)

    def test_multi_index_header(self):
        columns = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
        df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
        df.columns = columns
        header = ['a', 'b', 'c', 'd']
        result = df.to_csv(header=header)
        expected_rows = [',a,b,c,d', '0,1,2,3,4', '1,5,6,7,8']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (result == expected)

    def test_to_csv_single_level_multi_index(self):
        index = Index([(1,), (2,), (3,)])
        df = DataFrame([[1, 2, 3]], columns=index)
        df = df.reindex(columns=[(1,), (3,)])
        expected = ',1,3\n0,1,3\n'
        result = df.to_csv(line_terminator='\n')
        tm.assert_almost_equal(result, expected)

    def test_gz_lineend(self):
        df = DataFrame({'a': [1, 2]})
        expected_rows = ['a', '1', '2']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        with tm.ensure_clean('__test_gz_lineend.csv.gz') as path:
            df.to_csv(path, index=False)
            with tm.decompress_file(path, compression='gzip') as f:
                result = f.read().decode('utf-8')
        assert (result == expected)

    def test_to_csv_numpy_16_bug(self):
        frame = DataFrame({'a': date_range('1/1/2000', periods=10)})
        buf = StringIO()
        frame.to_csv(buf)
        result = buf.getvalue()
        assert ('2000-01-01' in result)
