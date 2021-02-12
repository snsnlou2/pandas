
from datetime import datetime
import io
import os
from pathlib import Path
import dateutil.parser
import numpy as np
import pytest
from pandas.errors import EmptyDataError, PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm

@pytest.mark.filterwarnings("ignore:can't resolve package:ImportWarning")
class TestSAS7BDAT():

    @pytest.fixture(autouse=True)
    def setup_method(self, datapath):
        self.dirpath = datapath('io', 'sas', 'data')
        self.data = []
        self.test_ix = [list(range(1, 16)), [16]]
        for j in (1, 2):
            fname = os.path.join(self.dirpath, f'test_sas7bdat_{j}.csv')
            df = pd.read_csv(fname)
            epoch = datetime(1960, 1, 1)
            t1 = pd.to_timedelta(df['Column4'], unit='d')
            df['Column4'] = (epoch + t1)
            t2 = pd.to_timedelta(df['Column12'], unit='d')
            df['Column12'] = (epoch + t2)
            for k in range(df.shape[1]):
                col = df.iloc[:, k]
                if (col.dtype == np.int64):
                    df.iloc[:, k] = df.iloc[:, k].astype(np.float64)
            self.data.append(df)

    def test_from_file(self):
        for j in (0, 1):
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = os.path.join(self.dirpath, f'test{k}.sas7bdat')
                df = pd.read_sas(fname, encoding='utf-8')
                tm.assert_frame_equal(df, df0)

    def test_from_buffer(self):
        for j in (0, 1):
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = os.path.join(self.dirpath, f'test{k}.sas7bdat')
                with open(fname, 'rb') as f:
                    byts = f.read()
                buf = io.BytesIO(byts)
                with pd.read_sas(buf, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
                    df = rdr.read()
                tm.assert_frame_equal(df, df0, check_exact=False)

    def test_from_iterator(self):
        for j in (0, 1):
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = os.path.join(self.dirpath, f'test{k}.sas7bdat')
                with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
                    df = rdr.read(2)
                    tm.assert_frame_equal(df, df0.iloc[0:2, :])
                    df = rdr.read(3)
                    tm.assert_frame_equal(df, df0.iloc[2:5, :])

    def test_path_pathlib(self):
        for j in (0, 1):
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = Path(os.path.join(self.dirpath, f'test{k}.sas7bdat'))
                df = pd.read_sas(fname, encoding='utf-8')
                tm.assert_frame_equal(df, df0)

    @td.skip_if_no('py.path')
    def test_path_localpath(self):
        from py.path import local as LocalPath
        for j in (0, 1):
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = LocalPath(os.path.join(self.dirpath, f'test{k}.sas7bdat'))
                df = pd.read_sas(fname, encoding='utf-8')
                tm.assert_frame_equal(df, df0)

    def test_iterator_loop(self):
        for j in (0, 1):
            for k in self.test_ix[j]:
                for chunksize in (3, 5, 10, 11):
                    fname = os.path.join(self.dirpath, f'test{k}.sas7bdat')
                    with pd.read_sas(fname, chunksize=10, encoding='utf-8') as rdr:
                        y = 0
                        for x in rdr:
                            y += x.shape[0]
                    assert (y == rdr.row_count)

    def test_iterator_read_too_much(self):
        k = self.test_ix[0][0]
        fname = os.path.join(self.dirpath, f'test{k}.sas7bdat')
        with pd.read_sas(fname, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
            d1 = rdr.read((rdr.row_count + 20))
        with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
            d2 = rdr.read((rdr.row_count + 20))
        tm.assert_frame_equal(d1, d2)

def test_encoding_options(datapath):
    fname = datapath('io', 'sas', 'data', 'test1.sas7bdat')
    df1 = pd.read_sas(fname)
    df2 = pd.read_sas(fname, encoding='utf-8')
    for col in df1.columns:
        try:
            df1[col] = df1[col].str.decode('utf-8')
        except AttributeError:
            pass
    tm.assert_frame_equal(df1, df2)
    from pandas.io.sas.sas7bdat import SAS7BDATReader
    rdr = SAS7BDATReader(fname, convert_header_text=False)
    df3 = rdr.read()
    rdr.close()
    for (x, y) in zip(df1.columns, df3.columns):
        assert (x == y.decode())

def test_productsales(datapath):
    fname = datapath('io', 'sas', 'data', 'productsales.sas7bdat')
    df = pd.read_sas(fname, encoding='utf-8')
    fname = datapath('io', 'sas', 'data', 'productsales.csv')
    df0 = pd.read_csv(fname, parse_dates=['MONTH'])
    vn = ['ACTUAL', 'PREDICT', 'QUARTER', 'YEAR']
    df0[vn] = df0[vn].astype(np.float64)
    tm.assert_frame_equal(df, df0)

def test_12659(datapath):
    fname = datapath('io', 'sas', 'data', 'test_12659.sas7bdat')
    df = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'test_12659.csv')
    df0 = pd.read_csv(fname)
    df0 = df0.astype(np.float64)
    tm.assert_frame_equal(df, df0)

def test_airline(datapath):
    fname = datapath('io', 'sas', 'data', 'airline.sas7bdat')
    df = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'airline.csv')
    df0 = pd.read_csv(fname)
    df0 = df0.astype(np.float64)
    tm.assert_frame_equal(df, df0, check_exact=False)

def test_date_time(datapath):
    fname = datapath('io', 'sas', 'data', 'datetime.sas7bdat')
    df = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'datetime.csv')
    df0 = pd.read_csv(fname, parse_dates=['Date1', 'Date2', 'DateTime', 'DateTimeHi', 'Taiw'])
    df.iloc[:, 3] = df.iloc[:, 3].dt.round('us')
    tm.assert_frame_equal(df, df0)

def test_compact_numerical_values(datapath):
    fname = datapath('io', 'sas', 'data', 'cars.sas7bdat')
    df = pd.read_sas(fname, encoding='latin-1')
    result = df['WGT']
    expected = df['WGT'].round()
    tm.assert_series_equal(result, expected, check_exact=True)
    result = df['CYL']
    expected = df['CYL'].round()
    tm.assert_series_equal(result, expected, check_exact=True)

def test_many_columns(datapath):
    fname = datapath('io', 'sas', 'data', 'many_columns.sas7bdat')
    with tm.assert_produces_warning(PerformanceWarning):
        df = pd.read_sas(fname, encoding='latin-1')
    fname = datapath('io', 'sas', 'data', 'many_columns.csv')
    df0 = pd.read_csv(fname, encoding='latin-1')
    tm.assert_frame_equal(df, df0)

def test_inconsistent_number_of_rows(datapath):
    fname = datapath('io', 'sas', 'data', 'load_log.sas7bdat')
    df = pd.read_sas(fname, encoding='latin-1')
    assert (len(df) == 2097)

def test_zero_variables(datapath):
    fname = datapath('io', 'sas', 'data', 'zero_variables.sas7bdat')
    with pytest.raises(EmptyDataError, match='No columns to parse from file'):
        pd.read_sas(fname)

def test_corrupt_read(datapath):
    fname = datapath('io', 'sas', 'data', 'corrupt.sas7bdat')
    msg = "'SAS7BDATReader' object has no attribute 'row_count'"
    with pytest.raises(AttributeError, match=msg):
        pd.read_sas(fname)

def round_datetime_to_ms(ts):
    if isinstance(ts, datetime):
        return ts.replace(microsecond=(int((round(ts.microsecond, (- 3)) / 1000)) * 1000))
    elif isinstance(ts, str):
        _ts = dateutil.parser.parse(timestr=ts)
        return _ts.replace(microsecond=(int((round(_ts.microsecond, (- 3)) / 1000)) * 1000))
    else:
        return ts

def test_max_sas_date(datapath):
    fname = datapath('io', 'sas', 'data', 'max_sas_date.sas7bdat')
    df = pd.read_sas(fname, encoding='iso-8859-1')
    df = df.applymap((lambda x: (x.lstrip() if isinstance(x, str) else x)))
    try:
        df['dt_as_dt'] = df['dt_as_dt'].dt.round('us')
    except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
        df = df.applymap(round_datetime_to_ms)
    except AttributeError:
        df['dt_as_dt'] = df['dt_as_dt'].apply(round_datetime_to_ms)
    expected = pd.DataFrame({'text': ['max', 'normal'], 'dt_as_float': [253717747199.999, 1880323199.999], 'dt_as_dt': [datetime(9999, 12, 29, 23, 59, 59, 999000), datetime(2019, 8, 1, 23, 59, 59, 999000)], 'date_as_float': [2936547.0, 21762.0], 'date_as_date': [datetime(9999, 12, 29), datetime(2019, 8, 1)]}, columns=['text', 'dt_as_float', 'dt_as_dt', 'date_as_float', 'date_as_date'])
    tm.assert_frame_equal(df, expected)

def test_max_sas_date_iterator(datapath):
    col_order = ['text', 'dt_as_float', 'dt_as_dt', 'date_as_float', 'date_as_date']
    fname = datapath('io', 'sas', 'data', 'max_sas_date.sas7bdat')
    results = []
    for df in pd.read_sas(fname, encoding='iso-8859-1', chunksize=1):
        df = df.applymap((lambda x: (x.lstrip() if isinstance(x, str) else x)))
        try:
            df['dt_as_dt'] = df['dt_as_dt'].dt.round('us')
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            df = df.applymap(round_datetime_to_ms)
        except AttributeError:
            df['dt_as_dt'] = df['dt_as_dt'].apply(round_datetime_to_ms)
        df.reset_index(inplace=True, drop=True)
        results.append(df)
    expected = [pd.DataFrame({'text': ['max'], 'dt_as_float': [253717747199.999], 'dt_as_dt': [datetime(9999, 12, 29, 23, 59, 59, 999000)], 'date_as_float': [2936547.0], 'date_as_date': [datetime(9999, 12, 29)]}, columns=col_order), pd.DataFrame({'text': ['normal'], 'dt_as_float': [1880323199.999], 'dt_as_dt': [np.datetime64('2019-08-01 23:59:59.999')], 'date_as_float': [21762.0], 'date_as_date': [np.datetime64('2019-08-01')]}, columns=col_order)]
    for (result, expected) in zip(results, expected):
        tm.assert_frame_equal(result, expected)
