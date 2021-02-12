
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas

def numeric_as_float(data):
    for v in data.columns:
        if (data[v].dtype is np.dtype('int64')):
            data[v] = data[v].astype(np.float64)

class TestXport():

    @pytest.fixture(autouse=True)
    def setup_method(self, datapath):
        self.dirpath = datapath('io', 'sas', 'data')
        self.file01 = os.path.join(self.dirpath, 'DEMO_G.xpt')
        self.file02 = os.path.join(self.dirpath, 'SSHSV1_A.xpt')
        self.file03 = os.path.join(self.dirpath, 'DRXFCD_G.xpt')
        self.file04 = os.path.join(self.dirpath, 'paxraw_d_short.xpt')
        with td.file_leak_context():
            (yield)

    def test1_basic(self):
        data_csv = pd.read_csv(self.file01.replace('.xpt', '.csv'))
        numeric_as_float(data_csv)
        data = read_sas(self.file01, format='xport')
        tm.assert_frame_equal(data, data_csv)
        num_rows = data.shape[0]
        with read_sas(self.file01, format='xport', iterator=True) as reader:
            data = reader.read((num_rows + 100))
        assert (data.shape[0] == num_rows)
        with read_sas(self.file01, format='xport', iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])
        with read_sas(self.file01, format='xport', chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])
        m = 0
        with read_sas(self.file01, format='xport', chunksize=100) as reader:
            for x in reader:
                m += x.shape[0]
        assert (m == num_rows)
        data = read_sas(self.file01)
        tm.assert_frame_equal(data, data_csv)

    def test1_index(self):
        data_csv = pd.read_csv(self.file01.replace('.xpt', '.csv'))
        data_csv = data_csv.set_index('SEQN')
        numeric_as_float(data_csv)
        data = read_sas(self.file01, index='SEQN', format='xport')
        tm.assert_frame_equal(data, data_csv, check_index_type=False)
        with read_sas(self.file01, index='SEQN', format='xport', iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)
        with read_sas(self.file01, index='SEQN', format='xport', chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

    def test1_incremental(self):
        data_csv = pd.read_csv(self.file01.replace('.xpt', '.csv'))
        data_csv = data_csv.set_index('SEQN')
        numeric_as_float(data_csv)
        with read_sas(self.file01, index='SEQN', chunksize=1000) as reader:
            all_data = list(reader)
        data = pd.concat(all_data, axis=0)
        tm.assert_frame_equal(data, data_csv, check_index_type=False)

    def test2(self):
        data_csv = pd.read_csv(self.file02.replace('.xpt', '.csv'))
        numeric_as_float(data_csv)
        data = read_sas(self.file02)
        tm.assert_frame_equal(data, data_csv)

    def test2_binary(self):
        data_csv = pd.read_csv(self.file02.replace('.xpt', '.csv'))
        numeric_as_float(data_csv)
        with open(self.file02, 'rb') as fd:
            with td.file_leak_context():
                data = read_sas(fd, format='xport')
        tm.assert_frame_equal(data, data_csv)

    def test_multiple_types(self):
        data_csv = pd.read_csv(self.file03.replace('.xpt', '.csv'))
        data = read_sas(self.file03, encoding='utf-8')
        tm.assert_frame_equal(data, data_csv)

    def test_truncated_float_support(self):
        data_csv = pd.read_csv(self.file04.replace('.xpt', '.csv'))
        data = read_sas(self.file04, format='xport')
        tm.assert_frame_equal(data.astype('int64'), data_csv)
