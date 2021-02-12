
import pytest
import pandas.util._test_decorators as td
import pandas._testing as tm
from pandas.io.parsers import read_csv

@pytest.fixture
def frame(float_frame):
    '\n    Returns the first ten items in fixture "float_frame".\n    '
    return float_frame[:10]

@pytest.fixture
def tsframe():
    return tm.makeTimeDataFrame()[:5]

@pytest.fixture(params=[True, False])
def merge_cells(request):
    return request.param

@pytest.fixture
def df_ref(datapath):
    '\n    Obtain the reference data from read_csv with the Python engine.\n    '
    filepath = datapath('io', 'data', 'csv', 'test1.csv')
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine='python')
    return df_ref

@pytest.fixture(params=['.xls', '.xlsx', '.xlsm', '.ods', '.xlsb'])
def read_ext(request):
    '\n    Valid extensions for reading Excel files.\n    '
    return request.param

@pytest.fixture(autouse=True)
def check_for_file_leaks():
    '\n    Fixture to run around every test to ensure that we are not leaking files.\n\n    See also\n    --------\n    _test_decorators.check_file_leaks\n    '
    psutil = td.safe_import('psutil')
    if (not psutil):
        (yield)
    else:
        proc = psutil.Process()
        flist = proc.open_files()
        (yield)
        flist2 = proc.open_files()
        assert (flist == flist2)
