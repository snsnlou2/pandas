
'\nTests multithreading behaviour for reading and\nparsing files for each parser defined in parsers.py\n'
from contextlib import ExitStack
from io import BytesIO
from multiprocessing.pool import ThreadPool
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

def _construct_dataframe(num_rows):
    '\n    Construct a DataFrame for testing.\n\n    Parameters\n    ----------\n    num_rows : int\n        The number of rows for our DataFrame.\n\n    Returns\n    -------\n    df : DataFrame\n    '
    df = DataFrame(np.random.rand(num_rows, 5), columns=list('abcde'))
    df['foo'] = 'foo'
    df['bar'] = 'bar'
    df['baz'] = 'baz'
    df['date'] = pd.date_range('20000101 09:00:00', periods=num_rows, freq='s')
    df['int'] = np.arange(num_rows, dtype='int64')
    return df

@pytest.mark.slow
def test_multi_thread_string_io_read_csv(all_parsers):
    parser = all_parsers
    max_row_range = 10000
    num_files = 100
    bytes_to_df = ['\n'.join([f'{i:d},{i:d},{i:d}' for i in range(max_row_range)]).encode() for _ in range(num_files)]
    with ExitStack() as stack:
        files = [stack.enter_context(BytesIO(b)) for b in bytes_to_df]
        pool = stack.enter_context(ThreadPool(8))
        results = pool.map(parser.read_csv, files)
        first_result = results[0]
        for result in results:
            tm.assert_frame_equal(first_result, result)

def _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks):
    '\n    Generate a DataFrame via multi-thread.\n\n    Parameters\n    ----------\n    parser : BaseParser\n        The parser object to use for reading the data.\n    path : str\n        The location of the CSV file to read.\n    num_rows : int\n        The number of rows to read per task.\n    num_tasks : int\n        The number of tasks to use for reading this DataFrame.\n\n    Returns\n    -------\n    df : DataFrame\n    '

    def reader(arg):
        '\n        Create a reader for part of the CSV.\n\n        Parameters\n        ----------\n        arg : tuple\n            A tuple of the following:\n\n            * start : int\n                The starting row to start for parsing CSV\n            * nrows : int\n                The number of rows to read.\n\n        Returns\n        -------\n        df : DataFrame\n        '
        (start, nrows) = arg
        if (not start):
            return parser.read_csv(path, index_col=0, header=0, nrows=nrows, parse_dates=['date'])
        return parser.read_csv(path, index_col=0, header=None, skiprows=(int(start) + 1), nrows=nrows, parse_dates=[9])
    tasks = [(((num_rows * i) // num_tasks), (num_rows // num_tasks)) for i in range(num_tasks)]
    with ThreadPool(processes=num_tasks) as pool:
        results = pool.map(reader, tasks)
    header = results[0].columns
    for r in results[1:]:
        r.columns = header
    final_dataframe = pd.concat(results)
    return final_dataframe

@pytest.mark.slow
def test_multi_thread_path_multipart_read_csv(all_parsers):
    num_tasks = 4
    num_rows = 100000
    parser = all_parsers
    file_name = '__thread_pool_reader__.csv'
    df = _construct_dataframe(num_rows)
    with tm.ensure_clean(file_name) as path:
        df.to_csv(path)
        final_dataframe = _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks)
        tm.assert_frame_equal(df, final_dataframe)
