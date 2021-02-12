
"\nThis file is very long and growing, but it was decided to not split it yet, as\nit's still manageable (2020-03-17, ~1.1k LoC). See gh-31989\n\nInstead of splitting it was decided to define sections here:\n- Configuration / Settings\n- Autouse fixtures\n- Common arguments\n- Missing values & co.\n- Classes\n- Indices\n- Series'\n- DataFrames\n- Operators & Operations\n- Data sets/files\n- Time zones\n- Dtypes\n- Misc\n"
from collections import abc
from datetime import date, time, timedelta, timezone
from decimal import Decimal
import operator
import os
from dateutil.tz import tzlocal, tzutc
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import FixedOffset, utc
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import DatetimeTZDtype, IntervalDtype
import pandas as pd
from pandas import DataFrame, Interval, Period, Series, Timedelta, Timestamp
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import Index, MultiIndex

def pytest_configure(config):
    config.addinivalue_line('markers', 'single: mark a test as single cpu only')
    config.addinivalue_line('markers', 'slow: mark a test as slow')
    config.addinivalue_line('markers', 'network: mark a test as network')
    config.addinivalue_line('markers', 'db: tests requiring a database (mysql or postgres)')
    config.addinivalue_line('markers', 'high_memory: mark a test as a high-memory only')
    config.addinivalue_line('markers', 'clipboard: mark a pd.read_clipboard test')
    config.addinivalue_line('markers', 'arm_slow: mark a test as slow for arm64 architecture')

def pytest_addoption(parser):
    parser.addoption('--skip-slow', action='store_true', help='skip slow tests')
    parser.addoption('--skip-network', action='store_true', help='skip network tests')
    parser.addoption('--skip-db', action='store_true', help='skip db tests')
    parser.addoption('--run-high-memory', action='store_true', help='run high memory tests')
    parser.addoption('--only-slow', action='store_true', help='run only slow tests')
    parser.addoption('--strict-data-files', action='store_true', help='Fail if a test is skipped for missing data file.')

def pytest_runtest_setup(item):
    if (('slow' in item.keywords) and item.config.getoption('--skip-slow')):
        pytest.skip('skipping due to --skip-slow')
    if (('slow' not in item.keywords) and item.config.getoption('--only-slow')):
        pytest.skip('skipping due to --only-slow')
    if (('network' in item.keywords) and item.config.getoption('--skip-network')):
        pytest.skip('skipping due to --skip-network')
    if (('db' in item.keywords) and item.config.getoption('--skip-db')):
        pytest.skip('skipping due to --skip-db')
    if (('high_memory' in item.keywords) and (not item.config.getoption('--run-high-memory'))):
        pytest.skip('skipping high memory test since --run-high-memory was not set')
hypothesis.settings.register_profile('ci', deadline=500, suppress_health_check=(hypothesis.HealthCheck.too_slow,))
hypothesis.settings.load_profile('ci')
for name in 'MonthBegin MonthEnd BMonthBegin BMonthEnd'.split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(cls, st.builds(cls, n=st.integers((- 99), 99), normalize=st.booleans()))
for name in 'YearBegin YearEnd BYearBegin BYearEnd'.split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(cls, st.builds(cls, n=st.integers((- 5), 5), normalize=st.booleans(), month=st.integers(min_value=1, max_value=12)))
for name in 'QuarterBegin QuarterEnd BQuarterBegin BQuarterEnd'.split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(cls, st.builds(cls, n=st.integers((- 24), 24), normalize=st.booleans(), startingMonth=st.integers(min_value=1, max_value=12)))

@pytest.fixture(autouse=True)
def configure_tests():
    '\n    Configure settings for all tests and test modules.\n    '
    pd.set_option('chained_assignment', 'raise')

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    '\n    Make `np` and `pd` names available for doctests.\n    '
    doctest_namespace['np'] = np
    doctest_namespace['pd'] = pd

@pytest.fixture(params=[0, 1, 'index', 'columns'], ids=(lambda x: f'axis {repr(x)}'))
def axis(request):
    '\n    Fixture for returning the axis numbers of a DataFrame.\n    '
    return request.param
axis_frame = axis

@pytest.fixture(params=[True, False, None])
def observed(request):
    '\n    Pass in the observed keyword to groupby for [True, False]\n    This indicates whether categoricals should return values for\n    values which are not in the grouper [False / None], or only values which\n    appear in the grouper [True]. [None] is supported for future compatibility\n    if we decide to change the default (and would need to warn if this\n    parameter is not passed).\n    '
    return request.param

@pytest.fixture(params=[True, False, None])
def ordered(request):
    "\n    Boolean 'ordered' parameter for Categorical.\n    "
    return request.param

@pytest.fixture(params=['first', 'last', False])
def keep(request):
    "\n    Valid values for the 'keep' parameter used in\n    .duplicated or .drop_duplicates\n    "
    return request.param

@pytest.fixture(params=['left', 'right', 'both', 'neither'])
def closed(request):
    '\n    Fixture for trying all interval closed parameters.\n    '
    return request.param

@pytest.fixture(params=['left', 'right', 'both', 'neither'])
def other_closed(request):
    '\n    Secondary closed fixture to allow parametrizing over all pairs of closed.\n    '
    return request.param

@pytest.fixture(params=[None, 'gzip', 'bz2', 'zip', 'xz'])
def compression(request):
    '\n    Fixture for trying common compression types in compression tests.\n    '
    return request.param

@pytest.fixture(params=['gzip', 'bz2', 'zip', 'xz'])
def compression_only(request):
    '\n    Fixture for trying common compression types in compression tests excluding\n    uncompressed case.\n    '
    return request.param

@pytest.fixture(params=[True, False])
def writable(request):
    '\n    Fixture that an array is writable.\n    '
    return request.param

@pytest.fixture(params=['inner', 'outer', 'left', 'right'])
def join_type(request):
    '\n    Fixture for trying all types of join operations.\n    '
    return request.param

@pytest.fixture(params=['nlargest', 'nsmallest'])
def nselect_method(request):
    '\n    Fixture for trying all nselect methods.\n    '
    return request.param

@pytest.fixture(params=tm.NULL_OBJECTS, ids=str)
def nulls_fixture(request):
    '\n    Fixture for each null type in pandas.\n    '
    return request.param
nulls_fixture2 = nulls_fixture

@pytest.fixture(params=[None, np.nan, pd.NaT])
def unique_nulls_fixture(request):
    '\n    Fixture for each null type in pandas, each null type exactly once.\n    '
    return request.param
unique_nulls_fixture2 = unique_nulls_fixture

@pytest.fixture(params=[pd.DataFrame, pd.Series])
def frame_or_series(request):
    '\n    Fixture to parametrize over DataFrame and Series.\n    '
    return request.param

@pytest.fixture(params=[pd.Index, pd.Series], ids=['index', 'series'])
def index_or_series(request):
    '\n    Fixture to parametrize over Index and Series, made necessary by a mypy\n    bug, giving an error:\n\n    List item 0 has incompatible type "Type[Series]"; expected "Type[PandasObject]"\n\n    See GH#29725\n    '
    return request.param
index_or_series2 = index_or_series

@pytest.fixture(params=[pd.Index, pd.Series, pd.array], ids=['index', 'series', 'array'])
def index_or_series_or_array(request):
    '\n    Fixture to parametrize over Index, Series, and ExtensionArray\n    '
    return request.param

@pytest.fixture
def dict_subclass():
    '\n    Fixture for a dictionary subclass.\n    '

    class TestSubDict(dict):

        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)
    return TestSubDict

@pytest.fixture
def non_dict_mapping_subclass():
    '\n    Fixture for a non-mapping dictionary subclass.\n    '

    class TestNonDictMapping(abc.Mapping):

        def __init__(self, underlying_dict):
            self._data = underlying_dict

        def __getitem__(self, key):
            return self._data.__getitem__(key)

        def __iter__(self):
            return self._data.__iter__()

        def __len__(self):
            return self._data.__len__()
    return TestNonDictMapping

@pytest.fixture
def multiindex_year_month_day_dataframe_random_data():
    '\n    DataFrame with 3 level MultiIndex (year, month, day) covering\n    first 100 business days from 2000-01-01 with random data\n    '
    tdf = tm.makeTimeDataFrame(100)
    ymd = tdf.groupby([(lambda x: x.year), (lambda x: x.month), (lambda x: x.day)]).sum()
    ymd.index = ymd.index.set_levels([lev.astype('i8') for lev in ymd.index.levels])
    ymd.index.set_names(['year', 'month', 'day'], inplace=True)
    return ymd

@pytest.fixture
def multiindex_dataframe_random_data():
    'DataFrame with 2 level MultiIndex with random data'
    index = MultiIndex(levels=[['foo', 'bar', 'baz', 'qux'], ['one', 'two', 'three']], codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]], names=['first', 'second'])
    return DataFrame(np.random.randn(10, 3), index=index, columns=Index(['A', 'B', 'C'], name='exp'))

def _create_multiindex():
    '\n    MultiIndex used to test the general functionality of this object\n    '
    major_axis = Index(['foo', 'bar', 'baz', 'qux'])
    minor_axis = Index(['one', 'two'])
    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ['first', 'second']
    return MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes], names=index_names, verify_integrity=False)

def _create_mi_with_dt64tz_level():
    '\n    MultiIndex with a level that is a tzaware DatetimeIndex.\n    '
    return MultiIndex.from_product([[1, 2], ['a', 'b'], pd.date_range('20130101', periods=3, tz='US/Eastern')], names=['one', 'two', 'three'])
indices_dict = {'unicode': tm.makeUnicodeIndex(100), 'string': tm.makeStringIndex(100), 'datetime': tm.makeDateIndex(100), 'datetime-tz': tm.makeDateIndex(100, tz='US/Pacific'), 'period': tm.makePeriodIndex(100), 'timedelta': tm.makeTimedeltaIndex(100), 'int': tm.makeIntIndex(100), 'uint': tm.makeUIntIndex(100), 'range': tm.makeRangeIndex(100), 'float': tm.makeFloatIndex(100), 'bool': tm.makeBoolIndex(10), 'categorical': tm.makeCategoricalIndex(100), 'interval': tm.makeIntervalIndex(100), 'empty': Index([]), 'tuples': MultiIndex.from_tuples(zip(['foo', 'bar', 'baz'], [1, 2, 3])), 'mi-with-dt64tz-level': _create_mi_with_dt64tz_level(), 'multi': _create_multiindex(), 'repeats': Index([0, 0, 1, 1, 2, 2])}

@pytest.fixture(params=indices_dict.keys())
def index(request):
    '\n    Fixture for many "simple" kinds of indices.\n\n    These indices are unlikely to cover corner cases, e.g.\n        - no names\n        - no NaTs/NaNs\n        - no values near implementation bounds\n        - ...\n    '
    return indices_dict[request.param].copy()
index_fixture2 = index

@pytest.fixture(params=indices_dict.keys())
def index_with_missing(request):
    '\n    Fixture for indices with missing values\n    '
    if (request.param in ['int', 'uint', 'range', 'empty', 'repeats']):
        pytest.xfail('missing values not supported')
    ind = indices_dict[request.param].copy(deep=True)
    vals = ind.values
    if (request.param in ['tuples', 'mi-with-dt64tz-level', 'multi']):
        vals = ind.tolist()
        vals[0] = ((None,) + vals[0][1:])
        vals[(- 1)] = ((None,) + vals[(- 1)][1:])
        return MultiIndex.from_tuples(vals)
    else:
        vals[0] = None
        vals[(- 1)] = None
        return type(ind)(vals)

@pytest.fixture
def empty_series():
    return pd.Series([], index=[], dtype=np.float64)

@pytest.fixture
def string_series():
    '\n    Fixture for Series of floats with Index of unique strings\n    '
    s = tm.makeStringSeries()
    s.name = 'series'
    return s

@pytest.fixture
def object_series():
    '\n    Fixture for Series of dtype object with Index of unique strings\n    '
    s = tm.makeObjectSeries()
    s.name = 'objects'
    return s

@pytest.fixture
def datetime_series():
    '\n    Fixture for Series of floats with DatetimeIndex\n    '
    s = tm.makeTimeSeries()
    s.name = 'ts'
    return s

def _create_series(index):
    ' Helper for the _series dict '
    size = len(index)
    data = np.random.randn(size)
    return pd.Series(data, index=index, name='a')
_series = {f'series-with-{index_id}-index': _create_series(index) for (index_id, index) in indices_dict.items()}

@pytest.fixture
def series_with_simple_index(index):
    '\n    Fixture for tests on series with changing types of indices.\n    '
    return _create_series(index)

@pytest.fixture
def series_with_multilevel_index():
    '\n    Fixture with a Series with a 2-level MultiIndex.\n    '
    arrays = [['bar', 'bar', 'baz', 'baz', 'qux', 'qux', 'foo', 'foo'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    tuples = zip(*arrays)
    index = MultiIndex.from_tuples(tuples)
    data = np.random.randn(8)
    ser = Series(data, index=index)
    ser[3] = np.NaN
    return ser
_narrow_dtypes = [np.float16, np.float32, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]
_narrow_series = {f'{dtype.__name__}-series': tm.makeFloatSeries(name='a').astype(dtype) for dtype in _narrow_dtypes}

@pytest.fixture(params=_narrow_series.keys())
def narrow_series(request):
    '\n    Fixture for Series with low precision data types\n    '
    return _narrow_series[request.param].copy()
_index_or_series_objs = {**indices_dict, **_series, **_narrow_series}

@pytest.fixture(params=_index_or_series_objs.keys())
def index_or_series_obj(request):
    '\n    Fixture for tests on indexes, series and series with a narrow dtype\n    copy to avoid mutation, e.g. setting .name\n    '
    return _index_or_series_objs[request.param].copy(deep=True)

@pytest.fixture
def empty_frame():
    return DataFrame()

@pytest.fixture
def int_frame():
    "\n    Fixture for DataFrame of ints with index of unique strings\n\n    Columns are ['A', 'B', 'C', 'D']\n\n                A  B  C  D\n    vpBeWjM651  1  0  1  0\n    5JyxmrP1En -1  0  0  0\n    qEDaoD49U2 -1  1  0  0\n    m66TkTfsFe  0  0  0  0\n    EHPaNzEUFm -1  0 -1  0\n    fpRJCevQhi  2  0  0  0\n    OlQvnmfi3Q  0  0 -2  0\n    ...        .. .. .. ..\n    uB1FPlz4uP  0  0  0  1\n    EcSe6yNzCU  0  0 -1  0\n    L50VudaiI8 -1  1 -2  0\n    y3bpw4nwIp  0 -1  0  0\n    H0RdLLwrCT  1  1  0  0\n    rY82K0vMwm  0  0  0  0\n    1OPIUjnkjk  2  0  0  0\n\n    [30 rows x 4 columns]\n    "
    return DataFrame(tm.getSeriesData()).astype('int64')

@pytest.fixture
def datetime_frame():
    "\n    Fixture for DataFrame of floats with DatetimeIndex\n\n    Columns are ['A', 'B', 'C', 'D']\n\n                       A         B         C         D\n    2000-01-03 -1.122153  0.468535  0.122226  1.693711\n    2000-01-04  0.189378  0.486100  0.007864 -1.216052\n    2000-01-05  0.041401 -0.835752 -0.035279 -0.414357\n    2000-01-06  0.430050  0.894352  0.090719  0.036939\n    2000-01-07 -0.620982 -0.668211 -0.706153  1.466335\n    2000-01-10 -0.752633  0.328434 -0.815325  0.699674\n    2000-01-11 -2.236969  0.615737 -0.829076 -1.196106\n    ...              ...       ...       ...       ...\n    2000-02-03  1.642618 -0.579288  0.046005  1.385249\n    2000-02-04 -0.544873 -1.160962 -0.284071 -1.418351\n    2000-02-07 -2.656149 -0.601387  1.410148  0.444150\n    2000-02-08 -1.201881 -1.289040  0.772992 -1.445300\n    2000-02-09  1.377373  0.398619  1.008453 -0.928207\n    2000-02-10  0.473194 -0.636677  0.984058  0.511519\n    2000-02-11 -0.965556  0.408313 -1.312844 -0.381948\n\n    [30 rows x 4 columns]\n    "
    return DataFrame(tm.getTimeSeriesData())

@pytest.fixture
def float_frame():
    "\n    Fixture for DataFrame of floats with index of unique strings\n\n    Columns are ['A', 'B', 'C', 'D'].\n\n                       A         B         C         D\n    P7GACiRnxd -0.465578 -0.361863  0.886172 -0.053465\n    qZKh6afn8n -0.466693 -0.373773  0.266873  1.673901\n    tkp0r6Qble  0.148691 -0.059051  0.174817  1.598433\n    wP70WOCtv8  0.133045 -0.581994 -0.992240  0.261651\n    M2AeYQMnCz -1.207959 -0.185775  0.588206  0.563938\n    QEPzyGDYDo -0.381843 -0.758281  0.502575 -0.565053\n    r78Jwns6dn -0.653707  0.883127  0.682199  0.206159\n    ...              ...       ...       ...       ...\n    IHEGx9NO0T -0.277360  0.113021 -1.018314  0.196316\n    lPMj8K27FA -1.313667 -0.604776 -1.305618 -0.863999\n    qa66YMWQa5  1.110525  0.475310 -0.747865  0.032121\n    yOa0ATsmcE -0.431457  0.067094  0.096567 -0.264962\n    65znX3uRNG  1.528446  0.160416 -0.109635 -0.032987\n    eCOBvKqf3e  0.235281  1.622222  0.781255  0.392871\n    xSucinXxuV -1.263557  0.252799 -0.552247  0.400426\n\n    [30 rows x 4 columns]\n    "
    return DataFrame(tm.getSeriesData())

@pytest.fixture(params=[(Interval(left=0, right=5), IntervalDtype('int64')), (Interval(left=0.1, right=0.5), IntervalDtype('float64')), (Period('2012-01', freq='M'), 'period[M]'), (Period('2012-02-01', freq='D'), 'period[D]'), (Timestamp('2011-01-01', tz='US/Eastern'), DatetimeTZDtype(tz='US/Eastern')), (Timedelta(seconds=500), 'timedelta64[ns]')])
def ea_scalar_and_dtype(request):
    return request.param
_all_arithmetic_operators = ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__floordiv__', '__rfloordiv__', '__truediv__', '__rtruediv__', '__pow__', '__rpow__', '__mod__', '__rmod__']

@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    '\n    Fixture for dunder names for common arithmetic operations.\n    '
    return request.param

@pytest.fixture(params=[operator.add, ops.radd, operator.sub, ops.rsub, operator.mul, ops.rmul, operator.truediv, ops.rtruediv, operator.floordiv, ops.rfloordiv, operator.mod, ops.rmod, operator.pow, ops.rpow, operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge, operator.and_, ops.rand_, operator.xor, ops.rxor, operator.or_, ops.ror_])
def all_binary_operators(request):
    '\n    Fixture for operator and roperator arithmetic, comparison, and logical ops.\n    '
    return request.param

@pytest.fixture(params=[operator.add, ops.radd, operator.sub, ops.rsub, operator.mul, ops.rmul, operator.truediv, ops.rtruediv, operator.floordiv, ops.rfloordiv, operator.mod, ops.rmod, operator.pow, ops.rpow])
def all_arithmetic_functions(request):
    '\n    Fixture for operator and roperator arithmetic functions.\n\n    Notes\n    -----\n    This includes divmod and rdivmod, whereas all_arithmetic_operators\n    does not.\n    '
    return request.param
_all_numeric_reductions = ['sum', 'max', 'min', 'mean', 'prod', 'std', 'var', 'median', 'kurt', 'skew']

@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    '\n    Fixture for numeric reduction names.\n    '
    return request.param
_all_boolean_reductions = ['all', 'any']

@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    '\n    Fixture for boolean reduction names.\n    '
    return request.param
_all_reductions = (_all_numeric_reductions + _all_boolean_reductions)

@pytest.fixture(params=_all_reductions)
def all_reductions(request):
    '\n    Fixture for all (boolean + numeric) reduction names.\n    '
    return request.param

@pytest.fixture(params=['__eq__', '__ne__', '__le__', '__lt__', '__ge__', '__gt__'])
def all_compare_operators(request):
    '\n    Fixture for dunder names for common compare operations\n\n    * >=\n    * >\n    * ==\n    * !=\n    * <\n    * <=\n    '
    return request.param

@pytest.fixture(params=['__le__', '__lt__', '__ge__', '__gt__'])
def compare_operators_no_eq_ne(request):
    '\n    Fixture for dunder names for compare operations except == and !=\n\n    * >=\n    * >\n    * <\n    * <=\n    '
    return request.param

@pytest.fixture(params=['__and__', '__rand__', '__or__', '__ror__', '__xor__', '__rxor__'])
def all_logical_operators(request):
    '\n    Fixture for dunder names for common logical operations\n\n    * |\n    * &\n    * ^\n    '
    return request.param

@pytest.fixture
def strict_data_files(pytestconfig):
    '\n    Returns the configuration for the test setting `--strict-data-files`.\n    '
    return pytestconfig.getoption('--strict-data-files')

@pytest.fixture
def datapath(strict_data_files):
    "\n    Get the path to a data file.\n\n    Parameters\n    ----------\n    path : str\n        Path to the file, relative to ``pandas/tests/``\n\n    Returns\n    -------\n    path including ``pandas/tests``.\n\n    Raises\n    ------\n    ValueError\n        If the path doesn't exist and the --strict-data-files option is set.\n    "
    BASE_PATH = os.path.join(os.path.dirname(__file__), 'tests')

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        if (not os.path.exists(path)):
            if strict_data_files:
                raise ValueError(f'Could not find file {path} and --strict-data-files is set.')
            else:
                pytest.skip(f'Could not find {path}.')
        return path
    return deco

@pytest.fixture
def iris(datapath):
    '\n    The iris dataset as a DataFrame.\n    '
    return pd.read_csv(datapath('io', 'data', 'csv', 'iris.csv'))
TIMEZONES = [None, 'UTC', 'US/Eastern', 'Asia/Tokyo', 'dateutil/US/Pacific', 'dateutil/Asia/Singapore', '+01:15', '-02:15', 'UTC+01:15', 'UTC-02:15', tzutc(), tzlocal(), FixedOffset(300), FixedOffset(0), FixedOffset((- 300)), timezone.utc, timezone(timedelta(hours=1)), timezone(timedelta(hours=(- 1)), name='foo')]
TIMEZONE_IDS = [repr(i) for i in TIMEZONES]

@td.parametrize_fixture_doc(str(TIMEZONE_IDS))
@pytest.fixture(params=TIMEZONES, ids=TIMEZONE_IDS)
def tz_naive_fixture(request):
    '\n    Fixture for trying timezones including default (None): {0}\n    '
    return request.param

@td.parametrize_fixture_doc(str(TIMEZONE_IDS[1:]))
@pytest.fixture(params=TIMEZONES[1:], ids=TIMEZONE_IDS[1:])
def tz_aware_fixture(request):
    '\n    Fixture for trying explicit timezones: {0}\n    '
    return request.param
tz_aware_fixture2 = tz_aware_fixture

@pytest.fixture(scope='module')
def datetime_tz_utc():
    '\n    Yields the UTC timezone object from the datetime module.\n    '
    return timezone.utc

@pytest.fixture(params=['utc', 'dateutil/UTC', utc, tzutc(), timezone.utc])
def utc_fixture(request):
    '\n    Fixture to provide variants of UTC timezone strings and tzinfo objects.\n    '
    return request.param

@pytest.fixture(params=tm.STRING_DTYPES)
def string_dtype(request):
    "\n    Parametrized fixture for string dtypes.\n\n    * str\n    * 'str'\n    * 'U'\n    "
    return request.param

@pytest.fixture(params=tm.BYTES_DTYPES)
def bytes_dtype(request):
    "\n    Parametrized fixture for bytes dtypes.\n\n    * bytes\n    * 'bytes'\n    "
    return request.param

@pytest.fixture(params=tm.OBJECT_DTYPES)
def object_dtype(request):
    "\n    Parametrized fixture for object dtypes.\n\n    * object\n    * 'object'\n    "
    return request.param

@pytest.fixture(params=tm.DATETIME64_DTYPES)
def datetime64_dtype(request):
    "\n    Parametrized fixture for datetime64 dtypes.\n\n    * 'datetime64[ns]'\n    * 'M8[ns]'\n    "
    return request.param

@pytest.fixture(params=tm.TIMEDELTA64_DTYPES)
def timedelta64_dtype(request):
    "\n    Parametrized fixture for timedelta64 dtypes.\n\n    * 'timedelta64[ns]'\n    * 'm8[ns]'\n    "
    return request.param

@pytest.fixture(params=tm.FLOAT_DTYPES)
def float_dtype(request):
    "\n    Parameterized fixture for float dtypes.\n\n    * float\n    * 'float32'\n    * 'float64'\n    "
    return request.param

@pytest.fixture(params=tm.FLOAT_EA_DTYPES)
def float_ea_dtype(request):
    "\n    Parameterized fixture for float dtypes.\n\n    * 'Float32'\n    * 'Float64'\n    "
    return request.param

@pytest.fixture(params=(tm.FLOAT_DTYPES + tm.FLOAT_EA_DTYPES))
def any_float_allowed_nullable_dtype(request):
    "\n    Parameterized fixture for float dtypes.\n\n    * float\n    * 'float32'\n    * 'float64'\n    * 'Float32'\n    * 'Float64'\n    "
    return request.param

@pytest.fixture(params=tm.COMPLEX_DTYPES)
def complex_dtype(request):
    "\n    Parameterized fixture for complex dtypes.\n\n    * complex\n    * 'complex64'\n    * 'complex128'\n    "
    return request.param

@pytest.fixture(params=tm.SIGNED_INT_DTYPES)
def sint_dtype(request):
    "\n    Parameterized fixture for signed integer dtypes.\n\n    * int\n    * 'int8'\n    * 'int16'\n    * 'int32'\n    * 'int64'\n    "
    return request.param

@pytest.fixture(params=tm.UNSIGNED_INT_DTYPES)
def uint_dtype(request):
    "\n    Parameterized fixture for unsigned integer dtypes.\n\n    * 'uint8'\n    * 'uint16'\n    * 'uint32'\n    * 'uint64'\n    "
    return request.param

@pytest.fixture(params=tm.ALL_INT_DTYPES)
def any_int_dtype(request):
    "\n    Parameterized fixture for any integer dtype.\n\n    * int\n    * 'int8'\n    * 'uint8'\n    * 'int16'\n    * 'uint16'\n    * 'int32'\n    * 'uint32'\n    * 'int64'\n    * 'uint64'\n    "
    return request.param

@pytest.fixture(params=tm.ALL_EA_INT_DTYPES)
def any_nullable_int_dtype(request):
    "\n    Parameterized fixture for any nullable integer dtype.\n\n    * 'UInt8'\n    * 'Int8'\n    * 'UInt16'\n    * 'Int16'\n    * 'UInt32'\n    * 'Int32'\n    * 'UInt64'\n    * 'Int64'\n    "
    return request.param

@pytest.fixture(params=(tm.ALL_EA_INT_DTYPES + tm.FLOAT_EA_DTYPES))
def any_numeric_dtype(request):
    "\n    Parameterized fixture for any nullable integer dtype and\n    any float ea dtypes.\n\n    * 'UInt8'\n    * 'Int8'\n    * 'UInt16'\n    * 'Int16'\n    * 'UInt32'\n    * 'Int32'\n    * 'UInt64'\n    * 'Int64'\n    * 'Float32'\n    * 'Float64'\n    "
    return request.param

@pytest.fixture(params=tm.SIGNED_EA_INT_DTYPES)
def any_signed_nullable_int_dtype(request):
    "\n    Parameterized fixture for any signed nullable integer dtype.\n\n    * 'Int8'\n    * 'Int16'\n    * 'Int32'\n    * 'Int64'\n    "
    return request.param

@pytest.fixture(params=tm.ALL_REAL_DTYPES)
def any_real_dtype(request):
    "\n    Parameterized fixture for any (purely) real numeric dtype.\n\n    * int\n    * 'int8'\n    * 'uint8'\n    * 'int16'\n    * 'uint16'\n    * 'int32'\n    * 'uint32'\n    * 'int64'\n    * 'uint64'\n    * float\n    * 'float32'\n    * 'float64'\n    "
    return request.param

@pytest.fixture(params=tm.ALL_NUMPY_DTYPES)
def any_numpy_dtype(request):
    "\n    Parameterized fixture for all numpy dtypes.\n\n    * bool\n    * 'bool'\n    * int\n    * 'int8'\n    * 'uint8'\n    * 'int16'\n    * 'uint16'\n    * 'int32'\n    * 'uint32'\n    * 'int64'\n    * 'uint64'\n    * float\n    * 'float32'\n    * 'float64'\n    * complex\n    * 'complex64'\n    * 'complex128'\n    * str\n    * 'str'\n    * 'U'\n    * bytes\n    * 'bytes'\n    * 'datetime64[ns]'\n    * 'M8[ns]'\n    * 'timedelta64[ns]'\n    * 'm8[ns]'\n    * object\n    * 'object'\n    "
    return request.param
_any_skipna_inferred_dtype = [('string', ['a', np.nan, 'c']), ('string', ['a', pd.NA, 'c']), ('bytes', [b'a', np.nan, b'c']), ('empty', [np.nan, np.nan, np.nan]), ('empty', []), ('mixed-integer', ['a', np.nan, 2]), ('mixed', ['a', np.nan, 2.0]), ('floating', [1.0, np.nan, 2.0]), ('integer', [1, np.nan, 2]), ('mixed-integer-float', [1, np.nan, 2.0]), ('decimal', [Decimal(1), np.nan, Decimal(2)]), ('boolean', [True, np.nan, False]), ('boolean', [True, pd.NA, False]), ('datetime64', [np.datetime64('2013-01-01'), np.nan, np.datetime64('2018-01-01')]), ('datetime', [pd.Timestamp('20130101'), np.nan, pd.Timestamp('20180101')]), ('date', [date(2013, 1, 1), np.nan, date(2018, 1, 1)]), ('timedelta', [timedelta(1), np.nan, timedelta(2)]), ('time', [time(1), np.nan, time(2)]), ('period', [pd.Period(2013), pd.NaT, pd.Period(2018)]), ('interval', [pd.Interval(0, 1), np.nan, pd.Interval(0, 2)])]
(ids, _) = zip(*_any_skipna_inferred_dtype)

@pytest.fixture(params=_any_skipna_inferred_dtype, ids=ids)
def any_skipna_inferred_dtype(request):
    "\n    Fixture for all inferred dtypes from _libs.lib.infer_dtype\n\n    The covered (inferred) types are:\n    * 'string'\n    * 'empty'\n    * 'bytes'\n    * 'mixed'\n    * 'mixed-integer'\n    * 'mixed-integer-float'\n    * 'floating'\n    * 'integer'\n    * 'decimal'\n    * 'boolean'\n    * 'datetime64'\n    * 'datetime'\n    * 'date'\n    * 'timedelta'\n    * 'time'\n    * 'period'\n    * 'interval'\n\n    Returns\n    -------\n    inferred_dtype : str\n        The string for the inferred dtype from _libs.lib.infer_dtype\n    values : np.ndarray\n        An array of object dtype that will be inferred to have\n        `inferred_dtype`\n\n    Examples\n    --------\n    >>> import pandas._libs.lib as lib\n    >>>\n    >>> def test_something(any_skipna_inferred_dtype):\n    ...     inferred_dtype, values = any_skipna_inferred_dtype\n    ...     # will pass\n    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype\n    "
    (inferred_dtype, values) = request.param
    values = np.array(values, dtype=object)
    return (inferred_dtype, values)

@pytest.fixture
def ip():
    '\n    Get an instance of IPython.InteractiveShell.\n\n    Will raise a skip if IPython is not installed.\n    '
    pytest.importorskip('IPython', minversion='6.0.0')
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config import Config
    c = Config()
    c.HistoryManager.hist_file = ':memory:'
    return InteractiveShell(config=c)

@pytest.fixture(params=['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'])
def spmatrix(request):
    '\n    Yields scipy sparse matrix classes.\n    '
    from scipy import sparse
    return getattr(sparse, (request.param + '_matrix'))

@pytest.fixture(params=[getattr(pd.offsets, o) for o in pd.offsets.__all__ if issubclass(getattr(pd.offsets, o), pd.offsets.Tick)])
def tick_classes(request):
    '\n    Fixture for Tick based datetime offsets available for a time series.\n    '
    return request.param

@pytest.fixture(params=[None, (lambda x: x)])
def sort_by_key(request):
    '\n    Simple fixture for testing keys in sorting methods.\n    Tests None (no key) and the identity key.\n    '
    return request.param

@pytest.fixture()
def fsspectest():
    pytest.importorskip('fsspec')
    from fsspec import register_implementation
    from fsspec.implementations.memory import MemoryFileSystem
    from fsspec.registry import _registry as registry

    class TestMemoryFS(MemoryFileSystem):
        protocol = 'testmem'
        test = [None]

        def __init__(self, **kwargs):
            self.test[0] = kwargs.pop('test', None)
            super().__init__(**kwargs)
    register_implementation('testmem', TestMemoryFS, clobber=True)
    (yield TestMemoryFS())
    registry.pop('testmem', None)
    TestMemoryFS.test[0] = None
    TestMemoryFS.store.clear()

@pytest.fixture(params=[('foo', None, None), ('Egon', 'Venkman', None), ('NCC1701D', 'NCC1701D', 'NCC1701D')])
def names(request):
    '\n    A 3-tuple of names, the first two for operands, the last for a result.\n    '
    return request.param

@pytest.fixture(params=[tm.setitem, tm.loc, tm.iloc])
def indexer_sli(request):
    '\n    Parametrize over __setitem__, loc.__setitem__, iloc.__setitem__\n    '
    return request.param

@pytest.fixture(params=[tm.setitem, tm.iloc])
def indexer_si(request):
    '\n    Parametrize over __setitem__, iloc.__setitem__\n    '
    return request.param
