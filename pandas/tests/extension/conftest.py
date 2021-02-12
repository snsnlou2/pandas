
import operator
import pytest
from pandas import Series

@pytest.fixture
def dtype():
    'A fixture providing the ExtensionDtype to validate.'
    raise NotImplementedError

@pytest.fixture
def data():
    '\n    Length-100 array for this type.\n\n    * data[0] and data[1] should both be non missing\n    * data[0] and data[1] should not be equal\n    '
    raise NotImplementedError

@pytest.fixture
def data_for_twos():
    'Length-100 array in which all the elements are two.'
    raise NotImplementedError

@pytest.fixture
def data_missing():
    'Length-2 array with [NA, Valid]'
    raise NotImplementedError

@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    "Parametrized fixture giving 'data' and 'data_missing'"
    if (request.param == 'data'):
        return data
    elif (request.param == 'data_missing'):
        return data_missing

@pytest.fixture
def data_repeated(data):
    '\n    Generate many datasets.\n\n    Parameters\n    ----------\n    data : fixture implementing `data`\n\n    Returns\n    -------\n    Callable[[int], Generator]:\n        A callable that takes a `count` argument and\n        returns a generator yielding `count` datasets.\n    '

    def gen(count):
        for _ in range(count):
            (yield data)
    return gen

@pytest.fixture
def data_for_sorting():
    '\n    Length-3 array with a known sort order.\n\n    This should be three items [B, C, A] with\n    A < B < C\n    '
    raise NotImplementedError

@pytest.fixture
def data_missing_for_sorting():
    '\n    Length-3 array with a known sort order.\n\n    This should be three items [B, NA, A] with\n    A < B and NA missing.\n    '
    raise NotImplementedError

@pytest.fixture
def na_cmp():
    '\n    Binary operator for comparing NA values.\n\n    Should return a function of two arguments that returns\n    True if both arguments are (scalar) NA for your type.\n\n    By default, uses ``operator.is_``\n    '
    return operator.is_

@pytest.fixture
def na_value():
    "The scalar missing value for this type. Default 'None'"
    return None

@pytest.fixture
def data_for_grouping():
    '\n    Data for factorization, grouping, and unique tests.\n\n    Expected to be like [B, B, NA, NA, A, A, B, C]\n\n    Where A < B < C and NA is missing\n    '
    raise NotImplementedError

@pytest.fixture(params=[True, False])
def box_in_series(request):
    'Whether to box the data in a Series'
    return request.param

@pytest.fixture(params=[(lambda x: 1), (lambda x: ([1] * len(x))), (lambda x: Series(([1] * len(x)))), (lambda x: x)], ids=['scalar', 'list', 'series', 'object'])
def groupby_apply_op(request):
    '\n    Functions to test groupby.apply().\n    '
    return request.param

@pytest.fixture(params=[True, False])
def as_frame(request):
    '\n    Boolean fixture to support Series and Series.to_frame() comparison testing.\n    '
    return request.param

@pytest.fixture(params=[True, False])
def as_series(request):
    '\n    Boolean fixture to support arr and Series(arr) comparison testing.\n    '
    return request.param

@pytest.fixture(params=[True, False])
def use_numpy(request):
    '\n    Boolean fixture to support comparison testing of ExtensionDtype array\n    and numpy array.\n    '
    return request.param

@pytest.fixture(params=['ffill', 'bfill'])
def fillna_method(request):
    "\n    Parametrized fixture giving method parameters 'ffill' and 'bfill' for\n    Series.fillna(method=<method>) testing.\n    "
    return request.param

@pytest.fixture(params=[True, False])
def as_array(request):
    '\n    Boolean fixture to support ExtensionDtype _from_sequence method testing.\n    '
    return request.param
