
import pytest

@pytest.fixture(params=['split', 'records', 'index', 'columns', 'values'])
def orient(request):
    '\n    Fixture for orients excluding the table format.\n    '
    return request.param
