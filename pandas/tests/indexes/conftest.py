
import pytest

@pytest.fixture(params=[None, False])
def sort(request):
    '\n    Valid values for the \'sort\' parameter used in the Index\n    setops methods (intersection, union, etc.)\n\n    Caution:\n        Don\'t confuse this one with the "sort" fixture used\n        for DataFrame.append or concat. That one has\n        parameters [True, False].\n\n        We can\'t combine them as sort=True is not permitted\n        in the Index setops methods.\n    '
    return request.param

@pytest.fixture(params=['D', '3D', '-3D', 'H', '2H', '-2H', 'T', '2T', 'S', '-3S'])
def freq_sample(request):
    "\n    Valid values for 'freq' parameter used to create date_range and\n    timedelta_range..\n    "
    return request.param
