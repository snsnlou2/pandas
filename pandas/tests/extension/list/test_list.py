
import pytest
import pandas as pd
from .array import ListArray, ListDtype, make_data

@pytest.fixture
def dtype():
    return ListDtype()

@pytest.fixture
def data():
    'Length-100 ListArray for semantics test.'
    data = make_data()
    while (len(data[0]) == len(data[1])):
        data = make_data()
    return ListArray(data)

def test_to_csv(data):
    df = pd.DataFrame({'a': data})
    res = df.to_csv()
    assert (str(data[0]) in res)
