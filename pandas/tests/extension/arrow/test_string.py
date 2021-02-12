
import pytest
import pandas as pd
pytest.importorskip('pyarrow', minversion='0.13.0')
from .arrays import ArrowStringDtype

def test_constructor_from_list():
    result = pd.Series(['E'], dtype=ArrowStringDtype())
    assert isinstance(result.dtype, ArrowStringDtype)
