
import numpy as np
import pytest
from pandas import Categorical, DataFrame

@pytest.mark.parametrize('data, expected', [(DataFrame(), True), (DataFrame({'A': [1, 2], 'B': [1, 2]}), True), (DataFrame({'A': np.array([1, 2], dtype=object), 'B': np.array(['a', 'b'], dtype=object)}), True), (DataFrame({'A': Categorical(['a', 'b']), 'B': Categorical(['a', 'b'])}), True), (DataFrame({'A': [1, 2], 'B': [1.0, 2.0]}), False), (DataFrame({'A': np.array([1, 2], dtype=np.int32), 'B': np.array([1, 2], dtype=np.int64)}), False), (DataFrame({'A': Categorical(['a', 'b']), 'B': Categorical(['b', 'c'])}), False)])
def test_is_homogeneous_type(data, expected):
    assert (data._is_homogeneous_type is expected)
