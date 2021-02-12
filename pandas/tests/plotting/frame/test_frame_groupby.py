
' Test cases for DataFrame.plot '
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import TestPlotBase
pytestmark = pytest.mark.slow

@td.skip_if_no_mpl
class TestDataFramePlotsGroupby(TestPlotBase):

    def setup_method(self, method):
        TestPlotBase.setup_method(self, method)
        import matplotlib as mpl
        mpl.rcdefaults()
        self.tdf = tm.makeTimeDataFrame()
        self.hexbin_df = DataFrame({'A': np.random.uniform(size=20), 'B': np.random.uniform(size=20), 'C': (np.arange(20) + np.random.uniform(size=20))})

    def _assert_ytickslabels_visibility(self, axes, expected):
        for (ax, exp) in zip(axes, expected):
            self._check_visible(ax.get_yticklabels(), visible=exp)

    def _assert_xtickslabels_visibility(self, axes, expected):
        for (ax, exp) in zip(axes, expected):
            self._check_visible(ax.get_xticklabels(), visible=exp)

    @pytest.mark.parametrize('kwargs, expected', [({}, [True, False, True, False]), ({'sharey': True}, [True, False, True, False]), ({'sharey': False}, [True, True, True, True])])
    def test_groupby_boxplot_sharey(self, kwargs, expected):
        df = DataFrame({'a': [(- 1.43), (- 0.15), (- 3.7), (- 1.43), (- 0.14)], 'b': [0.56, 0.84, 0.29, 0.56, 0.85], 'c': [0, 1, 2, 3, 1]}, index=[0, 1, 2, 3, 4])
        axes = df.groupby('c').boxplot(**kwargs)
        self._assert_ytickslabels_visibility(axes, expected)

    @pytest.mark.parametrize('kwargs, expected', [({}, [True, True, True, True]), ({'sharex': False}, [True, True, True, True]), ({'sharex': True}, [False, False, True, True])])
    def test_groupby_boxplot_sharex(self, kwargs, expected):
        df = DataFrame({'a': [(- 1.43), (- 0.15), (- 3.7), (- 1.43), (- 0.14)], 'b': [0.56, 0.84, 0.29, 0.56, 0.85], 'c': [0, 1, 2, 3, 1]}, index=[0, 1, 2, 3, 4])
        axes = df.groupby('c').boxplot(**kwargs)
        self._assert_xtickslabels_visibility(axes, expected)
