
"\nBase test suite for extension arrays.\n\nThese tests are intended for third-party libraries to subclass to validate\nthat their extension arrays and dtypes satisfy the interface. Moving or\nrenaming the tests should not be done lightly.\n\nLibraries are expected to implement a few pytest fixtures to provide data\nfor the tests. The fixtures may be located in either\n\n* The same module as your test class.\n* A ``conftest.py`` in the same directory as your test class.\n\nThe full list of fixtures may be found in the ``conftest.py`` next to this\nfile.\n\n.. code-block:: python\n\n   import pytest\n   from pandas.tests.extension.base import BaseDtypeTests\n\n\n   @pytest.fixture\n   def dtype():\n       return MyDtype()\n\n\n   class TestMyDtype(BaseDtypeTests):\n       pass\n\n\nYour class ``TestDtype`` will inherit all the tests defined on\n``BaseDtypeTests``. pytest's fixture discover will supply your ``dtype``\nwherever the test requires it. You're free to implement additional tests.\n\nAll the tests in these modules use ``self.assert_frame_equal`` or\n``self.assert_series_equal`` for dataframe or series comparisons. By default,\nthey use the usual ``pandas.testing.assert_frame_equal`` and\n``pandas.testing.assert_series_equal``. You can override the checks used\nby defining the staticmethods ``assert_frame_equal`` and\n``assert_series_equal`` on your base test class.\n\n"
from .casting import BaseCastingTests
from .constructors import BaseConstructorsTests
from .dtype import BaseDtypeTests
from .getitem import BaseGetitemTests
from .groupby import BaseGroupbyTests
from .interface import BaseInterfaceTests
from .io import BaseParsingTests
from .methods import BaseMethodsTests
from .missing import BaseMissingTests
from .ops import BaseArithmeticOpsTests, BaseComparisonOpsTests, BaseOpsUtil, BaseUnaryOpsTests
from .printing import BasePrintingTests
from .reduce import BaseBooleanReduceTests, BaseNoReduceTests, BaseNumericReduceTests
from .reshaping import BaseReshapingTests
from .setitem import BaseSetitemTests
