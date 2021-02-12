
from distutils.version import LooseVersion
import pytest
from pandas.compat._optional import import_optional_dependency
pytestmark = [pytest.mark.filterwarnings('ignore:This method will be removed in future versions:PendingDeprecationWarning'), pytest.mark.filterwarnings('ignore:This method will be removed in future versions:DeprecationWarning'), pytest.mark.filterwarnings('ignore:As the xlwt package is no longer maintained:FutureWarning'), pytest.mark.filterwarnings('ignore:.*In xlrd >= 2.0, only the xls format is supported:FutureWarning')]
if (import_optional_dependency('xlrd', raise_on_missing=False, on_version='ignore') is None):
    xlrd_version = None
else:
    import xlrd
    xlrd_version = LooseVersion(xlrd.__version__)
