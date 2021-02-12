
import pytest
pytestmark = [pytest.mark.filterwarnings('ignore:PY_SSIZE_T_CLEAN will be required.*:DeprecationWarning'), pytest.mark.filterwarnings('ignore:This method will be removed in future versions:DeprecationWarning'), pytest.mark.filterwarnings("ignore:This method will be removed in future versions.  Use 'tree.iter\\(\\)' or 'list\\(tree.iter\\(\\)\\)' instead.:PendingDeprecationWarning"), pytest.mark.filterwarnings('ignore:As the xlwt package is no longer maintained:FutureWarning')]
