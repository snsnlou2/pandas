
import pytest
pytestmark = [pytest.mark.filterwarnings('ignore:a closed node found in the registry:UserWarning'), pytest.mark.filterwarnings('ignore:tostring\\(\\) is deprecated:DeprecationWarning')]
