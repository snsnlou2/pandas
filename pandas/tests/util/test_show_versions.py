
import re
import pytest
import pandas as pd

@pytest.mark.filterwarnings('ignore:defusedxml.lxml is no longer supported:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Using or importing the ABCs from:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:pandas.core.index is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:pandas.util.testing is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:Distutils:UserWarning')
@pytest.mark.filterwarnings('ignore:Setuptools is replacing distutils:UserWarning')
def test_show_versions(capsys):
    pd.show_versions()
    captured = capsys.readouterr()
    result = captured.out
    assert ('INSTALLED VERSIONS' in result)
    assert re.search('commit\\s*:\\s[0-9a-f]{40}\\n', result)
    assert re.search('numpy\\s*:\\s([0-9\\.\\+a-g\\_]|dev)+(dirty)?\\n', result)
    assert re.search('pyarrow\\s*:\\s([0-9\\.]+|None)\\n', result)
