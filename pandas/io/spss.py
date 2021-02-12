
from pathlib import Path
from typing import Optional, Sequence, Union
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.inference import is_list_like
from pandas.core.api import DataFrame
from pandas.io.common import stringify_path

def read_spss(path, usecols=None, convert_categoricals=True):
    '\n    Load an SPSS file from the file path, returning a DataFrame.\n\n    .. versionadded:: 0.25.0\n\n    Parameters\n    ----------\n    path : str or Path\n        File path.\n    usecols : list-like, optional\n        Return a subset of the columns. If None, return all columns.\n    convert_categoricals : bool, default is True\n        Convert categorical columns into pd.Categorical.\n\n    Returns\n    -------\n    DataFrame\n    '
    pyreadstat = import_optional_dependency('pyreadstat')
    if (usecols is not None):
        if (not is_list_like(usecols)):
            raise TypeError('usecols must be list-like.')
        else:
            usecols = list(usecols)
    (df, _) = pyreadstat.read_sav(stringify_path(path), usecols=usecols, apply_value_formats=convert_categoricals)
    return df
