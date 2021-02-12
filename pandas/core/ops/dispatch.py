
'\nFunctions for defining unary operations.\n'
from typing import Any
from pandas._typing import ArrayLike
from pandas.core.dtypes.generic import ABCExtensionArray

def should_extension_dispatch(left, right):
    '\n    Identify cases where Series operation should dispatch to ExtensionArray method.\n\n    Parameters\n    ----------\n    left : np.ndarray or ExtensionArray\n    right : object\n\n    Returns\n    -------\n    bool\n    '
    return (isinstance(left, ABCExtensionArray) or isinstance(right, ABCExtensionArray))
