
from functools import reduce
import numpy as np
from pandas._config import get_option

def ensure_decoded(s):
    '\n    If we have bytes, decode them to unicode.\n    '
    if isinstance(s, (np.bytes_, bytes)):
        s = s.decode(get_option('display.encoding'))
    return s

def result_type_many(*arrays_and_dtypes):
    '\n    Wrapper around numpy.result_type which overcomes the NPY_MAXARGS (32)\n    argument limit.\n    '
    try:
        return np.result_type(*arrays_and_dtypes)
    except ValueError:
        return reduce(np.result_type, arrays_and_dtypes)
