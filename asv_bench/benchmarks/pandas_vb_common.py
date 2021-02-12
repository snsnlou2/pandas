
from importlib import import_module
import os
import numpy as np
import pandas as pd
for imp in ['pandas._libs.lib', 'pandas.lib']:
    try:
        lib = import_module(imp)
        break
    except (ImportError, TypeError, ValueError):
        pass
try:
    import pandas._testing as tm
except ImportError:
    import pandas.util.testing as tm
numeric_dtypes = [np.int64, np.int32, np.uint32, np.uint64, np.float32, np.float64, np.int16, np.int8, np.uint16, np.uint8]
datetime_dtypes = [np.datetime64, np.timedelta64]
string_dtypes = [object]
try:
    extension_dtypes = [pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype, pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype, pd.CategoricalDtype, pd.IntervalDtype, pd.DatetimeTZDtype('ns', 'UTC'), pd.PeriodDtype('D')]
except AttributeError:
    extension_dtypes = []

def setup(*args, **kwargs):
    np.random.seed(1234)

class BaseIO():
    '\n    Base class for IO benchmarks\n    '
    fname = None

    def remove(self, f):
        'Remove created files'
        try:
            os.remove(f)
        except OSError:
            pass

    def teardown(self, *args, **kwargs):
        self.remove(self.fname)
