
try:
    from pandas._libs.tslibs import is_date_array_normalized, normalize_i8_timestamps
except ImportError:
    from pandas._libs.tslibs.conversion import normalize_i8_timestamps, is_date_array_normalized
import pandas as pd
from .tslib import _sizes, _tzs

class Normalize():
    params = [_sizes, _tzs]
    param_names = ['size', 'tz']

    def setup(self, size, tz):
        dti = pd.date_range('2016-01-01', periods=10, tz=tz).repeat((size // 10))
        self.i8data = dti.asi8

    def time_normalize_i8_timestamps(self, size, tz):
        normalize_i8_timestamps(self.i8data, tz)

    def time_is_date_array_normalized(self, size, tz):
        is_date_array_normalized(self.i8data, tz)
