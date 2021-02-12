
'\nipython analogue:\n\ntr = TimeIntsToPydatetime()\nmi = pd.MultiIndex.from_product(\n    tr.params[:-1] + ([str(x) for x in tr.params[-1]],)\n)\ndf = pd.DataFrame(np.nan, index=mi, columns=["mean", "stdev"])\nfor box in tr.params[0]:\n    for size in tr.params[1]:\n        for tz in tr.params[2]:\n            tr.setup(box, size, tz)\n            key = (box, size, str(tz))\n            print(key)\n            val = %timeit -o tr.time_ints_to_pydatetime(box, size, tz)\n            df.loc[key] = (val.average, val.stdev)\n'
from datetime import timedelta, timezone
from dateutil.tz import gettz, tzlocal
import numpy as np
import pytz
try:
    from pandas._libs.tslibs import ints_to_pydatetime
except ImportError:
    from pandas._libs.tslib import ints_to_pydatetime
_tzs = [None, timezone.utc, timezone(timedelta(minutes=60)), pytz.timezone('US/Pacific'), gettz('Asia/Tokyo'), tzlocal()]
_sizes = [0, 1, 100, (10 ** 4), (10 ** 6)]

class TimeIntsToPydatetime():
    params = (['time', 'date', 'datetime', 'timestamp'], _sizes, _tzs)
    param_names = ['box', 'size', 'tz']

    def setup(self, box, size, tz):
        arr = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr

    def time_ints_to_pydatetime(self, box, size, tz):
        if (box == 'date'):
            tz = None
        ints_to_pydatetime(self.i8data, tz, box=box)
