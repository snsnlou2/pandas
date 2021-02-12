
'\nipython analogue:\n\ntr = TimeResolution()\nmi = pd.MultiIndex.from_product(tr.params[:-1] + ([str(x) for x in tr.params[-1]],))\ndf = pd.DataFrame(np.nan, index=mi, columns=["mean", "stdev"])\n\nfor unit in tr.params[0]:\n    for size in tr.params[1]:\n        for tz in tr.params[2]:\n            tr.setup(unit, size, tz)\n            key = (unit, size, str(tz))\n            print(key)\n\n            val = %timeit -o tr.time_get_resolution(unit, size, tz)\n\n            df.loc[key] = (val.average, val.stdev)\n\n'
from datetime import timedelta, timezone
from dateutil.tz import gettz, tzlocal
import numpy as np
import pytz
try:
    from pandas._libs.tslibs import get_resolution
except ImportError:
    from pandas._libs.tslibs.resolution import get_resolution

class TimeResolution():
    params = (['D', 'h', 'm', 's', 'us', 'ns'], [1, 100, (10 ** 4), (10 ** 6)], [None, timezone.utc, timezone(timedelta(minutes=60)), pytz.timezone('US/Pacific'), gettz('Asia/Tokyo'), tzlocal()])
    param_names = ['unit', 'size', 'tz']

    def setup(self, unit, size, tz):
        arr = np.random.randint(0, 10, size=size, dtype='i8')
        arr = arr.view(f'M8[{unit}]').astype('M8[ns]').view('i8')
        self.i8data = arr

    def time_get_resolution(self, unit, size, tz):
        get_resolution(self.i8data, tz)
