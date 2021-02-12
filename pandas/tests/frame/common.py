
from typing import List
from pandas import DataFrame, concat

def _check_mixed_float(df, dtype=None):
    dtypes = {'A': 'float32', 'B': 'float32', 'C': 'float16', 'D': 'float64'}
    if isinstance(dtype, str):
        dtypes = {k: dtype for (k, v) in dtypes.items()}
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    if dtypes.get('A'):
        assert (df.dtypes['A'] == dtypes['A'])
    if dtypes.get('B'):
        assert (df.dtypes['B'] == dtypes['B'])
    if dtypes.get('C'):
        assert (df.dtypes['C'] == dtypes['C'])
    if dtypes.get('D'):
        assert (df.dtypes['D'] == dtypes['D'])

def _check_mixed_int(df, dtype=None):
    dtypes = {'A': 'int32', 'B': 'uint64', 'C': 'uint8', 'D': 'int64'}
    if isinstance(dtype, str):
        dtypes = {k: dtype for (k, v) in dtypes.items()}
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    if dtypes.get('A'):
        assert (df.dtypes['A'] == dtypes['A'])
    if dtypes.get('B'):
        assert (df.dtypes['B'] == dtypes['B'])
    if dtypes.get('C'):
        assert (df.dtypes['C'] == dtypes['C'])
    if dtypes.get('D'):
        assert (df.dtypes['D'] == dtypes['D'])

def zip_frames(frames, axis=1):
    "\n    take a list of frames, zip them together under the\n    assumption that these all have the first frames' index/columns.\n\n    Returns\n    -------\n    new_frame : DataFrame\n    "
    if (axis == 1):
        columns = frames[0].columns
        zipped = [f.loc[:, c] for c in columns for f in frames]
        return concat(zipped, axis=1)
    else:
        index = frames[0].index
        zipped = [f.loc[i, :] for i in index for f in frames]
        return DataFrame(zipped)
