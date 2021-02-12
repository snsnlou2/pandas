
'\nSupport pre-0.12 series pickle compatibility.\n'
import contextlib
import copy
import io
import pickle as pkl
from typing import TYPE_CHECKING, Optional
import warnings
from pandas._libs.tslibs import BaseOffset
from pandas import Index
if TYPE_CHECKING:
    from pandas import DataFrame, Series

def load_reduce(self):
    stack = self.stack
    args = stack.pop()
    func = stack[(- 1)]
    if (len(args) and (type(args[0]) is type)):
        n = args[0].__name__
    try:
        stack[(- 1)] = func(*args)
        return
    except TypeError as err:
        msg = '_reconstruct: First argument must be a sub-type of ndarray'
        if (msg in str(err)):
            try:
                cls = args[0]
                stack[(- 1)] = object.__new__(cls)
                return
            except TypeError:
                pass
        elif (args and issubclass(args[0], BaseOffset)):
            cls = args[0]
            stack[(- 1)] = cls.__new__(*args)
            return
        raise
_sparse_msg = "\nLoading a saved '{cls}' as a {new} with sparse values.\n'{cls}' is now removed. You should re-save this dataset in its new format.\n"

class _LoadSparseSeries():

    def __new__(cls):
        from pandas import Series
        warnings.warn(_sparse_msg.format(cls='SparseSeries', new='Series'), FutureWarning, stacklevel=6)
        return Series(dtype=object)

class _LoadSparseFrame():

    def __new__(cls):
        from pandas import DataFrame
        warnings.warn(_sparse_msg.format(cls='SparseDataFrame', new='DataFrame'), FutureWarning, stacklevel=6)
        return DataFrame()
_class_locations_map = {('pandas.core.sparse.array', 'SparseArray'): ('pandas.core.arrays', 'SparseArray'), ('pandas.core.base', 'FrozenNDArray'): ('numpy', 'ndarray'), ('pandas.core.indexes.frozen', 'FrozenNDArray'): ('numpy', 'ndarray'), ('pandas.core.base', 'FrozenList'): ('pandas.core.indexes.frozen', 'FrozenList'), ('pandas.core.series', 'TimeSeries'): ('pandas.core.series', 'Series'), ('pandas.sparse.series', 'SparseTimeSeries'): ('pandas.core.sparse.series', 'SparseSeries'), ('pandas._sparse', 'BlockIndex'): ('pandas._libs.sparse', 'BlockIndex'), ('pandas.tslib', 'Timestamp'): ('pandas._libs.tslib', 'Timestamp'), ('pandas._period', 'Period'): ('pandas._libs.tslibs.period', 'Period'), ('pandas._libs.period', 'Period'): ('pandas._libs.tslibs.period', 'Period'), ('pandas.tslib', '__nat_unpickle'): ('pandas._libs.tslibs.nattype', '__nat_unpickle'), ('pandas._libs.tslib', '__nat_unpickle'): ('pandas._libs.tslibs.nattype', '__nat_unpickle'), ('pandas.sparse.array', 'SparseArray'): ('pandas.core.arrays.sparse', 'SparseArray'), ('pandas.sparse.series', 'SparseSeries'): ('pandas.compat.pickle_compat', '_LoadSparseSeries'), ('pandas.sparse.frame', 'SparseDataFrame'): ('pandas.core.sparse.frame', '_LoadSparseFrame'), ('pandas.indexes.base', '_new_Index'): ('pandas.core.indexes.base', '_new_Index'), ('pandas.indexes.base', 'Index'): ('pandas.core.indexes.base', 'Index'), ('pandas.indexes.numeric', 'Int64Index'): ('pandas.core.indexes.numeric', 'Int64Index'), ('pandas.indexes.range', 'RangeIndex'): ('pandas.core.indexes.range', 'RangeIndex'), ('pandas.indexes.multi', 'MultiIndex'): ('pandas.core.indexes.multi', 'MultiIndex'), ('pandas.tseries.index', '_new_DatetimeIndex'): ('pandas.core.indexes.datetimes', '_new_DatetimeIndex'), ('pandas.tseries.index', 'DatetimeIndex'): ('pandas.core.indexes.datetimes', 'DatetimeIndex'), ('pandas.tseries.period', 'PeriodIndex'): ('pandas.core.indexes.period', 'PeriodIndex'), ('pandas.core.categorical', 'Categorical'): ('pandas.core.arrays', 'Categorical'), ('pandas.tseries.tdi', 'TimedeltaIndex'): ('pandas.core.indexes.timedeltas', 'TimedeltaIndex'), ('pandas.indexes.numeric', 'Float64Index'): ('pandas.core.indexes.numeric', 'Float64Index'), ('pandas.core.sparse.series', 'SparseSeries'): ('pandas.compat.pickle_compat', '_LoadSparseSeries'), ('pandas.core.sparse.frame', 'SparseDataFrame'): ('pandas.compat.pickle_compat', '_LoadSparseFrame')}

class Unpickler(pkl._Unpickler):

    def find_class(self, module, name):
        key = (module, name)
        (module, name) = _class_locations_map.get(key, key)
        return super().find_class(module, name)
Unpickler.dispatch = copy.copy(Unpickler.dispatch)
Unpickler.dispatch[pkl.REDUCE[0]] = load_reduce

def load_newobj(self):
    args = self.stack.pop()
    cls = self.stack[(- 1)]
    if issubclass(cls, Index):
        obj = object.__new__(cls)
    else:
        obj = cls.__new__(cls, *args)
    self.stack[(- 1)] = obj
Unpickler.dispatch[pkl.NEWOBJ[0]] = load_newobj

def load_newobj_ex(self):
    kwargs = self.stack.pop()
    args = self.stack.pop()
    cls = self.stack.pop()
    if issubclass(cls, Index):
        obj = object.__new__(cls)
    else:
        obj = cls.__new__(cls, *args, **kwargs)
    self.append(obj)
try:
    Unpickler.dispatch[pkl.NEWOBJ_EX[0]] = load_newobj_ex
except (AttributeError, KeyError):
    pass

def load(fh, encoding=None, is_verbose=False):
    '\n    Load a pickle, with a provided encoding,\n\n    Parameters\n    ----------\n    fh : a filelike object\n    encoding : an optional encoding\n    is_verbose : show exception output\n    '
    try:
        fh.seek(0)
        if (encoding is not None):
            up = Unpickler(fh, encoding=encoding)
        else:
            up = Unpickler(fh)
        up.is_verbose = is_verbose
        return up.load()
    except (ValueError, TypeError):
        raise

def loads(bytes_object, *, fix_imports=True, encoding='ASCII', errors='strict'):
    '\n    Analogous to pickle._loads.\n    '
    fd = io.BytesIO(bytes_object)
    return Unpickler(fd, fix_imports=fix_imports, encoding=encoding, errors=errors).load()

@contextlib.contextmanager
def patch_pickle():
    '\n    Temporarily patch pickle to use our unpickler.\n    '
    orig_loads = pkl.loads
    try:
        setattr(pkl, 'loads', loads)
        (yield)
    finally:
        setattr(pkl, 'loads', orig_loads)
