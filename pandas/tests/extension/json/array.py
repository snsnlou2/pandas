
'\nTest extension array for storing nested data in a pandas container.\n\nThe JSONArray stores lists of dictionaries. The storage mechanism is a list,\nnot an ndarray.\n\nNote\n----\nWe currently store lists of UserDicts. Pandas has a few places\ninternally that specifically check for dicts, and does non-scalar things\nin that case. We *want* the dictionaries to be treated as scalars, so we\nhack around pandas by using UserDicts.\n'
from collections import UserDict, abc
import itertools
import numbers
import random
import string
import sys
from typing import Any, Mapping, Type
import numpy as np
from pandas.core.dtypes.common import pandas_dtype
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

class JSONDtype(ExtensionDtype):
    type = abc.Mapping
    name = 'json'
    na_value = UserDict()

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        return JSONArray

class JSONArray(ExtensionArray):
    dtype = JSONDtype()
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False):
        for val in values:
            if (not isinstance(val, self.dtype.type)):
                raise TypeError(('All values must be of type ' + str(self.dtype.type)))
        self.data = values
        self._items = self._data = self.data

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls([UserDict(x) for x in values if (x != ())])

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        elif (isinstance(item, slice) and (item == slice(None))):
            return type(self)(self.data)
        elif isinstance(item, slice):
            return type(self)(self.data[item])
        else:
            item = pd.api.indexers.check_array_indexer(self, item)
            if pd.api.types.is_bool_dtype(item.dtype):
                return self._from_sequence([x for (x, m) in zip(self, item) if m])
            return type(self)([self.data[i] for i in item])

    def __setitem__(self, key, value):
        if isinstance(key, numbers.Integral):
            self.data[key] = value
        else:
            if (not isinstance(value, (type(self), abc.Sequence))):
                value = itertools.cycle([value])
            if (isinstance(key, np.ndarray) and (key.dtype == 'bool')):
                for (i, (k, v)) in enumerate(zip(key, value)):
                    if k:
                        assert isinstance(v, self.dtype.type)
                        self.data[i] = v
            else:
                for (k, v) in zip(key, value):
                    assert isinstance(v, self.dtype.type)
                    self.data[k] = v

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return NotImplemented

    def __array__(self, dtype=None):
        if (dtype is None):
            dtype = object
        return np.asarray(self.data, dtype=dtype)

    @property
    def nbytes(self):
        return sys.getsizeof(self.data)

    def isna(self):
        return np.array([(x == self.dtype.na_value) for x in self.data], dtype=bool)

    def take(self, indexer, allow_fill=False, fill_value=None):
        indexer = np.asarray(indexer)
        msg = 'Index is out of bounds or cannot do a non-empty take from an empty array.'
        if allow_fill:
            if (fill_value is None):
                fill_value = self.dtype.na_value
            if (indexer < (- 1)).any():
                raise ValueError
            try:
                output = [(self.data[loc] if (loc != (- 1)) else fill_value) for loc in indexer]
            except IndexError as err:
                raise IndexError(msg) from err
        else:
            try:
                output = [self.data[loc] for loc in indexer]
            except IndexError as err:
                raise IndexError(msg) from err
        return self._from_sequence(output)

    def copy(self):
        return type(self)(self.data[:])

    def astype(self, dtype, copy=True):
        from pandas.core.arrays.string_ import StringDtype
        dtype = pandas_dtype(dtype)
        if (isinstance(dtype, type(self.dtype)) and (dtype == self.dtype)):
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, StringDtype):
            value = self.astype(str)
            return dtype.construct_array_type()._from_sequence(value, copy=False)
        return np.array([dict(x) for x in self], dtype=dtype, copy=copy)

    def unique(self):
        return type(self)([dict(x) for x in {tuple(d.items()) for d in self.data}])

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = list(itertools.chain.from_iterable((x.data for x in to_concat)))
        return cls(data)

    def _values_for_factorize(self):
        frozen = self._values_for_argsort()
        if (len(frozen) == 0):
            frozen = frozen.ravel()
        return (frozen, ())

    def _values_for_argsort(self):
        frozen = ([()] + [tuple(x.items()) for x in self])
        return np.array(frozen, dtype=object)[1:]

def make_data():
    return [UserDict([(random.choice(string.ascii_letters), random.randint(0, 100)) for _ in range(random.randint(0, 10))]) for _ in range(100)]
