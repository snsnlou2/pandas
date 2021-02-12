
'\nfrozen (immutable) data structures to support MultiIndexing\n\nThese are used for:\n\n- .names (FrozenList)\n\n'
from typing import Any
from pandas.core.base import PandasObject
from pandas.io.formats.printing import pprint_thing

class FrozenList(PandasObject, list):
    "\n    Container that doesn't allow setting item *but*\n    because it's technically non-hashable, will be used\n    for lookups, appropriately, etc.\n    "

    def union(self, other):
        '\n        Returns a FrozenList with other concatenated to the end of self.\n\n        Parameters\n        ----------\n        other : array-like\n            The array-like whose elements we are concatenating.\n\n        Returns\n        -------\n        FrozenList\n            The collection difference between self and other.\n        '
        if isinstance(other, tuple):
            other = list(other)
        return type(self)(super().__add__(other))

    def difference(self, other):
        '\n        Returns a FrozenList with elements from other removed from self.\n\n        Parameters\n        ----------\n        other : array-like\n            The array-like whose elements we are removing self.\n\n        Returns\n        -------\n        FrozenList\n            The collection difference between self and other.\n        '
        other = set(other)
        temp = [x for x in self if (x not in other)]
        return type(self)(temp)
    __add__ = __iadd__ = union

    def __getitem__(self, n):
        if isinstance(n, slice):
            return type(self)(super().__getitem__(n))
        return super().__getitem__(n)

    def __radd__(self, other):
        if isinstance(other, tuple):
            other = list(other)
        return type(self)((other + list(self)))

    def __eq__(self, other):
        if isinstance(other, (tuple, FrozenList)):
            other = list(other)
        return super().__eq__(other)
    __req__ = __eq__

    def __mul__(self, other):
        return type(self)(super().__mul__(other))
    __imul__ = __mul__

    def __reduce__(self):
        return (type(self), (list(self),))

    def __hash__(self):
        return hash(tuple(self))

    def _disabled(self, *args, **kwargs):
        '\n        This method will not function because object is immutable.\n        '
        raise TypeError(f"'{type(self).__name__}' does not support mutable operations.")

    def __str__(self):
        return pprint_thing(self, quote_strings=True, escape_chars=('\t', '\r', '\n'))

    def __repr__(self):
        return f'{type(self).__name__}({str(self)})'
    __setitem__ = __setslice__ = _disabled
    __delitem__ = __delslice__ = _disabled
    pop = append = extend = _disabled
    remove = sort = insert = _disabled
