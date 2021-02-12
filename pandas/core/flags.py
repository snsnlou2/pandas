
import weakref

class Flags():
    "\n    Flags that apply to pandas objects.\n\n    .. versionadded:: 1.2.0\n\n    Parameters\n    ----------\n    obj : Series or DataFrame\n        The object these flags are associated with.\n    allows_duplicate_labels : bool, default True\n        Whether to allow duplicate labels in this object. By default,\n        duplicate labels are permitted. Setting this to ``False`` will\n        cause an :class:`errors.DuplicateLabelError` to be raised when\n        `index` (or columns for DataFrame) is not unique, or any\n        subsequent operation on introduces duplicates.\n        See :ref:`duplicates.disallow` for more.\n\n        .. warning::\n\n           This is an experimental feature. Currently, many methods fail to\n           propagate the ``allows_duplicate_labels`` value. In future versions\n           it is expected that every method taking or returning one or more\n           DataFrame or Series objects will propagate ``allows_duplicate_labels``.\n\n    Notes\n    -----\n    Attributes can be set in two ways\n\n    >>> df = pd.DataFrame()\n    >>> df.flags\n    <Flags(allows_duplicate_labels=True)>\n    >>> df.flags.allows_duplicate_labels = False\n    >>> df.flags\n    <Flags(allows_duplicate_labels=False)>\n\n    >>> df.flags['allows_duplicate_labels'] = True\n    >>> df.flags\n    <Flags(allows_duplicate_labels=True)>\n    "
    _keys = {'allows_duplicate_labels'}

    def __init__(self, obj, *, allows_duplicate_labels):
        self._allows_duplicate_labels = allows_duplicate_labels
        self._obj = weakref.ref(obj)

    @property
    def allows_duplicate_labels(self):
        '\n        Whether this object allows duplicate labels.\n\n        Setting ``allows_duplicate_labels=False`` ensures that the\n        index (and columns of a DataFrame) are unique. Most methods\n        that accept and return a Series or DataFrame will propagate\n        the value of ``allows_duplicate_labels``.\n\n        See :ref:`duplicates` for more.\n\n        See Also\n        --------\n        DataFrame.attrs : Set global metadata on this object.\n        DataFrame.set_flags : Set global flags on this object.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2]}, index=[\'a\', \'a\'])\n        >>> df.allows_duplicate_labels\n        True\n        >>> df.allows_duplicate_labels = False\n        Traceback (most recent call last):\n            ...\n        pandas.errors.DuplicateLabelError: Index has duplicates.\n              positions\n        label\n        a        [0, 1]\n        '
        return self._allows_duplicate_labels

    @allows_duplicate_labels.setter
    def allows_duplicate_labels(self, value):
        value = bool(value)
        obj = self._obj()
        if (obj is None):
            raise ValueError("This flag's object has been deleted.")
        if (not value):
            for ax in obj.axes:
                ax._maybe_check_unique()
        self._allows_duplicate_labels = value

    def __getitem__(self, key):
        if (key not in self._keys):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if (key not in self._keys):
            raise ValueError(f'Unknown flag {key}. Must be one of {self._keys}')
        setattr(self, key, value)

    def __repr__(self):
        return f'<Flags(allows_duplicate_labels={self.allows_duplicate_labels})>'

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.allows_duplicate_labels == other.allows_duplicate_labels)
        return False
