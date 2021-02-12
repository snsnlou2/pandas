
'\n\naccessor.py contains base classes for implementing accessor properties\nthat can be mixed into or pinned onto other pandas classes.\n\n'
from typing import FrozenSet, List, Set
import warnings
from pandas.util._decorators import doc

class DirNamesMixin():
    _accessors = set()
    _hidden_attrs = frozenset()

    def _dir_deletions(self):
        '\n        Delete unwanted __dir__ for this object.\n        '
        return (self._accessors | self._hidden_attrs)

    def _dir_additions(self):
        '\n        Add additional __dir__ for this object.\n        '
        return {accessor for accessor in self._accessors if hasattr(self, accessor)}

    def __dir__(self):
        "\n        Provide method name lookup and completion.\n\n        Notes\n        -----\n        Only provide 'public' methods.\n        "
        rv = set(super().__dir__())
        rv = ((rv - self._dir_deletions()) | self._dir_additions())
        return sorted(rv)

class PandasDelegate():
    '\n    Abstract base class for delegating methods/properties.\n    '

    def _delegate_property_get(self, name, *args, **kwargs):
        raise TypeError(f'You cannot access the property {name}')

    def _delegate_property_set(self, name, value, *args, **kwargs):
        raise TypeError(f'The property {name} cannot be set')

    def _delegate_method(self, name, *args, **kwargs):
        raise TypeError(f'You cannot call method {name}')

    @classmethod
    def _add_delegate_accessors(cls, delegate, accessors, typ, overwrite=False):
        "\n        Add accessors to cls from the delegate class.\n\n        Parameters\n        ----------\n        cls\n            Class to add the methods/properties to.\n        delegate\n            Class to get methods/properties and doc-strings.\n        accessors : list of str\n            List of accessors to add.\n        typ : {'property', 'method'}\n        overwrite : bool, default False\n            Overwrite the method/property in the target class if it exists.\n        "

        def _create_delegator_property(name):

            def _getter(self):
                return self._delegate_property_get(name)

            def _setter(self, new_values):
                return self._delegate_property_set(name, new_values)
            _getter.__name__ = name
            _setter.__name__ = name
            return property(fget=_getter, fset=_setter, doc=getattr(delegate, name).__doc__)

        def _create_delegator_method(name):

            def f(self, *args, **kwargs):
                return self._delegate_method(name, *args, **kwargs)
            f.__name__ = name
            f.__doc__ = getattr(delegate, name).__doc__
            return f
        for name in accessors:
            if (typ == 'property'):
                f = _create_delegator_property(name)
            else:
                f = _create_delegator_method(name)
            if (overwrite or (not hasattr(cls, name))):
                setattr(cls, name, f)

def delegate_names(delegate, accessors, typ, overwrite=False):
    '\n    Add delegated names to a class using a class decorator.  This provides\n    an alternative usage to directly calling `_add_delegate_accessors`\n    below a class definition.\n\n    Parameters\n    ----------\n    delegate : object\n        The class to get methods/properties & doc-strings.\n    accessors : Sequence[str]\n        List of accessor to add.\n    typ : {\'property\', \'method\'}\n    overwrite : bool, default False\n       Overwrite the method/property in the target class if it exists.\n\n    Returns\n    -------\n    callable\n        A class decorator.\n\n    Examples\n    --------\n    @delegate_names(Categorical, ["categories", "ordered"], "property")\n    class CategoricalAccessor(PandasDelegate):\n        [...]\n    '

    def add_delegate_accessors(cls):
        cls._add_delegate_accessors(delegate, accessors, typ, overwrite=overwrite)
        return cls
    return add_delegate_accessors

class CachedAccessor():
    "\n    Custom property-like object.\n\n    A descriptor for caching accessors.\n\n    Parameters\n    ----------\n    name : str\n        Namespace that will be accessed under, e.g. ``df.foo``.\n    accessor : cls\n        Class with the extension methods.\n\n    Notes\n    -----\n    For accessor, The class's __init__ method assumes that one of\n    ``Series``, ``DataFrame`` or ``Index`` as the\n    single argument ``data``.\n    "

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if (obj is None):
            return self._accessor
        accessor_obj = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj

@doc(klass='', others='')
def _register_accessor(name, cls):
    '\n    Register a custom accessor on {klass} objects.\n\n    Parameters\n    ----------\n    name : str\n        Name under which the accessor should be registered. A warning is issued\n        if this name conflicts with a preexisting attribute.\n\n    Returns\n    -------\n    callable\n        A class decorator.\n\n    See Also\n    --------\n    register_dataframe_accessor : Register a custom accessor on DataFrame objects.\n    register_series_accessor : Register a custom accessor on Series objects.\n    register_index_accessor : Register a custom accessor on Index objects.\n\n    Notes\n    -----\n    When accessed, your accessor will be initialized with the pandas object\n    the user is interacting with. So the signature must be\n\n    .. code-block:: python\n\n        def __init__(self, pandas_object):  # noqa: E999\n            ...\n\n    For consistency with pandas methods, you should raise an ``AttributeError``\n    if the data passed to your accessor has an incorrect dtype.\n\n    >>> pd.Series([\'a\', \'b\']).dt\n    Traceback (most recent call last):\n    ...\n    AttributeError: Can only use .dt accessor with datetimelike values\n\n    Examples\n    --------\n    In your library code::\n\n        import pandas as pd\n\n        @pd.api.extensions.register_dataframe_accessor("geo")\n        class GeoAccessor:\n            def __init__(self, pandas_obj):\n                self._obj = pandas_obj\n\n            @property\n            def center(self):\n                # return the geographic center point of this DataFrame\n                lat = self._obj.latitude\n                lon = self._obj.longitude\n                return (float(lon.mean()), float(lat.mean()))\n\n            def plot(self):\n                # plot this array\'s data on a map, e.g., using Cartopy\n                pass\n\n    Back in an interactive IPython session:\n\n        .. code-block:: ipython\n\n            In [1]: ds = pd.DataFrame({{"longitude": np.linspace(0, 10),\n               ...:                    "latitude": np.linspace(0, 20)}})\n            In [2]: ds.geo.center\n            Out[2]: (5.0, 10.0)\n            In [3]: ds.geo.plot()  # plots data on a map\n    '

    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(f'registration of accessor {repr(accessor)} under name {repr(name)} for type {repr(cls)} is overriding a preexisting attribute with the same name.', UserWarning, stacklevel=2)
        setattr(cls, name, CachedAccessor(name, accessor))
        cls._accessors.add(name)
        return accessor
    return decorator

@doc(_register_accessor, klass='DataFrame')
def register_dataframe_accessor(name):
    from pandas import DataFrame
    return _register_accessor(name, DataFrame)

@doc(_register_accessor, klass='Series')
def register_series_accessor(name):
    from pandas import Series
    return _register_accessor(name, Series)

@doc(_register_accessor, klass='Index')
def register_index_accessor(name):
    from pandas import Index
    return _register_accessor(name, Index)
