
'\nModule for scope operations\n'
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import List
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.compat.chainmap import DeepChainMap

def ensure_scope(level, global_dict=None, local_dict=None, resolvers=(), target=None, **kwargs):
    'Ensure that we are grabbing the correct scope.'
    return Scope((level + 1), global_dict=global_dict, local_dict=local_dict, resolvers=resolvers, target=target)

def _replacer(x):
    "\n    Replace a number with its hexadecimal representation. Used to tag\n    temporary variables with their calling scope's id.\n    "
    try:
        hexin = ord(x)
    except TypeError:
        hexin = x
    return hex(hexin)

def _raw_hex_id(obj):
    'Return the padded hexadecimal id of ``obj``.'
    packed = struct.pack('@P', id(obj))
    return ''.join((_replacer(x) for x in packed))
DEFAULT_GLOBALS = {'Timestamp': Timestamp, 'datetime': datetime.datetime, 'True': True, 'False': False, 'list': list, 'tuple': tuple, 'inf': np.inf, 'Inf': np.inf}

def _get_pretty_string(obj):
    '\n    Return a prettier version of obj.\n\n    Parameters\n    ----------\n    obj : object\n        Object to pretty print\n\n    Returns\n    -------\n    str\n        Pretty print object repr\n    '
    sio = StringIO()
    pprint.pprint(obj, stream=sio)
    return sio.getvalue()

class Scope():
    '\n    Object to hold scope, with a few bells to deal with some custom syntax\n    and contexts added by pandas.\n\n    Parameters\n    ----------\n    level : int\n    global_dict : dict or None, optional, default None\n    local_dict : dict or Scope or None, optional, default None\n    resolvers : list-like or None, optional, default None\n    target : object\n\n    Attributes\n    ----------\n    level : int\n    scope : DeepChainMap\n    target : object\n    temps : dict\n    '
    __slots__ = ['level', 'scope', 'target', 'resolvers', 'temps']

    def __init__(self, level, global_dict=None, local_dict=None, resolvers=(), target=None):
        self.level = (level + 1)
        self.scope = DeepChainMap(DEFAULT_GLOBALS.copy())
        self.target = target
        if isinstance(local_dict, Scope):
            self.scope.update(local_dict.scope)
            if (local_dict.target is not None):
                self.target = local_dict.target
            self._update(local_dict.level)
        frame = sys._getframe(self.level)
        try:
            self.scope = self.scope.new_child((global_dict or frame.f_globals).copy())
            if (not isinstance(local_dict, Scope)):
                self.scope = self.scope.new_child((local_dict or frame.f_locals).copy())
        finally:
            del frame
        if isinstance(local_dict, Scope):
            resolvers += tuple(local_dict.resolvers.maps)
        self.resolvers = DeepChainMap(*resolvers)
        self.temps = {}

    def __repr__(self):
        scope_keys = _get_pretty_string(list(self.scope.keys()))
        res_keys = _get_pretty_string(list(self.resolvers.keys()))
        return f'{type(self).__name__}(scope={scope_keys}, resolvers={res_keys})'

    @property
    def has_resolvers(self):
        '\n        Return whether we have any extra scope.\n\n        For example, DataFrames pass Their columns as resolvers during calls to\n        ``DataFrame.eval()`` and ``DataFrame.query()``.\n\n        Returns\n        -------\n        hr : bool\n        '
        return bool(len(self.resolvers))

    def resolve(self, key, is_local):
        "\n        Resolve a variable name in a possibly local context.\n\n        Parameters\n        ----------\n        key : str\n            A variable name\n        is_local : bool\n            Flag indicating whether the variable is local or not (prefixed with\n            the '@' symbol)\n\n        Returns\n        -------\n        value : object\n            The value of a particular variable\n        "
        try:
            if is_local:
                return self.scope[key]
            if self.has_resolvers:
                return self.resolvers[key]
            assert ((not is_local) and (not self.has_resolvers))
            return self.scope[key]
        except KeyError:
            try:
                return self.temps[key]
            except KeyError as err:
                from pandas.core.computation.ops import UndefinedVariableError
                raise UndefinedVariableError(key, is_local) from err

    def swapkey(self, old_key, new_key, new_value=None):
        '\n        Replace a variable name, with a potentially new value.\n\n        Parameters\n        ----------\n        old_key : str\n            Current variable name to replace\n        new_key : str\n            New variable name to replace `old_key` with\n        new_value : object\n            Value to be replaced along with the possible renaming\n        '
        if self.has_resolvers:
            maps = (self.resolvers.maps + self.scope.maps)
        else:
            maps = self.scope.maps
        maps.append(self.temps)
        for mapping in maps:
            if (old_key in mapping):
                mapping[new_key] = new_value
                return

    def _get_vars(self, stack, scopes):
        "\n        Get specifically scoped variables from a list of stack frames.\n\n        Parameters\n        ----------\n        stack : list\n            A list of stack frames as returned by ``inspect.stack()``\n        scopes : sequence of strings\n            A sequence containing valid stack frame attribute names that\n            evaluate to a dictionary. For example, ('locals', 'globals')\n        "
        variables = itertools.product(scopes, stack)
        for (scope, (frame, _, _, _, _, _)) in variables:
            try:
                d = getattr(frame, ('f_' + scope))
                self.scope = self.scope.new_child(d)
            finally:
                del frame

    def _update(self, level):
        '\n        Update the current scope by going back `level` levels.\n\n        Parameters\n        ----------\n        level : int\n        '
        sl = (level + 1)
        stack = inspect.stack()
        try:
            self._get_vars(stack[:sl], scopes=['locals'])
        finally:
            del stack[:], stack

    def add_tmp(self, value):
        '\n        Add a temporary variable to the scope.\n\n        Parameters\n        ----------\n        value : object\n            An arbitrary object to be assigned to a temporary variable.\n\n        Returns\n        -------\n        str\n            The name of the temporary variable created.\n        '
        name = f'{type(value).__name__}_{self.ntemps}_{_raw_hex_id(self)}'
        assert (name not in self.temps)
        self.temps[name] = value
        assert (name in self.temps)
        return name

    @property
    def ntemps(self):
        'The number of temporary variables in this scope'
        return len(self.temps)

    @property
    def full_scope(self):
        '\n        Return the full scope for use with passing to engines transparently\n        as a mapping.\n\n        Returns\n        -------\n        vars : DeepChainMap\n            All variables in this scope.\n        '
        maps = (([self.temps] + self.resolvers.maps) + self.scope.maps)
        return DeepChainMap(*maps)
