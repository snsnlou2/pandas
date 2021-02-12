
'\nThe config module holds package-wide configurables and provides\na uniform API for working with them.\n\nOverview\n========\n\nThis module supports the following requirements:\n- options are referenced using keys in dot.notation, e.g. "x.y.option - z".\n- keys are case-insensitive.\n- functions should accept partial/regex keys, when unambiguous.\n- options can be registered by modules at import time.\n- options can be registered at init-time (via core.config_init)\n- options have a default value, and (optionally) a description and\n  validation function associated with them.\n- options can be deprecated, in which case referencing them\n  should produce a warning.\n- deprecated options can optionally be rerouted to a replacement\n  so that accessing a deprecated option reroutes to a differently\n  named option.\n- options can be reset to their default value.\n- all option can be reset to their default value at once.\n- all options in a certain sub - namespace can be reset at once.\n- the user can set / get / reset or ask for the description of an option.\n- a developer can register and mark an option as deprecated.\n- you can register a callback to be invoked when the option value\n  is set or reset. Changing the stored value is considered misuse, but\n  is not verboten.\n\nImplementation\n==============\n\n- Data is stored using nested dictionaries, and should be accessed\n  through the provided API.\n\n- "Registered options" and "Deprecated options" have metadata associated\n  with them, which are stored in auxiliary dictionaries keyed on the\n  fully-qualified key, e.g. "x.y.z.option".\n\n- the config_init module is imported by the package\'s __init__.py file.\n  placing any register_option() calls there will ensure those options\n  are available as soon as pandas is loaded. If you use register_option\n  in a module, it will only be available after that module is imported,\n  which you should be aware of.\n\n- `config_prefix` is a context_manager (for use with the `with` keyword)\n  which can save developers some typing, see the docstring.\n\n'
from collections import namedtuple
from contextlib import ContextDecorator, contextmanager
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, cast
import warnings
from pandas._typing import F
DeprecatedOption = namedtuple('DeprecatedOption', 'key msg rkey removal_ver')
RegisteredOption = namedtuple('RegisteredOption', 'key defval doc validator cb')
_deprecated_options = {}
_registered_options = {}
_global_config = {}
_reserved_keys = ['all']

class OptionError(AttributeError, KeyError):
    '\n    Exception for pandas.options, backwards compatible with KeyError\n    checks\n    '

def _get_single_key(pat, silent):
    keys = _select_options(pat)
    if (len(keys) == 0):
        if (not silent):
            _warn_if_deprecated(pat)
        raise OptionError(f'No such keys(s): {repr(pat)}')
    if (len(keys) > 1):
        raise OptionError('Pattern matched multiple keys')
    key = keys[0]
    if (not silent):
        _warn_if_deprecated(key)
    key = _translate_key(key)
    return key

def _get_option(pat, silent=False):
    key = _get_single_key(pat, silent)
    (root, k) = _get_root(key)
    return root[k]

def _set_option(*args, **kwargs):
    nargs = len(args)
    if ((not nargs) or ((nargs % 2) != 0)):
        raise ValueError('Must provide an even number of non-keyword arguments')
    silent = kwargs.pop('silent', False)
    if kwargs:
        kwarg = list(kwargs.keys())[0]
        raise TypeError(f'_set_option() got an unexpected keyword argument "{kwarg}"')
    for (k, v) in zip(args[::2], args[1::2]):
        key = _get_single_key(k, silent)
        o = _get_registered_option(key)
        if (o and o.validator):
            o.validator(v)
        (root, k) = _get_root(key)
        root[k] = v
        if o.cb:
            if silent:
                with warnings.catch_warnings(record=True):
                    o.cb(key)
            else:
                o.cb(key)

def _describe_option(pat='', _print_desc=True):
    keys = _select_options(pat)
    if (len(keys) == 0):
        raise OptionError('No such keys(s)')
    s = '\n'.join([_build_option_description(k) for k in keys])
    if _print_desc:
        print(s)
    else:
        return s

def _reset_option(pat, silent=False):
    keys = _select_options(pat)
    if (len(keys) == 0):
        raise OptionError('No such keys(s)')
    if ((len(keys) > 1) and (len(pat) < 4) and (pat != 'all')):
        raise ValueError('You must specify at least 4 characters when resetting multiple keys, use the special keyword "all" to reset all the options to their default value')
    for k in keys:
        _set_option(k, _registered_options[k].defval, silent=silent)

def get_default_val(pat):
    key = _get_single_key(pat, silent=True)
    return _get_registered_option(key).defval

class DictWrapper():
    ' provide attribute-style access to a nested dict'

    def __init__(self, d, prefix=''):
        object.__setattr__(self, 'd', d)
        object.__setattr__(self, 'prefix', prefix)

    def __setattr__(self, key, val):
        prefix = object.__getattribute__(self, 'prefix')
        if prefix:
            prefix += '.'
        prefix += key
        if ((key in self.d) and (not isinstance(self.d[key], dict))):
            _set_option(prefix, val)
        else:
            raise OptionError('You can only set the value of existing options')

    def __getattr__(self, key):
        prefix = object.__getattribute__(self, 'prefix')
        if prefix:
            prefix += '.'
        prefix += key
        try:
            v = object.__getattribute__(self, 'd')[key]
        except KeyError as err:
            raise OptionError('No such option') from err
        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return _get_option(prefix)

    def __dir__(self):
        return list(self.d.keys())

class CallableDynamicDoc():

    def __init__(self, func, doc_tmpl):
        self.__doc_tmpl__ = doc_tmpl
        self.__func__ = func

    def __call__(self, *args, **kwds):
        return self.__func__(*args, **kwds)

    @property
    def __doc__(self):
        opts_desc = _describe_option('all', _print_desc=False)
        opts_list = pp_options_list(list(_registered_options.keys()))
        return self.__doc_tmpl__.format(opts_desc=opts_desc, opts_list=opts_list)
_get_option_tmpl = '\nget_option(pat)\n\nRetrieves the value of the specified option.\n\nAvailable options:\n\n{opts_list}\n\nParameters\n----------\npat : str\n    Regexp which should match a single option.\n    Note: partial matches are supported for convenience, but unless you use the\n    full option name (e.g. x.y.z.option_name), your code may break in future\n    versions if new options with similar names are introduced.\n\nReturns\n-------\nresult : the value of the option\n\nRaises\n------\nOptionError : if no such option exists\n\nNotes\n-----\nThe available options with its descriptions:\n\n{opts_desc}\n'
_set_option_tmpl = '\nset_option(pat, value)\n\nSets the value of the specified option.\n\nAvailable options:\n\n{opts_list}\n\nParameters\n----------\npat : str\n    Regexp which should match a single option.\n    Note: partial matches are supported for convenience, but unless you use the\n    full option name (e.g. x.y.z.option_name), your code may break in future\n    versions if new options with similar names are introduced.\nvalue : object\n    New value of option.\n\nReturns\n-------\nNone\n\nRaises\n------\nOptionError if no such option exists\n\nNotes\n-----\nThe available options with its descriptions:\n\n{opts_desc}\n'
_describe_option_tmpl = '\ndescribe_option(pat, _print_desc=False)\n\nPrints the description for one or more registered options.\n\nCall with not arguments to get a listing for all registered options.\n\nAvailable options:\n\n{opts_list}\n\nParameters\n----------\npat : str\n    Regexp pattern. All matching keys will have their description displayed.\n_print_desc : bool, default True\n    If True (default) the description(s) will be printed to stdout.\n    Otherwise, the description(s) will be returned as a unicode string\n    (for testing).\n\nReturns\n-------\nNone by default, the description(s) as a unicode string if _print_desc\nis False\n\nNotes\n-----\nThe available options with its descriptions:\n\n{opts_desc}\n'
_reset_option_tmpl = '\nreset_option(pat)\n\nReset one or more options to their default value.\n\nPass "all" as argument to reset all options.\n\nAvailable options:\n\n{opts_list}\n\nParameters\n----------\npat : str/regex\n    If specified only options matching `prefix*` will be reset.\n    Note: partial matches are supported for convenience, but unless you\n    use the full option name (e.g. x.y.z.option_name), your code may break\n    in future versions if new options with similar names are introduced.\n\nReturns\n-------\nNone\n\nNotes\n-----\nThe available options with its descriptions:\n\n{opts_desc}\n'
get_option = CallableDynamicDoc(_get_option, _get_option_tmpl)
set_option = CallableDynamicDoc(_set_option, _set_option_tmpl)
reset_option = CallableDynamicDoc(_reset_option, _reset_option_tmpl)
describe_option = CallableDynamicDoc(_describe_option, _describe_option_tmpl)
options = DictWrapper(_global_config)

class option_context(ContextDecorator):
    "\n    Context manager to temporarily set options in the `with` statement context.\n\n    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.\n\n    Examples\n    --------\n    >>> with option_context('display.max_rows', 10, 'display.max_columns', 5):\n    ...     ...\n    "

    def __init__(self, *args):
        if (((len(args) % 2) != 0) or (len(args) < 2)):
            raise ValueError('Need to invoke as option_context(pat, val, [(pat, val), ...]).')
        self.ops = list(zip(args[::2], args[1::2]))

    def __enter__(self):
        self.undo = [(pat, _get_option(pat, silent=True)) for (pat, val) in self.ops]
        for (pat, val) in self.ops:
            _set_option(pat, val, silent=True)

    def __exit__(self, *args):
        if self.undo:
            for (pat, val) in self.undo:
                _set_option(pat, val, silent=True)

def register_option(key, defval, doc='', validator=None, cb=None):
    '\n    Register an option in the package-wide pandas config object\n\n    Parameters\n    ----------\n    key : str\n        Fully-qualified key, e.g. "x.y.option - z".\n    defval : object\n        Default value of the option.\n    doc : str\n        Description of the option.\n    validator : Callable, optional\n        Function of a single argument, should raise `ValueError` if\n        called with a value which is not a legal value for the option.\n    cb\n        a function of a single argument "key", which is called\n        immediately after an option value is set/reset. key is\n        the full name of the option.\n\n    Raises\n    ------\n    ValueError if `validator` is specified and `defval` is not a valid value.\n\n    '
    import keyword
    import tokenize
    key = key.lower()
    if (key in _registered_options):
        raise OptionError(f"Option '{key}' has already been registered")
    if (key in _reserved_keys):
        raise OptionError(f"Option '{key}' is a reserved key")
    if validator:
        validator(defval)
    path = key.split('.')
    for k in path:
        if (not re.match((('^' + tokenize.Name) + '$'), k)):
            raise ValueError(f'{k} is not a valid identifier')
        if keyword.iskeyword(k):
            raise ValueError(f'{k} is a python keyword')
    cursor = _global_config
    msg = "Path prefix to option '{option}' is already an option"
    for (i, p) in enumerate(path[:(- 1)]):
        if (not isinstance(cursor, dict)):
            raise OptionError(msg.format(option='.'.join(path[:i])))
        if (p not in cursor):
            cursor[p] = {}
        cursor = cursor[p]
    if (not isinstance(cursor, dict)):
        raise OptionError(msg.format(option='.'.join(path[:(- 1)])))
    cursor[path[(- 1)]] = defval
    _registered_options[key] = RegisteredOption(key=key, defval=defval, doc=doc, validator=validator, cb=cb)

def deprecate_option(key, msg=None, rkey=None, removal_ver=None):
    '\n    Mark option `key` as deprecated, if code attempts to access this option,\n    a warning will be produced, using `msg` if given, or a default message\n    if not.\n    if `rkey` is given, any access to the key will be re-routed to `rkey`.\n\n    Neither the existence of `key` nor that if `rkey` is checked. If they\n    do not exist, any subsequence access will fail as usual, after the\n    deprecation warning is given.\n\n    Parameters\n    ----------\n    key : str\n        Name of the option to be deprecated.\n        must be a fully-qualified option name (e.g "x.y.z.rkey").\n    msg : str, optional\n        Warning message to output when the key is referenced.\n        if no message is given a default message will be emitted.\n    rkey : str, optional\n        Name of an option to reroute access to.\n        If specified, any referenced `key` will be\n        re-routed to `rkey` including set/get/reset.\n        rkey must be a fully-qualified option name (e.g "x.y.z.rkey").\n        used by the default message if no `msg` is specified.\n    removal_ver : optional\n        Specifies the version in which this option will\n        be removed. used by the default message if no `msg` is specified.\n\n    Raises\n    ------\n    OptionError\n        If the specified key has already been deprecated.\n    '
    key = key.lower()
    if (key in _deprecated_options):
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")
    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)

def _select_options(pat):
    '\n    returns a list of keys matching `pat`\n\n    if pat=="all", returns all registered options\n    '
    if (pat in _registered_options):
        return [pat]
    keys = sorted(_registered_options.keys())
    if (pat == 'all'):
        return keys
    return [k for k in keys if re.search(pat, k, re.I)]

def _get_root(key):
    path = key.split('.')
    cursor = _global_config
    for p in path[:(- 1)]:
        cursor = cursor[p]
    return (cursor, path[(- 1)])

def _is_deprecated(key):
    ' Returns True if the given option has been deprecated '
    key = key.lower()
    return (key in _deprecated_options)

def _get_deprecated_option(key):
    '\n    Retrieves the metadata for a deprecated option, if `key` is deprecated.\n\n    Returns\n    -------\n    DeprecatedOption (namedtuple) if key is deprecated, None otherwise\n    '
    try:
        d = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d

def _get_registered_option(key):
    '\n    Retrieves the option metadata if `key` is a registered option.\n\n    Returns\n    -------\n    RegisteredOption (namedtuple) if key is deprecated, None otherwise\n    '
    return _registered_options.get(key)

def _translate_key(key):
    '\n    if key id deprecated and a replacement key defined, will return the\n    replacement key, otherwise returns `key` as - is\n    '
    d = _get_deprecated_option(key)
    if d:
        return (d.rkey or key)
    else:
        return key

def _warn_if_deprecated(key):
    '\n    Checks if `key` is a deprecated option and if so, prints a warning.\n\n    Returns\n    -------\n    bool - True if `key` is deprecated, False otherwise.\n    '
    d = _get_deprecated_option(key)
    if d:
        if d.msg:
            print(d.msg)
            warnings.warn(d.msg, FutureWarning)
        else:
            msg = f"'{key}' is deprecated"
            if d.removal_ver:
                msg += f' and will be removed in {d.removal_ver}'
            if d.rkey:
                msg += f", please use '{d.rkey}' instead."
            else:
                msg += ', please refrain from using it.'
            warnings.warn(msg, FutureWarning)
        return True
    return False

def _build_option_description(k):
    ' Builds a formatted description of a registered option and prints it '
    o = _get_registered_option(k)
    d = _get_deprecated_option(k)
    s = f'{k} '
    if o.doc:
        s += '\n'.join(o.doc.strip().split('\n'))
    else:
        s += 'No description available.'
    if o:
        s += f'''
    [default: {o.defval}] [currently: {_get_option(k, True)}]'''
    if d:
        rkey = (d.rkey or '')
        s += '\n    (Deprecated'
        s += f', use `{rkey}` instead.'
        s += ')'
    return s

def pp_options_list(keys, width=80, _print=False):
    ' Builds a concise listing of available options, grouped by prefix '
    from itertools import groupby
    from textwrap import wrap

    def pp(name: str, ks: Iterable[str]) -> List[str]:
        pfx = ((('- ' + name) + '.[') if name else '')
        ls = wrap(', '.join(ks), width, initial_indent=pfx, subsequent_indent='  ', break_long_words=False)
        if (ls and ls[(- 1)] and name):
            ls[(- 1)] = (ls[(- 1)] + ']')
        return ls
    ls: List[str] = []
    singles = [x for x in sorted(keys) if (x.find('.') < 0)]
    if singles:
        ls += pp('', singles)
    keys = [x for x in keys if (x.find('.') >= 0)]
    for (k, g) in groupby(sorted(keys), (lambda x: x[:x.rfind('.')])):
        ks = [x[(len(k) + 1):] for x in list(g)]
        ls += pp(k, ks)
    s = '\n'.join(ls)
    if _print:
        print(s)
    else:
        return s

@contextmanager
def config_prefix(prefix):
    '\n    contextmanager for multiple invocations of API with a common prefix\n\n    supported API functions: (register / get / set )__option\n\n    Warning: This is not thread - safe, and won\'t work properly if you import\n    the API functions into your module using the "from x import y" construct.\n\n    Example\n    -------\n    import pandas._config.config as cf\n    with cf.config_prefix("display.font"):\n        cf.register_option("color", "red")\n        cf.register_option("size", " 5 pt")\n        cf.set_option(size, " 6 pt")\n        cf.get_option(size)\n        ...\n\n        etc\'\n\n    will register options "display.font.color", "display.font.size", set the\n    value of "display.font.size"... and so on.\n    '
    global register_option, get_option, set_option, reset_option

    def wrap(func: F) -> F:

        def inner(key: str, *args, **kwds):
            pkey = f'{prefix}.{key}'
            return func(pkey, *args, **kwds)
        return cast(F, inner)
    _register_option = register_option
    _get_option = get_option
    _set_option = set_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    (yield None)
    set_option = _set_option
    get_option = _get_option
    register_option = _register_option

def is_type_factory(_type):
    '\n\n    Parameters\n    ----------\n    `_type` - a type to be compared against (e.g. type(x) == `_type`)\n\n    Returns\n    -------\n    validator - a function of a single argument x , which raises\n                ValueError if type(x) is not equal to `_type`\n\n    '

    def inner(x) -> None:
        if (type(x) != _type):
            raise ValueError(f"Value must have type '{_type}'")
    return inner

def is_instance_factory(_type):
    '\n\n    Parameters\n    ----------\n    `_type` - the type to be checked against\n\n    Returns\n    -------\n    validator - a function of a single argument x , which raises\n                ValueError if x is not an instance of `_type`\n\n    '
    if isinstance(_type, (tuple, list)):
        _type = tuple(_type)
        type_repr = '|'.join(map(str, _type))
    else:
        type_repr = f"'{_type}'"

    def inner(x) -> None:
        if (not isinstance(x, _type)):
            raise ValueError(f'Value must be an instance of {type_repr}')
    return inner

def is_one_of_factory(legal_values):
    callables = [c for c in legal_values if callable(c)]
    legal_values = [c for c in legal_values if (not callable(c))]

    def inner(x) -> None:
        if (x not in legal_values):
            if (not any((c(x) for c in callables))):
                uvals = [str(lval) for lval in legal_values]
                pp_values = '|'.join(uvals)
                msg = f'Value must be one of {pp_values}'
                if len(callables):
                    msg += ' or a callable'
                raise ValueError(msg)
    return inner

def is_nonnegative_int(value):
    '\n    Verify that value is None or a positive int.\n\n    Parameters\n    ----------\n    value : None or int\n            The `value` to be checked.\n\n    Raises\n    ------\n    ValueError\n        When the value is not None or is a negative integer\n    '
    if (value is None):
        return
    elif isinstance(value, int):
        if (value >= 0):
            return
    msg = 'Value must be a nonnegative integer or None'
    raise ValueError(msg)
is_int = is_type_factory(int)
is_bool = is_type_factory(bool)
is_float = is_type_factory(float)
is_str = is_type_factory(str)
is_text = is_instance_factory((str, bytes))

def is_callable(obj):
    '\n\n    Parameters\n    ----------\n    `obj` - the object to be checked\n\n    Returns\n    -------\n    validator - returns True if object is callable\n        raises ValueError otherwise.\n\n    '
    if (not callable(obj)):
        raise ValueError('Value must be a callable')
    return True
