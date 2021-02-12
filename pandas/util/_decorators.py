
from functools import wraps
import inspect
from textwrap import dedent
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, Union, cast
import warnings
from pandas._libs.properties import cache_readonly
from pandas._typing import F

def deprecate(name, alternative, version, alt_name=None, klass=None, stacklevel=2, msg=None):
    "\n    Return a new function that emits a deprecation warning on use.\n\n    To use this method for a deprecated function, another function\n    `alternative` with the same signature must exist. The deprecated\n    function will emit a deprecation warning, and in the docstring\n    it will contain the deprecation directive with the provided version\n    so it can be detected for future removal.\n\n    Parameters\n    ----------\n    name : str\n        Name of function to deprecate.\n    alternative : func\n        Function to use instead.\n    version : str\n        Version of pandas in which the method has been deprecated.\n    alt_name : str, optional\n        Name to use in preference of alternative.__name__.\n    klass : Warning, default FutureWarning\n    stacklevel : int, default 2\n    msg : str\n        The message to display in the warning.\n        Default is '{name} is deprecated. Use {alt_name} instead.'\n    "
    alt_name = (alt_name or alternative.__name__)
    klass = (klass or FutureWarning)
    warning_msg = (msg or f'{name} is deprecated, use {alt_name} instead')

    @wraps(alternative)
    def wrapper(*args, **kwargs) -> Callable[(..., Any)]:
        warnings.warn(warning_msg, klass, stacklevel=stacklevel)
        return alternative(*args, **kwargs)
    msg = (msg or f'Use `{alt_name}` instead.')
    doc_error_msg = f'''deprecate needs a correctly formatted docstring in the target function (should have a one liner short summary, and opening quotes should be in their own line). Found:
{alternative.__doc__}'''
    if alternative.__doc__:
        if (alternative.__doc__.count('\n') < 3):
            raise AssertionError(doc_error_msg)
        (empty1, summary, empty2, doc) = alternative.__doc__.split('\n', 3)
        if (empty1 or (empty2 and (not summary))):
            raise AssertionError(doc_error_msg)
        wrapper.__doc__ = dedent(f'''
        {summary.strip()}

        .. deprecated:: {version}
            {msg}

        {dedent(doc)}''')
    return wrapper

def deprecate_kwarg(old_arg_name, new_arg_name, mapping=None, stacklevel=2):
    '\n    Decorator to deprecate a keyword argument of a function.\n\n    Parameters\n    ----------\n    old_arg_name : str\n        Name of argument in function to deprecate\n    new_arg_name : str or None\n        Name of preferred argument in function. Use None to raise warning that\n        ``old_arg_name`` keyword is deprecated.\n    mapping : dict or callable\n        If mapping is present, use it to translate old arguments to\n        new arguments. A callable must do its own value checking;\n        values not found in a dict will be forwarded unchanged.\n\n    Examples\n    --------\n    The following deprecates \'cols\', using \'columns\' instead\n\n    >>> @deprecate_kwarg(old_arg_name=\'cols\', new_arg_name=\'columns\')\n    ... def f(columns=\'\'):\n    ...     print(columns)\n    ...\n    >>> f(columns=\'should work ok\')\n    should work ok\n\n    >>> f(cols=\'should raise warning\')\n    FutureWarning: cols is deprecated, use columns instead\n      warnings.warn(msg, FutureWarning)\n    should raise warning\n\n    >>> f(cols=\'should error\', columns="can\'t pass do both")\n    TypeError: Can only specify \'cols\' or \'columns\', not both\n\n    >>> @deprecate_kwarg(\'old\', \'new\', {\'yes\': True, \'no\': False})\n    ... def f(new=False):\n    ...     print(\'yes!\' if new else \'no!\')\n    ...\n    >>> f(old=\'yes\')\n    FutureWarning: old=\'yes\' is deprecated, use new=True instead\n      warnings.warn(msg, FutureWarning)\n    yes!\n\n    To raise a warning that a keyword will be removed entirely in the future\n\n    >>> @deprecate_kwarg(old_arg_name=\'cols\', new_arg_name=None)\n    ... def f(cols=\'\', another_param=\'\'):\n    ...     print(cols)\n    ...\n    >>> f(cols=\'should raise warning\')\n    FutureWarning: the \'cols\' keyword is deprecated and will be removed in a\n    future version please takes steps to stop use of \'cols\'\n    should raise warning\n    >>> f(another_param=\'should not raise warning\')\n    should not raise warning\n\n    >>> f(cols=\'should raise warning\', another_param=\'\')\n    FutureWarning: the \'cols\' keyword is deprecated and will be removed in a\n    future version please takes steps to stop use of \'cols\'\n    should raise warning\n    '
    if ((mapping is not None) and (not hasattr(mapping, 'get')) and (not callable(mapping))):
        raise TypeError('mapping from old to new argument values must be dict or callable!')

    def _deprecate_kwarg(func: F) -> F:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[(..., Any)]:
            old_arg_value = kwargs.pop(old_arg_name, None)
            if (old_arg_value is not None):
                if (new_arg_name is None):
                    msg = f'the {repr(old_arg_name)} keyword is deprecated and will be removed in a future version. Please take steps to stop the use of {repr(old_arg_name)}'
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)
                elif (mapping is not None):
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = f'the {old_arg_name}={repr(old_arg_value)} keyword is deprecated, use {new_arg_name}={repr(new_arg_value)} instead'
                else:
                    new_arg_value = old_arg_value
                    msg = f"the {repr(old_arg_name)}' keyword is deprecated, use {repr(new_arg_name)} instead"
                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if (kwargs.get(new_arg_name) is not None):
                    msg = f'Can only specify {repr(old_arg_name)} or {repr(new_arg_name)}, not both'
                    raise TypeError(msg)
                else:
                    kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return _deprecate_kwarg

def _format_argument_list(allow_args):
    '\n    Convert the allow_args argument (either string or integer) of\n    `deprecate_nonkeyword_arguments` function to a string describing\n    it to be inserted into warning message.\n\n    Parameters\n    ----------\n    allowed_args : list, tuple or int\n        The `allowed_args` argument for `deprecate_nonkeyword_arguments`,\n        but None value is not allowed.\n\n    Returns\n    -------\n    s : str\n        The substring describing the argument list in best way to be\n        inserted to the warning message.\n\n    Examples\n    --------\n    `format_argument_list(0)` -> \'\'\n    `format_argument_list(1)` -> \'except for the first argument\'\n    `format_argument_list(2)` -> \'except for the first 2 arguments\'\n    `format_argument_list([])` -> \'\'\n    `format_argument_list([\'a\'])` -> "except for the arguments \'a\'"\n    `format_argument_list([\'a\', \'b\'])` -> "except for the arguments \'a\' and \'b\'"\n    `format_argument_list([\'a\', \'b\', \'c\'])` ->\n        "except for the arguments \'a\', \'b\' and \'c\'"\n    '
    if (not allow_args):
        return ''
    elif (allow_args == 1):
        return ' except for the first argument'
    elif isinstance(allow_args, int):
        return f' except for the first {allow_args} arguments'
    elif (len(allow_args) == 1):
        return f" except for the argument '{allow_args[0]}'"
    else:
        last = allow_args[(- 1)]
        args = ', '.join([(("'" + x) + "'") for x in allow_args[:(- 1)]])
        return f" except for the arguments {args} and '{last}'"

def deprecate_nonkeyword_arguments(version, allowed_args=None, stacklevel=2):
    '\n    Decorator to deprecate a use of non-keyword arguments of a function.\n\n    Parameters\n    ----------\n    version : str\n        The version in which positional arguments will become\n        keyword-only.\n\n    allowed_args : list or int, optional\n        In case of list, it must be the list of names of some\n        first arguments of the decorated functions that are\n        OK to be given as positional arguments. In case of an\n        integer, this is the number of positional arguments\n        that will stay positional. In case of None value,\n        defaults to list of all arguments not having the\n        default value.\n\n    stacklevel : int, default=2\n        The stack level for warnings.warn\n    '

    def decorate(func):
        if (allowed_args is not None):
            allow_args = allowed_args
        else:
            spec = inspect.getfullargspec(func)
            assert (spec.defaults is not None)
            allow_args = spec.args[:(- len(spec.defaults))]

        @wraps(func)
        def wrapper(*args, **kwargs):
            arguments = _format_argument_list(allow_args)
            if isinstance(allow_args, (list, tuple)):
                num_allow_args = len(allow_args)
            else:
                num_allow_args = allow_args
            if (len(args) > num_allow_args):
                msg = f'Starting with Pandas version {version} all arguments of {func.__name__}{arguments} will be keyword-only'
                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)
        return wrapper
    return decorate

def rewrite_axis_style_signature(name, extra_params):

    def decorate(func: F) -> F:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[(..., Any)]:
            return func(*args, **kwargs)
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        params = [inspect.Parameter('self', kind), inspect.Parameter(name, kind, default=None), inspect.Parameter('index', kind, default=None), inspect.Parameter('columns', kind, default=None), inspect.Parameter('axis', kind, default=None)]
        for (pname, default) in extra_params:
            params.append(inspect.Parameter(pname, kind, default=default))
        sig = inspect.Signature(params)
        func.__signature__ = sig
        return cast(F, wrapper)
    return decorate

def doc(*docstrings, **params):
    '\n    A decorator take docstring templates, concatenate them and perform string\n    substitution on it.\n\n    This decorator will add a variable "_docstring_components" to the wrapped\n    callable to keep track the original docstring template for potential usage.\n    If it should be consider as a template, it will be saved as a string.\n    Otherwise, it will be saved as callable, and later user __doc__ and dedent\n    to get docstring.\n\n    Parameters\n    ----------\n    *docstrings : str or callable\n        The string / docstring / docstring template to be appended in order\n        after default docstring under callable.\n    **params\n        The string which would be used to format docstring template.\n    '

    def decorator(decorated: F) -> F:
        docstring_components: List[Union[(str, Callable)]] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))
        for docstring in docstrings:
            if hasattr(docstring, '_docstring_components'):
                docstring_components.extend(docstring._docstring_components)
            elif (isinstance(docstring, str) or docstring.__doc__):
                docstring_components.append(docstring)
        decorated.__doc__ = ''.join([(component.format(**params) if isinstance(component, str) else dedent((component.__doc__ or ''))) for component in docstring_components])
        decorated._docstring_components = docstring_components
        return decorated
    return decorator

class Substitution():
    '\n    A decorator to take a function\'s docstring and perform string\n    substitution on it.\n\n    This decorator should be robust even if func.__doc__ is None\n    (for example, if -OO was passed to the interpreter)\n\n    Usage: construct a docstring.Substitution with a sequence or\n    dictionary suitable for performing substitution; then\n    decorate a suitable function with the constructed object. e.g.\n\n    sub_author_name = Substitution(author=\'Jason\')\n\n    @sub_author_name\n    def some_function(x):\n        "%(author)s wrote this function"\n\n    # note that some_function.__doc__ is now "Jason wrote this function"\n\n    One can also use positional arguments.\n\n    sub_first_last_names = Substitution(\'Edgar Allen\', \'Poe\')\n\n    @sub_first_last_names\n    def some_function(x):\n        "%s %s wrote the Raven"\n    '

    def __init__(self, *args, **kwargs):
        if (args and kwargs):
            raise AssertionError('Only positional or keyword args are allowed')
        self.params = (args or kwargs)

    def __call__(self, func):
        func.__doc__ = (func.__doc__ and (func.__doc__ % self.params))
        return func

    def update(self, *args, **kwargs):
        '\n        Update self.params with supplied args.\n        '
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)

class Appender():
    '\n    A function decorator that will append an addendum to the docstring\n    of the target function.\n\n    This decorator should be robust even if func.__doc__ is None\n    (for example, if -OO was passed to the interpreter).\n\n    Usage: construct a docstring.Appender with a string to be joined to\n    the original docstring. An optional \'join\' parameter may be supplied\n    which will be used to join the docstring and addendum. e.g.\n\n    add_copyright = Appender("Copyright (c) 2009", join=\'\n\')\n\n    @add_copyright\n    def my_dog(has=\'fleas\'):\n        "This docstring will have a copyright below"\n        pass\n    '

    def __init__(self, addendum, join='', indents=0):
        if (indents > 0):
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func):
        func.__doc__ = (func.__doc__ if func.__doc__ else '')
        self.addendum = (self.addendum if self.addendum else '')
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func

def indent(text, indents=1):
    if ((not text) or (not isinstance(text, str))):
        return ''
    jointext = ''.join((['\n'] + (['    '] * indents)))
    return jointext.join(text.split('\n'))
