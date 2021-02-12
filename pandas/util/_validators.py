
'\nModule that contains many useful utilities\nfor validating data or function arguments\n'
from typing import Iterable, Union
import warnings
import numpy as np
from pandas.core.dtypes.common import is_bool

def _check_arg_length(fname, args, max_fname_arg_count, compat_args):
    "\n    Checks whether 'args' has length of at most 'compat_args'. Raises\n    a TypeError if that is not the case, similar to in Python when a\n    function is called with too many arguments.\n    "
    if (max_fname_arg_count < 0):
        raise ValueError("'max_fname_arg_count' must be non-negative")
    if (len(args) > len(compat_args)):
        max_arg_count = (len(compat_args) + max_fname_arg_count)
        actual_arg_count = (len(args) + max_fname_arg_count)
        argument = ('argument' if (max_arg_count == 1) else 'arguments')
        raise TypeError(f'{fname}() takes at most {max_arg_count} {argument} ({actual_arg_count} given)')

def _check_for_default_values(fname, arg_val_dict, compat_args):
    '\n    Check that the keys in `arg_val_dict` are mapped to their\n    default values as specified in `compat_args`.\n\n    Note that this function is to be called only when it has been\n    checked that arg_val_dict.keys() is a subset of compat_args\n    '
    for key in arg_val_dict:
        try:
            v1 = arg_val_dict[key]
            v2 = compat_args[key]
            if (((v1 is not None) and (v2 is None)) or ((v1 is None) and (v2 is not None))):
                match = False
            else:
                match = (v1 == v2)
            if (not is_bool(match)):
                raise ValueError("'match' is not a boolean")
        except ValueError:
            match = (arg_val_dict[key] is compat_args[key])
        if (not match):
            raise ValueError(f"the '{key}' parameter is not supported in the pandas implementation of {fname}()")

def validate_args(fname, args, max_fname_arg_count, compat_args):
    '\n    Checks whether the length of the `*args` argument passed into a function\n    has at most `len(compat_args)` arguments and whether or not all of these\n    elements in `args` are set to their default values.\n\n    Parameters\n    ----------\n    fname : str\n        The name of the function being passed the `*args` parameter\n    args : tuple\n        The `*args` parameter passed into a function\n    max_fname_arg_count : int\n        The maximum number of arguments that the function `fname`\n        can accept, excluding those in `args`. Used for displaying\n        appropriate error messages. Must be non-negative.\n    compat_args : dict\n        A dictionary of keys and their associated default values.\n        In order to accommodate buggy behaviour in some versions of `numpy`,\n        where a signature displayed keyword arguments but then passed those\n        arguments **positionally** internally when calling downstream\n        implementations, a dict ensures that the original\n        order of the keyword arguments is enforced.\n\n    Raises\n    ------\n    TypeError\n        If `args` contains more values than there are `compat_args`\n    ValueError\n        If `args` contains values that do not correspond to those\n        of the default values specified in `compat_args`\n    '
    _check_arg_length(fname, args, max_fname_arg_count, compat_args)
    kwargs = dict(zip(compat_args, args))
    _check_for_default_values(fname, kwargs, compat_args)

def _check_for_invalid_keys(fname, kwargs, compat_args):
    "\n    Checks whether 'kwargs' contains any keys that are not\n    in 'compat_args' and raises a TypeError if there is one.\n    "
    diff = (set(kwargs) - set(compat_args))
    if diff:
        bad_arg = list(diff)[0]
        raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")

def validate_kwargs(fname, kwargs, compat_args):
    '\n    Checks whether parameters passed to the **kwargs argument in a\n    function `fname` are valid parameters as specified in `*compat_args`\n    and whether or not they are set to their default values.\n\n    Parameters\n    ----------\n    fname : str\n        The name of the function being passed the `**kwargs` parameter\n    kwargs : dict\n        The `**kwargs` parameter passed into `fname`\n    compat_args: dict\n        A dictionary of keys that `kwargs` is allowed to have and their\n        associated default values\n\n    Raises\n    ------\n    TypeError if `kwargs` contains keys not in `compat_args`\n    ValueError if `kwargs` contains keys in `compat_args` that do not\n    map to the default values specified in `compat_args`\n    '
    kwds = kwargs.copy()
    _check_for_invalid_keys(fname, kwargs, compat_args)
    _check_for_default_values(fname, kwds, compat_args)

def validate_args_and_kwargs(fname, args, kwargs, max_fname_arg_count, compat_args):
    '\n    Checks whether parameters passed to the *args and **kwargs argument in a\n    function `fname` are valid parameters as specified in `*compat_args`\n    and whether or not they are set to their default values.\n\n    Parameters\n    ----------\n    fname: str\n        The name of the function being passed the `**kwargs` parameter\n    args: tuple\n        The `*args` parameter passed into a function\n    kwargs: dict\n        The `**kwargs` parameter passed into `fname`\n    max_fname_arg_count: int\n        The minimum number of arguments that the function `fname`\n        requires, excluding those in `args`. Used for displaying\n        appropriate error messages. Must be non-negative.\n    compat_args: dict\n        A dictionary of keys that `kwargs` is allowed to\n        have and their associated default values.\n\n    Raises\n    ------\n    TypeError if `args` contains more values than there are\n    `compat_args` OR `kwargs` contains keys not in `compat_args`\n    ValueError if `args` contains values not at the default value (`None`)\n    `kwargs` contains keys in `compat_args` that do not map to the default\n    value as specified in `compat_args`\n\n    See Also\n    --------\n    validate_args : Purely args validation.\n    validate_kwargs : Purely kwargs validation.\n\n    '
    _check_arg_length(fname, (args + tuple(kwargs.values())), max_fname_arg_count, compat_args)
    args_dict = dict(zip(compat_args, args))
    for key in args_dict:
        if (key in kwargs):
            raise TypeError(f"{fname}() got multiple values for keyword argument '{key}'")
    kwargs.update(args_dict)
    validate_kwargs(fname, kwargs, compat_args)

def validate_bool_kwarg(value, arg_name):
    ' Ensures that argument passed in arg_name is of type bool. '
    if (not (is_bool(value) or (value is None))):
        raise ValueError(f'For argument "{arg_name}" expected type bool, received type {type(value).__name__}.')
    return value

def validate_axis_style_args(data, args, kwargs, arg_name, method_name):
    "\n    Argument handler for mixed index, columns / axis functions\n\n    In an attempt to handle both `.method(index, columns)`, and\n    `.method(arg, axis=.)`, we have to do some bad things to argument\n    parsing. This translates all arguments to `{index=., columns=.}` style.\n\n    Parameters\n    ----------\n    data : DataFrame\n    args : tuple\n        All positional arguments from the user\n    kwargs : dict\n        All keyword arguments from the user\n    arg_name, method_name : str\n        Used for better error messages\n\n    Returns\n    -------\n    kwargs : dict\n        A dictionary of keyword arguments. Doesn't modify ``kwargs``\n        inplace, so update them with the return value here.\n\n    Examples\n    --------\n    >>> df._validate_axis_style_args((str.upper,), {'columns': id},\n    ...                              'mapper', 'rename')\n    {'columns': <function id>, 'index': <method 'upper' of 'str' objects>}\n\n    This emits a warning\n    >>> df._validate_axis_style_args((str.upper, id), {},\n    ...                              'mapper', 'rename')\n    {'columns': <function id>, 'index': <method 'upper' of 'str' objects>}\n    "
    out = {}
    if (('axis' in kwargs) and any(((x in kwargs) for x in data._AXIS_TO_AXIS_NUMBER))):
        msg = "Cannot specify both 'axis' and any of 'index' or 'columns'."
        raise TypeError(msg)
    if (arg_name in kwargs):
        if args:
            msg = f"{method_name} got multiple values for argument '{arg_name}'"
            raise TypeError(msg)
        axis = data._get_axis_name(kwargs.get('axis', 0))
        out[axis] = kwargs[arg_name]
    for (k, v) in kwargs.items():
        try:
            ax = data._get_axis_name(k)
        except ValueError:
            pass
        else:
            out[ax] = v
    if (len(args) == 0):
        pass
    elif (len(args) == 1):
        axis = data._get_axis_name(kwargs.get('axis', 0))
        out[axis] = args[0]
    elif (len(args) == 2):
        if ('axis' in kwargs):
            msg = "Cannot specify both 'axis' and any of 'index' or 'columns'"
            raise TypeError(msg)
        msg = f'''Interpreting call
	'.{method_name}(a, b)' as 
	'.{method_name}(index=a, columns=b)'.
Use named arguments to remove any ambiguity. In the future, using positional arguments for 'index' or 'columns' will raise a 'TypeError'.'''
        warnings.warn(msg, FutureWarning, stacklevel=4)
        out[data._get_axis_name(0)] = args[0]
        out[data._get_axis_name(1)] = args[1]
    else:
        msg = f"Cannot specify all of '{arg_name}', 'index', 'columns'."
        raise TypeError(msg)
    return out

def validate_fillna_kwargs(value, method, validate_scalar_dict_value=True):
    "\n    Validate the keyword arguments to 'fillna'.\n\n    This checks that exactly one of 'value' and 'method' is specified.\n    If 'method' is specified, this validates that it's a valid method.\n\n    Parameters\n    ----------\n    value, method : object\n        The 'value' and 'method' keyword arguments for 'fillna'.\n    validate_scalar_dict_value : bool, default True\n        Whether to validate that 'value' is a scalar or dict. Specifically,\n        validate that it is not a list or tuple.\n\n    Returns\n    -------\n    value, method : object\n    "
    from pandas.core.missing import clean_fill_method
    if ((value is None) and (method is None)):
        raise ValueError("Must specify a fill 'value' or 'method'.")
    elif ((value is None) and (method is not None)):
        method = clean_fill_method(method)
    elif ((value is not None) and (method is None)):
        if (validate_scalar_dict_value and isinstance(value, (list, tuple))):
            raise TypeError(f'"value" parameter must be a scalar or dict, but you passed a "{type(value).__name__}"')
    elif ((value is not None) and (method is not None)):
        raise ValueError("Cannot specify both 'value' and 'method'.")
    return (value, method)

def validate_percentile(q):
    '\n    Validate percentiles (used by describe and quantile).\n\n    This function checks if the given float or iterable of floats is a valid percentile\n    otherwise raises a ValueError.\n\n    Parameters\n    ----------\n    q: float or iterable of floats\n        A single percentile or an iterable of percentiles.\n\n    Returns\n    -------\n    ndarray\n        An ndarray of the percentiles if valid.\n\n    Raises\n    ------\n    ValueError if percentiles are not in given interval([0, 1]).\n    '
    q_arr = np.asarray(q)
    msg = 'percentiles should all be in the interval [0, 1]. Try {} instead.'
    if (q_arr.ndim == 0):
        if (not (0 <= q_arr <= 1)):
            raise ValueError(msg.format((q_arr / 100.0)))
    elif (not all(((0 <= qs <= 1) for qs in q_arr))):
        raise ValueError(msg.format((q_arr / 100.0)))
    return q_arr
