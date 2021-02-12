
'\nTop level ``eval`` module.\n'
import tokenize
from typing import Optional
import warnings
from pandas._libs.lib import no_default
from pandas.util._validators import validate_bool_kwarg
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import PARSERS, Expr
from pandas.core.computation.parsing import tokenize_string
from pandas.core.computation.scope import ensure_scope
from pandas.io.formats.printing import pprint_thing

def _check_engine(engine):
    "\n    Make sure a valid engine is passed.\n\n    Parameters\n    ----------\n    engine : str\n        String to validate.\n\n    Raises\n    ------\n    KeyError\n      * If an invalid engine is passed.\n    ImportError\n      * If numexpr was requested but doesn't exist.\n\n    Returns\n    -------\n    str\n        Engine name.\n    "
    from pandas.core.computation.check import NUMEXPR_INSTALLED
    if (engine is None):
        engine = ('numexpr' if NUMEXPR_INSTALLED else 'python')
    if (engine not in ENGINES):
        valid_engines = list(ENGINES.keys())
        raise KeyError(f"Invalid engine '{engine}' passed, valid engines are {valid_engines}")
    if ((engine == 'numexpr') and (not NUMEXPR_INSTALLED)):
        raise ImportError("'numexpr' is not installed or an unsupported version. Cannot use engine='numexpr' for query/eval if 'numexpr' is not installed")
    return engine

def _check_parser(parser):
    '\n    Make sure a valid parser is passed.\n\n    Parameters\n    ----------\n    parser : str\n\n    Raises\n    ------\n    KeyError\n      * If an invalid parser is passed\n    '
    if (parser not in PARSERS):
        raise KeyError(f"Invalid parser '{parser}' passed, valid parsers are {PARSERS.keys()}")

def _check_resolvers(resolvers):
    if (resolvers is not None):
        for resolver in resolvers:
            if (not hasattr(resolver, '__getitem__')):
                name = type(resolver).__name__
                raise TypeError(f"Resolver of type '{name}' does not implement the __getitem__ method")

def _check_expression(expr):
    '\n    Make sure an expression is not an empty string\n\n    Parameters\n    ----------\n    expr : object\n        An object that can be converted to a string\n\n    Raises\n    ------\n    ValueError\n      * If expr is an empty string\n    '
    if (not expr):
        raise ValueError('expr cannot be an empty string')

def _convert_expression(expr):
    "\n    Convert an object to an expression.\n\n    This function converts an object to an expression (a unicode string) and\n    checks to make sure it isn't empty after conversion. This is used to\n    convert operators to their string representation for recursive calls to\n    :func:`~pandas.eval`.\n\n    Parameters\n    ----------\n    expr : object\n        The object to be converted to a string.\n\n    Returns\n    -------\n    str\n        The string representation of an object.\n\n    Raises\n    ------\n    ValueError\n      * If the expression is empty.\n    "
    s = pprint_thing(expr)
    _check_expression(s)
    return s

def _check_for_locals(expr, stack_level, parser):
    at_top_of_stack = (stack_level == 0)
    not_pandas_parser = (parser != 'pandas')
    if not_pandas_parser:
        msg = "The '@' prefix is only supported by the pandas parser"
    elif at_top_of_stack:
        msg = "The '@' prefix is not allowed in top-level eval calls.\nplease refer to your variables by name without the '@' prefix."
    if (at_top_of_stack or not_pandas_parser):
        for (toknum, tokval) in tokenize_string(expr):
            if ((toknum == tokenize.OP) and (tokval == '@')):
                raise SyntaxError(msg)

def eval(expr, parser='pandas', engine=None, truediv=no_default, local_dict=None, global_dict=None, resolvers=(), level=0, target=None, inplace=False):
    '\n    Evaluate a Python expression as a string using various backends.\n\n    The following arithmetic operations are supported: ``+``, ``-``, ``*``,\n    ``/``, ``**``, ``%``, ``//`` (python engine only) along with the following\n    boolean operations: ``|`` (or), ``&`` (and), and ``~`` (not).\n    Additionally, the ``\'pandas\'`` parser allows the use of :keyword:`and`,\n    :keyword:`or`, and :keyword:`not` with the same semantics as the\n    corresponding bitwise operators.  :class:`~pandas.Series` and\n    :class:`~pandas.DataFrame` objects are supported and behave as they would\n    with plain ol\' Python evaluation.\n\n    Parameters\n    ----------\n    expr : str\n        The expression to evaluate. This string cannot contain any Python\n        `statements\n        <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,\n        only Python `expressions\n        <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.\n    parser : {\'pandas\', \'python\'}, default \'pandas\'\n        The parser to use to construct the syntax tree from the expression. The\n        default of ``\'pandas\'`` parses code slightly different than standard\n        Python. Alternatively, you can parse an expression using the\n        ``\'python\'`` parser to retain strict Python semantics.  See the\n        :ref:`enhancing performance <enhancingperf.eval>` documentation for\n        more details.\n    engine : {\'python\', \'numexpr\'}, default \'numexpr\'\n\n        The engine used to evaluate the expression. Supported engines are\n\n        - None         : tries to use ``numexpr``, falls back to ``python``\n        - ``\'numexpr\'``: This default engine evaluates pandas objects using\n                         numexpr for large speed ups in complex expressions\n                         with large frames.\n        - ``\'python\'``: Performs operations as if you had ``eval``\'d in top\n                        level python. This engine is generally not that useful.\n\n        More backends may be available in the future.\n\n    truediv : bool, optional\n        Whether to use true division, like in Python >= 3.\n\n        .. deprecated:: 1.0.0\n\n    local_dict : dict or None, optional\n        A dictionary of local variables, taken from locals() by default.\n    global_dict : dict or None, optional\n        A dictionary of global variables, taken from globals() by default.\n    resolvers : list of dict-like or None, optional\n        A list of objects implementing the ``__getitem__`` special method that\n        you can use to inject an additional collection of namespaces to use for\n        variable lookup. For example, this is used in the\n        :meth:`~DataFrame.query` method to inject the\n        ``DataFrame.index`` and ``DataFrame.columns``\n        variables that refer to their respective :class:`~pandas.DataFrame`\n        instance attributes.\n    level : int, optional\n        The number of prior stack frames to traverse and add to the current\n        scope. Most users will **not** need to change this parameter.\n    target : object, optional, default None\n        This is the target object for assignment. It is used when there is\n        variable assignment in the expression. If so, then `target` must\n        support item assignment with string keys, and if a copy is being\n        returned, it must also support `.copy()`.\n    inplace : bool, default False\n        If `target` is provided, and the expression mutates `target`, whether\n        to modify `target` inplace. Otherwise, return a copy of `target` with\n        the mutation.\n\n    Returns\n    -------\n    ndarray, numeric scalar, DataFrame, Series, or None\n        The completion value of evaluating the given code or None if ``inplace=True``.\n\n    Raises\n    ------\n    ValueError\n        There are many instances where such an error can be raised:\n\n        - `target=None`, but the expression is multiline.\n        - The expression is multiline, but not all them have item assignment.\n          An example of such an arrangement is this:\n\n          a = b + 1\n          a + 2\n\n          Here, there are expressions on different lines, making it multiline,\n          but the last line has no variable assigned to the output of `a + 2`.\n        - `inplace=True`, but the expression is missing item assignment.\n        - Item assignment is provided, but the `target` does not support\n          string item assignment.\n        - Item assignment is provided and `inplace=False`, but the `target`\n          does not support the `.copy()` method\n\n    See Also\n    --------\n    DataFrame.query : Evaluates a boolean expression to query the columns\n            of a frame.\n    DataFrame.eval : Evaluate a string describing operations on\n            DataFrame columns.\n\n    Notes\n    -----\n    The ``dtype`` of any objects involved in an arithmetic ``%`` operation are\n    recursively cast to ``float64``.\n\n    See the :ref:`enhancing performance <enhancingperf.eval>` documentation for\n    more details.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})\n    >>> df\n      animal  age\n    0    dog   10\n    1    pig   20\n\n    We can add a new column using ``pd.eval``:\n\n    >>> pd.eval("double_age = df.age * 2", target=df)\n      animal  age  double_age\n    0    dog   10          20\n    1    pig   20          40\n    '
    inplace = validate_bool_kwarg(inplace, 'inplace')
    if (truediv is not no_default):
        warnings.warn('The `truediv` parameter in pd.eval is deprecated and will be removed in a future version.', FutureWarning, stacklevel=2)
    if isinstance(expr, str):
        _check_expression(expr)
        exprs = [e.strip() for e in expr.splitlines() if (e.strip() != '')]
    else:
        exprs = [expr]
    multi_line = (len(exprs) > 1)
    if (multi_line and (target is None)):
        raise ValueError('multi-line expressions are only valid in the context of data, use DataFrame.eval')
    engine = _check_engine(engine)
    _check_parser(parser)
    _check_resolvers(resolvers)
    ret = None
    first_expr = True
    target_modified = False
    for expr in exprs:
        expr = _convert_expression(expr)
        _check_for_locals(expr, level, parser)
        env = ensure_scope((level + 1), global_dict=global_dict, local_dict=local_dict, resolvers=resolvers, target=target)
        parsed_expr = Expr(expr, engine=engine, parser=parser, env=env)
        eng = ENGINES[engine]
        eng_inst = eng(parsed_expr)
        ret = eng_inst.evaluate()
        if (parsed_expr.assigner is None):
            if multi_line:
                raise ValueError('Multi-line expressions are only valid if all expressions contain an assignment')
            elif inplace:
                raise ValueError('Cannot operate inplace if there is no assignment')
        assigner = parsed_expr.assigner
        if ((env.target is not None) and (assigner is not None)):
            target_modified = True
            if ((not inplace) and first_expr):
                try:
                    target = env.target.copy()
                except AttributeError as err:
                    raise ValueError('Cannot return a copy of the target') from err
            else:
                target = env.target
            try:
                with warnings.catch_warnings(record=True):
                    target[assigner] = ret
            except (TypeError, IndexError) as err:
                raise ValueError('Cannot assign expression output to target') from err
            if (not resolvers):
                resolvers = ({assigner: ret},)
            else:
                for resolver in resolvers:
                    if (assigner in resolver):
                        resolver[assigner] = ret
                        break
                else:
                    resolvers += ({assigner: ret},)
            ret = None
            first_expr = False
    if (inplace is False):
        return (target if target_modified else ret)
