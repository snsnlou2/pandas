
'\nPrinting tools.\n'
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Sized, Tuple, TypeVar, Union
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
EscapeChars = Union[(Mapping[(str, str)], Iterable[str])]
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

def adjoin(space, *lists, **kwargs):
    '\n    Glues together two sets of strings using the amount of space requested.\n    The idea is to prettify.\n\n    ----------\n    space : int\n        number of spaces for padding\n    lists : str\n        list of str which being joined\n    strlen : callable\n        function used to calculate the length of each str. Needed for unicode\n        handling.\n    justfunc : callable\n        function used to justify str. Needed for unicode handling.\n    '
    strlen = kwargs.pop('strlen', len)
    justfunc = kwargs.pop('justfunc', justify)
    out_lines = []
    newLists = []
    lengths = [(max(map(strlen, x)) + space) for x in lists[:(- 1)]]
    lengths.append(max(map(len, lists[(- 1)])))
    maxLen = max(map(len, lists))
    for (i, lst) in enumerate(lists):
        nl = justfunc(lst, lengths[i], mode='left')
        nl.extend(([(' ' * lengths[i])] * (maxLen - len(lst))))
        newLists.append(nl)
    toJoin = zip(*newLists)
    for lines in toJoin:
        out_lines.append(''.join(lines))
    return '\n'.join(out_lines)

def justify(texts, max_len, mode='right'):
    '\n    Perform ljust, center, rjust against string or list-like\n    '
    if (mode == 'left'):
        return [x.ljust(max_len) for x in texts]
    elif (mode == 'center'):
        return [x.center(max_len) for x in texts]
    else:
        return [x.rjust(max_len) for x in texts]

def _pprint_seq(seq, _nest_lvl=0, max_seq_items=None, **kwds):
    '\n    internal. pprinter for iterables. you should probably use pprint_thing()\n    rather than calling this directly.\n\n    bounds length of printed sequence, depending on options\n    '
    if isinstance(seq, set):
        fmt = '{{{body}}}'
    else:
        fmt = ('[{body}]' if hasattr(seq, '__setitem__') else '({body})')
    if (max_seq_items is False):
        nitems = len(seq)
    else:
        nitems = (max_seq_items or get_option('max_seq_items') or len(seq))
    s = iter(seq)
    r = [pprint_thing(next(s), (_nest_lvl + 1), max_seq_items=max_seq_items, **kwds) for i in range(min(nitems, len(seq)))]
    body = ', '.join(r)
    if (nitems < len(seq)):
        body += ', ...'
    elif (isinstance(seq, tuple) and (len(seq) == 1)):
        body += ','
    return fmt.format(body=body)

def _pprint_dict(seq, _nest_lvl=0, max_seq_items=None, **kwds):
    '\n    internal. pprinter for iterables. you should probably use pprint_thing()\n    rather than calling this directly.\n    '
    fmt = '{{{things}}}'
    pairs = []
    pfmt = '{key}: {val}'
    if (max_seq_items is False):
        nitems = len(seq)
    else:
        nitems = (max_seq_items or get_option('max_seq_items') or len(seq))
    for (k, v) in list(seq.items())[:nitems]:
        pairs.append(pfmt.format(key=pprint_thing(k, (_nest_lvl + 1), max_seq_items=max_seq_items, **kwds), val=pprint_thing(v, (_nest_lvl + 1), max_seq_items=max_seq_items, **kwds)))
    if (nitems < len(seq)):
        return fmt.format(things=(', '.join(pairs) + ', ...'))
    else:
        return fmt.format(things=', '.join(pairs))

def pprint_thing(thing, _nest_lvl=0, escape_chars=None, default_escapes=False, quote_strings=False, max_seq_items=None):
    '\n    This function is the sanctioned way of converting objects\n    to a string representation and properly handles nested sequences.\n\n    Parameters\n    ----------\n    thing : anything to be formatted\n    _nest_lvl : internal use only. pprint_thing() is mutually-recursive\n        with pprint_sequence, this argument is used to keep track of the\n        current nesting level, and limit it.\n    escape_chars : list or dict, optional\n        Characters to escape. If a dict is passed the values are the\n        replacements\n    default_escapes : bool, default False\n        Whether the input escape characters replaces or adds to the defaults\n    max_seq_items : int or None, default None\n        Pass through to other pretty printers to limit sequence printing\n\n    Returns\n    -------\n    str\n    '

    def as_escaped_string(thing: Any, escape_chars: Optional[EscapeChars]=escape_chars) -> str:
        translate = {'\t': '\\t', '\n': '\\n', '\r': '\\r'}
        if isinstance(escape_chars, dict):
            if default_escapes:
                translate.update(escape_chars)
            else:
                translate = escape_chars
            escape_chars = list(escape_chars.keys())
        else:
            escape_chars = (escape_chars or ())
        result = str(thing)
        for c in escape_chars:
            result = result.replace(c, translate[c])
        return result
    if hasattr(thing, '__next__'):
        return str(thing)
    elif (isinstance(thing, dict) and (_nest_lvl < get_option('display.pprint_nest_depth'))):
        result = _pprint_dict(thing, _nest_lvl, quote_strings=True, max_seq_items=max_seq_items)
    elif (is_sequence(thing) and (_nest_lvl < get_option('display.pprint_nest_depth'))):
        result = _pprint_seq(thing, _nest_lvl, escape_chars=escape_chars, quote_strings=quote_strings, max_seq_items=max_seq_items)
    elif (isinstance(thing, str) and quote_strings):
        result = f"'{as_escaped_string(thing)}'"
    else:
        result = as_escaped_string(thing)
    return result

def pprint_thing_encoded(object, encoding='utf-8', errors='replace'):
    value = pprint_thing(object)
    return value.encode(encoding, errors)

def enable_data_resource_formatter(enable):
    if ('IPython' not in sys.modules):
        return
    from IPython import get_ipython
    ip = get_ipython()
    if (ip is None):
        return
    formatters = ip.display_formatter.formatters
    mimetype = 'application/vnd.dataresource+json'
    if enable:
        if (mimetype not in formatters):
            from IPython.core.formatters import BaseFormatter

            class TableSchemaFormatter(BaseFormatter):
                print_method = '_repr_data_resource_'
                _return_type = (dict,)
            formatters[mimetype] = TableSchemaFormatter()
        formatters[mimetype].enabled = True
    elif (mimetype in formatters):
        formatters[mimetype].enabled = False

def default_pprint(thing, max_seq_items=None):
    return pprint_thing(thing, escape_chars=('\t', '\r', '\n'), quote_strings=True, max_seq_items=max_seq_items)

def format_object_summary(obj, formatter, is_justify=True, name=None, indent_for_name=True, line_break_each_value=False):
    '\n    Return the formatted obj as a unicode string\n\n    Parameters\n    ----------\n    obj : object\n        must be iterable and support __getitem__\n    formatter : callable\n        string formatter for an element\n    is_justify : boolean\n        should justify the display\n    name : name, optional\n        defaults to the class name of the obj\n    indent_for_name : bool, default True\n        Whether subsequent lines should be indented to\n        align with the name.\n    line_break_each_value : bool, default False\n        If True, inserts a line break for each value of ``obj``.\n        If False, only break lines when the a line of values gets wider\n        than the display width.\n\n        .. versionadded:: 0.25.0\n\n    Returns\n    -------\n    summary string\n    '
    from pandas.io.formats.console import get_console_size
    from pandas.io.formats.format import get_adjustment
    (display_width, _) = get_console_size()
    if (display_width is None):
        display_width = (get_option('display.width') or 80)
    if (name is None):
        name = type(obj).__name__
    if indent_for_name:
        name_len = len(name)
        space1 = f'''
{(' ' * (name_len + 1))}'''
        space2 = f'''
{(' ' * (name_len + 2))}'''
    else:
        space1 = '\n'
        space2 = '\n '
    n = len(obj)
    if line_break_each_value:
        sep = (',\n ' + (' ' * len(name)))
    else:
        sep = ','
    max_seq_items = (get_option('display.max_seq_items') or n)
    is_truncated = (n > max_seq_items)
    adj = get_adjustment()

    def _extend_line(s: str, line: str, value: str, display_width: int, next_line_prefix: str) -> Tuple[(str, str)]:
        if ((adj.len(line.rstrip()) + adj.len(value.rstrip())) >= display_width):
            s += line.rstrip()
            line = next_line_prefix
        line += value
        return (s, line)

    def best_len(values: List[str]) -> int:
        if values:
            return max((adj.len(x) for x in values))
        else:
            return 0
    close = ', '
    if (n == 0):
        summary = f'[]{close}'
    elif ((n == 1) and (not line_break_each_value)):
        first = formatter(obj[0])
        summary = f'[{first}]{close}'
    elif ((n == 2) and (not line_break_each_value)):
        first = formatter(obj[0])
        last = formatter(obj[(- 1)])
        summary = f'[{first}, {last}]{close}'
    else:
        if (max_seq_items == 1):
            head = []
            tail = [formatter(x) for x in obj[(- 1):]]
        elif (n > max_seq_items):
            n = min((max_seq_items // 2), 10)
            head = [formatter(x) for x in obj[:n]]
            tail = [formatter(x) for x in obj[(- n):]]
        else:
            head = []
            tail = [formatter(x) for x in obj]
        if is_justify:
            if line_break_each_value:
                (head, tail) = _justify(head, tail)
            elif (is_truncated or (not ((len(', '.join(head)) < display_width) and (len(', '.join(tail)) < display_width)))):
                max_length = max(best_len(head), best_len(tail))
                head = [x.rjust(max_length) for x in head]
                tail = [x.rjust(max_length) for x in tail]
        if line_break_each_value:
            max_space = (display_width - len(space2))
            value = tail[0]
            for max_items in reversed(range(1, (len(value) + 1))):
                pprinted_seq = _pprint_seq(value, max_seq_items=max_items)
                if (len(pprinted_seq) < max_space):
                    break
            head = [_pprint_seq(x, max_seq_items=max_items) for x in head]
            tail = [_pprint_seq(x, max_seq_items=max_items) for x in tail]
        summary = ''
        line = space2
        for max_items in range(len(head)):
            word = ((head[max_items] + sep) + ' ')
            (summary, line) = _extend_line(summary, line, word, display_width, space2)
        if is_truncated:
            summary += ((line.rstrip() + space2) + '...')
            line = space2
        for max_items in range((len(tail) - 1)):
            word = ((tail[max_items] + sep) + ' ')
            (summary, line) = _extend_line(summary, line, word, display_width, space2)
        (summary, line) = _extend_line(summary, line, tail[(- 1)], (display_width - 2), space2)
        summary += line
        close = (']' + close.rstrip(' '))
        summary += close
        if ((len(summary) > display_width) or line_break_each_value):
            summary += space1
        else:
            summary += ' '
        summary = ('[' + summary[len(space2):])
    return summary

def _justify(head, tail):
    "\n    Justify items in head and tail, so they are right-aligned when stacked.\n\n    Parameters\n    ----------\n    head : list-like of list-likes of strings\n    tail : list-like of list-likes of strings\n\n    Returns\n    -------\n    tuple of list of tuples of strings\n        Same as head and tail, but items are right aligned when stacked\n        vertically.\n\n    Examples\n    --------\n    >>> _justify([['a', 'b']], [['abc', 'abcd']])\n    ([('  a', '   b')], [('abc', 'abcd')])\n    "
    combined = (head + tail)
    max_length = ([0] * len(combined[0]))
    for inner_seq in combined:
        length = [len(item) for item in inner_seq]
        max_length = [max(x, y) for (x, y) in zip(max_length, length)]
    head = [tuple((x.rjust(max_len) for (x, max_len) in zip(seq, max_length))) for seq in head]
    tail = [tuple((x.rjust(max_len) for (x, max_len) in zip(seq, max_length))) for seq in tail]
    return (head, tail)

def format_object_attrs(obj, include_dtype=True):
    "\n    Return a list of tuples of the (attr, formatted_value)\n    for common attrs, including dtype, name, length\n\n    Parameters\n    ----------\n    obj : object\n        Must be sized.\n    include_dtype : bool\n        If False, dtype won't be in the returned list\n\n    Returns\n    -------\n    list of 2-tuple\n\n    "
    attrs: List[Tuple[(str, Union[(str, int)])]] = []
    if (hasattr(obj, 'dtype') and include_dtype):
        attrs.append(('dtype', f"'{obj.dtype}'"))
    if (getattr(obj, 'name', None) is not None):
        attrs.append(('name', default_pprint(obj.name)))
    elif ((getattr(obj, 'names', None) is not None) and any(obj.names)):
        attrs.append(('names', default_pprint(obj.names)))
    max_seq_items = (get_option('display.max_seq_items') or len(obj))
    if (len(obj) > max_seq_items):
        attrs.append(('length', len(obj)))
    return attrs

class PrettyDict(Dict[(_KT, _VT)]):
    'Dict extension to support abbreviated __repr__'

    def __repr__(self):
        return pprint_thing(self)
