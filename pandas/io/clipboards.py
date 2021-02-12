
' io on the clipboard '
from io import StringIO
import warnings
from pandas.core.dtypes.generic import ABCDataFrame
from pandas import get_option, option_context

def read_clipboard(sep='\\s+', **kwargs):
    "\n    Read text from clipboard and pass to read_csv.\n\n    Parameters\n    ----------\n    sep : str, default '\\s+'\n        A string or regex delimiter. The default of '\\s+' denotes\n        one or more whitespace characters.\n\n    **kwargs\n        See read_csv for the full argument list.\n\n    Returns\n    -------\n    DataFrame\n        A parsed DataFrame object.\n    "
    encoding = kwargs.pop('encoding', 'utf-8')
    if ((encoding is not None) and (encoding.lower().replace('-', '') != 'utf8')):
        raise NotImplementedError('reading from clipboard only supports utf-8 encoding')
    from pandas.io.clipboard import clipboard_get
    from pandas.io.parsers import read_csv
    text = clipboard_get()
    try:
        text = text.decode((kwargs.get('encoding') or get_option('display.encoding')))
    except AttributeError:
        pass
    lines = text[:10000].split('\n')[:(- 1)][:10]
    counts = {x.lstrip().count('\t') for x in lines}
    if ((len(lines) > 1) and (len(counts) == 1) and (counts.pop() != 0)):
        sep = '\t'
    if ((sep is None) and (kwargs.get('delim_whitespace') is None)):
        sep = '\\s+'
    if ((len(sep) > 1) and (kwargs.get('engine') is None)):
        kwargs['engine'] = 'python'
    elif ((len(sep) > 1) and (kwargs.get('engine') == 'c')):
        warnings.warn('read_clipboard with regex separator does not work properly with c engine')
    return read_csv(StringIO(text), sep=sep, **kwargs)

def to_clipboard(obj, excel=True, sep=None, **kwargs):
    '\n    Attempt to write text representation of object to the system clipboard\n    The clipboard can be then pasted into Excel for example.\n\n    Parameters\n    ----------\n    obj : the object to write to the clipboard\n    excel : boolean, defaults to True\n            if True, use the provided separator, writing in a csv\n            format for allowing easy pasting into excel.\n            if False, write a string representation of the object\n            to the clipboard\n    sep : optional, defaults to tab\n    other keywords are passed to to_csv\n\n    Notes\n    -----\n    Requirements for your platform\n      - Linux: xclip, or xsel (with PyQt4 modules)\n      - Windows:\n      - OS X:\n    '
    encoding = kwargs.pop('encoding', 'utf-8')
    if ((encoding is not None) and (encoding.lower().replace('-', '') != 'utf8')):
        raise ValueError('clipboard only supports utf-8 encoding')
    from pandas.io.clipboard import clipboard_set
    if (excel is None):
        excel = True
    if excel:
        try:
            if (sep is None):
                sep = '\t'
            buf = StringIO()
            obj.to_csv(buf, sep=sep, encoding='utf-8', **kwargs)
            text = buf.getvalue()
            clipboard_set(text)
            return
        except TypeError:
            warnings.warn('to_clipboard in excel mode requires a single character separator.')
    elif (sep is not None):
        warnings.warn('to_clipboard with excel=False ignores the sep argument')
    if isinstance(obj, ABCDataFrame):
        with option_context('display.max_colwidth', None):
            objstr = obj.to_string(**kwargs)
    else:
        objstr = str(obj)
    clipboard_set(objstr)
