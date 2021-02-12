
'\n:mod:`pandas.io.html` is a module containing functionality for dealing with\nHTML IO.\n\n'
from collections import abc
import numbers
import os
import re
from typing import Dict, List, Optional, Pattern, Sequence, Tuple, Union
from pandas._typing import FilePathOrBuffer
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError, EmptyDataError
from pandas.util._decorators import deprecate_nonkeyword_arguments
from pandas.core.dtypes.common import is_list_like
from pandas.core.construction import create_series_with_explicit_dtype
from pandas.core.frame import DataFrame
from pandas.io.common import is_url, stringify_path, urlopen, validate_header_arg
from pandas.io.formats.printing import pprint_thing
from pandas.io.parsers import TextParser
_IMPORTS = False
_HAS_BS4 = False
_HAS_LXML = False
_HAS_HTML5LIB = False

def _importers():
    global _IMPORTS
    if _IMPORTS:
        return
    global _HAS_BS4, _HAS_LXML, _HAS_HTML5LIB
    bs4 = import_optional_dependency('bs4', raise_on_missing=False, on_version='ignore')
    _HAS_BS4 = (bs4 is not None)
    lxml = import_optional_dependency('lxml.etree', raise_on_missing=False, on_version='ignore')
    _HAS_LXML = (lxml is not None)
    html5lib = import_optional_dependency('html5lib', raise_on_missing=False, on_version='ignore')
    _HAS_HTML5LIB = (html5lib is not None)
    _IMPORTS = True
_RE_WHITESPACE = re.compile('[\\r\\n]+|\\s{2,}')

def _remove_whitespace(s, regex=_RE_WHITESPACE):
    '\n    Replace extra whitespace inside of a string with a single space.\n\n    Parameters\n    ----------\n    s : str or unicode\n        The string from which to remove extra whitespace.\n    regex : re.Pattern\n        The regular expression to use to remove extra whitespace.\n\n    Returns\n    -------\n    subd : str or unicode\n        `s` with all extra whitespace replaced with a single space.\n    '
    return regex.sub(' ', s.strip())

def _get_skiprows(skiprows):
    '\n    Get an iterator given an integer, slice or container.\n\n    Parameters\n    ----------\n    skiprows : int, slice, container\n        The iterator to use to skip rows; can also be a slice.\n\n    Raises\n    ------\n    TypeError\n        * If `skiprows` is not a slice, integer, or Container\n\n    Returns\n    -------\n    it : iterable\n        A proper iterator to use to skip rows of a DataFrame.\n    '
    if isinstance(skiprows, slice):
        (start, step) = ((skiprows.start or 0), (skiprows.step or 1))
        return list(range(start, skiprows.stop, step))
    elif (isinstance(skiprows, numbers.Integral) or is_list_like(skiprows)):
        return skiprows
    elif (skiprows is None):
        return 0
    raise TypeError(f'{type(skiprows).__name__} is not a valid type for skipping rows')

def _read(obj):
    '\n    Try to read from a url, file or string.\n\n    Parameters\n    ----------\n    obj : str, unicode, or file-like\n\n    Returns\n    -------\n    raw_text : str\n    '
    if is_url(obj):
        with urlopen(obj) as url:
            text = url.read()
    elif hasattr(obj, 'read'):
        text = obj.read()
    elif isinstance(obj, (str, bytes)):
        text = obj
        try:
            if os.path.isfile(text):
                with open(text, 'rb') as f:
                    return f.read()
        except (TypeError, ValueError):
            pass
    else:
        raise TypeError(f"Cannot read object of type '{type(obj).__name__}'")
    return text

class _HtmlFrameParser():
    '\n    Base class for parsers that parse HTML into DataFrames.\n\n    Parameters\n    ----------\n    io : str or file-like\n        This can be either a string of raw HTML, a valid URL using the HTTP,\n        FTP, or FILE protocols or a file-like object.\n\n    match : str or regex\n        The text to match in the document.\n\n    attrs : dict\n        List of HTML <table> element attributes to match.\n\n    encoding : str\n        Encoding to be used by parser\n\n    displayed_only : bool\n        Whether or not items with "display:none" should be ignored\n\n    Attributes\n    ----------\n    io : str or file-like\n        raw HTML, URL, or file-like object\n\n    match : regex\n        The text to match in the raw HTML\n\n    attrs : dict-like\n        A dictionary of valid table attributes to use to search for table\n        elements.\n\n    encoding : str\n        Encoding to be used by parser\n\n    displayed_only : bool\n        Whether or not items with "display:none" should be ignored\n\n    Notes\n    -----\n    To subclass this class effectively you must override the following methods:\n        * :func:`_build_doc`\n        * :func:`_attr_getter`\n        * :func:`_text_getter`\n        * :func:`_parse_td`\n        * :func:`_parse_thead_tr`\n        * :func:`_parse_tbody_tr`\n        * :func:`_parse_tfoot_tr`\n        * :func:`_parse_tables`\n        * :func:`_equals_tag`\n    See each method\'s respective documentation for details on their\n    functionality.\n    '

    def __init__(self, io, match, attrs, encoding, displayed_only):
        self.io = io
        self.match = match
        self.attrs = attrs
        self.encoding = encoding
        self.displayed_only = displayed_only

    def parse_tables(self):
        '\n        Parse and return all tables from the DOM.\n\n        Returns\n        -------\n        list of parsed (header, body, footer) tuples from tables.\n        '
        tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        return (self._parse_thead_tbody_tfoot(table) for table in tables)

    def _attr_getter(self, obj, attr):
        '\n        Return the attribute value of an individual DOM node.\n\n        Parameters\n        ----------\n        obj : node-like\n            A DOM node.\n\n        attr : str or unicode\n            The attribute, such as "colspan"\n\n        Returns\n        -------\n        str or unicode\n            The attribute value.\n        '
        return obj.get(attr)

    def _text_getter(self, obj):
        '\n        Return the text of an individual DOM node.\n\n        Parameters\n        ----------\n        obj : node-like\n            A DOM node.\n\n        Returns\n        -------\n        text : str or unicode\n            The text from an individual DOM node.\n        '
        raise AbstractMethodError(self)

    def _parse_td(self, obj):
        '\n        Return the td elements from a row element.\n\n        Parameters\n        ----------\n        obj : node-like\n            A DOM <tr> node.\n\n        Returns\n        -------\n        list of node-like\n            These are the elements of each row, i.e., the columns.\n        '
        raise AbstractMethodError(self)

    def _parse_thead_tr(self, table):
        '\n        Return the list of thead row elements from the parsed table element.\n\n        Parameters\n        ----------\n        table : a table element that contains zero or more thead elements.\n\n        Returns\n        -------\n        list of node-like\n            These are the <tr> row elements of a table.\n        '
        raise AbstractMethodError(self)

    def _parse_tbody_tr(self, table):
        '\n        Return the list of tbody row elements from the parsed table element.\n\n        HTML5 table bodies consist of either 0 or more <tbody> elements (which\n        only contain <tr> elements) or 0 or more <tr> elements. This method\n        checks for both structures.\n\n        Parameters\n        ----------\n        table : a table element that contains row elements.\n\n        Returns\n        -------\n        list of node-like\n            These are the <tr> row elements of a table.\n        '
        raise AbstractMethodError(self)

    def _parse_tfoot_tr(self, table):
        '\n        Return the list of tfoot row elements from the parsed table element.\n\n        Parameters\n        ----------\n        table : a table element that contains row elements.\n\n        Returns\n        -------\n        list of node-like\n            These are the <tr> row elements of a table.\n        '
        raise AbstractMethodError(self)

    def _parse_tables(self, doc, match, attrs):
        '\n        Return all tables from the parsed DOM.\n\n        Parameters\n        ----------\n        doc : the DOM from which to parse the table element.\n\n        match : str or regular expression\n            The text to search for in the DOM tree.\n\n        attrs : dict\n            A dictionary of table attributes that can be used to disambiguate\n            multiple tables on a page.\n\n        Raises\n        ------\n        ValueError : `match` does not match any text in the document.\n\n        Returns\n        -------\n        list of node-like\n            HTML <table> elements to be parsed into raw data.\n        '
        raise AbstractMethodError(self)

    def _equals_tag(self, obj, tag):
        "\n        Return whether an individual DOM node matches a tag\n\n        Parameters\n        ----------\n        obj : node-like\n            A DOM node.\n\n        tag : str\n            Tag name to be checked for equality.\n\n        Returns\n        -------\n        boolean\n            Whether `obj`'s tag name is `tag`\n        "
        raise AbstractMethodError(self)

    def _build_doc(self):
        '\n        Return a tree-like object that can be used to iterate over the DOM.\n\n        Returns\n        -------\n        node-like\n            The DOM from which to parse the table element.\n        '
        raise AbstractMethodError(self)

    def _parse_thead_tbody_tfoot(self, table_html):
        '\n        Given a table, return parsed header, body, and foot.\n\n        Parameters\n        ----------\n        table_html : node-like\n\n        Returns\n        -------\n        tuple of (header, body, footer), each a list of list-of-text rows.\n\n        Notes\n        -----\n        Header and body are lists-of-lists. Top level list is a list of\n        rows. Each row is a list of str text.\n\n        Logic: Use <thead>, <tbody>, <tfoot> elements to identify\n               header, body, and footer, otherwise:\n               - Put all rows into body\n               - Move rows from top of body to header only if\n                 all elements inside row are <th>\n               - Move rows from bottom of body to footer only if\n                 all elements inside row are <th>\n        '
        header_rows = self._parse_thead_tr(table_html)
        body_rows = self._parse_tbody_tr(table_html)
        footer_rows = self._parse_tfoot_tr(table_html)

        def row_is_all_th(row):
            return all((self._equals_tag(t, 'th') for t in self._parse_td(row)))
        if (not header_rows):
            while (body_rows and row_is_all_th(body_rows[0])):
                header_rows.append(body_rows.pop(0))
        header = self._expand_colspan_rowspan(header_rows)
        body = self._expand_colspan_rowspan(body_rows)
        footer = self._expand_colspan_rowspan(footer_rows)
        return (header, body, footer)

    def _expand_colspan_rowspan(self, rows):
        '\n        Given a list of <tr>s, return a list of text rows.\n\n        Parameters\n        ----------\n        rows : list of node-like\n            List of <tr>s\n\n        Returns\n        -------\n        list of list\n            Each returned row is a list of str text.\n\n        Notes\n        -----\n        Any cell with ``rowspan`` or ``colspan`` will have its contents copied\n        to subsequent cells.\n        '
        all_texts = []
        remainder: List[Tuple[(int, str, int)]] = []
        for tr in rows:
            texts = []
            next_remainder = []
            index = 0
            tds = self._parse_td(tr)
            for td in tds:
                while (remainder and (remainder[0][0] <= index)):
                    (prev_i, prev_text, prev_rowspan) = remainder.pop(0)
                    texts.append(prev_text)
                    if (prev_rowspan > 1):
                        next_remainder.append((prev_i, prev_text, (prev_rowspan - 1)))
                    index += 1
                text = _remove_whitespace(self._text_getter(td))
                rowspan = int((self._attr_getter(td, 'rowspan') or 1))
                colspan = int((self._attr_getter(td, 'colspan') or 1))
                for _ in range(colspan):
                    texts.append(text)
                    if (rowspan > 1):
                        next_remainder.append((index, text, (rowspan - 1)))
                    index += 1
            for (prev_i, prev_text, prev_rowspan) in remainder:
                texts.append(prev_text)
                if (prev_rowspan > 1):
                    next_remainder.append((prev_i, prev_text, (prev_rowspan - 1)))
            all_texts.append(texts)
            remainder = next_remainder
        while remainder:
            next_remainder = []
            texts = []
            for (prev_i, prev_text, prev_rowspan) in remainder:
                texts.append(prev_text)
                if (prev_rowspan > 1):
                    next_remainder.append((prev_i, prev_text, (prev_rowspan - 1)))
            all_texts.append(texts)
            remainder = next_remainder
        return all_texts

    def _handle_hidden_tables(self, tbl_list, attr_name):
        '\n        Return list of tables, potentially removing hidden elements\n\n        Parameters\n        ----------\n        tbl_list : list of node-like\n            Type of list elements will vary depending upon parser used\n        attr_name : str\n            Name of the accessor for retrieving HTML attributes\n\n        Returns\n        -------\n        list of node-like\n            Return type matches `tbl_list`\n        '
        if (not self.displayed_only):
            return tbl_list
        return [x for x in tbl_list if ('display:none' not in getattr(x, attr_name).get('style', '').replace(' ', ''))]

class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    '\n    HTML to DataFrame parser that uses BeautifulSoup under the hood.\n\n    See Also\n    --------\n    pandas.io.html._HtmlFrameParser\n    pandas.io.html._LxmlFrameParser\n\n    Notes\n    -----\n    Documentation strings for this class are in the base class\n    :class:`pandas.io.html._HtmlFrameParser`.\n    '

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from bs4 import SoupStrainer
        self._strainer = SoupStrainer('table')

    def _parse_tables(self, doc, match, attrs):
        element_name = self._strainer.name
        tables = doc.find_all(element_name, attrs=attrs)
        if (not tables):
            raise ValueError('No tables found')
        result = []
        unique_tables = set()
        tables = self._handle_hidden_tables(tables, 'attrs')
        for table in tables:
            if self.displayed_only:
                for elem in table.find_all(style=re.compile('display:\\s*none')):
                    elem.decompose()
            if ((table not in unique_tables) and (table.find(text=match) is not None)):
                result.append(table)
            unique_tables.add(table)
        if (not result):
            raise ValueError(f'No tables found matching pattern {repr(match.pattern)}')
        return result

    def _text_getter(self, obj):
        return obj.text

    def _equals_tag(self, obj, tag):
        return (obj.name == tag)

    def _parse_td(self, row):
        return row.find_all(('td', 'th'), recursive=False)

    def _parse_thead_tr(self, table):
        return table.select('thead tr')

    def _parse_tbody_tr(self, table):
        from_tbody = table.select('tbody tr')
        from_root = table.find_all('tr', recursive=False)
        return (from_tbody + from_root)

    def _parse_tfoot_tr(self, table):
        return table.select('tfoot tr')

    def _setup_build_doc(self):
        raw_text = _read(self.io)
        if (not raw_text):
            raise ValueError(f'No text parsed from document: {self.io}')
        return raw_text

    def _build_doc(self):
        from bs4 import BeautifulSoup
        bdoc = self._setup_build_doc()
        if (isinstance(bdoc, bytes) and (self.encoding is not None)):
            udoc = bdoc.decode(self.encoding)
            from_encoding = None
        else:
            udoc = bdoc
            from_encoding = self.encoding
        return BeautifulSoup(udoc, features='html5lib', from_encoding=from_encoding)

def _build_xpath_expr(attrs):
    "\n    Build an xpath expression to simulate bs4's ability to pass in kwargs to\n    search for attributes when using the lxml parser.\n\n    Parameters\n    ----------\n    attrs : dict\n        A dict of HTML attributes. These are NOT checked for validity.\n\n    Returns\n    -------\n    expr : unicode\n        An XPath expression that checks for the given HTML attributes.\n    "
    if ('class_' in attrs):
        attrs['class'] = attrs.pop('class_')
    s = ' and '.join([f'@{k}={repr(v)}' for (k, v) in attrs.items()])
    return f'[{s}]'
_re_namespace = {'re': 'http://exslt.org/regular-expressions'}
_valid_schemes = ('http', 'file', 'ftp')

class _LxmlFrameParser(_HtmlFrameParser):
    '\n    HTML to DataFrame parser that uses lxml under the hood.\n\n    Warning\n    -------\n    This parser can only handle HTTP, FTP, and FILE urls.\n\n    See Also\n    --------\n    _HtmlFrameParser\n    _BeautifulSoupLxmlFrameParser\n\n    Notes\n    -----\n    Documentation strings for this class are in the base class\n    :class:`_HtmlFrameParser`.\n    '

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _text_getter(self, obj):
        return obj.text_content()

    def _parse_td(self, row):
        return row.xpath('./td|./th')

    def _parse_tables(self, doc, match, kwargs):
        pattern = match.pattern
        xpath_expr = f'//table//*[re:test(text(), {repr(pattern)})]/ancestor::table'
        if kwargs:
            xpath_expr += _build_xpath_expr(kwargs)
        tables = doc.xpath(xpath_expr, namespaces=_re_namespace)
        tables = self._handle_hidden_tables(tables, 'attrib')
        if self.displayed_only:
            for table in tables:
                for elem in table.xpath('.//*[@style]'):
                    if ('display:none' in elem.attrib.get('style', '').replace(' ', '')):
                        elem.getparent().remove(elem)
        if (not tables):
            raise ValueError(f'No tables found matching regex {repr(pattern)}')
        return tables

    def _equals_tag(self, obj, tag):
        return (obj.tag == tag)

    def _build_doc(self):
        '\n        Raises\n        ------\n        ValueError\n            * If a URL that lxml cannot parse is passed.\n\n        Exception\n            * Any other ``Exception`` thrown. For example, trying to parse a\n              URL that is syntactically correct on a machine with no internet\n              connection will fail.\n\n        See Also\n        --------\n        pandas.io.html._HtmlFrameParser._build_doc\n        '
        from lxml.etree import XMLSyntaxError
        from lxml.html import HTMLParser, fromstring, parse
        parser = HTMLParser(recover=True, encoding=self.encoding)
        try:
            if is_url(self.io):
                with urlopen(self.io) as f:
                    r = parse(f, parser=parser)
            else:
                r = parse(self.io, parser=parser)
            try:
                r = r.getroot()
            except AttributeError:
                pass
        except (UnicodeDecodeError, OSError) as e:
            if (not is_url(self.io)):
                r = fromstring(self.io, parser=parser)
                try:
                    r = r.getroot()
                except AttributeError:
                    pass
            else:
                raise e
        else:
            if (not hasattr(r, 'text_content')):
                raise XMLSyntaxError('no text parsed from document', 0, 0, 0)
        return r

    def _parse_thead_tr(self, table):
        rows = []
        for thead in table.xpath('.//thead'):
            rows.extend(thead.xpath('./tr'))
            elements_at_root = thead.xpath('./td|./th')
            if elements_at_root:
                rows.append(thead)
        return rows

    def _parse_tbody_tr(self, table):
        from_tbody = table.xpath('.//tbody//tr')
        from_root = table.xpath('./tr')
        return (from_tbody + from_root)

    def _parse_tfoot_tr(self, table):
        return table.xpath('.//tfoot//tr')

def _expand_elements(body):
    data = [len(elem) for elem in body]
    lens = create_series_with_explicit_dtype(data, dtype_if_empty=object)
    lens_max = lens.max()
    not_max = lens[(lens != lens_max)]
    empty = ['']
    for (ind, length) in not_max.items():
        body[ind] += (empty * (lens_max - length))

def _data_to_frame(**kwargs):
    (head, body, foot) = kwargs.pop('data')
    header = kwargs.pop('header')
    kwargs['skiprows'] = _get_skiprows(kwargs['skiprows'])
    if head:
        body = (head + body)
        if (header is None):
            if (len(head) == 1):
                header = 0
            else:
                header = [i for (i, row) in enumerate(head) if any((text for text in row))]
    if foot:
        body += foot
    _expand_elements(body)
    with TextParser(body, header=header, **kwargs) as tp:
        return tp.read()
_valid_parsers = {'lxml': _LxmlFrameParser, None: _LxmlFrameParser, 'html5lib': _BeautifulSoupHtml5LibFrameParser, 'bs4': _BeautifulSoupHtml5LibFrameParser}

def _parser_dispatch(flavor):
    '\n    Choose the parser based on the input flavor.\n\n    Parameters\n    ----------\n    flavor : str\n        The type of parser to use. This must be a valid backend.\n\n    Returns\n    -------\n    cls : _HtmlFrameParser subclass\n        The parser class based on the requested input flavor.\n\n    Raises\n    ------\n    ValueError\n        * If `flavor` is not a valid backend.\n    ImportError\n        * If you do not have the requested `flavor`\n    '
    valid_parsers = list(_valid_parsers.keys())
    if (flavor not in valid_parsers):
        raise ValueError(f'{repr(flavor)} is not a valid flavor, valid flavors are {valid_parsers}')
    if (flavor in ('bs4', 'html5lib')):
        if (not _HAS_HTML5LIB):
            raise ImportError('html5lib not found, please install it')
        if (not _HAS_BS4):
            raise ImportError('BeautifulSoup4 (bs4) not found, please install it')
        bs4 = import_optional_dependency('bs4')
    elif (not _HAS_LXML):
        raise ImportError('lxml not found, please install it')
    return _valid_parsers[flavor]

def _print_as_set(s):
    arg = ', '.join((pprint_thing(el) for el in s))
    return f'{{{arg}}}'

def _validate_flavor(flavor):
    if (flavor is None):
        flavor = ('lxml', 'bs4')
    elif isinstance(flavor, str):
        flavor = (flavor,)
    elif isinstance(flavor, abc.Iterable):
        if (not all((isinstance(flav, str) for flav in flavor))):
            raise TypeError(f'Object of type {repr(type(flavor).__name__)} is not an iterable of strings')
    else:
        msg = (repr(flavor) if isinstance(flavor, str) else str(flavor))
        msg += ' is not a valid flavor'
        raise ValueError(msg)
    flavor = tuple(flavor)
    valid_flavors = set(_valid_parsers)
    flavor_set = set(flavor)
    if (not (flavor_set & valid_flavors)):
        raise ValueError(f'{_print_as_set(flavor_set)} is not a valid set of flavors, valid flavors are {_print_as_set(valid_flavors)}')
    return flavor

def _parse(flavor, io, match, attrs, encoding, displayed_only, **kwargs):
    flavor = _validate_flavor(flavor)
    compiled_match = re.compile(match)
    retained = None
    for flav in flavor:
        parser = _parser_dispatch(flav)
        p = parser(io, compiled_match, attrs, encoding, displayed_only)
        try:
            tables = p.parse_tables()
        except ValueError as caught:
            if (hasattr(io, 'seekable') and io.seekable()):
                io.seek(0)
            elif (hasattr(io, 'seekable') and (not io.seekable())):
                raise ValueError(f"The flavor {flav} failed to parse your input. Since you passed a non-rewindable file object, we can't rewind it to try another parser. Try read_html() with a different flavor.") from caught
            retained = caught
        else:
            break
    else:
        assert (retained is not None)
        raise retained
    ret = []
    for table in tables:
        try:
            ret.append(_data_to_frame(data=table, **kwargs))
        except EmptyDataError:
            continue
    return ret

@deprecate_nonkeyword_arguments(version='2.0')
def read_html(io, match='.+', flavor=None, header=None, index_col=None, skiprows=None, attrs=None, parse_dates=False, thousands=',', encoding=None, decimal='.', converters=None, na_values=None, keep_default_na=True, displayed_only=True):
    '\n    Read HTML tables into a ``list`` of ``DataFrame`` objects.\n\n    Parameters\n    ----------\n    io : str, path object or file-like object\n        A URL, a file-like object, or a raw string containing HTML. Note that\n        lxml only accepts the http, ftp and file url protocols. If you have a\n        URL that starts with ``\'https\'`` you might try removing the ``\'s\'``.\n\n    match : str or compiled regular expression, optional\n        The set of tables containing text matching this regex or string will be\n        returned. Unless the HTML is extremely simple you will probably need to\n        pass a non-empty string here. Defaults to \'.+\' (match any non-empty\n        string). The default value will return all tables contained on a page.\n        This value is converted to a regular expression so that there is\n        consistent behavior between Beautiful Soup and lxml.\n\n    flavor : str, optional\n        The parsing engine to use. \'bs4\' and \'html5lib\' are synonymous with\n        each other, they are both there for backwards compatibility. The\n        default of ``None`` tries to use ``lxml`` to parse and if that fails it\n        falls back on ``bs4`` + ``html5lib``.\n\n    header : int or list-like, optional\n        The row (or list of rows for a :class:`~pandas.MultiIndex`) to use to\n        make the columns headers.\n\n    index_col : int or list-like, optional\n        The column (or list of columns) to use to create the index.\n\n    skiprows : int, list-like or slice, optional\n        Number of rows to skip after parsing the column integer. 0-based. If a\n        sequence of integers or a slice is given, will skip the rows indexed by\n        that sequence.  Note that a single element sequence means \'skip the nth\n        row\' whereas an integer means \'skip n rows\'.\n\n    attrs : dict, optional\n        This is a dictionary of attributes that you can pass to use to identify\n        the table in the HTML. These are not checked for validity before being\n        passed to lxml or Beautiful Soup. However, these attributes must be\n        valid HTML table attributes to work correctly. For example, ::\n\n            attrs = {\'id\': \'table\'}\n\n        is a valid attribute dictionary because the \'id\' HTML tag attribute is\n        a valid HTML attribute for *any* HTML tag as per `this document\n        <https://html.spec.whatwg.org/multipage/dom.html#global-attributes>`__. ::\n\n            attrs = {\'asdf\': \'table\'}\n\n        is *not* a valid attribute dictionary because \'asdf\' is not a valid\n        HTML attribute even if it is a valid XML attribute.  Valid HTML 4.01\n        table attributes can be found `here\n        <http://www.w3.org/TR/REC-html40/struct/tables.html#h-11.2>`__. A\n        working draft of the HTML 5 spec can be found `here\n        <https://html.spec.whatwg.org/multipage/tables.html>`__. It contains the\n        latest information on table attributes for the modern web.\n\n    parse_dates : bool, optional\n        See :func:`~read_csv` for more details.\n\n    thousands : str, optional\n        Separator to use to parse thousands. Defaults to ``\',\'``.\n\n    encoding : str, optional\n        The encoding used to decode the web page. Defaults to ``None``.``None``\n        preserves the previous encoding behavior, which depends on the\n        underlying parser library (e.g., the parser library will try to use\n        the encoding provided by the document).\n\n    decimal : str, default \'.\'\n        Character to recognize as decimal point (e.g. use \',\' for European\n        data).\n\n    converters : dict, default None\n        Dict of functions for converting values in certain columns. Keys can\n        either be integers or column labels, values are functions that take one\n        input argument, the cell (not column) content, and return the\n        transformed content.\n\n    na_values : iterable, default None\n        Custom NA values.\n\n    keep_default_na : bool, default True\n        If na_values are specified and keep_default_na is False the default NaN\n        values are overridden, otherwise they\'re appended to.\n\n    displayed_only : bool, default True\n        Whether elements with "display: none" should be parsed.\n\n    Returns\n    -------\n    dfs\n        A list of DataFrames.\n\n    See Also\n    --------\n    read_csv : Read a comma-separated values (csv) file into DataFrame.\n\n    Notes\n    -----\n    Before using this function you should read the :ref:`gotchas about the\n    HTML parsing libraries <io.html.gotchas>`.\n\n    Expect to do some cleanup after you call this function. For example, you\n    might need to manually assign column names if the column names are\n    converted to NaN when you pass the `header=0` argument. We try to assume as\n    little as possible about the structure of the table and push the\n    idiosyncrasies of the HTML contained in the table to the user.\n\n    This function searches for ``<table>`` elements and only for ``<tr>``\n    and ``<th>`` rows and ``<td>`` elements within each ``<tr>`` or ``<th>``\n    element in the table. ``<td>`` stands for "table data". This function\n    attempts to properly handle ``colspan`` and ``rowspan`` attributes.\n    If the function has a ``<thead>`` argument, it is used to construct\n    the header, otherwise the function attempts to find the header within\n    the body (by putting rows with only ``<th>`` elements into the header).\n\n    Similar to :func:`~read_csv` the `header` argument is applied\n    **after** `skiprows` is applied.\n\n    This function will *always* return a list of :class:`DataFrame` *or*\n    it will fail, e.g., it will *not* return an empty list.\n\n    Examples\n    --------\n    See the :ref:`read_html documentation in the IO section of the docs\n    <io.read_html>` for some examples of reading in HTML tables.\n    '
    _importers()
    if (isinstance(skiprows, numbers.Integral) and (skiprows < 0)):
        raise ValueError('cannot skip rows starting from the end of the data (you passed a negative value)')
    validate_header_arg(header)
    io = stringify_path(io)
    return _parse(flavor=flavor, io=io, match=match, header=header, index_col=index_col, skiprows=skiprows, parse_dates=parse_dates, thousands=thousands, attrs=attrs, encoding=encoding, decimal=decimal, converters=converters, na_values=na_values, keep_default_na=keep_default_na, displayed_only=displayed_only)
