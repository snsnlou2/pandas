
'\nCollection of query wrappers / abstractions to both facilitate data\nretrieval and to reduce dependency on DB-specific API.\n'
from contextlib import contextmanager
from datetime import date, datetime, time
from functools import partial
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast, overload
import warnings
import numpy as np
import pandas._libs.lib as lib
from pandas._typing import DtypeArg
from pandas.core.dtypes.common import is_datetime64tz_dtype, is_dict_like, is_list_like
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core.api import DataFrame, Series
from pandas.core.base import PandasObject
from pandas.core.tools.datetimes import to_datetime

class SQLAlchemyRequired(ImportError):
    pass

class DatabaseError(IOError):
    pass
_SQLALCHEMY_INSTALLED = None

def _is_sqlalchemy_connectable(con):
    global _SQLALCHEMY_INSTALLED
    if (_SQLALCHEMY_INSTALLED is None):
        try:
            import sqlalchemy
            _SQLALCHEMY_INSTALLED = True
        except ImportError:
            _SQLALCHEMY_INSTALLED = False
    if _SQLALCHEMY_INSTALLED:
        import sqlalchemy
        return isinstance(con, sqlalchemy.engine.Connectable)
    else:
        return False

def _convert_params(sql, params):
    'Convert SQL and params args to DBAPI2.0 compliant format.'
    args = [sql]
    if (params is not None):
        if hasattr(params, 'keys'):
            args += [params]
        else:
            args += [list(params)]
    return args

def _process_parse_dates_argument(parse_dates):
    'Process parse_dates argument for read_sql functions'
    if ((parse_dates is True) or (parse_dates is None) or (parse_dates is False)):
        parse_dates = []
    elif (not hasattr(parse_dates, '__iter__')):
        parse_dates = [parse_dates]
    return parse_dates

def _handle_date_column(col, utc=None, format=None):
    if isinstance(format, dict):
        error = (format.pop('errors', None) or 'ignore')
        return to_datetime(col, errors=error, **format)
    else:
        if ((format is None) and (issubclass(col.dtype.type, np.floating) or issubclass(col.dtype.type, np.integer))):
            format = 's'
        if (format in ['D', 'd', 'h', 'm', 's', 'ms', 'us', 'ns']):
            return to_datetime(col, errors='coerce', unit=format, utc=utc)
        elif is_datetime64tz_dtype(col.dtype):
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors='coerce', format=format, utc=utc)

def _parse_date_columns(data_frame, parse_dates):
    '\n    Force non-datetime columns to be read as such.\n    Supports both string formatted and integer timestamp columns.\n    '
    parse_dates = _process_parse_dates_argument(parse_dates)
    for (col_name, df_col) in data_frame.items():
        if (is_datetime64tz_dtype(df_col.dtype) or (col_name in parse_dates)):
            try:
                fmt = parse_dates[col_name]
            except TypeError:
                fmt = None
            data_frame[col_name] = _handle_date_column(df_col, format=fmt)
    return data_frame

def _wrap_result(data, columns, index_col=None, coerce_float=True, parse_dates=None, dtype=None):
    'Wrap result set of query in a DataFrame.'
    frame = DataFrame.from_records(data, columns=columns, coerce_float=coerce_float)
    if dtype:
        frame = frame.astype(dtype)
    frame = _parse_date_columns(frame, parse_dates)
    if (index_col is not None):
        frame.set_index(index_col, inplace=True)
    return frame

def execute(sql, con, cur=None, params=None):
    '\n    Execute the given SQL query using the provided connection object.\n\n    Parameters\n    ----------\n    sql : string\n        SQL query to be executed.\n    con : SQLAlchemy connectable(engine/connection) or sqlite3 connection\n        Using SQLAlchemy makes it possible to use any DB supported by the\n        library.\n        If a DBAPI2 object, only sqlite3 is supported.\n    cur : deprecated, cursor is obtained from connection, default: None\n    params : list or tuple, optional, default: None\n        List of parameters to pass to execute method.\n\n    Returns\n    -------\n    Results Iterable\n    '
    if (cur is None):
        pandas_sql = pandasSQL_builder(con)
    else:
        pandas_sql = pandasSQL_builder(cur, is_cursor=True)
    args = _convert_params(sql, params)
    return pandas_sql.execute(*args)

@overload
def read_sql_table(table_name, con, schema=None, index_col=None, coerce_float=True, parse_dates=None, columns=None, chunksize=None):
    ...

@overload
def read_sql_table(table_name, con, schema=None, index_col=None, coerce_float=True, parse_dates=None, columns=None, chunksize=1):
    ...

def read_sql_table(table_name, con, schema=None, index_col=None, coerce_float=True, parse_dates=None, columns=None, chunksize=None):
    "\n    Read SQL database table into a DataFrame.\n\n    Given a table name and a SQLAlchemy connectable, returns a DataFrame.\n    This function does not support DBAPI connections.\n\n    Parameters\n    ----------\n    table_name : str\n        Name of SQL table in database.\n    con : SQLAlchemy connectable or str\n        A database URI could be provided as str.\n        SQLite DBAPI connection mode not supported.\n    schema : str, default None\n        Name of SQL schema in database to query (if database flavor\n        supports this). Uses default schema if None (default).\n    index_col : str or list of str, optional, default: None\n        Column(s) to set as index(MultiIndex).\n    coerce_float : bool, default True\n        Attempts to convert values of non-string, non-numeric objects (like\n        decimal.Decimal) to floating point. Can result in loss of Precision.\n    parse_dates : list or dict, default None\n        - List of column names to parse as dates.\n        - Dict of ``{column_name: format string}`` where format string is\n          strftime compatible in case of parsing string times or is one of\n          (D, s, ns, ms, us) in case of parsing integer timestamps.\n        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds\n          to the keyword arguments of :func:`pandas.to_datetime`\n          Especially useful with databases without native Datetime support,\n          such as SQLite.\n    columns : list, default None\n        List of column names to select from SQL table.\n    chunksize : int, default None\n        If specified, returns an iterator where `chunksize` is the number of\n        rows to include in each chunk.\n\n    Returns\n    -------\n    DataFrame or Iterator[DataFrame]\n        A SQL table is returned as two-dimensional data structure with labeled\n        axes.\n\n    See Also\n    --------\n    read_sql_query : Read SQL query into a DataFrame.\n    read_sql : Read SQL query or database table into a DataFrame.\n\n    Notes\n    -----\n    Any datetime values with time zone information will be converted to UTC.\n\n    Examples\n    --------\n    >>> pd.read_sql_table('table_name', 'postgres:///db_name')  # doctest:+SKIP\n    "
    con = _engine_builder(con)
    if (not _is_sqlalchemy_connectable(con)):
        raise NotImplementedError('read_sql_table only supported for SQLAlchemy connectable.')
    import sqlalchemy
    from sqlalchemy.schema import MetaData
    meta = MetaData(con, schema=schema)
    try:
        meta.reflect(only=[table_name], views=True)
    except sqlalchemy.exc.InvalidRequestError as err:
        raise ValueError(f'Table {table_name} not found') from err
    pandas_sql = SQLDatabase(con, meta=meta)
    table = pandas_sql.read_table(table_name, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, columns=columns, chunksize=chunksize)
    if (table is not None):
        return table
    else:
        raise ValueError(f'Table {table_name} not found', con)

@overload
def read_sql_query(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None, dtype=None):
    ...

@overload
def read_sql_query(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=1, dtype=None):
    ...

def read_sql_query(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None, dtype=None):
    "\n    Read SQL query into a DataFrame.\n\n    Returns a DataFrame corresponding to the result set of the query\n    string. Optionally provide an `index_col` parameter to use one of the\n    columns as the index, otherwise default integer index will be used.\n\n    Parameters\n    ----------\n    sql : str SQL query or SQLAlchemy Selectable (select or text object)\n        SQL query to be executed.\n    con : SQLAlchemy connectable, str, or sqlite3 connection\n        Using SQLAlchemy makes it possible to use any DB supported by that\n        library. If a DBAPI2 object, only sqlite3 is supported.\n    index_col : str or list of str, optional, default: None\n        Column(s) to set as index(MultiIndex).\n    coerce_float : bool, default True\n        Attempts to convert values of non-string, non-numeric objects (like\n        decimal.Decimal) to floating point. Useful for SQL result sets.\n    params : list, tuple or dict, optional, default: None\n        List of parameters to pass to execute method.  The syntax used\n        to pass parameters is database driver dependent. Check your\n        database driver documentation for which of the five syntax styles,\n        described in PEP 249's paramstyle, is supported.\n        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.\n    parse_dates : list or dict, default: None\n        - List of column names to parse as dates.\n        - Dict of ``{column_name: format string}`` where format string is\n          strftime compatible in case of parsing string times, or is one of\n          (D, s, ns, ms, us) in case of parsing integer timestamps.\n        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds\n          to the keyword arguments of :func:`pandas.to_datetime`\n          Especially useful with databases without native Datetime support,\n          such as SQLite.\n    chunksize : int, default None\n        If specified, return an iterator where `chunksize` is the number of\n        rows to include in each chunk.\n    dtype : Type name or dict of columns\n        Data type for data or columns. E.g. np.float64 or\n        {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}\n\n        .. versionadded:: 1.3.0\n\n    Returns\n    -------\n    DataFrame or Iterator[DataFrame]\n\n    See Also\n    --------\n    read_sql_table : Read SQL database table into a DataFrame.\n    read_sql : Read SQL query or database table into a DataFrame.\n\n    Notes\n    -----\n    Any datetime values with time zone information parsed via the `parse_dates`\n    parameter will be converted to UTC.\n    "
    pandas_sql = pandasSQL_builder(con)
    return pandas_sql.read_query(sql, index_col=index_col, params=params, coerce_float=coerce_float, parse_dates=parse_dates, chunksize=chunksize, dtype=dtype)

@overload
def read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None):
    ...

@overload
def read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=1):
    ...

def read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None):
    '\n    Read SQL query or database table into a DataFrame.\n\n    This function is a convenience wrapper around ``read_sql_table`` and\n    ``read_sql_query`` (for backward compatibility). It will delegate\n    to the specific function depending on the provided input. A SQL query\n    will be routed to ``read_sql_query``, while a database table name will\n    be routed to ``read_sql_table``. Note that the delegated function might\n    have more specific notes about their functionality not listed here.\n\n    Parameters\n    ----------\n    sql : str or SQLAlchemy Selectable (select or text object)\n        SQL query to be executed or a table name.\n    con : SQLAlchemy connectable, str, or sqlite3 connection\n        Using SQLAlchemy makes it possible to use any DB supported by that\n        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible\n        for engine disposal and connection closure for the SQLAlchemy connectable; str\n        connections are closed automatically. See\n        `here <https://docs.sqlalchemy.org/en/13/core/connections.html>`_.\n    index_col : str or list of str, optional, default: None\n        Column(s) to set as index(MultiIndex).\n    coerce_float : bool, default True\n        Attempts to convert values of non-string, non-numeric objects (like\n        decimal.Decimal) to floating point, useful for SQL result sets.\n    params : list, tuple or dict, optional, default: None\n        List of parameters to pass to execute method.  The syntax used\n        to pass parameters is database driver dependent. Check your\n        database driver documentation for which of the five syntax styles,\n        described in PEP 249\'s paramstyle, is supported.\n        Eg. for psycopg2, uses %(name)s so use params={\'name\' : \'value\'}.\n    parse_dates : list or dict, default: None\n        - List of column names to parse as dates.\n        - Dict of ``{column_name: format string}`` where format string is\n          strftime compatible in case of parsing string times, or is one of\n          (D, s, ns, ms, us) in case of parsing integer timestamps.\n        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds\n          to the keyword arguments of :func:`pandas.to_datetime`\n          Especially useful with databases without native Datetime support,\n          such as SQLite.\n    columns : list, default: None\n        List of column names to select from SQL table (only used when reading\n        a table).\n    chunksize : int, default None\n        If specified, return an iterator where `chunksize` is the\n        number of rows to include in each chunk.\n\n    Returns\n    -------\n    DataFrame or Iterator[DataFrame]\n\n    See Also\n    --------\n    read_sql_table : Read SQL database table into a DataFrame.\n    read_sql_query : Read SQL query into a DataFrame.\n\n    Examples\n    --------\n    Read data from SQL via either a SQL query or a SQL tablename.\n    When using a SQLite database only SQL queries are accepted,\n    providing only the SQL tablename will result in an error.\n\n    >>> from sqlite3 import connect\n    >>> conn = connect(\':memory:\')\n    >>> df = pd.DataFrame(data=[[0, \'10/11/12\'], [1, \'12/11/10\']],\n    ...                   columns=[\'int_column\', \'date_column\'])\n    >>> df.to_sql(\'test_data\', conn)\n\n    >>> pd.read_sql(\'SELECT int_column, date_column FROM test_data\', conn)\n       int_column date_column\n    0           0    10/11/12\n    1           1    12/11/10\n\n    >>> pd.read_sql(\'test_data\', \'postgres:///db_name\')  # doctest:+SKIP\n\n    Apply date parsing to columns through the ``parse_dates`` argument\n\n    >>> pd.read_sql(\'SELECT int_column, date_column FROM test_data\',\n    ...             conn,\n    ...             parse_dates=["date_column"])\n       int_column date_column\n    0           0  2012-10-11\n    1           1  2010-12-11\n\n    The ``parse_dates`` argument calls ``pd.to_datetime`` on the provided columns.\n    Custom argument values for applying ``pd.to_datetime`` on a column are specified\n    via a dictionary format:\n    1. Ignore errors while parsing the values of "date_column"\n\n    >>> pd.read_sql(\'SELECT int_column, date_column FROM test_data\',\n    ...             conn,\n    ...             parse_dates={"date_column": {"errors": "ignore"}})\n       int_column date_column\n    0           0  2012-10-11\n    1           1  2010-12-11\n\n    2. Apply a dayfirst date parsing order on the values of "date_column"\n\n    >>> pd.read_sql(\'SELECT int_column, date_column FROM test_data\',\n    ...             conn,\n    ...             parse_dates={"date_column": {"dayfirst": True}})\n       int_column date_column\n    0           0  2012-11-10\n    1           1  2010-11-12\n\n    3. Apply custom formatting when date parsing the values of "date_column"\n\n    >>> pd.read_sql(\'SELECT int_column, date_column FROM test_data\',\n    ...             conn,\n    ...             parse_dates={"date_column": {"format": "%d/%m/%y"}})\n       int_column date_column\n    0           0  2012-11-10\n    1           1  2010-11-12\n    '
    pandas_sql = pandasSQL_builder(con)
    if isinstance(pandas_sql, SQLiteDatabase):
        return pandas_sql.read_query(sql, index_col=index_col, params=params, coerce_float=coerce_float, parse_dates=parse_dates, chunksize=chunksize)
    try:
        _is_table_name = pandas_sql.has_table(sql)
    except Exception:
        _is_table_name = False
    if _is_table_name:
        pandas_sql.meta.reflect(only=[sql])
        return pandas_sql.read_table(sql, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, columns=columns, chunksize=chunksize)
    else:
        return pandas_sql.read_query(sql, index_col=index_col, params=params, coerce_float=coerce_float, parse_dates=parse_dates, chunksize=chunksize)

def to_sql(frame, name, con, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None, method=None):
    "\n    Write records stored in a DataFrame to a SQL database.\n\n    Parameters\n    ----------\n    frame : DataFrame, Series\n    name : str\n        Name of SQL table.\n    con : SQLAlchemy connectable(engine/connection) or database string URI\n        or sqlite3 DBAPI2 connection\n        Using SQLAlchemy makes it possible to use any DB supported by that\n        library.\n        If a DBAPI2 object, only sqlite3 is supported.\n    schema : str, optional\n        Name of SQL schema in database to write to (if database flavor\n        supports this). If None, use default schema (default).\n    if_exists : {'fail', 'replace', 'append'}, default 'fail'\n        - fail: If table exists, do nothing.\n        - replace: If table exists, drop it, recreate it, and insert data.\n        - append: If table exists, insert data. Create if does not exist.\n    index : boolean, default True\n        Write DataFrame index as a column.\n    index_label : str or sequence, optional\n        Column label for index column(s). If None is given (default) and\n        `index` is True, then the index names are used.\n        A sequence should be given if the DataFrame uses MultiIndex.\n    chunksize : int, optional\n        Specify the number of rows in each batch to be written at a time.\n        By default, all rows will be written at once.\n    dtype : dict or scalar, optional\n        Specifying the datatype for columns. If a dictionary is used, the\n        keys should be the column names and the values should be the\n        SQLAlchemy types or strings for the sqlite3 fallback mode. If a\n        scalar is provided, it will be applied to all columns.\n    method : {None, 'multi', callable}, optional\n        Controls the SQL insertion clause used:\n\n        - None : Uses standard SQL ``INSERT`` clause (one per row).\n        - 'multi': Pass multiple values in a single ``INSERT`` clause.\n        - callable with signature ``(pd_table, conn, keys, data_iter)``.\n\n        Details and a sample callable implementation can be found in the\n        section :ref:`insert method <io.sql.method>`.\n\n        .. versionadded:: 0.24.0\n    "
    if (if_exists not in ('fail', 'replace', 'append')):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")
    pandas_sql = pandasSQL_builder(con, schema=schema)
    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif (not isinstance(frame, DataFrame)):
        raise NotImplementedError("'frame' argument should be either a Series or a DataFrame")
    pandas_sql.to_sql(frame, name, if_exists=if_exists, index=index, index_label=index_label, schema=schema, chunksize=chunksize, dtype=dtype, method=method)

def has_table(table_name, con, schema=None):
    '\n    Check if DataBase has named table.\n\n    Parameters\n    ----------\n    table_name: string\n        Name of SQL table.\n    con: SQLAlchemy connectable(engine/connection) or sqlite3 DBAPI2 connection\n        Using SQLAlchemy makes it possible to use any DB supported by that\n        library.\n        If a DBAPI2 object, only sqlite3 is supported.\n    schema : string, default None\n        Name of SQL schema in database to write to (if database flavor supports\n        this). If None, use default schema (default).\n\n    Returns\n    -------\n    boolean\n    '
    pandas_sql = pandasSQL_builder(con, schema=schema)
    return pandas_sql.has_table(table_name)
table_exists = has_table

def _engine_builder(con):
    '\n    Returns a SQLAlchemy engine from a URI (if con is a string)\n    else it just return con without modifying it.\n    '
    global _SQLALCHEMY_INSTALLED
    if isinstance(con, str):
        try:
            import sqlalchemy
        except ImportError:
            _SQLALCHEMY_INSTALLED = False
        else:
            con = sqlalchemy.create_engine(con)
            return con
    return con

def pandasSQL_builder(con, schema=None, meta=None, is_cursor=False):
    '\n    Convenience function to return the correct PandasSQL subclass based on the\n    provided parameters.\n    '
    con = _engine_builder(con)
    if _is_sqlalchemy_connectable(con):
        return SQLDatabase(con, schema=schema, meta=meta)
    elif isinstance(con, str):
        raise ImportError('Using URI string without sqlalchemy installed.')
    else:
        return SQLiteDatabase(con, is_cursor=is_cursor)

class SQLTable(PandasObject):
    '\n    For mapping Pandas tables to SQL tables.\n    Uses fact that table is reflected by SQLAlchemy to\n    do better type conversions.\n    Also holds various flags needed to avoid having to\n    pass them between functions all the time.\n    '

    def __init__(self, name, pandas_sql_engine, frame=None, index=True, if_exists='fail', prefix='pandas', index_label=None, schema=None, keys=None, dtype=None):
        self.name = name
        self.pd_sql = pandas_sql_engine
        self.prefix = prefix
        self.frame = frame
        self.index = self._index_name(index, index_label)
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype
        if (frame is not None):
            self.table = self._create_table_setup()
        else:
            self.table = self.pd_sql.get_table(self.name, self.schema)
        if (self.table is None):
            raise ValueError(f"Could not init table '{name}'")

    def exists(self):
        return self.pd_sql.has_table(self.name, self.schema)

    def sql_schema(self):
        from sqlalchemy.schema import CreateTable
        return str(CreateTable(self.table).compile(self.pd_sql.connectable))

    def _execute_create(self):
        self.table = self.table.tometadata(self.pd_sql.meta)
        self.table.create()

    def create(self):
        if self.exists():
            if (self.if_exists == 'fail'):
                raise ValueError(f"Table '{self.name}' already exists.")
            elif (self.if_exists == 'replace'):
                self.pd_sql.drop_table(self.name, self.schema)
                self._execute_create()
            elif (self.if_exists == 'append'):
                pass
            else:
                raise ValueError(f"'{self.if_exists}' is not valid for if_exists")
        else:
            self._execute_create()

    def _execute_insert(self, conn, keys, data_iter):
        '\n        Execute SQL statement inserting data\n\n        Parameters\n        ----------\n        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection\n        keys : list of str\n           Column names\n        data_iter : generator of list\n           Each item contains a list of values to be inserted\n        '
        data = [dict(zip(keys, row)) for row in data_iter]
        conn.execute(self.table.insert(), data)

    def _execute_insert_multi(self, conn, keys, data_iter):
        '\n        Alternative to _execute_insert for DBs support multivalue INSERT.\n\n        Note: multi-value insert is usually faster for analytics DBs\n        and tables containing a few columns\n        but performance degrades quickly with increase of columns.\n        '
        data = [dict(zip(keys, row)) for row in data_iter]
        conn.execute(self.table.insert(data))

    def insert_data(self):
        if (self.index is not None):
            temp = self.frame.copy()
            temp.index.names = self.index
            try:
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f'duplicate name in index/columns: {err}') from err
        else:
            temp = self.frame
        column_names = list(map(str, temp.columns))
        ncols = len(column_names)
        data_list = ([None] * ncols)
        for (i, (_, ser)) in enumerate(temp.items()):
            vals = ser._values
            if (vals.dtype.kind == 'M'):
                d = vals.to_pydatetime()
            elif (vals.dtype.kind == 'm'):
                d = vals.view('i8').astype(object)
            else:
                d = vals.astype(object)
            assert isinstance(d, np.ndarray), type(d)
            if ser._can_hold_na:
                mask = isna(d)
                d[mask] = None
            data_list[i] = d
        return (column_names, data_list)

    def insert(self, chunksize=None, method=None):
        if (method is None):
            exec_insert = self._execute_insert
        elif (method == 'multi'):
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f'Invalid parameter `method`: {method}')
        (keys, data_list) = self.insert_data()
        nrows = len(self.frame)
        if (nrows == 0):
            return
        if (chunksize is None):
            chunksize = nrows
        elif (chunksize == 0):
            raise ValueError('chunksize argument should be non-zero')
        chunks = (int((nrows / chunksize)) + 1)
        with self.pd_sql.run_transaction() as conn:
            for i in range(chunks):
                start_i = (i * chunksize)
                end_i = min(((i + 1) * chunksize), nrows)
                if (start_i >= end_i):
                    break
                chunk_iter = zip(*[arr[start_i:end_i] for arr in data_list])
                exec_insert(conn, keys, chunk_iter)

    def _query_iterator(self, result, chunksize, columns, coerce_float=True, parse_dates=None):
        'Return generator through chunked result set.'
        while True:
            data = result.fetchmany(chunksize)
            if (not data):
                break
            else:
                self.frame = DataFrame.from_records(data, columns=columns, coerce_float=coerce_float)
                self._harmonize_columns(parse_dates=parse_dates)
                if (self.index is not None):
                    self.frame.set_index(self.index, inplace=True)
                (yield self.frame)

    def read(self, coerce_float=True, parse_dates=None, columns=None, chunksize=None):
        if ((columns is not None) and (len(columns) > 0)):
            from sqlalchemy import select
            cols = [self.table.c[n] for n in columns]
            if (self.index is not None):
                for idx in self.index[::(- 1)]:
                    cols.insert(0, self.table.c[idx])
            sql_select = select(cols)
        else:
            sql_select = self.table.select()
        result = self.pd_sql.execute(sql_select)
        column_names = result.keys()
        if (chunksize is not None):
            return self._query_iterator(result, chunksize, column_names, coerce_float=coerce_float, parse_dates=parse_dates)
        else:
            data = result.fetchall()
            self.frame = DataFrame.from_records(data, columns=column_names, coerce_float=coerce_float)
            self._harmonize_columns(parse_dates=parse_dates)
            if (self.index is not None):
                self.frame.set_index(self.index, inplace=True)
            return self.frame

    def _index_name(self, index, index_label):
        if (index is True):
            nlevels = self.frame.index.nlevels
            if (index_label is not None):
                if (not isinstance(index_label, list)):
                    index_label = [index_label]
                if (len(index_label) != nlevels):
                    raise ValueError(f"Length of 'index_label' should match number of levels, which is {nlevels}")
                else:
                    return index_label
            if ((nlevels == 1) and ('index' not in self.frame.columns) and (self.frame.index.name is None)):
                return ['index']
            else:
                return [(l if (l is not None) else f'level_{i}') for (i, l) in enumerate(self.frame.index.names)]
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None

    def _get_column_names_and_types(self, dtype_mapper):
        column_names_and_types = []
        if (self.index is not None):
            for (i, idx_label) in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))
                column_names_and_types.append((str(idx_label), idx_type, True))
        column_names_and_types += [(str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False) for i in range(len(self.frame.columns))]
        return column_names_and_types

    def _create_table_setup(self):
        from sqlalchemy import Column, PrimaryKeyConstraint, Table
        column_names_and_types = self._get_column_names_and_types(self._sqlalchemy_type)
        columns = [Column(name, typ, index=is_index) for (name, typ, is_index) in column_names_and_types]
        if (self.keys is not None):
            if (not is_list_like(self.keys)):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=(self.name + '_pk'))
            columns.append(pkc)
        schema = (self.schema or self.pd_sql.meta.schema)
        from sqlalchemy.schema import MetaData
        meta = MetaData(self.pd_sql, schema=schema)
        return Table(self.name, meta, *columns, schema=schema)

    def _harmonize_columns(self, parse_dates=None):
        "\n        Make the DataFrame's column types align with the SQL table\n        column types.\n        Need to work around limited NA value support. Floats are always\n        fine, ints must always be floats if there are Null values.\n        Booleans are hard because converting bool column with None replaces\n        all Nones with false. Therefore only convert bool if there are no\n        NA values.\n        Datetimes should already be converted to np.datetime64 if supported,\n        but here we also force conversion if required.\n        "
        parse_dates = _process_parse_dates_argument(parse_dates)
        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                df_col = self.frame[col_name]
                if (col_name in parse_dates):
                    try:
                        fmt = parse_dates[col_name]
                    except TypeError:
                        fmt = None
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue
                col_type = self._get_dtype(sql_col.type)
                if ((col_type is datetime) or (col_type is date) or (col_type is DatetimeTZDtype)):
                    utc = (col_type is DatetimeTZDtype)
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                elif (col_type is float):
                    self.frame[col_name] = df_col.astype(col_type, copy=False)
                elif (len(df_col) == df_col.count()):
                    if ((col_type is np.dtype('int64')) or (col_type is bool)):
                        self.frame[col_name] = df_col.astype(col_type, copy=False)
            except KeyError:
                pass

    def _sqlalchemy_type(self, col):
        dtype: DtypeArg = (self.dtype or {})
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if (col.name in dtype):
                return dtype[col.name]
        col_type = lib.infer_dtype(col, skipna=True)
        from sqlalchemy.types import TIMESTAMP, BigInteger, Boolean, Date, DateTime, Float, Integer, SmallInteger, Text, Time
        if ((col_type == 'datetime64') or (col_type == 'datetime')):
            try:
                if (col.dt.tz is not None):
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                if (getattr(col, 'tz', None) is not None):
                    return TIMESTAMP(timezone=True)
            return DateTime
        if (col_type == 'timedelta64'):
            warnings.warn("the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.", UserWarning, stacklevel=8)
            return BigInteger
        elif (col_type == 'floating'):
            if (col.dtype == 'float32'):
                return Float(precision=23)
            else:
                return Float(precision=53)
        elif (col_type == 'integer'):
            if (col.dtype.name.lower() in ('int8', 'uint8', 'int16')):
                return SmallInteger
            elif (col.dtype.name.lower() in ('uint16', 'int32')):
                return Integer
            elif (col.dtype.name.lower() == 'uint64'):
                raise ValueError('Unsigned 64 bit integer datatype is not supported')
            else:
                return BigInteger
        elif (col_type == 'boolean'):
            return Boolean
        elif (col_type == 'date'):
            return Date
        elif (col_type == 'time'):
            return Time
        elif (col_type == 'complex'):
            raise ValueError('Complex datatypes not supported')
        return Text

    def _get_dtype(self, sqltype):
        from sqlalchemy.types import TIMESTAMP, Boolean, Date, DateTime, Float, Integer
        if isinstance(sqltype, Float):
            return float
        elif isinstance(sqltype, Integer):
            return np.dtype('int64')
        elif isinstance(sqltype, TIMESTAMP):
            if (not sqltype.timezone):
                return datetime
            return DatetimeTZDtype
        elif isinstance(sqltype, DateTime):
            return datetime
        elif isinstance(sqltype, Date):
            return date
        elif isinstance(sqltype, Boolean):
            return bool
        return object

class PandasSQL(PandasObject):
    '\n    Subclasses Should define read_sql and to_sql.\n    '

    def read_sql(self, *args, **kwargs):
        raise ValueError('PandasSQL must be created with an SQLAlchemy connectable or sqlite connection')

    def to_sql(self, frame, name, if_exists='fail', index=True, index_label=None, schema=None, chunksize=None, dtype=None, method=None):
        raise ValueError('PandasSQL must be created with an SQLAlchemy connectable or sqlite connection')

class SQLDatabase(PandasSQL):
    '\n    This class enables conversion between DataFrame and SQL databases\n    using SQLAlchemy to handle DataBase abstraction.\n\n    Parameters\n    ----------\n    engine : SQLAlchemy connectable\n        Connectable to connect with the database. Using SQLAlchemy makes it\n        possible to use any DB supported by that library.\n    schema : string, default None\n        Name of SQL schema in database to write to (if database flavor\n        supports this). If None, use default schema (default).\n    meta : SQLAlchemy MetaData object, default None\n        If provided, this MetaData object is used instead of a newly\n        created. This allows to specify database flavor specific\n        arguments in the MetaData object.\n\n    '

    def __init__(self, engine, schema=None, meta=None):
        self.connectable = engine
        if (not meta):
            from sqlalchemy.schema import MetaData
            meta = MetaData(self.connectable, schema=schema)
        self.meta = meta

    @contextmanager
    def run_transaction(self):
        with self.connectable.begin() as tx:
            if hasattr(tx, 'execute'):
                (yield tx)
            else:
                (yield self.connectable)

    def execute(self, *args, **kwargs):
        'Simple passthrough to SQLAlchemy connectable'
        return self.connectable.execution_options().execute(*args, **kwargs)

    def read_table(self, table_name, index_col=None, coerce_float=True, parse_dates=None, columns=None, schema=None, chunksize=None):
        '\n        Read SQL database table into a DataFrame.\n\n        Parameters\n        ----------\n        table_name : string\n            Name of SQL table in database.\n        index_col : string, optional, default: None\n            Column to set as index.\n        coerce_float : boolean, default True\n            Attempts to convert values of non-string, non-numeric objects\n            (like decimal.Decimal) to floating point. This can result in\n            loss of precision.\n        parse_dates : list or dict, default: None\n            - List of column names to parse as dates.\n            - Dict of ``{column_name: format string}`` where format string is\n              strftime compatible in case of parsing string times, or is one of\n              (D, s, ns, ms, us) in case of parsing integer timestamps.\n            - Dict of ``{column_name: arg}``, where the arg corresponds\n              to the keyword arguments of :func:`pandas.to_datetime`.\n              Especially useful with databases without native Datetime support,\n              such as SQLite.\n        columns : list, default: None\n            List of column names to select from SQL table.\n        schema : string, default None\n            Name of SQL schema in database to query (if database flavor\n            supports this).  If specified, this overwrites the default\n            schema of the SQL database object.\n        chunksize : int, default None\n            If specified, return an iterator where `chunksize` is the number\n            of rows to include in each chunk.\n\n        Returns\n        -------\n        DataFrame\n\n        See Also\n        --------\n        pandas.read_sql_table\n        SQLDatabase.read_query\n\n        '
        table = SQLTable(table_name, self, index=index_col, schema=schema)
        return table.read(coerce_float=coerce_float, parse_dates=parse_dates, columns=columns, chunksize=chunksize)

    @staticmethod
    def _query_iterator(result, chunksize, columns, index_col=None, coerce_float=True, parse_dates=None, dtype=None):
        'Return generator through chunked result set'
        while True:
            data = result.fetchmany(chunksize)
            if (not data):
                break
            else:
                (yield _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype))

    def read_query(self, sql, index_col=None, coerce_float=True, parse_dates=None, params=None, chunksize=None, dtype=None):
        "\n        Read SQL query into a DataFrame.\n\n        Parameters\n        ----------\n        sql : string\n            SQL query to be executed.\n        index_col : string, optional, default: None\n            Column name to use as index for the returned DataFrame object.\n        coerce_float : boolean, default True\n            Attempt to convert values of non-string, non-numeric objects (like\n            decimal.Decimal) to floating point, useful for SQL result sets.\n        params : list, tuple or dict, optional, default: None\n            List of parameters to pass to execute method.  The syntax used\n            to pass parameters is database driver dependent. Check your\n            database driver documentation for which of the five syntax styles,\n            described in PEP 249's paramstyle, is supported.\n            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}\n        parse_dates : list or dict, default: None\n            - List of column names to parse as dates.\n            - Dict of ``{column_name: format string}`` where format string is\n              strftime compatible in case of parsing string times, or is one of\n              (D, s, ns, ms, us) in case of parsing integer timestamps.\n            - Dict of ``{column_name: arg dict}``, where the arg dict\n              corresponds to the keyword arguments of\n              :func:`pandas.to_datetime` Especially useful with databases\n              without native Datetime support, such as SQLite.\n        chunksize : int, default None\n            If specified, return an iterator where `chunksize` is the number\n            of rows to include in each chunk.\n        dtype : Type name or dict of columns\n            Data type for data or columns. E.g. np.float64 or\n            {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}\n\n            .. versionadded:: 1.3.0\n\n        Returns\n        -------\n        DataFrame\n\n        See Also\n        --------\n        read_sql_table : Read SQL database table into a DataFrame.\n        read_sql\n\n        "
        args = _convert_params(sql, params)
        result = self.execute(*args)
        columns = result.keys()
        if (chunksize is not None):
            return self._query_iterator(result, chunksize, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype)
        else:
            data = result.fetchall()
            frame = _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype)
            return frame
    read_sql = read_query

    def to_sql(self, frame, name, if_exists='fail', index=True, index_label=None, schema=None, chunksize=None, dtype=None, method=None):
        "\n        Write records stored in a DataFrame to a SQL database.\n\n        Parameters\n        ----------\n        frame : DataFrame\n        name : string\n            Name of SQL table.\n        if_exists : {'fail', 'replace', 'append'}, default 'fail'\n            - fail: If table exists, do nothing.\n            - replace: If table exists, drop it, recreate it, and insert data.\n            - append: If table exists, insert data. Create if does not exist.\n        index : boolean, default True\n            Write DataFrame index as a column.\n        index_label : string or sequence, default None\n            Column label for index column(s). If None is given (default) and\n            `index` is True, then the index names are used.\n            A sequence should be given if the DataFrame uses MultiIndex.\n        schema : string, default None\n            Name of SQL schema in database to write to (if database flavor\n            supports this). If specified, this overwrites the default\n            schema of the SQLDatabase object.\n        chunksize : int, default None\n            If not None, then rows will be written in batches of this size at a\n            time.  If None, all rows will be written at once.\n        dtype : single type or dict of column name to SQL type, default None\n            Optional specifying the datatype for columns. The SQL type should\n            be a SQLAlchemy type. If all columns are of the same type, one\n            single value can be used.\n        method : {None', 'multi', callable}, default None\n            Controls the SQL insertion clause used:\n\n            * None : Uses standard SQL ``INSERT`` clause (one per row).\n            * 'multi': Pass multiple values in a single ``INSERT`` clause.\n            * callable with signature ``(pd_table, conn, keys, data_iter)``.\n\n            Details and a sample callable implementation can be found in the\n            section :ref:`insert method <io.sql.method>`.\n\n            .. versionadded:: 0.24.0\n        "
        if dtype:
            if (not is_dict_like(dtype)):
                dtype = {col_name: dtype for col_name in frame}
            else:
                dtype = cast(dict, dtype)
            from sqlalchemy.types import TypeEngine, to_instance
            for (col, my_type) in dtype.items():
                if (not isinstance(to_instance(my_type), TypeEngine)):
                    raise ValueError(f'The type of {col} is not a SQLAlchemy type')
        table = SQLTable(name, self, frame=frame, index=index, if_exists=if_exists, index_label=index_label, schema=schema, dtype=dtype)
        table.create()
        from sqlalchemy import exc
        try:
            table.insert(chunksize, method=method)
        except exc.SQLAlchemyError as err:
            msg = '(1054, "Unknown column \'inf\' in \'field list\'")'
            err_text = str(err.orig)
            if re.search(msg, err_text):
                raise ValueError('inf cannot be used with MySQL') from err
            else:
                raise err
        if ((not name.isdigit()) and (not name.islower())):
            engine = self.connectable.engine
            with self.connectable.connect() as conn:
                table_names = engine.table_names(schema=(schema or self.meta.schema), connection=conn)
            if (name not in table_names):
                msg = f"The provided table name '{name}' is not found exactly as such in the database after writing the table, possibly due to case sensitivity issues. Consider using lower case table names."
                warnings.warn(msg, UserWarning)

    @property
    def tables(self):
        return self.meta.tables

    def has_table(self, name, schema=None):
        return self.connectable.run_callable(self.connectable.dialect.has_table, name, (schema or self.meta.schema))

    def get_table(self, table_name, schema=None):
        schema = (schema or self.meta.schema)
        if schema:
            tbl = self.meta.tables.get('.'.join([schema, table_name]))
        else:
            tbl = self.meta.tables.get(table_name)
        from sqlalchemy import Numeric
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False
        return tbl

    def drop_table(self, table_name, schema=None):
        schema = (schema or self.meta.schema)
        if self.has_table(table_name, schema):
            self.meta.reflect(only=[table_name], schema=schema)
            self.get_table(table_name, schema).drop()
            self.meta.clear()

    def _create_sql_schema(self, frame, table_name, keys=None, dtype=None, schema=None):
        table = SQLTable(table_name, self, frame=frame, index=False, keys=keys, dtype=dtype, schema=schema)
        return str(table.sql_schema())
_SQL_TYPES = {'string': 'TEXT', 'floating': 'REAL', 'integer': 'INTEGER', 'datetime': 'TIMESTAMP', 'date': 'DATE', 'time': 'TIME', 'boolean': 'INTEGER'}

def _get_unicode_name(name):
    try:
        uname = str(name).encode('utf-8', 'strict').decode('utf-8')
    except UnicodeError as err:
        raise ValueError(f"Cannot convert identifier to UTF-8: '{name}'") from err
    return uname

def _get_valid_sqlite_name(name):
    uname = _get_unicode_name(name)
    if (not len(uname)):
        raise ValueError('Empty table or column name specified')
    nul_index = uname.find('\x00')
    if (nul_index >= 0):
        raise ValueError('SQLite identifier cannot contain NULs')
    return (('"' + uname.replace('"', '""')) + '"')
_SAFE_NAMES_WARNING = 'The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.'

class SQLiteTable(SQLTable):
    '\n    Patch the SQLTable for fallback support.\n    Instead of a table variable just use the Create Table statement.\n    '

    def __init__(self, *args, **kwargs):
        import sqlite3
        sqlite3.register_adapter(time, (lambda _: _.strftime('%H:%M:%S.%f')))
        super().__init__(*args, **kwargs)

    def sql_schema(self):
        return str(';\n'.join(self.table))

    def _execute_create(self):
        with self.pd_sql.run_transaction() as conn:
            for stmt in self.table:
                conn.execute(stmt)

    def insert_statement(self, *, num_rows):
        names = list(map(str, self.frame.columns))
        wld = '?'
        escape = _get_valid_sqlite_name
        if (self.index is not None):
            for idx in self.index[::(- 1)]:
                names.insert(0, idx)
        bracketed_names = [escape(column) for column in names]
        col_names = ','.join(bracketed_names)
        row_wildcards = ','.join(([wld] * len(names)))
        wildcards = ','.join((f'({row_wildcards})' for _ in range(num_rows)))
        insert_statement = f'INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}'
        return insert_statement

    def _execute_insert(self, conn, keys, data_iter):
        data_list = list(data_iter)
        conn.executemany(self.insert_statement(num_rows=1), data_list)

    def _execute_insert_multi(self, conn, keys, data_iter):
        data_list = list(data_iter)
        flattened_data = [x for row in data_list for x in row]
        conn.execute(self.insert_statement(num_rows=len(data_list)), flattened_data)

    def _create_table_setup(self):
        '\n        Return a list of SQL statements that creates a table reflecting the\n        structure of a DataFrame.  The first entry will be a CREATE TABLE\n        statement while the rest will be CREATE INDEX statements.\n        '
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        pat = re.compile('\\s+')
        column_names = [col_name for (col_name, _, _) in column_names_and_types]
        if any(map(pat.search, column_names)):
            warnings.warn(_SAFE_NAMES_WARNING, stacklevel=6)
        escape = _get_valid_sqlite_name
        create_tbl_stmts = [((escape(cname) + ' ') + ctype) for (cname, ctype, _) in column_names_and_types]
        if ((self.keys is not None) and len(self.keys)):
            if (not is_list_like(self.keys)):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ', '.join((escape(c) for c in keys))
            create_tbl_stmts.append(f'CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})')
        if self.schema:
            schema_name = (self.schema + '.')
        else:
            schema_name = ''
        create_stmts = [((((('CREATE TABLE ' + schema_name) + escape(self.name)) + ' (\n') + ',\n  '.join(create_tbl_stmts)) + '\n)')]
        ix_cols = [cname for (cname, _, is_index) in column_names_and_types if is_index]
        if len(ix_cols):
            cnames = '_'.join(ix_cols)
            cnames_br = ','.join((escape(c) for c in ix_cols))
            create_stmts.append((((((('CREATE INDEX ' + escape(((('ix_' + self.name) + '_') + cnames))) + 'ON ') + escape(self.name)) + ' (') + cnames_br) + ')'))
        return create_stmts

    def _sql_type_name(self, col):
        dtype: DtypeArg = (self.dtype or {})
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if (col.name in dtype):
                return dtype[col.name]
        col_type = lib.infer_dtype(col, skipna=True)
        if (col_type == 'timedelta64'):
            warnings.warn("the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.", UserWarning, stacklevel=8)
            col_type = 'integer'
        elif (col_type == 'datetime64'):
            col_type = 'datetime'
        elif (col_type == 'empty'):
            col_type = 'string'
        elif (col_type == 'complex'):
            raise ValueError('Complex datatypes not supported')
        if (col_type not in _SQL_TYPES):
            col_type = 'string'
        return _SQL_TYPES[col_type]

class SQLiteDatabase(PandasSQL):
    '\n    Version of SQLDatabase to support SQLite connections (fallback without\n    SQLAlchemy). This should only be used internally.\n\n    Parameters\n    ----------\n    con : sqlite connection object\n\n    '

    def __init__(self, con, is_cursor=False):
        self.is_cursor = is_cursor
        self.con = con

    @contextmanager
    def run_transaction(self):
        cur = self.con.cursor()
        try:
            (yield cur)
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        finally:
            cur.close()

    def execute(self, *args, **kwargs):
        if self.is_cursor:
            cur = self.con
        else:
            cur = self.con.cursor()
        try:
            cur.execute(*args, **kwargs)
            return cur
        except Exception as exc:
            try:
                self.con.rollback()
            except Exception as inner_exc:
                ex = DatabaseError(f'''Execution failed on sql: {args[0]}
{exc}
unable to rollback''')
                raise ex from inner_exc
            ex = DatabaseError(f"Execution failed on sql '{args[0]}': {exc}")
            raise ex from exc

    @staticmethod
    def _query_iterator(cursor, chunksize, columns, index_col=None, coerce_float=True, parse_dates=None, dtype=None):
        'Return generator through chunked result set'
        while True:
            data = cursor.fetchmany(chunksize)
            if (type(data) == tuple):
                data = list(data)
            if (not data):
                cursor.close()
                break
            else:
                (yield _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype))

    def read_query(self, sql, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None, dtype=None):
        args = _convert_params(sql, params)
        cursor = self.execute(*args)
        columns = [col_desc[0] for col_desc in cursor.description]
        if (chunksize is not None):
            return self._query_iterator(cursor, chunksize, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype)
        else:
            data = self._fetchall_as_list(cursor)
            cursor.close()
            frame = _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype)
            return frame

    def _fetchall_as_list(self, cur):
        result = cur.fetchall()
        if (not isinstance(result, list)):
            result = list(result)
        return result

    def to_sql(self, frame, name, if_exists='fail', index=True, index_label=None, schema=None, chunksize=None, dtype=None, method=None):
        "\n        Write records stored in a DataFrame to a SQL database.\n\n        Parameters\n        ----------\n        frame: DataFrame\n        name: string\n            Name of SQL table.\n        if_exists: {'fail', 'replace', 'append'}, default 'fail'\n            fail: If table exists, do nothing.\n            replace: If table exists, drop it, recreate it, and insert data.\n            append: If table exists, insert data. Create if it does not exist.\n        index : boolean, default True\n            Write DataFrame index as a column\n        index_label : string or sequence, default None\n            Column label for index column(s). If None is given (default) and\n            `index` is True, then the index names are used.\n            A sequence should be given if the DataFrame uses MultiIndex.\n        schema : string, default None\n            Ignored parameter included for compatibility with SQLAlchemy\n            version of ``to_sql``.\n        chunksize : int, default None\n            If not None, then rows will be written in batches of this\n            size at a time. If None, all rows will be written at once.\n        dtype : single type or dict of column name to SQL type, default None\n            Optional specifying the datatype for columns. The SQL type should\n            be a string. If all columns are of the same type, one single value\n            can be used.\n        method : {None, 'multi', callable}, default None\n            Controls the SQL insertion clause used:\n\n            * None : Uses standard SQL ``INSERT`` clause (one per row).\n            * 'multi': Pass multiple values in a single ``INSERT`` clause.\n            * callable with signature ``(pd_table, conn, keys, data_iter)``.\n\n            Details and a sample callable implementation can be found in the\n            section :ref:`insert method <io.sql.method>`.\n\n            .. versionadded:: 0.24.0\n        "
        if dtype:
            if (not is_dict_like(dtype)):
                dtype = {col_name: dtype for col_name in frame}
            else:
                dtype = cast(dict, dtype)
            for (col, my_type) in dtype.items():
                if (not isinstance(my_type, str)):
                    raise ValueError(f'{col} ({my_type}) not a string')
        table = SQLiteTable(name, self, frame=frame, index=index, if_exists=if_exists, index_label=index_label, dtype=dtype)
        table.create()
        table.insert(chunksize, method)

    def has_table(self, name, schema=None):
        wld = '?'
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name={wld};"
        return (len(self.execute(query, [name]).fetchall()) > 0)

    def get_table(self, table_name, schema=None):
        return None

    def drop_table(self, name, schema=None):
        drop_sql = f'DROP TABLE {_get_valid_sqlite_name(name)}'
        self.execute(drop_sql)

    def _create_sql_schema(self, frame, table_name, keys=None, dtype=None, schema=None):
        table = SQLiteTable(table_name, self, frame=frame, index=False, keys=keys, dtype=dtype, schema=schema)
        return str(table.sql_schema())

def get_schema(frame, name, keys=None, con=None, dtype=None, schema=None):
    '\n    Get the SQL db table schema for the given frame.\n\n    Parameters\n    ----------\n    frame : DataFrame\n    name : string\n        name of SQL table\n    keys : string or sequence, default: None\n        columns to use a primary key\n    con: an open SQL database connection object or a SQLAlchemy connectable\n        Using SQLAlchemy makes it possible to use any DB supported by that\n        library, default: None\n        If a DBAPI2 object, only sqlite3 is supported.\n    dtype : dict of column name to SQL type, default None\n        Optional specifying the datatype for columns. The SQL type should\n        be a SQLAlchemy type, or a string for sqlite3 fallback connection.\n    schema: str, default: None\n        Optional specifying the schema to be used in creating the table.\n\n        .. versionadded:: 1.2.0\n    '
    pandas_sql = pandasSQL_builder(con=con)
    return pandas_sql._create_sql_schema(frame, name, keys=keys, dtype=dtype, schema=schema)
