
'\nModule for formatting output data into CSV files.\n'
import csv as csvlib
import os
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Union
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import CompressionOptions, FilePathOrBuffer, FloatFormatType, IndexLabel, Label, StorageOptions
from pandas.core.dtypes.generic import ABCDatetimeIndex, ABCIndex, ABCMultiIndex, ABCPeriodIndex
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle
if TYPE_CHECKING:
    from pandas.io.formats.format import DataFrameFormatter

class CSVFormatter():

    def __init__(self, formatter, path_or_buf='', sep=',', cols=None, index_label=None, mode='w', encoding=None, errors='strict', compression='infer', quoting=None, line_terminator='\n', chunksize=None, quotechar='"', date_format=None, doublequote=True, escapechar=None, storage_options=None):
        self.fmt = formatter
        self.obj = self.fmt.frame
        self.filepath_or_buffer = path_or_buf
        self.encoding = encoding
        self.compression = compression
        self.mode = mode
        self.storage_options = storage_options
        self.sep = sep
        self.index_label = self._initialize_index_label(index_label)
        self.errors = errors
        self.quoting = (quoting or csvlib.QUOTE_MINIMAL)
        self.quotechar = self._initialize_quotechar(quotechar)
        self.doublequote = doublequote
        self.escapechar = escapechar
        self.line_terminator = (line_terminator or os.linesep)
        self.date_format = date_format
        self.cols = self._initialize_columns(cols)
        self.chunksize = self._initialize_chunksize(chunksize)

    @property
    def na_rep(self):
        return self.fmt.na_rep

    @property
    def float_format(self):
        return self.fmt.float_format

    @property
    def decimal(self):
        return self.fmt.decimal

    @property
    def header(self):
        return self.fmt.header

    @property
    def index(self):
        return self.fmt.index

    def _initialize_index_label(self, index_label):
        if (index_label is not False):
            if (index_label is None):
                return self._get_index_label_from_obj()
            elif (not isinstance(index_label, (list, tuple, np.ndarray, ABCIndex))):
                return [index_label]
        return index_label

    def _get_index_label_from_obj(self):
        if isinstance(self.obj.index, ABCMultiIndex):
            return self._get_index_label_multiindex()
        else:
            return self._get_index_label_flat()

    def _get_index_label_multiindex(self):
        return [(name or '') for name in self.obj.index.names]

    def _get_index_label_flat(self):
        index_label = self.obj.index.name
        return ([''] if (index_label is None) else [index_label])

    def _initialize_quotechar(self, quotechar):
        if (self.quoting != csvlib.QUOTE_NONE):
            return quotechar
        return None

    @property
    def has_mi_columns(self):
        return bool(isinstance(self.obj.columns, ABCMultiIndex))

    def _initialize_columns(self, cols):
        if self.has_mi_columns:
            if (cols is not None):
                msg = 'cannot specify cols with a MultiIndex on the columns'
                raise TypeError(msg)
        if (cols is not None):
            if isinstance(cols, ABCIndex):
                cols = cols._format_native_types(**self._number_format)
            else:
                cols = list(cols)
            self.obj = self.obj.loc[:, cols]
        new_cols = self.obj.columns
        if isinstance(new_cols, ABCIndex):
            return new_cols._format_native_types(**self._number_format)
        else:
            return list(new_cols)

    def _initialize_chunksize(self, chunksize):
        if (chunksize is None):
            return ((100000 // (len(self.cols) or 1)) or 1)
        return int(chunksize)

    @property
    def _number_format(self):
        'Dictionary used for storing number formatting settings.'
        return {'na_rep': self.na_rep, 'float_format': self.float_format, 'date_format': self.date_format, 'quoting': self.quoting, 'decimal': self.decimal}

    @property
    def data_index(self):
        data_index = self.obj.index
        if (isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex)) and (self.date_format is not None)):
            data_index = Index([(x.strftime(self.date_format) if notna(x) else '') for x in data_index])
        return data_index

    @property
    def nlevels(self):
        if self.index:
            return getattr(self.data_index, 'nlevels', 1)
        else:
            return 0

    @property
    def _has_aliases(self):
        return isinstance(self.header, (tuple, list, np.ndarray, ABCIndex))

    @property
    def _need_to_save_header(self):
        return bool((self._has_aliases or self.header))

    @property
    def write_cols(self):
        if self._has_aliases:
            assert (not isinstance(self.header, bool))
            if (len(self.header) != len(self.cols)):
                raise ValueError(f'Writing {len(self.cols)} cols but got {len(self.header)} aliases')
            else:
                return self.header
        else:
            return self.cols

    @property
    def encoded_labels(self):
        encoded_labels: List[Label] = []
        if (self.index and self.index_label):
            assert isinstance(self.index_label, Sequence)
            encoded_labels = list(self.index_label)
        if ((not self.has_mi_columns) or self._has_aliases):
            encoded_labels += list(self.write_cols)
        return encoded_labels

    def save(self):
        '\n        Create the writer & save.\n        '
        with get_handle(self.filepath_or_buffer, self.mode, encoding=self.encoding, errors=self.errors, compression=self.compression, storage_options=self.storage_options) as handles:
            self.writer = csvlib.writer(handles.handle, lineterminator=self.line_terminator, delimiter=self.sep, quoting=self.quoting, doublequote=self.doublequote, escapechar=self.escapechar, quotechar=self.quotechar)
            self._save()

    def _save(self):
        if self._need_to_save_header:
            self._save_header()
        self._save_body()

    def _save_header(self):
        if ((not self.has_mi_columns) or self._has_aliases):
            self.writer.writerow(self.encoded_labels)
        else:
            for row in self._generate_multiindex_header_rows():
                self.writer.writerow(row)

    def _generate_multiindex_header_rows(self):
        columns = self.obj.columns
        for i in range(columns.nlevels):
            col_line = []
            if self.index:
                col_line.append(columns.names[i])
                if (isinstance(self.index_label, list) and (len(self.index_label) > 1)):
                    col_line.extend(([''] * (len(self.index_label) - 1)))
            col_line.extend(columns._get_level_values(i))
            (yield col_line)
        if (self.encoded_labels and (set(self.encoded_labels) != {''})):
            (yield (self.encoded_labels + ([''] * len(columns))))

    def _save_body(self):
        nrows = len(self.data_index)
        chunks = (int((nrows / self.chunksize)) + 1)
        for i in range(chunks):
            start_i = (i * self.chunksize)
            end_i = min((start_i + self.chunksize), nrows)
            if (start_i >= end_i):
                break
            self._save_chunk(start_i, end_i)

    def _save_chunk(self, start_i, end_i):
        slicer = slice(start_i, end_i)
        df = self.obj.iloc[slicer]
        res = df._mgr.to_native_types(**self._number_format)
        data = [res.iget_values(i) for i in range(len(res.items))]
        ix = self.data_index[slicer]._format_native_types(**self._number_format)
        libwriters.write_csv_rows(data, ix, self.nlevels, self.cols, self.writer)
