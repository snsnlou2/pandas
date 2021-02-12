
from typing import List
from pandas._typing import FilePathOrBuffer, Scalar, StorageOptions
from pandas.compat._optional import import_optional_dependency
from pandas.io.excel._base import BaseExcelReader

class PyxlsbReader(BaseExcelReader):

    def __init__(self, filepath_or_buffer, storage_options=None):
        '\n        Reader using pyxlsb engine.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object, or Workbook\n            Object to be parsed.\n        storage_options : dict, optional\n            passed to fsspec for appropriate URLs (see ``_get_filepath_or_buffer``)\n        '
        import_optional_dependency('pyxlsb')
        super().__init__(filepath_or_buffer, storage_options=storage_options)

    @property
    def _workbook_class(self):
        from pyxlsb import Workbook
        return Workbook

    def load_workbook(self, filepath_or_buffer):
        from pyxlsb import open_workbook
        return open_workbook(filepath_or_buffer)

    @property
    def sheet_names(self):
        return self.book.sheets

    def get_sheet_by_name(self, name):
        return self.book.get_sheet(name)

    def get_sheet_by_index(self, index):
        return self.book.get_sheet((index + 1))

    def _convert_cell(self, cell, convert_float):
        if (cell.v is None):
            return ''
        if (isinstance(cell.v, float) and convert_float):
            val = int(cell.v)
            if (val == cell.v):
                return val
            else:
                return float(cell.v)
        return cell.v

    def get_sheet_data(self, sheet, convert_float):
        return [[self._convert_cell(c, convert_float) for c in r] for r in sheet.rows(sparse=False)]
