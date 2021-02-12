
'\nValidate that the titles in the rst files follow the proper capitalization convention.\n\nPrint the titles that do not follow the convention.\n\nUsage::\n./scripts/validate_rst_title_capitalization.py doc/source/development/contributing.rst\n./scripts/validate_rst_title_capitalization.py doc/source/\n\n'
import argparse
import glob
import os
import re
import sys
from typing import Iterable, List, Tuple
CAPITALIZATION_EXCEPTIONS = {'pandas', 'pd', 'Python', 'IPython', 'PyTables', 'Excel', 'JSON', 'HTML', 'SAS', 'SQL', 'BigQuery', 'STATA', 'Interval', 'IntervalArray', 'PEP8', 'Period', 'Series', 'Index', 'DataFrame', 'DataFrames', 'C', 'Git', 'GitHub', 'NumPy', 'Apache', 'Arrow', 'Parquet', 'MultiIndex', 'NumFOCUS', 'sklearn', 'Docker', 'PeriodIndex', 'NA', 'NaN', 'NaT', 'ValueError', 'Boolean', 'BooleanArray', 'KeyError', 'API', 'FAQ', 'IO', 'Timedelta', 'TimedeltaIndex', 'DatetimeIndex', 'IntervalIndex', 'Categorical', 'CategoricalIndex', 'Categorical', 'GroupBy', 'SPSS', 'ORC', 'R', 'HDF5', 'HDFStore', 'CDay', 'CBMonthBegin', 'CBMonthEnd', 'BMonthBegin', 'BMonthEnd', 'BDay', 'FY5253Quarter', 'FY5253', 'YearBegin', 'YearEnd', 'BYearBegin', 'BYearEnd', 'YearOffset', 'QuarterBegin', 'QuarterEnd', 'BQuarterBegin', 'BQuarterEnd', 'QuarterOffset', 'LastWeekOfMonth', 'WeekOfMonth', 'SemiMonthBegin', 'SemiMonthEnd', 'SemiMonthOffset', 'CustomBusinessMonthBegin', 'CustomBusinessMonthEnd', 'BusinessMonthBegin', 'BusinessMonthEnd', 'MonthBegin', 'MonthEnd', 'MonthOffset', 'CustomBusinessHour', 'CustomBusinessDay', 'BusinessHour', 'BusinessDay', 'DateOffset', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Float64Index', 'FloatIndex', 'TZ', 'GIL', 'strftime', 'XPORT', 'Unicode', 'East', 'Asian', 'None', 'URLs', 'UInt64', 'SciPy', 'Matplotlib', 'PyPy', 'SparseDataFrame', 'Google', 'CategoricalDtype', 'UTC', 'False', 'Styler', 'os', 'UTC', 'str', 'msgpack', 'ExtensionArray', 'LZMA', 'Numba', 'Timestamp'}
CAP_EXCEPTIONS_DICT = {word.lower(): word for word in CAPITALIZATION_EXCEPTIONS}
err_msg = 'Heading capitalization formatted incorrectly. Please correctly capitalize'
symbols = ('*', '=', '-', '^', '~', '#', '"')

def correct_title_capitalization(title):
    '\n    Algorithm to create the correct capitalization for a given title.\n\n    Parameters\n    ----------\n    title : str\n        Heading string to correct.\n\n    Returns\n    -------\n    str\n        Correctly capitalized heading.\n    '
    if (title[0] == ':'):
        return title
    correct_title: str = re.sub('^\\W*', '', title).capitalize()
    removed_https_title = re.sub('<https?:\\/\\/.*[\\r\\n]*>', '', correct_title)
    word_list = re.split('\\W', removed_https_title)
    for word in word_list:
        if (word.lower() in CAP_EXCEPTIONS_DICT):
            correct_title = re.sub(f'\b{word}\b', CAP_EXCEPTIONS_DICT[word.lower()], correct_title)
    return correct_title

def find_titles(rst_file):
    '\n    Algorithm to identify particular text that should be considered headings in an\n    RST file.\n\n    See <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html> for details\n    on what constitutes a string as a heading in RST.\n\n    Parameters\n    ----------\n    rst_file : str\n        RST file to scan through for headings.\n\n    Yields\n    -------\n    title : str\n        A heading found in the rst file.\n\n    line_number : int\n        The corresponding line number of the heading.\n    '
    with open(rst_file) as fd:
        previous_line = ''
        for (i, line) in enumerate(fd):
            line = line[:(- 1)]
            line_chars = set(line)
            if ((len(line_chars) == 1) and (line_chars.pop() in symbols) and (len(line) == len(previous_line))):
                (yield (re.sub('[`\\*_]', '', previous_line), i))
            previous_line = line

def find_rst_files(source_paths):
    '\n    Given the command line arguments of directory paths, this method\n    yields the strings of the .rst file directories that these paths contain.\n\n    Parameters\n    ----------\n    source_paths : str\n        List of directories to validate, provided through command line arguments.\n\n    Yields\n    -------\n    str\n        Directory address of a .rst files found in command line argument directories.\n    '
    for directory_address in source_paths:
        if (not os.path.exists(directory_address)):
            raise ValueError('Please enter a valid path, pointing to a valid file/directory.')
        elif directory_address.endswith('.rst'):
            (yield directory_address)
        else:
            (yield from glob.glob(pathname=f'{directory_address}/**/*.rst', recursive=True))

def main(source_paths, output_format):
    '\n    The main method to print all headings with incorrect capitalization.\n\n    Parameters\n    ----------\n    source_paths : str\n        List of directories to validate, provided through command line arguments.\n    output_format : str\n        Output format of the script.\n\n    Returns\n    -------\n    int\n        Number of incorrect headings found overall.\n    '
    number_of_errors: int = 0
    for filename in find_rst_files(source_paths):
        for (title, line_number) in find_titles(filename):
            if (title != correct_title_capitalization(title)):
                print(f'{filename}:{line_number}:{err_msg} "{title}" to "{correct_title_capitalization(title)}" ')
                number_of_errors += 1
    return number_of_errors
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Validate heading capitalization')
    parser.add_argument('paths', nargs='+', default='.', help='Source paths of file/directory to check.')
    parser.add_argument('--format', '-f', default='{source_path}:{line_number}:{msg}:{heading}:{correct_heading}', help='Output format of incorrectly capitalized titles')
    args = parser.parse_args()
    sys.exit(main(args.paths, args.format))
