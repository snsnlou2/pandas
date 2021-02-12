
"\nCheck that test suite file doesn't use the pandas namespace inconsistently.\n\nWe check for cases of ``Series`` and ``pd.Series`` appearing in the same file\n(likewise for some other common classes).\n\nThis is meant to be run as a pre-commit hook - to run it manually, you can do:\n\n    pre-commit run inconsistent-namespace-usage --all-files\n"
import argparse
from pathlib import Path
import re
from typing import Optional, Sequence
PATTERN = "\n    (\n        (?<!pd\\.)(?<!\\w)    # check class_name doesn't start with pd. or character\n        ([A-Z]\\w+)\\(        # match DataFrame but not pd.DataFrame or tm.makeDataFrame\n        .*                  # match anything\n        pd\\.\\2\\(            # only match e.g. pd.DataFrame\n    )|\n    (\n        pd\\.([A-Z]\\w+)\\(    # only match e.g. pd.DataFrame\n        .*                  # match anything\n        (?<!pd\\.)(?<!\\w)    # check class_name doesn't start with pd. or character\n        \\4\\(                # match DataFrame but not pd.DataFrame or tm.makeDataFrame\n    )\n    "
ERROR_MESSAGE = 'Found both `pd.{class_name}` and `{class_name}` in {path}'

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', type=Path)
    args = parser.parse_args(argv)
    pattern = re.compile(PATTERN.encode(), flags=((re.MULTILINE | re.DOTALL) | re.VERBOSE))
    for path in args.paths:
        contents = path.read_bytes()
        match = pattern.search(contents)
        if (match is None):
            continue
        if (match.group(2) is not None):
            raise AssertionError(ERROR_MESSAGE.format(class_name=match.group(2).decode(), path=str(path)))
        if (match.group(4) is not None):
            raise AssertionError(ERROR_MESSAGE.format(class_name=match.group(4).decode(), path=str(path)))
if (__name__ == '__main__'):
    main()
