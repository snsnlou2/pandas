
'\nUnwanted patterns test cases.\n\nThe reason this file exist despite the fact we already have\n`ci/code_checks.sh`,\n(see https://github.com/pandas-dev/pandas/blob/master/ci/code_checks.sh)\n\nis that some of the test cases are more complex/imposible to validate via regex.\nSo this file is somewhat an extensions to `ci/code_checks.sh`\n'
import argparse
import ast
import sys
import token
import tokenize
from typing import IO, Callable, Iterable, List, Set, Tuple
PRIVATE_IMPORTS_TO_IGNORE = {'_extension_array_shared_docs', '_index_shared_docs', '_interval_shared_docs', '_merge_doc', '_shared_docs', '_apply_docs', '_new_Index', '_new_PeriodIndex', '_doc_template', '_agg_template', '_pipe_template', '_get_version', '__main__', '_transform_template', '_flex_comp_doc_FRAME', '_op_descriptions', '_IntegerDtype', '_use_inf_as_na', '_get_plot_backend', '_matplotlib', '_arrow_utils', '_registry', '_get_offset', '_test_parse_iso8601', '_json_normalize', '_testing', '_test_decorators', '__version__'}

def _get_literal_string_prefix_len(token_string):
    '\n    Getting the length of the literal string prefix.\n\n    Parameters\n    ----------\n    token_string : str\n        String to check.\n\n    Returns\n    -------\n    int\n        Length of the literal string prefix.\n\n    Examples\n    --------\n    >>> example_string = "\'Hello world\'"\n    >>> _get_literal_string_prefix_len(example_string)\n    0\n    >>> example_string = "r\'Hello world\'"\n    >>> _get_literal_string_prefix_len(example_string)\n    1\n    '
    try:
        return min((token_string.find(quote) for quote in ("'", '"') if (token_string.find(quote) >= 0)))
    except ValueError:
        return 0

def bare_pytest_raises(file_obj):
    '\n    Test Case for bare pytest raises.\n\n    For example, this is wrong:\n\n    >>> with pytest.raise(ValueError):\n    ...     # Some code that raises ValueError\n\n    And this is what we want instead:\n\n    >>> with pytest.raise(ValueError, match="foo"):\n    ...     # Some code that raises ValueError\n\n    Parameters\n    ----------\n    file_obj : IO\n        File-like object containing the Python code to validate.\n\n    Yields\n    ------\n    line_number : int\n        Line number of unconcatenated string.\n    msg : str\n        Explenation of the error.\n\n    Notes\n    -----\n    GH #23922\n    '
    contents = file_obj.read()
    tree = ast.parse(contents)
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Call)):
            continue
        try:
            if (not ((node.func.value.id == 'pytest') and (node.func.attr == 'raises'))):
                continue
        except AttributeError:
            continue
        if (not node.keywords):
            (yield (node.lineno, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception."))
        elif (not any(((keyword.arg == 'match') for keyword in node.keywords))):
            (yield (node.lineno, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception."))
PRIVATE_FUNCTIONS_ALLOWED = {'sys._getframe'}

def private_function_across_module(file_obj):
    '\n    Checking that a private function is not used across modules.\n    Parameters\n    ----------\n    file_obj : IO\n        File-like object containing the Python code to validate.\n    Yields\n    ------\n    line_number : int\n        Line number of the private function that is used across modules.\n    msg : str\n        Explenation of the error.\n    '
    contents = file_obj.read()
    tree = ast.parse(contents)
    imported_modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for module in node.names:
                module_fqdn = (module.name if (module.asname is None) else module.asname)
                imported_modules.add(module_fqdn)
        if (not isinstance(node, ast.Call)):
            continue
        try:
            module_name = node.func.value.id
            function_name = node.func.attr
        except AttributeError:
            continue
        if module_name[0].isupper():
            continue
        elif (function_name.startswith('__') and function_name.endswith('__')):
            continue
        elif (((module_name + '.') + function_name) in PRIVATE_FUNCTIONS_ALLOWED):
            continue
        if ((module_name in imported_modules) and function_name.startswith('_')):
            (yield (node.lineno, f"Private function '{module_name}.{function_name}'"))

def private_import_across_module(file_obj):
    '\n    Checking that a private function is not imported across modules.\n    Parameters\n    ----------\n    file_obj : IO\n        File-like object containing the Python code to validate.\n    Yields\n    ------\n    line_number : int\n        Line number of import statement, that imports the private function.\n    msg : str\n        Explenation of the error.\n    '
    contents = file_obj.read()
    tree = ast.parse(contents)
    for node in ast.walk(tree):
        if (not (isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom))):
            continue
        for module in node.names:
            module_name = module.name.split('.')[(- 1)]
            if (module_name in PRIVATE_IMPORTS_TO_IGNORE):
                continue
            if module_name.startswith('_'):
                (yield (node.lineno, f'Import of internal function {repr(module_name)}'))

def strings_to_concatenate(file_obj):
    '\n    This test case is necessary after \'Black\' (https://github.com/psf/black),\n    is formating strings over multiple lines.\n\n    For example, when this:\n\n    >>> foo = (\n    ...     "bar "\n    ...     "baz"\n    ... )\n\n    Is becoming this:\n\n    >>> foo = ("bar " "baz")\n\n    \'Black\' is not considering this as an\n    issue (see https://github.com/psf/black/issues/1051),\n    so we are checking it here instead.\n\n    Parameters\n    ----------\n    file_obj : IO\n        File-like object containing the Python code to validate.\n\n    Yields\n    ------\n    line_number : int\n        Line number of unconcatenated string.\n    msg : str\n        Explenation of the error.\n\n    Notes\n    -----\n    GH #30454\n    '
    tokens: List = list(tokenize.generate_tokens(file_obj.readline))
    for (current_token, next_token) in zip(tokens, tokens[1:]):
        if (current_token.type == next_token.type == token.STRING):
            (yield (current_token.start[0], 'String unnecessarily split in two by black. Please merge them manually.'))

def strings_with_wrong_placed_whitespace(file_obj):
    '\n    Test case for leading spaces in concated strings.\n\n    For example:\n\n    >>> rule = (\n    ...    "We want the space at the end of the line, "\n    ...    "not at the beginning"\n    ... )\n\n    Instead of:\n\n    >>> rule = (\n    ...    "We want the space at the end of the line,"\n    ...    " not at the beginning"\n    ... )\n\n    Parameters\n    ----------\n    file_obj : IO\n        File-like object containing the Python code to validate.\n\n    Yields\n    ------\n    line_number : int\n        Line number of unconcatenated string.\n    msg : str\n        Explenation of the error.\n    '

    def has_wrong_whitespace(first_line: str, second_line: str) -> bool:
        '\n        Checking if the two lines are mattching the unwanted pattern.\n\n        Parameters\n        ----------\n        first_line : str\n            First line to check.\n        second_line : str\n            Second line to check.\n\n        Returns\n        -------\n        bool\n            True if the two recived string match, an unwanted pattern.\n\n        Notes\n        -----\n        The unwanted pattern that we are trying to catch is if the spaces in\n        a string that is concatenated over multiple lines are placed at the\n        end of each string, unless this string is ending with a\n        newline character (\n).\n\n        For example, this is bad:\n\n        >>> rule = (\n        ...    "We want the space at the end of the line,"\n        ...    " not at the beginning"\n        ... )\n\n        And what we want is:\n\n        >>> rule = (\n        ...    "We want the space at the end of the line, "\n        ...    "not at the beginning"\n        ... )\n\n        And if the string is ending with a new line character (\n) we\n        do not want any trailing whitespaces after it.\n\n        For example, this is bad:\n\n        >>> rule = (\n        ...    "We want the space at the begging of "\n        ...    "the line if the previous line is ending with a \n "\n        ...    "not at the end, like always"\n        ... )\n\n        And what we do want is:\n\n        >>> rule = (\n        ...    "We want the space at the begging of "\n        ...    "the line if the previous line is ending with a \n"\n        ...    " not at the end, like always"\n        ... )\n        '
        if first_line.endswith('\\n'):
            return False
        elif (first_line.startswith('  ') or second_line.startswith('  ')):
            return False
        elif (first_line.endswith('  ') or second_line.endswith('  ')):
            return False
        elif ((not first_line.endswith(' ')) and second_line.startswith(' ')):
            return True
        return False
    tokens: List = list(tokenize.generate_tokens(file_obj.readline))
    for (first_token, second_token, third_token) in zip(tokens, tokens[1:], tokens[2:]):
        if ((first_token.type == third_token.type == token.STRING) and (second_token.type == token.NL)):
            first_string: str = first_token.string[(_get_literal_string_prefix_len(first_token.string) + 1):(- 1)]
            second_string: str = third_token.string[(_get_literal_string_prefix_len(third_token.string) + 1):(- 1)]
            if has_wrong_whitespace(first_string, second_string):
                (yield (third_token.start[0], 'String has a space at the beginning instead of the end of the previous string.'))

def main(function, source_path, output_format):
    '\n    Main entry point of the script.\n\n    Parameters\n    ----------\n    function : Callable\n        Function to execute for the specified validation type.\n    source_path : str\n        Source path representing path to a file/directory.\n    output_format : str\n        Output format of the error message.\n    file_extensions_to_check : str\n        Comma separated values of what file extensions to check.\n    excluded_file_paths : str\n        Comma separated values of what file paths to exclude during the check.\n\n    Returns\n    -------\n    bool\n        True if found any patterns are found related to the given function.\n\n    Raises\n    ------\n    ValueError\n        If the `source_path` is not pointing to existing file/directory.\n    '
    is_failed: bool = False
    for file_path in source_path:
        with open(file_path, encoding='utf-8') as file_obj:
            for (line_number, msg) in function(file_obj):
                is_failed = True
                print(output_format.format(source_path=file_path, line_number=line_number, msg=msg))
    return is_failed
if (__name__ == '__main__'):
    available_validation_types = ['bare_pytest_raises', 'private_function_across_module', 'private_import_across_module', 'strings_to_concatenate', 'strings_with_wrong_placed_whitespace']
    parser = argparse.ArgumentParser(description='Unwanted patterns checker.')
    parser.add_argument('paths', nargs='*', help='Source paths of files to check.')
    parser.add_argument('--format', '-f', default='{source_path}:{line_number}:{msg}', help='Output format of the error message.')
    parser.add_argument('--validation-type', '-vt', choices=available_validation_types, required=True, help='Validation test case to check.')
    args = parser.parse_args()
    sys.exit(main(function=globals().get(args.validation_type), source_path=args.paths, output_format=args.format))
