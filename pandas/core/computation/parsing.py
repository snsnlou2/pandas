
'\n:func:`~pandas.eval` source string parsing functions\n'
from io import StringIO
from keyword import iskeyword
import token
import tokenize
from typing import Iterator, Tuple
from pandas._typing import Label
BACKTICK_QUOTED_STRING = 100

def create_valid_python_identifier(name):
    '\n    Create valid Python identifiers from any string.\n\n    Check if name contains any special characters. If it contains any\n    special characters, the special characters will be replaced by\n    a special string and a prefix is added.\n\n    Raises\n    ------\n    SyntaxError\n        If the returned name is not a Python valid identifier, raise an exception.\n        This can happen if there is a hashtag in the name, as the tokenizer will\n        than terminate and not find the backtick.\n        But also for characters that fall out of the range of (U+0001..U+007F).\n    '
    if (name.isidentifier() and (not iskeyword(name))):
        return name
    special_characters_replacements = {char: f'_{token.tok_name[tokval]}_' for (char, tokval) in tokenize.EXACT_TOKEN_TYPES.items()}
    special_characters_replacements.update({' ': '_', '?': '_QUESTIONMARK_', '!': '_EXCLAMATIONMARK_', '$': '_DOLLARSIGN_', '€': '_EUROSIGN_', "'": '_SINGLEQUOTE_', '"': '_DOUBLEQUOTE_'})
    name = ''.join((special_characters_replacements.get(char, char) for char in name))
    name = ('BACKTICK_QUOTED_STRING_' + name)
    if (not name.isidentifier()):
        raise SyntaxError(f"Could not convert '{name}' to a valid Python identifier.")
    return name

def clean_backtick_quoted_toks(tok):
    '\n    Clean up a column name if surrounded by backticks.\n\n    Backtick quoted string are indicated by a certain tokval value. If a string\n    is a backtick quoted token it will processed by\n    :func:`_create_valid_python_identifier` so that the parser can find this\n    string when the query is executed.\n    In this case the tok will get the NAME tokval.\n\n    Parameters\n    ----------\n    tok : tuple of int, str\n        ints correspond to the all caps constants in the tokenize module\n\n    Returns\n    -------\n    tok : Tuple[int, str]\n        Either the input or token or the replacement values\n    '
    (toknum, tokval) = tok
    if (toknum == BACKTICK_QUOTED_STRING):
        return (tokenize.NAME, create_valid_python_identifier(tokval))
    return (toknum, tokval)

def clean_column_name(name):
    '\n    Function to emulate the cleaning of a backtick quoted name.\n\n    The purpose for this function is to see what happens to the name of\n    identifier if it goes to the process of being parsed a Python code\n    inside a backtick quoted string and than being cleaned\n    (removed of any special characters).\n\n    Parameters\n    ----------\n    name : hashable\n        Name to be cleaned.\n\n    Returns\n    -------\n    name : hashable\n        Returns the name after tokenizing and cleaning.\n\n    Notes\n    -----\n        For some cases, a name cannot be converted to a valid Python identifier.\n        In that case :func:`tokenize_string` raises a SyntaxError.\n        In that case, we just return the name unmodified.\n\n        If this name was used in the query string (this makes the query call impossible)\n        an error will be raised by :func:`tokenize_backtick_quoted_string` instead,\n        which is not caught and propagates to the user level.\n    '
    try:
        tokenized = tokenize_string(f'`{name}`')
        tokval = next(tokenized)[1]
        return create_valid_python_identifier(tokval)
    except SyntaxError:
        return name

def tokenize_backtick_quoted_string(token_generator, source, string_start):
    '\n    Creates a token from a backtick quoted string.\n\n    Moves the token_generator forwards till right after the next backtick.\n\n    Parameters\n    ----------\n    token_generator : Iterator[tokenize.TokenInfo]\n        The generator that yields the tokens of the source string (Tuple[int, str]).\n        The generator is at the first token after the backtick (`)\n\n    source : str\n        The Python source code string.\n\n    string_start : int\n        This is the start of backtick quoted string inside the source string.\n\n    Returns\n    -------\n    tok: Tuple[int, str]\n        The token that represents the backtick quoted string.\n        The integer is equal to BACKTICK_QUOTED_STRING (100).\n    '
    for (_, tokval, start, _, _) in token_generator:
        if (tokval == '`'):
            string_end = start[1]
            break
    return (BACKTICK_QUOTED_STRING, source[string_start:string_end])

def tokenize_string(source):
    '\n    Tokenize a Python source code string.\n\n    Parameters\n    ----------\n    source : str\n        The Python source code string.\n\n    Returns\n    -------\n    tok_generator : Iterator[Tuple[int, str]]\n        An iterator yielding all tokens with only toknum and tokval (Tuple[ing, str]).\n    '
    line_reader = StringIO(source).readline
    token_generator = tokenize.generate_tokens(line_reader)
    for (toknum, tokval, start, _, _) in token_generator:
        if (tokval == '`'):
            try:
                (yield tokenize_backtick_quoted_string(token_generator, source, string_start=(start[1] + 1)))
            except Exception as err:
                raise SyntaxError(f"Failed to parse backticks in '{source}'.") from err
        else:
            (yield (toknum, tokval))
