
import contextlib
from typing import Tuple

@contextlib.contextmanager
def rewrite_exception(old_name, new_name):
    '\n    Rewrite the message of an exception.\n    '
    try:
        (yield)
    except Exception as err:
        msg = err.args[0]
        msg = msg.replace(old_name, new_name)
        args: Tuple[(str, ...)] = (msg,)
        if (len(err.args) > 1):
            args = (args + err.args[1:])
        err.args = args
        raise
