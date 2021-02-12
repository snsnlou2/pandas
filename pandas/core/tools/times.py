
from datetime import datetime, time
from typing import List, Optional
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.core.dtypes.generic import ABCIndex, ABCSeries
from pandas.core.dtypes.missing import notna

def to_time(arg, format=None, infer_time_format=False, errors='raise'):
    '\n    Parse time strings to time objects using fixed strptime formats ("%H:%M",\n    "%H%M", "%I:%M%p", "%I%M%p", "%H:%M:%S", "%H%M%S", "%I:%M:%S%p",\n    "%I%M%S%p")\n\n    Use infer_time_format if all the strings are in the same format to speed\n    up conversion.\n\n    Parameters\n    ----------\n    arg : string in time format, datetime.time, list, tuple, 1-d array,  Series\n    format : str, default None\n        Format used to convert arg into a time object.  If None, fixed formats\n        are used.\n    infer_time_format: bool, default False\n        Infer the time format based on the first non-NaN element.  If all\n        strings are in the same format, this will speed up conversion.\n    errors : {\'ignore\', \'raise\', \'coerce\'}, default \'raise\'\n        - If \'raise\', then invalid parsing will raise an exception\n        - If \'coerce\', then invalid parsing will be set as None\n        - If \'ignore\', then invalid parsing will return the input\n\n    Returns\n    -------\n    datetime.time\n    '

    def _convert_listlike(arg, format):
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg, dtype='O')
        elif (getattr(arg, 'ndim', 1) > 1):
            raise TypeError('arg must be a string, datetime, list, tuple, 1-d array, or Series')
        arg = np.asarray(arg, dtype='O')
        if (infer_time_format and (format is None)):
            format = _guess_time_format_for_array(arg)
        times: List[Optional[time]] = []
        if (format is not None):
            for element in arg:
                try:
                    times.append(datetime.strptime(element, format).time())
                except (ValueError, TypeError) as err:
                    if (errors == 'raise'):
                        msg = f'Cannot convert {element} to a time with given format {format}'
                        raise ValueError(msg) from err
                    elif (errors == 'ignore'):
                        return arg
                    else:
                        times.append(None)
        else:
            formats = _time_formats[:]
            format_found = False
            for element in arg:
                time_object = None
                for time_format in formats:
                    try:
                        time_object = datetime.strptime(element, time_format).time()
                        if (not format_found):
                            fmt = formats.pop(formats.index(time_format))
                            formats.insert(0, fmt)
                            format_found = True
                        break
                    except (ValueError, TypeError):
                        continue
                if (time_object is not None):
                    times.append(time_object)
                elif (errors == 'raise'):
                    raise ValueError(f'Cannot convert arg {arg} to a time')
                elif (errors == 'ignore'):
                    return arg
                else:
                    times.append(None)
        return times
    if (arg is None):
        return arg
    elif isinstance(arg, time):
        return arg
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, format)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, format)
    elif is_list_like(arg):
        return _convert_listlike(arg, format)
    return _convert_listlike(np.array([arg]), format)[0]
_time_formats = ['%H:%M', '%H%M', '%I:%M%p', '%I%M%p', '%H:%M:%S', '%H%M%S', '%I:%M:%S%p', '%I%M%S%p']

def _guess_time_format_for_array(arr):
    non_nan_elements = notna(arr).nonzero()[0]
    if len(non_nan_elements):
        element = arr[non_nan_elements[0]]
        for time_format in _time_formats:
            try:
                datetime.strptime(element, time_format)
                return time_format
            except ValueError:
                pass
    return None
