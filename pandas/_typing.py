
from datetime import datetime, timedelta, tzinfo
from io import BufferedIOBase, RawIOBase, TextIOBase, TextIOWrapper
from mmap import mmap
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, AnyStr, Callable, Collection, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union
import numpy as np
if TYPE_CHECKING:
    from typing import final
    from pandas._libs import Period, Timedelta, Timestamp
    from pandas.core.dtypes.dtypes import ExtensionDtype
    from pandas import Interval
    from pandas.core.arrays.base import ExtensionArray
    from pandas.core.frame import DataFrame
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
    from pandas.core.indexes.base import Index
    from pandas.core.resample import Resampler
    from pandas.core.series import Series
    from pandas.core.window.rolling import BaseWindow
    from pandas.io.formats.format import EngFormatter
else:
    final = (lambda x: x)
AnyArrayLike = TypeVar('AnyArrayLike', 'ExtensionArray', 'Index', 'Series', np.ndarray)
ArrayLike = TypeVar('ArrayLike', 'ExtensionArray', np.ndarray)
PythonScalar = Union[(str, int, float, bool)]
DatetimeLikeScalar = TypeVar('DatetimeLikeScalar', 'Period', 'Timestamp', 'Timedelta')
PandasScalar = Union[('Period', 'Timestamp', 'Timedelta', 'Interval')]
Scalar = Union[(PythonScalar, PandasScalar)]
TimestampConvertibleTypes = Union[('Timestamp', datetime, np.datetime64, int, np.int64, float, str)]
TimedeltaConvertibleTypes = Union[('Timedelta', timedelta, np.timedelta64, int, np.int64, float, str)]
Timezone = Union[(str, tzinfo)]
FrameOrSeriesUnion = Union[('DataFrame', 'Series')]
FrameOrSeries = TypeVar('FrameOrSeries', bound='NDFrame')
Axis = Union[(str, int)]
Label = Optional[Hashable]
IndexLabel = Union[(Label, Sequence[Label])]
Level = Union[(Label, int)]
Shape = Tuple[(int, ...)]
Suffixes = Tuple[(str, str)]
Ordered = Optional[bool]
JSONSerializable = Optional[Union[(PythonScalar, List, Dict)]]
Axes = Collection
NpDtype = Union[(str, np.dtype)]
Dtype = Union[('ExtensionDtype', NpDtype, Type[Union[(str, float, int, complex, bool, object)]])]
DtypeArg = Union[(Dtype, Dict[(Label, Dtype)])]
DtypeObj = Union[(np.dtype, 'ExtensionDtype')]
Renamer = Union[(Mapping[(Label, Any)], Callable[([Label], Label)])]
T = TypeVar('T')
FuncType = Callable[(..., Any)]
F = TypeVar('F', bound=FuncType)
ValueKeyFunc = Optional[Callable[(['Series'], Union[('Series', AnyArrayLike)])]]
IndexKeyFunc = Optional[Callable[(['Index'], Union[('Index', AnyArrayLike)])]]
AggFuncTypeBase = Union[(Callable, str)]
AggFuncTypeDict = Dict[(Label, Union[(AggFuncTypeBase, List[AggFuncTypeBase])])]
AggFuncType = Union[(AggFuncTypeBase, List[AggFuncTypeBase], AggFuncTypeDict)]
AggObjType = Union[('Series', 'DataFrame', 'SeriesGroupBy', 'DataFrameGroupBy', 'BaseWindow', 'Resampler')]
PythonFuncType = Callable[([Any], Any)]
Buffer = Union[(IO[AnyStr], RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper, mmap)]
FileOrBuffer = Union[(str, Buffer[T])]
FilePathOrBuffer = Union[('PathLike[str]', FileOrBuffer[T])]
StorageOptions = Optional[Dict[(str, Any)]]
CompressionDict = Dict[(str, Any)]
CompressionOptions = Optional[Union[(str, CompressionDict)]]
FormattersType = Union[(List[Callable], Tuple[(Callable, ...)], Mapping[(Union[(str, int)], Callable)])]
ColspaceType = Mapping[(Label, Union[(str, int)])]
FloatFormatType = Union[(str, Callable, 'EngFormatter')]
ColspaceArgType = Union[(str, int, Sequence[Union[(str, int)]], Mapping[(Label, Union[(str, int)])])]
