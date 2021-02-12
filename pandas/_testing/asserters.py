
from typing import Union, cast
import warnings
import numpy as np
from pandas._libs.lib import no_default
import pandas._libs.testing as _testing
from pandas.core.dtypes.common import is_bool, is_categorical_dtype, is_extension_array_dtype, is_interval_dtype, is_number, is_numeric_dtype, needs_i8_conversion
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import Categorical, DataFrame, DatetimeIndex, Index, IntervalIndex, MultiIndex, PeriodIndex, Series, TimedeltaIndex
from pandas.core.algorithms import take_1d
from pandas.core.arrays import DatetimeArray, ExtensionArray, IntervalArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.io.formats.printing import pprint_thing

def assert_almost_equal(left, right, check_dtype='equiv', check_less_precise=no_default, rtol=1e-05, atol=1e-08, **kwargs):
    "\n    Check that the left and right objects are approximately equal.\n\n    By approximately equal, we refer to objects that are numbers or that\n    contain numbers which may be equivalent to specific levels of precision.\n\n    Parameters\n    ----------\n    left : object\n    right : object\n    check_dtype : bool or {'equiv'}, default 'equiv'\n        Check dtype if both a and b are the same type. If 'equiv' is passed in,\n        then `RangeIndex` and `Int64Index` are also considered equivalent\n        when doing type checking.\n    check_less_precise : bool or int, default False\n        Specify comparison precision. 5 digits (False) or 3 digits (True)\n        after decimal points are compared. If int, then specify the number\n        of digits to compare.\n\n        When comparing two numbers, if the first number has magnitude less\n        than 1e-5, we compare the two numbers directly and check whether\n        they are equivalent within the specified precision. Otherwise, we\n        compare the **ratio** of the second number to the first number and\n        check whether it is equivalent to 1 within the specified precision.\n\n        .. deprecated:: 1.1.0\n           Use `rtol` and `atol` instead to define relative/absolute\n           tolerance, respectively. Similar to :func:`math.isclose`.\n    rtol : float, default 1e-5\n        Relative tolerance.\n\n        .. versionadded:: 1.1.0\n    atol : float, default 1e-8\n        Absolute tolerance.\n\n        .. versionadded:: 1.1.0\n    "
    if (check_less_precise is not no_default):
        warnings.warn("The 'check_less_precise' keyword in testing.assert_*_equal is deprecated and will be removed in a future version. You can stop passing 'check_less_precise' to silence this warning.", FutureWarning, stacklevel=2)
        rtol = atol = _get_tol_from_less_precise(check_less_precise)
    if isinstance(left, Index):
        assert_index_equal(left, right, check_exact=False, exact=check_dtype, rtol=rtol, atol=atol, **kwargs)
    elif isinstance(left, Series):
        assert_series_equal(left, right, check_exact=False, check_dtype=check_dtype, rtol=rtol, atol=atol, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, check_exact=False, check_dtype=check_dtype, rtol=rtol, atol=atol, **kwargs)
    else:
        if check_dtype:
            if (is_number(left) and is_number(right)):
                pass
            elif (is_bool(left) and is_bool(right)):
                pass
            else:
                if (isinstance(left, np.ndarray) or isinstance(right, np.ndarray)):
                    obj = 'numpy array'
                else:
                    obj = 'Input'
                assert_class_equal(left, right, obj=obj)
        _testing.assert_almost_equal(left, right, check_dtype=check_dtype, rtol=rtol, atol=atol, **kwargs)

def _get_tol_from_less_precise(check_less_precise):
    '\n    Return the tolerance equivalent to the deprecated `check_less_precise`\n    parameter.\n\n    Parameters\n    ----------\n    check_less_precise : bool or int\n\n    Returns\n    -------\n    float\n        Tolerance to be used as relative/absolute tolerance.\n\n    Examples\n    --------\n    >>> # Using check_less_precise as a bool:\n    >>> _get_tol_from_less_precise(False)\n    0.5e-5\n    >>> _get_tol_from_less_precise(True)\n    0.5e-3\n    >>> # Using check_less_precise as an int representing the decimal\n    >>> # tolerance intended:\n    >>> _get_tol_from_less_precise(2)\n    0.5e-2\n    >>> _get_tol_from_less_precise(8)\n    0.5e-8\n\n    '
    if isinstance(check_less_precise, bool):
        if check_less_precise:
            return 0.0005
        else:
            return 5e-06
    else:
        return (0.5 * (10 ** (- check_less_precise)))

def _check_isinstance(left, right, cls):
    '\n    Helper method for our assert_* methods that ensures that\n    the two objects being compared have the right type before\n    proceeding with the comparison.\n\n    Parameters\n    ----------\n    left : The first object being compared.\n    right : The second object being compared.\n    cls : The class type to check against.\n\n    Raises\n    ------\n    AssertionError : Either `left` or `right` is not an instance of `cls`.\n    '
    cls_name = cls.__name__
    if (not isinstance(left, cls)):
        raise AssertionError(f'{cls_name} Expected type {cls}, found {type(left)} instead')
    if (not isinstance(right, cls)):
        raise AssertionError(f'{cls_name} Expected type {cls}, found {type(right)} instead')

def assert_dict_equal(left, right, compare_keys=True):
    _check_isinstance(left, right, dict)
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)

def assert_index_equal(left, right, exact='equiv', check_names=True, check_less_precise=no_default, check_exact=True, check_categorical=True, check_order=True, rtol=1e-05, atol=1e-08, obj='Index'):
    "\n    Check that left and right Index are equal.\n\n    Parameters\n    ----------\n    left : Index\n    right : Index\n    exact : bool or {'equiv'}, default 'equiv'\n        Whether to check the Index class, dtype and inferred_type\n        are identical. If 'equiv', then RangeIndex can be substituted for\n        Int64Index as well.\n    check_names : bool, default True\n        Whether to check the names attribute.\n    check_less_precise : bool or int, default False\n        Specify comparison precision. Only used when check_exact is False.\n        5 digits (False) or 3 digits (True) after decimal points are compared.\n        If int, then specify the digits to compare.\n\n        .. deprecated:: 1.1.0\n           Use `rtol` and `atol` instead to define relative/absolute\n           tolerance, respectively. Similar to :func:`math.isclose`.\n    check_exact : bool, default True\n        Whether to compare number exactly.\n    check_categorical : bool, default True\n        Whether to compare internal Categorical exactly.\n    check_order : bool, default True\n        Whether to compare the order of index entries as well as their values.\n        If True, both indexes must contain the same elements, in the same order.\n        If False, both indexes must contain the same elements, but in any order.\n\n        .. versionadded:: 1.2.0\n    rtol : float, default 1e-5\n        Relative tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n    atol : float, default 1e-8\n        Absolute tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n    obj : str, default 'Index'\n        Specify object name being compared, internally used to show appropriate\n        assertion message.\n\n    Examples\n    --------\n    >>> from pandas.testing import assert_index_equal\n    >>> a = pd.Index([1, 2, 3])\n    >>> b = pd.Index([1, 2, 3])\n    >>> assert_index_equal(a, b)\n    "
    __tracebackhide__ = True

    def _check_types(left, right, obj='Index'):
        if exact:
            assert_class_equal(left, right, exact=exact, obj=obj)
            if check_categorical:
                assert_attr_equal('dtype', left, right, obj=obj)
            if (left.inferred_type in 'string'):
                assert (right.inferred_type in 'string')
            else:
                assert_attr_equal('inferred_type', left, right, obj=obj)

    def _get_ilevel_values(index, level):
        unique = index.levels[level]
        level_codes = index.codes[level]
        filled = take_1d(unique._values, level_codes, fill_value=unique._na_value)
        return unique._shallow_copy(filled, name=index.names[level])
    if (check_less_precise is not no_default):
        warnings.warn("The 'check_less_precise' keyword in testing.assert_*_equal is deprecated and will be removed in a future version. You can stop passing 'check_less_precise' to silence this warning.", FutureWarning, stacklevel=2)
        rtol = atol = _get_tol_from_less_precise(check_less_precise)
    _check_isinstance(left, right, Index)
    _check_types(left, right, obj=obj)
    if (left.nlevels != right.nlevels):
        msg1 = f'{obj} levels are different'
        msg2 = f'{left.nlevels}, {left}'
        msg3 = f'{right.nlevels}, {right}'
        raise_assert_detail(obj, msg1, msg2, msg3)
    if (len(left) != len(right)):
        msg1 = f'{obj} length are different'
        msg2 = f'{len(left)}, {left}'
        msg3 = f'{len(right)}, {right}'
        raise_assert_detail(obj, msg1, msg2, msg3)
    if (not check_order):
        left = left.sort_values()
        right = right.sort_values()
    if (left.nlevels > 1):
        left = cast(MultiIndex, left)
        right = cast(MultiIndex, right)
        for level in range(left.nlevels):
            llevel = _get_ilevel_values(left, level)
            rlevel = _get_ilevel_values(right, level)
            lobj = f'MultiIndex level [{level}]'
            assert_index_equal(llevel, rlevel, exact=exact, check_names=check_names, check_exact=check_exact, rtol=rtol, atol=atol, obj=lobj)
            _check_types(left.levels[level], right.levels[level], obj=obj)
    if (check_exact and check_categorical):
        if (not left.equals(right)):
            diff = ((np.sum((left._values != right._values).astype(int)) * 100.0) / len(left))
            msg = f'{obj} values are different ({np.round(diff, 5)} %)'
            raise_assert_detail(obj, msg, left, right)
    else:
        _testing.assert_almost_equal(left.values, right.values, rtol=rtol, atol=atol, check_dtype=exact, obj=obj, lobj=left, robj=right)
    if check_names:
        assert_attr_equal('names', left, right, obj=obj)
    if (isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex)):
        assert_attr_equal('freq', left, right, obj=obj)
    if (isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex)):
        assert_interval_array_equal(left._values, right._values)
    if check_categorical:
        if (is_categorical_dtype(left.dtype) or is_categorical_dtype(right.dtype)):
            assert_categorical_equal(left._values, right._values, obj=f'{obj} category')

def assert_class_equal(left, right, exact=True, obj='Input'):
    '\n    Checks classes are equal.\n    '
    __tracebackhide__ = True

    def repr_class(x):
        if isinstance(x, Index):
            return x
        return type(x).__name__
    if (exact == 'equiv'):
        if (type(left) != type(right)):
            types = {type(left).__name__, type(right).__name__}
            if len((types - {'Int64Index', 'RangeIndex'})):
                msg = f'{obj} classes are not equivalent'
                raise_assert_detail(obj, msg, repr_class(left), repr_class(right))
    elif exact:
        if (type(left) != type(right)):
            msg = f'{obj} classes are different'
            raise_assert_detail(obj, msg, repr_class(left), repr_class(right))

def assert_attr_equal(attr, left, right, obj='Attributes'):
    "\n    Check attributes are equal. Both objects must have attribute.\n\n    Parameters\n    ----------\n    attr : str\n        Attribute name being compared.\n    left : object\n    right : object\n    obj : str, default 'Attributes'\n        Specify object name being compared, internally used to show appropriate\n        assertion message\n    "
    __tracebackhide__ = True
    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)
    if (left_attr is right_attr):
        return True
    elif (is_number(left_attr) and np.isnan(left_attr) and is_number(right_attr) and np.isnan(right_attr)):
        return True
    try:
        result = (left_attr == right_attr)
    except TypeError:
        result = False
    if (not isinstance(result, bool)):
        result = result.all()
    if result:
        return True
    else:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)

def assert_is_valid_plot_return_object(objs):
    import matplotlib.pyplot as plt
    if isinstance(objs, (Series, np.ndarray)):
        for el in objs.ravel():
            msg = f"one of 'objs' is not a matplotlib Axes instance, type encountered {repr(type(el).__name__)}"
            assert isinstance(el, (plt.Axes, dict)), msg
    else:
        msg = f"objs is neither an ndarray of Artist instances nor a single ArtistArtist instance, tuple, or dict, 'objs' is a {repr(type(objs).__name__)}"
        assert isinstance(objs, (plt.Artist, tuple, dict)), msg

def assert_is_sorted(seq):
    'Assert that the sequence is sorted.'
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    assert_numpy_array_equal(seq, np.sort(np.array(seq)))

def assert_categorical_equal(left, right, check_dtype=True, check_category_order=True, obj='Categorical'):
    "\n    Test that Categoricals are equivalent.\n\n    Parameters\n    ----------\n    left : Categorical\n    right : Categorical\n    check_dtype : bool, default True\n        Check that integer dtype of the codes are the same\n    check_category_order : bool, default True\n        Whether the order of the categories should be compared, which\n        implies identical integer codes.  If False, only the resulting\n        values are compared.  The ordered attribute is\n        checked regardless.\n    obj : str, default 'Categorical'\n        Specify object name being compared, internally used to show appropriate\n        assertion message\n    "
    _check_isinstance(left, right, Categorical)
    if check_category_order:
        assert_index_equal(left.categories, right.categories, obj=f'{obj}.categories')
        assert_numpy_array_equal(left.codes, right.codes, check_dtype=check_dtype, obj=f'{obj}.codes')
    else:
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            (lc, rc) = (left.categories, right.categories)
        assert_index_equal(lc, rc, obj=f'{obj}.categories')
        assert_index_equal(left.categories.take(left.codes), right.categories.take(right.codes), obj=f'{obj}.values')
    assert_attr_equal('ordered', left, right, obj=obj)

def assert_interval_array_equal(left, right, exact='equiv', obj='IntervalArray'):
    "\n    Test that two IntervalArrays are equivalent.\n\n    Parameters\n    ----------\n    left, right : IntervalArray\n        The IntervalArrays to compare.\n    exact : bool or {'equiv'}, default 'equiv'\n        Whether to check the Index class, dtype and inferred_type\n        are identical. If 'equiv', then RangeIndex can be substituted for\n        Int64Index as well.\n    obj : str, default 'IntervalArray'\n        Specify object name being compared, internally used to show appropriate\n        assertion message\n    "
    _check_isinstance(left, right, IntervalArray)
    kwargs = {}
    if (left._left.dtype.kind in ['m', 'M']):
        kwargs['check_freq'] = False
    assert_equal(left._left, right._left, obj=f'{obj}.left', **kwargs)
    assert_equal(left._right, right._right, obj=f'{obj}.left', **kwargs)
    assert_attr_equal('closed', left, right, obj=obj)

def assert_period_array_equal(left, right, obj='PeriodArray'):
    _check_isinstance(left, right, PeriodArray)
    assert_numpy_array_equal(left._data, right._data, obj=f'{obj}._data')
    assert_attr_equal('freq', left, right, obj=obj)

def assert_datetime_array_equal(left, right, obj='DatetimeArray', check_freq=True):
    __tracebackhide__ = True
    _check_isinstance(left, right, DatetimeArray)
    assert_numpy_array_equal(left._data, right._data, obj=f'{obj}._data')
    if check_freq:
        assert_attr_equal('freq', left, right, obj=obj)
    assert_attr_equal('tz', left, right, obj=obj)

def assert_timedelta_array_equal(left, right, obj='TimedeltaArray', check_freq=True):
    __tracebackhide__ = True
    _check_isinstance(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._data, right._data, obj=f'{obj}._data')
    if check_freq:
        assert_attr_equal('freq', left, right, obj=obj)

def raise_assert_detail(obj, message, left, right, diff=None, index_values=None):
    __tracebackhide__ = True
    msg = f'''{obj} are different

{message}'''
    if isinstance(index_values, np.ndarray):
        msg += f'''
[index]: {pprint_thing(index_values)}'''
    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif is_categorical_dtype(left):
        left = repr(left)
    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif is_categorical_dtype(right):
        right = repr(right)
    msg += f'''
[left]:  {left}
[right]: {right}'''
    if (diff is not None):
        msg += f'''
[diff]: {diff}'''
    raise AssertionError(msg)

def assert_numpy_array_equal(left, right, strict_nan=False, check_dtype=True, err_msg=None, check_same=None, obj='numpy array', index_values=None):
    "\n    Check that 'np.ndarray' is equivalent.\n\n    Parameters\n    ----------\n    left, right : numpy.ndarray or iterable\n        The two arrays to be compared.\n    strict_nan : bool, default False\n        If True, consider NaN and None to be different.\n    check_dtype : bool, default True\n        Check dtype if both a and b are np.ndarray.\n    err_msg : str, default None\n        If provided, used as assertion message.\n    check_same : None|'copy'|'same', default None\n        Ensure left and right refer/do not refer to the same memory area.\n    obj : str, default 'numpy array'\n        Specify object name being compared, internally used to show appropriate\n        assertion message.\n    index_values : numpy.ndarray, default None\n        optional index (shared by both left and right), used in output.\n    "
    __tracebackhide__ = True
    assert_class_equal(left, right, obj=obj)
    _check_isinstance(left, right, np.ndarray)

    def _get_base(obj):
        return (obj.base if (getattr(obj, 'base', None) is not None) else obj)
    left_base = _get_base(left)
    right_base = _get_base(right)
    if (check_same == 'same'):
        if (left_base is not right_base):
            raise AssertionError(f'{repr(left_base)} is not {repr(right_base)}')
    elif (check_same == 'copy'):
        if (left_base is right_base):
            raise AssertionError(f'{repr(left_base)} is {repr(right_base)}')

    def _raise(left, right, err_msg):
        if (err_msg is None):
            if (left.shape != right.shape):
                raise_assert_detail(obj, f'{obj} shapes are different', left.shape, right.shape)
            diff = 0
            for (left_arr, right_arr) in zip(left, right):
                if (not array_equivalent(left_arr, right_arr, strict_nan=strict_nan)):
                    diff += 1
            diff = ((diff * 100.0) / left.size)
            msg = f'{obj} values are different ({np.round(diff, 5)} %)'
            raise_assert_detail(obj, msg, left, right, index_values=index_values)
        raise AssertionError(err_msg)
    if (not array_equivalent(left, right, strict_nan=strict_nan)):
        _raise(left, right, err_msg)
    if check_dtype:
        if (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            assert_attr_equal('dtype', left, right, obj=obj)

def assert_extension_array_equal(left, right, check_dtype=True, index_values=None, check_less_precise=no_default, check_exact=False, rtol=1e-05, atol=1e-08):
    '\n    Check that left and right ExtensionArrays are equal.\n\n    Parameters\n    ----------\n    left, right : ExtensionArray\n        The two arrays to compare.\n    check_dtype : bool, default True\n        Whether to check if the ExtensionArray dtypes are identical.\n    index_values : numpy.ndarray, default None\n        Optional index (shared by both left and right), used in output.\n    check_less_precise : bool or int, default False\n        Specify comparison precision. Only used when check_exact is False.\n        5 digits (False) or 3 digits (True) after decimal points are compared.\n        If int, then specify the digits to compare.\n\n        .. deprecated:: 1.1.0\n           Use `rtol` and `atol` instead to define relative/absolute\n           tolerance, respectively. Similar to :func:`math.isclose`.\n    check_exact : bool, default False\n        Whether to compare number exactly.\n    rtol : float, default 1e-5\n        Relative tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n    atol : float, default 1e-8\n        Absolute tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n\n    Notes\n    -----\n    Missing values are checked separately from valid values.\n    A mask of missing values is computed for each and checked to match.\n    The remaining all-valid values are cast to object dtype and checked.\n\n    Examples\n    --------\n    >>> from pandas.testing import assert_extension_array_equal\n    >>> a = pd.Series([1, 2, 3, 4])\n    >>> b, c = a.array, a.array\n    >>> assert_extension_array_equal(b, c)\n    '
    if (check_less_precise is not no_default):
        warnings.warn("The 'check_less_precise' keyword in testing.assert_*_equal is deprecated and will be removed in a future version. You can stop passing 'check_less_precise' to silence this warning.", FutureWarning, stacklevel=2)
        rtol = atol = _get_tol_from_less_precise(check_less_precise)
    assert isinstance(left, ExtensionArray), 'left is not an ExtensionArray'
    assert isinstance(right, ExtensionArray), 'right is not an ExtensionArray'
    if check_dtype:
        assert_attr_equal('dtype', left, right, obj='ExtensionArray')
    if (isinstance(left, DatetimeLikeArrayMixin) and isinstance(right, DatetimeLikeArrayMixin) and (type(right) == type(left))):
        assert_numpy_array_equal(np.asarray(left.asi8), np.asarray(right.asi8), index_values=index_values)
        return
    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    assert_numpy_array_equal(left_na, right_na, obj='ExtensionArray NA mask', index_values=index_values)
    left_valid = np.asarray(left[(~ left_na)].astype(object))
    right_valid = np.asarray(right[(~ right_na)].astype(object))
    if check_exact:
        assert_numpy_array_equal(left_valid, right_valid, obj='ExtensionArray', index_values=index_values)
    else:
        _testing.assert_almost_equal(left_valid, right_valid, check_dtype=check_dtype, rtol=rtol, atol=atol, obj='ExtensionArray', index_values=index_values)

def assert_series_equal(left, right, check_dtype=True, check_index_type='equiv', check_series_type=True, check_less_precise=no_default, check_names=True, check_exact=False, check_datetimelike_compat=False, check_categorical=True, check_category_order=True, check_freq=True, check_flags=True, rtol=1e-05, atol=1e-08, obj='Series', *, check_index=True):
    "\n    Check that left and right Series are equal.\n\n    Parameters\n    ----------\n    left : Series\n    right : Series\n    check_dtype : bool, default True\n        Whether to check the Series dtype is identical.\n    check_index_type : bool or {'equiv'}, default 'equiv'\n        Whether to check the Index class, dtype and inferred_type\n        are identical.\n    check_series_type : bool, default True\n         Whether to check the Series class is identical.\n    check_less_precise : bool or int, default False\n        Specify comparison precision. Only used when check_exact is False.\n        5 digits (False) or 3 digits (True) after decimal points are compared.\n        If int, then specify the digits to compare.\n\n        When comparing two numbers, if the first number has magnitude less\n        than 1e-5, we compare the two numbers directly and check whether\n        they are equivalent within the specified precision. Otherwise, we\n        compare the **ratio** of the second number to the first number and\n        check whether it is equivalent to 1 within the specified precision.\n\n        .. deprecated:: 1.1.0\n           Use `rtol` and `atol` instead to define relative/absolute\n           tolerance, respectively. Similar to :func:`math.isclose`.\n    check_names : bool, default True\n        Whether to check the Series and Index names attribute.\n    check_exact : bool, default False\n        Whether to compare number exactly.\n    check_datetimelike_compat : bool, default False\n        Compare datetime-like which is comparable ignoring dtype.\n    check_categorical : bool, default True\n        Whether to compare internal Categorical exactly.\n    check_category_order : bool, default True\n        Whether to compare category order of internal Categoricals.\n\n        .. versionadded:: 1.0.2\n    check_freq : bool, default True\n        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.\n    check_flags : bool, default True\n        Whether to check the `flags` attribute.\n\n        .. versionadded:: 1.2.0\n\n    rtol : float, default 1e-5\n        Relative tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n    atol : float, default 1e-8\n        Absolute tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n    obj : str, default 'Series'\n        Specify object name being compared, internally used to show appropriate\n        assertion message.\n    check_index : bool, default True\n        Whether to check index equivalence. If False, then compare only values.\n\n        .. versionadded:: 1.3.0\n\n    Examples\n    --------\n    >>> from pandas.testing import assert_series_equal\n    >>> a = pd.Series([1, 2, 3, 4])\n    >>> b = pd.Series([1, 2, 3, 4])\n    >>> assert_series_equal(a, b)\n    "
    __tracebackhide__ = True
    if (check_less_precise is not no_default):
        warnings.warn("The 'check_less_precise' keyword in testing.assert_*_equal is deprecated and will be removed in a future version. You can stop passing 'check_less_precise' to silence this warning.", FutureWarning, stacklevel=2)
        rtol = atol = _get_tol_from_less_precise(check_less_precise)
    _check_isinstance(left, right, Series)
    if check_series_type:
        assert_class_equal(left, right, obj=obj)
    if (len(left) != len(right)):
        msg1 = f'{len(left)}, {left.index}'
        msg2 = f'{len(right)}, {right.index}'
        raise_assert_detail(obj, 'Series length are different', msg1, msg2)
    if check_flags:
        assert (left.flags == right.flags), f'{repr(left.flags)} != {repr(right.flags)}'
    if check_index:
        assert_index_equal(left.index, right.index, exact=check_index_type, check_names=check_names, check_exact=check_exact, check_categorical=check_categorical, rtol=rtol, atol=atol, obj=f'{obj}.index')
    if (check_freq and isinstance(left.index, (DatetimeIndex, TimedeltaIndex))):
        lidx = left.index
        ridx = right.index
        assert (lidx.freq == ridx.freq), (lidx.freq, ridx.freq)
    if check_dtype:
        if (is_categorical_dtype(left.dtype) and is_categorical_dtype(right.dtype) and (not check_categorical)):
            pass
        else:
            assert_attr_equal('dtype', left, right, obj=f'Attributes of {obj}')
    if (check_exact and is_numeric_dtype(left.dtype) and is_numeric_dtype(right.dtype)):
        assert_numpy_array_equal(left._values, right._values, check_dtype=check_dtype, obj=str(obj), index_values=np.asarray(left.index))
    elif (check_datetimelike_compat and (needs_i8_conversion(left.dtype) or needs_i8_conversion(right.dtype))):
        if (not Index(left._values).equals(Index(right._values))):
            msg = f'[datetimelike_compat=True] {left._values} is not equal to {right._values}.'
            raise AssertionError(msg)
    elif (is_interval_dtype(left.dtype) and is_interval_dtype(right.dtype)):
        assert_interval_array_equal(left.array, right.array)
    elif (is_categorical_dtype(left.dtype) or is_categorical_dtype(right.dtype)):
        _testing.assert_almost_equal(left._values, right._values, rtol=rtol, atol=atol, check_dtype=check_dtype, obj=str(obj), index_values=np.asarray(left.index))
    elif (is_extension_array_dtype(left.dtype) and is_extension_array_dtype(right.dtype)):
        assert_extension_array_equal(left._values, right._values, check_dtype=check_dtype, index_values=np.asarray(left.index))
    elif (is_extension_array_dtype_and_needs_i8_conversion(left.dtype, right.dtype) or is_extension_array_dtype_and_needs_i8_conversion(right.dtype, left.dtype)):
        assert_extension_array_equal(left._values, right._values, check_dtype=check_dtype, index_values=np.asarray(left.index))
    elif (needs_i8_conversion(left.dtype) and needs_i8_conversion(right.dtype)):
        assert_extension_array_equal(left._values, right._values, check_dtype=check_dtype, index_values=np.asarray(left.index))
    else:
        _testing.assert_almost_equal(left._values, right._values, rtol=rtol, atol=atol, check_dtype=check_dtype, obj=str(obj), index_values=np.asarray(left.index))
    if check_names:
        assert_attr_equal('name', left, right, obj=obj)
    if check_categorical:
        if (is_categorical_dtype(left.dtype) or is_categorical_dtype(right.dtype)):
            assert_categorical_equal(left._values, right._values, obj=f'{obj} category', check_category_order=check_category_order)

def assert_frame_equal(left, right, check_dtype=True, check_index_type='equiv', check_column_type='equiv', check_frame_type=True, check_less_precise=no_default, check_names=True, by_blocks=False, check_exact=False, check_datetimelike_compat=False, check_categorical=True, check_like=False, check_freq=True, check_flags=True, rtol=1e-05, atol=1e-08, obj='DataFrame'):
    '\n    Check that left and right DataFrame are equal.\n\n    This function is intended to compare two DataFrames and output any\n    differences. Is is mostly intended for use in unit tests.\n    Additional parameters allow varying the strictness of the\n    equality checks performed.\n\n    Parameters\n    ----------\n    left : DataFrame\n        First DataFrame to compare.\n    right : DataFrame\n        Second DataFrame to compare.\n    check_dtype : bool, default True\n        Whether to check the DataFrame dtype is identical.\n    check_index_type : bool or {\'equiv\'}, default \'equiv\'\n        Whether to check the Index class, dtype and inferred_type\n        are identical.\n    check_column_type : bool or {\'equiv\'}, default \'equiv\'\n        Whether to check the columns class, dtype and inferred_type\n        are identical. Is passed as the ``exact`` argument of\n        :func:`assert_index_equal`.\n    check_frame_type : bool, default True\n        Whether to check the DataFrame class is identical.\n    check_less_precise : bool or int, default False\n        Specify comparison precision. Only used when check_exact is False.\n        5 digits (False) or 3 digits (True) after decimal points are compared.\n        If int, then specify the digits to compare.\n\n        When comparing two numbers, if the first number has magnitude less\n        than 1e-5, we compare the two numbers directly and check whether\n        they are equivalent within the specified precision. Otherwise, we\n        compare the **ratio** of the second number to the first number and\n        check whether it is equivalent to 1 within the specified precision.\n\n        .. deprecated:: 1.1.0\n           Use `rtol` and `atol` instead to define relative/absolute\n           tolerance, respectively. Similar to :func:`math.isclose`.\n    check_names : bool, default True\n        Whether to check that the `names` attribute for both the `index`\n        and `column` attributes of the DataFrame is identical.\n    by_blocks : bool, default False\n        Specify how to compare internal data. If False, compare by columns.\n        If True, compare by blocks.\n    check_exact : bool, default False\n        Whether to compare number exactly.\n    check_datetimelike_compat : bool, default False\n        Compare datetime-like which is comparable ignoring dtype.\n    check_categorical : bool, default True\n        Whether to compare internal Categorical exactly.\n    check_like : bool, default False\n        If True, ignore the order of index & columns.\n        Note: index labels must match their respective rows\n        (same as in columns) - same labels must be with the same data.\n    check_freq : bool, default True\n        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.\n    check_flags : bool, default True\n        Whether to check the `flags` attribute.\n    rtol : float, default 1e-5\n        Relative tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n    atol : float, default 1e-8\n        Absolute tolerance. Only used when check_exact is False.\n\n        .. versionadded:: 1.1.0\n    obj : str, default \'DataFrame\'\n        Specify object name being compared, internally used to show appropriate\n        assertion message.\n\n    See Also\n    --------\n    assert_series_equal : Equivalent method for asserting Series equality.\n    DataFrame.equals : Check DataFrame equality.\n\n    Examples\n    --------\n    This example shows comparing two DataFrames that are equal\n    but with columns of differing dtypes.\n\n    >>> from pandas._testing import assert_frame_equal\n    >>> df1 = pd.DataFrame({\'a\': [1, 2], \'b\': [3, 4]})\n    >>> df2 = pd.DataFrame({\'a\': [1, 2], \'b\': [3.0, 4.0]})\n\n    df1 equals itself.\n\n    >>> assert_frame_equal(df1, df1)\n\n    df1 differs from df2 as column \'b\' is of a different type.\n\n    >>> assert_frame_equal(df1, df2)\n    Traceback (most recent call last):\n    ...\n    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different\n\n    Attribute "dtype" are different\n    [left]:  int64\n    [right]: float64\n\n    Ignore differing dtypes in columns with check_dtype.\n\n    >>> assert_frame_equal(df1, df2, check_dtype=False)\n    '
    __tracebackhide__ = True
    if (check_less_precise is not no_default):
        warnings.warn("The 'check_less_precise' keyword in testing.assert_*_equal is deprecated and will be removed in a future version. You can stop passing 'check_less_precise' to silence this warning.", FutureWarning, stacklevel=2)
        rtol = atol = _get_tol_from_less_precise(check_less_precise)
    _check_isinstance(left, right, DataFrame)
    if check_frame_type:
        assert isinstance(left, type(right))
    if (left.shape != right.shape):
        raise_assert_detail(obj, f'{obj} shape mismatch', f'{repr(left.shape)}', f'{repr(right.shape)}')
    if check_flags:
        assert (left.flags == right.flags), f'{repr(left.flags)} != {repr(right.flags)}'
    assert_index_equal(left.index, right.index, exact=check_index_type, check_names=check_names, check_exact=check_exact, check_categorical=check_categorical, check_order=(not check_like), rtol=rtol, atol=atol, obj=f'{obj}.index')
    assert_index_equal(left.columns, right.columns, exact=check_column_type, check_names=check_names, check_exact=check_exact, check_categorical=check_categorical, check_order=(not check_like), rtol=rtol, atol=atol, obj=f'{obj}.columns')
    if check_like:
        (left, right) = (left.reindex_like(right), right)
    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set((list(lblocks.keys()) + list(rblocks.keys())))):
            assert (dtype in lblocks)
            assert (dtype in rblocks)
            assert_frame_equal(lblocks[dtype], rblocks[dtype], check_dtype=check_dtype, obj=obj)
    else:
        for (i, col) in enumerate(left.columns):
            assert (col in right)
            lcol = left.iloc[:, i]
            rcol = right.iloc[:, i]
            assert_series_equal(lcol, rcol, check_dtype=check_dtype, check_index_type=check_index_type, check_exact=check_exact, check_names=check_names, check_datetimelike_compat=check_datetimelike_compat, check_categorical=check_categorical, check_freq=check_freq, obj=f'{obj}.iloc[:, {i}] (column name="{col}")', rtol=rtol, atol=atol, check_index=False)

def assert_equal(left, right, **kwargs):
    '\n    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.\n\n    Parameters\n    ----------\n    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray\n        The two items to be compared.\n    **kwargs\n        All keyword arguments are passed through to the underlying assert method.\n    '
    __tracebackhide__ = True
    if isinstance(left, Index):
        assert_index_equal(left, right, **kwargs)
        if isinstance(left, (DatetimeIndex, TimedeltaIndex)):
            assert (left.freq == right.freq), (left.freq, right.freq)
    elif isinstance(left, Series):
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        assert_interval_array_equal(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        assert_period_array_equal(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        assert_datetime_array_equal(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        assert_timedelta_array_equal(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        assert_extension_array_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        assert_numpy_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        assert (kwargs == {})
        assert (left == right)
    else:
        raise NotImplementedError(type(left))

def assert_sp_array_equal(left, right):
    '\n    Check that the left and right SparseArray are equal.\n\n    Parameters\n    ----------\n    left : SparseArray\n    right : SparseArray\n    '
    _check_isinstance(left, right, pd.arrays.SparseArray)
    assert_numpy_array_equal(left.sp_values, right.sp_values)
    assert isinstance(left.sp_index, pd._libs.sparse.SparseIndex)
    assert isinstance(right.sp_index, pd._libs.sparse.SparseIndex)
    left_index = left.sp_index
    right_index = right.sp_index
    if (not left_index.equals(right_index)):
        raise_assert_detail('SparseArray.index', 'index are not equal', left_index, right_index)
    else:
        pass
    assert_attr_equal('fill_value', left, right)
    assert_attr_equal('dtype', left, right)
    assert_numpy_array_equal(left.to_dense(), right.to_dense())

def assert_contains_all(iterable, dic):
    for k in iterable:
        assert (k in dic), f'Did not contain item: {repr(k)}'

def assert_copy(iter1, iter2, **eql_kwargs):
    '\n    iter1, iter2: iterables that produce elements\n    comparable with assert_almost_equal\n\n    Checks that the elements are equal, but not\n    the same object. (Does not check that items\n    in sequences are also not the same object)\n    '
    for (elem1, elem2) in zip(iter1, iter2):
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        msg = f'Expected object {repr(type(elem1))} and object {repr(type(elem2))} to be different objects, but they were the same object.'
        assert (elem1 is not elem2), msg

def is_extension_array_dtype_and_needs_i8_conversion(left_dtype, right_dtype):
    '\n    Checks that we have the combination of an ExtensionArraydtype and\n    a dtype that should be converted to int64\n\n    Returns\n    -------\n    bool\n\n    Related to issue #37609\n    '
    return (is_extension_array_dtype(left_dtype) and needs_i8_conversion(right_dtype))
