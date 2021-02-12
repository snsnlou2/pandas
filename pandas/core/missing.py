
'\nRoutines for filling missing data.\n'
from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional, Set, Union
import numpy as np
from pandas._libs import algos, lib
from pandas._typing import ArrayLike, Axis, DtypeObj
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import ensure_float64, is_integer_dtype, is_numeric_v_string_like, needs_i8_conversion
from pandas.core.dtypes.missing import isna
if TYPE_CHECKING:
    from pandas import Index

def mask_missing(arr, values_to_mask):
    '\n    Return a masking array of same size/shape as arr\n    with entries equaling any member of values_to_mask set to True\n\n    Parameters\n    ----------\n    arr : ArrayLike\n    values_to_mask: list, tuple, or scalar\n\n    Returns\n    -------\n    np.ndarray[bool]\n    '
    (dtype, values_to_mask) = infer_dtype_from(values_to_mask)
    values_to_mask = np.array(values_to_mask, dtype=dtype)
    na_mask = isna(values_to_mask)
    nonna = values_to_mask[(~ na_mask)]
    mask = np.zeros(arr.shape, dtype=bool)
    for x in nonna:
        if is_numeric_v_string_like(arr, x):
            pass
        else:
            mask |= (arr == x)
    if na_mask.any():
        mask |= isna(arr)
    return mask

def clean_fill_method(method, allow_nearest=False):
    if (method in [None, 'asfreq']):
        return None
    if isinstance(method, str):
        method = method.lower()
        if (method == 'ffill'):
            method = 'pad'
        elif (method == 'bfill'):
            method = 'backfill'
    valid_methods = ['pad', 'backfill']
    expecting = 'pad (ffill) or backfill (bfill)'
    if allow_nearest:
        valid_methods.append('nearest')
        expecting = 'pad (ffill), backfill (bfill) or nearest'
    if (method not in valid_methods):
        raise ValueError(f'Invalid fill method. Expecting {expecting}. Got {method}')
    return method
NP_METHODS = ['linear', 'time', 'index', 'values']
SP_METHODS = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline']

def clean_interp_method(method, **kwargs):
    order = kwargs.get('order')
    if ((method in ('spline', 'polynomial')) and (order is None)):
        raise ValueError('You must specify the order of the spline or polynomial.')
    valid = (NP_METHODS + SP_METHODS)
    if (method not in valid):
        raise ValueError(f"method must be one of {valid}. Got '{method}' instead.")
    return method

def find_valid_index(values, how):
    "\n    Retrieves the index of the first valid value.\n\n    Parameters\n    ----------\n    values : ndarray or ExtensionArray\n    how : {'first', 'last'}\n        Use this parameter to change between the first or last valid index.\n\n    Returns\n    -------\n    int or None\n    "
    assert (how in ['first', 'last'])
    if (len(values) == 0):
        return None
    is_valid = (~ isna(values))
    if (values.ndim == 2):
        is_valid = is_valid.any(1)
    if (how == 'first'):
        idxpos = is_valid[:].argmax()
    if (how == 'last'):
        idxpos = ((len(values) - 1) - is_valid[::(- 1)].argmax())
    chk_notna = is_valid[idxpos]
    if (not chk_notna):
        return None
    return idxpos

def interpolate_1d(xvalues, yvalues, method='linear', limit=None, limit_direction='forward', limit_area=None, fill_value=None, bounds_error=False, order=None, **kwargs):
    "\n    Logic for the 1-d interpolation.  The result should be 1-d, inputs\n    xvalues and yvalues will each be 1-d arrays of the same length.\n\n    Bounds_error is currently hardcoded to False since non-scipy ones don't\n    take it as an argument.\n    "
    invalid = isna(yvalues)
    valid = (~ invalid)
    if (not valid.any()):
        result = np.empty(xvalues.shape, dtype=np.float64)
        result.fill(np.nan)
        return result
    if valid.all():
        return yvalues
    if (method == 'time'):
        if (not needs_i8_conversion(xvalues.dtype)):
            raise ValueError('time-weighted interpolation only works on Series or DataFrames with a DatetimeIndex')
        method = 'values'
    valid_limit_directions = ['forward', 'backward', 'both']
    limit_direction = limit_direction.lower()
    if (limit_direction not in valid_limit_directions):
        raise ValueError(f"Invalid limit_direction: expecting one of {valid_limit_directions}, got '{limit_direction}'.")
    if (limit_area is not None):
        valid_limit_areas = ['inside', 'outside']
        limit_area = limit_area.lower()
        if (limit_area not in valid_limit_areas):
            raise ValueError(f'Invalid limit_area: expecting one of {valid_limit_areas}, got {limit_area}.')
    limit = algos.validate_limit(nobs=None, limit=limit)
    all_nans = set(np.flatnonzero(invalid))
    start_nans = set(range(find_valid_index(yvalues, 'first')))
    end_nans = set(range((1 + find_valid_index(yvalues, 'last')), len(valid)))
    mid_nans = ((all_nans - start_nans) - end_nans)
    preserve_nans: Union[(List, Set)]
    if (limit_direction == 'forward'):
        preserve_nans = (start_nans | set(_interp_limit(invalid, limit, 0)))
    elif (limit_direction == 'backward'):
        preserve_nans = (end_nans | set(_interp_limit(invalid, 0, limit)))
    else:
        preserve_nans = set(_interp_limit(invalid, limit, limit))
    if (limit_area == 'inside'):
        preserve_nans |= (start_nans | end_nans)
    elif (limit_area == 'outside'):
        preserve_nans |= mid_nans
    preserve_nans = sorted(preserve_nans)
    result = yvalues.copy()
    xarr = xvalues._values
    if needs_i8_conversion(xarr.dtype):
        xarr = xarr.view('i8')
    if (method == 'linear'):
        inds = xarr
    else:
        inds = np.asarray(xarr)
        if (method in ('values', 'index')):
            if (inds.dtype == np.object_):
                inds = lib.maybe_convert_objects(inds)
    if (method in NP_METHODS):
        indexer = np.argsort(inds[valid])
        result[invalid] = np.interp(inds[invalid], inds[valid][indexer], yvalues[valid][indexer])
    else:
        result[invalid] = _interpolate_scipy_wrapper(inds[valid], yvalues[valid], inds[invalid], method=method, fill_value=fill_value, bounds_error=bounds_error, order=order, **kwargs)
    result[preserve_nans] = np.nan
    return result

def _interpolate_scipy_wrapper(x, y, new_x, method, fill_value=None, bounds_error=False, order=None, **kwargs):
    "\n    Passed off to scipy.interpolate.interp1d. method is scipy's kind.\n    Returns an array interpolated at new_x.  Add any new methods to\n    the list in _clean_interp_method.\n    "
    extra = f'{method} interpolation requires SciPy.'
    import_optional_dependency('scipy', extra=extra)
    from scipy import interpolate
    new_x = np.asarray(new_x)
    alt_methods = {'barycentric': interpolate.barycentric_interpolate, 'krogh': interpolate.krogh_interpolate, 'from_derivatives': _from_derivatives, 'piecewise_polynomial': _from_derivatives}
    if getattr(x, '_is_all_dates', False):
        (x, new_x) = (x._values.astype('i8'), new_x.astype('i8'))
    if (method == 'pchip'):
        alt_methods['pchip'] = interpolate.pchip_interpolate
    elif (method == 'akima'):
        alt_methods['akima'] = _akima_interpolate
    elif (method == 'cubicspline'):
        alt_methods['cubicspline'] = _cubicspline_interpolate
    interp1d_methods = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial']
    if (method in interp1d_methods):
        if (method == 'polynomial'):
            method = order
        terp = interpolate.interp1d(x, y, kind=method, fill_value=fill_value, bounds_error=bounds_error)
        new_y = terp(new_x)
    elif (method == 'spline'):
        if (isna(order) or (order <= 0)):
            raise ValueError(f'order needs to be specified and greater than 0; got order: {order}')
        terp = interpolate.UnivariateSpline(x, y, k=order, **kwargs)
        new_y = terp(new_x)
    else:
        if (not x.flags.writeable):
            x = x.copy()
        if (not y.flags.writeable):
            y = y.copy()
        if (not new_x.flags.writeable):
            new_x = new_x.copy()
        method = alt_methods[method]
        new_y = method(x, y, new_x, **kwargs)
    return new_y

def _from_derivatives(xi, yi, x, order=None, der=0, extrapolate=False):
    '\n    Convenience function for interpolate.BPoly.from_derivatives.\n\n    Construct a piecewise polynomial in the Bernstein basis, compatible\n    with the specified values and derivatives at breakpoints.\n\n    Parameters\n    ----------\n    xi : array_like\n        sorted 1D array of x-coordinates\n    yi : array_like or list of array-likes\n        yi[i][j] is the j-th derivative known at xi[i]\n    order: None or int or array_like of ints. Default: None.\n        Specifies the degree of local polynomials. If not None, some\n        derivatives are ignored.\n    der : int or list\n        How many derivatives to extract; None for all potentially nonzero\n        derivatives (that is a number equal to the number of points), or a\n        list of derivatives to extract. This number includes the function\n        value as 0th derivative.\n     extrapolate : bool, optional\n        Whether to extrapolate to ouf-of-bounds points based on first and last\n        intervals, or to return NaNs. Default: True.\n\n    See Also\n    --------\n    scipy.interpolate.BPoly.from_derivatives\n\n    Returns\n    -------\n    y : scalar or array_like\n        The result, of length R or length M or M by R.\n    '
    from scipy import interpolate
    method = interpolate.BPoly.from_derivatives
    m = method(xi, yi.reshape((- 1), 1), orders=order, extrapolate=extrapolate)
    return m(x)

def _akima_interpolate(xi, yi, x, der=0, axis=0):
    "\n    Convenience function for akima interpolation.\n    xi and yi are arrays of values used to approximate some function f,\n    with ``yi = f(xi)``.\n\n    See `Akima1DInterpolator` for details.\n\n    Parameters\n    ----------\n    xi : array_like\n        A sorted list of x-coordinates, of length N.\n    yi : array_like\n        A 1-D array of real values.  `yi`'s length along the interpolation\n        axis must be equal to the length of `xi`. If N-D array, use axis\n        parameter to select correct axis.\n    x : scalar or array_like\n        Of length M.\n    der : int, optional\n        How many derivatives to extract; None for all potentially\n        nonzero derivatives (that is a number equal to the number\n        of points), or a list of derivatives to extract. This number\n        includes the function value as 0th derivative.\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate values.\n\n    See Also\n    --------\n    scipy.interpolate.Akima1DInterpolator\n\n    Returns\n    -------\n    y : scalar or array_like\n        The result, of length R or length M or M by R,\n\n    "
    from scipy import interpolate
    P = interpolate.Akima1DInterpolator(xi, yi, axis=axis)
    return P(x, nu=der)

def _cubicspline_interpolate(xi, yi, x, axis=0, bc_type='not-a-knot', extrapolate=None):
    '\n    Convenience function for cubic spline data interpolator.\n\n    See `scipy.interpolate.CubicSpline` for details.\n\n    Parameters\n    ----------\n    xi : array_like, shape (n,)\n        1-d array containing values of the independent variable.\n        Values must be real, finite and in strictly increasing order.\n    yi : array_like\n        Array containing values of the dependent variable. It can have\n        arbitrary number of dimensions, but the length along ``axis``\n        (see below) must match the length of ``x``. Values must be finite.\n    x : scalar or array_like, shape (m,)\n    axis : int, optional\n        Axis along which `y` is assumed to be varying. Meaning that for\n        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.\n        Default is 0.\n    bc_type : string or 2-tuple, optional\n        Boundary condition type. Two additional equations, given by the\n        boundary conditions, are required to determine all coefficients of\n        polynomials on each segment [2]_.\n        If `bc_type` is a string, then the specified condition will be applied\n        at both ends of a spline. Available conditions are:\n        * \'not-a-knot\' (default): The first and second segment at a curve end\n          are the same polynomial. It is a good default when there is no\n          information on boundary conditions.\n        * \'periodic\': The interpolated functions is assumed to be periodic\n          of period ``x[-1] - x[0]``. The first and last value of `y` must be\n          identical: ``y[0] == y[-1]``. This boundary condition will result in\n          ``y\'[0] == y\'[-1]`` and ``y\'\'[0] == y\'\'[-1]``.\n        * \'clamped\': The first derivative at curves ends are zero. Assuming\n          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.\n        * \'natural\': The second derivative at curve ends are zero. Assuming\n          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.\n        If `bc_type` is a 2-tuple, the first and the second value will be\n        applied at the curve start and end respectively. The tuple values can\n        be one of the previously mentioned strings (except \'periodic\') or a\n        tuple `(order, deriv_values)` allowing to specify arbitrary\n        derivatives at curve ends:\n        * `order`: the derivative order, 1 or 2.\n        * `deriv_value`: array_like containing derivative values, shape must\n          be the same as `y`, excluding ``axis`` dimension. For example, if\n          `y` is 1D, then `deriv_value` must be a scalar. If `y` is 3D with\n          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D\n          and have the shape (n0, n1).\n    extrapolate : {bool, \'periodic\', None}, optional\n        If bool, determines whether to extrapolate to out-of-bounds points\n        based on first and last intervals, or to return NaNs. If \'periodic\',\n        periodic extrapolation is used. If None (default), ``extrapolate`` is\n        set to \'periodic\' for ``bc_type=\'periodic\'`` and to True otherwise.\n\n    See Also\n    --------\n    scipy.interpolate.CubicHermiteSpline\n\n    Returns\n    -------\n    y : scalar or array_like\n        The result, of shape (m,)\n\n    References\n    ----------\n    .. [1] `Cubic Spline Interpolation\n            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_\n            on Wikiversity.\n    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.\n    '
    from scipy import interpolate
    P = interpolate.CubicSpline(xi, yi, axis=axis, bc_type=bc_type, extrapolate=extrapolate)
    return P(x)

def _interpolate_with_limit_area(values, method, limit, limit_area):
    '\n    Apply interpolation and limit_area logic to values along a to-be-specified axis.\n\n    Parameters\n    ----------\n    values: array-like\n        Input array.\n    method: str\n        Interpolation method. Could be "bfill" or "pad"\n    limit: int, optional\n        Index limit on interpolation.\n    limit_area: str\n        Limit area for interpolation. Can be "inside" or "outside"\n\n    Returns\n    -------\n    values: array-like\n        Interpolated array.\n    '
    invalid = isna(values)
    if (not invalid.all()):
        first = find_valid_index(values, 'first')
        last = find_valid_index(values, 'last')
        values = interpolate_2d(values, method=method, limit=limit)
        if (limit_area == 'inside'):
            invalid[first:(last + 1)] = False
        elif (limit_area == 'outside'):
            invalid[:first] = invalid[(last + 1):] = False
        values[invalid] = np.nan
    return values

def interpolate_2d(values, method='pad', axis=0, limit=None, limit_area=None):
    '\n    Perform an actual interpolation of values, values will be make 2-d if\n    needed fills inplace, returns the result.\n\n       Parameters\n    ----------\n    values: array-like\n        Input array.\n    method: str, default "pad"\n        Interpolation method. Could be "bfill" or "pad"\n    axis: 0 or 1\n        Interpolation axis\n    limit: int, optional\n        Index limit on interpolation.\n    limit_area: str, optional\n        Limit area for interpolation. Can be "inside" or "outside"\n\n    Returns\n    -------\n    values: array-like\n        Interpolated array.\n    '
    if (limit_area is not None):
        return np.apply_along_axis(partial(_interpolate_with_limit_area, method=method, limit=limit, limit_area=limit_area), axis, values)
    orig_values = values
    transf = ((lambda x: x) if (axis == 0) else (lambda x: x.T))
    ndim = values.ndim
    if (values.ndim == 1):
        if (axis != 0):
            raise AssertionError('cannot interpolate on a ndim == 1 with axis != 0')
        values = values.reshape(tuple(((1,) + values.shape)))
    method = clean_fill_method(method)
    tvalues = transf(values)
    if (method == 'pad'):
        result = _pad_2d(tvalues, limit=limit)
    else:
        result = _backfill_2d(tvalues, limit=limit)
    result = transf(result)
    if (ndim == 1):
        result = result[0]
    if (orig_values.dtype.kind in ['m', 'M']):
        result = result.view(orig_values.dtype)
    return result

def _cast_values_for_fillna(values, dtype, has_mask):
    '\n    Cast values to a dtype that algos.pad and algos.backfill can handle.\n    '
    if needs_i8_conversion(dtype):
        values = values.view(np.int64)
    elif (is_integer_dtype(values) and (not has_mask)):
        values = ensure_float64(values)
    return values

def _fillna_prep(values, mask=None):
    dtype = values.dtype
    has_mask = (mask is not None)
    if (not has_mask):
        mask = isna(values)
    values = _cast_values_for_fillna(values, dtype, has_mask)
    mask = mask.view(np.uint8)
    return (values, mask)

def _pad_1d(values, limit=None, mask=None):
    (values, mask) = _fillna_prep(values, mask)
    algos.pad_inplace(values, mask, limit=limit)
    return values

def _backfill_1d(values, limit=None, mask=None):
    (values, mask) = _fillna_prep(values, mask)
    algos.backfill_inplace(values, mask, limit=limit)
    return values

def _pad_2d(values, limit=None, mask=None):
    (values, mask) = _fillna_prep(values, mask)
    if np.all(values.shape):
        algos.pad_2d_inplace(values, mask, limit=limit)
    else:
        pass
    return values

def _backfill_2d(values, limit=None, mask=None):
    (values, mask) = _fillna_prep(values, mask)
    if np.all(values.shape):
        algos.backfill_2d_inplace(values, mask, limit=limit)
    else:
        pass
    return values
_fill_methods = {'pad': _pad_1d, 'backfill': _backfill_1d}

def get_fill_func(method):
    method = clean_fill_method(method)
    return _fill_methods[method]

def clean_reindex_fill_method(method):
    return clean_fill_method(method, allow_nearest=True)

def _interp_limit(invalid, fw_limit, bw_limit):
    "\n    Get indexers of values that won't be filled\n    because they exceed the limits.\n\n    Parameters\n    ----------\n    invalid : boolean ndarray\n    fw_limit : int or None\n        forward limit to index\n    bw_limit : int or None\n        backward limit to index\n\n    Returns\n    -------\n    set of indexers\n\n    Notes\n    -----\n    This is equivalent to the more readable, but slower\n\n    .. code-block:: python\n\n        def _interp_limit(invalid, fw_limit, bw_limit):\n            for x in np.where(invalid)[0]:\n                if invalid[max(0, x - fw_limit):x + bw_limit + 1].all():\n                    yield x\n    "
    N = len(invalid)
    f_idx = set()
    b_idx = set()

    def inner(invalid, limit):
        limit = min(limit, N)
        windowed = _rolling_window(invalid, (limit + 1)).all(1)
        idx = (set((np.where(windowed)[0] + limit)) | set(np.where(((~ invalid[:(limit + 1)]).cumsum() == 0))[0]))
        return idx
    if (fw_limit is not None):
        if (fw_limit == 0):
            f_idx = set(np.where(invalid)[0])
        else:
            f_idx = inner(invalid, fw_limit)
    if (bw_limit is not None):
        if (bw_limit == 0):
            return f_idx
        else:
            b_idx_inv = list(inner(invalid[::(- 1)], bw_limit))
            b_idx = set(((N - 1) - np.asarray(b_idx_inv)))
            if (fw_limit == 0):
                return b_idx
    return (f_idx & b_idx)

def _rolling_window(a, window):
    '\n    [True, True, False, True, False], 2 ->\n\n    [\n        [True,  True],\n        [True, False],\n        [False, True],\n        [True, False],\n    ]\n    '
    shape = (a.shape[:(- 1)] + (((a.shape[(- 1)] - window) + 1), window))
    strides = (a.strides + (a.strides[(- 1)],))
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
