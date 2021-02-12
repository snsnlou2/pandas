
'Sparse accessor'
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import find_common_type
from pandas.core.accessor import PandasDelegate, delegate_names
from pandas.core.arrays.sparse.array import SparseArray
from pandas.core.arrays.sparse.dtype import SparseDtype

class BaseAccessor():
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data=None):
        self._parent = data
        self._validate(data)

    def _validate(self, data):
        raise NotImplementedError

@delegate_names(SparseArray, ['npoints', 'density', 'fill_value', 'sp_values'], typ='property')
class SparseAccessor(BaseAccessor, PandasDelegate):
    '\n    Accessor for SparseSparse from other sparse matrix data types.\n    '

    def _validate(self, data):
        if (not isinstance(data.dtype, SparseDtype)):
            raise AttributeError(self._validation_msg)

    def _delegate_property_get(self, name, *args, **kwargs):
        return getattr(self._parent.array, name)

    def _delegate_method(self, name, *args, **kwargs):
        if (name == 'from_coo'):
            return self.from_coo(*args, **kwargs)
        elif (name == 'to_coo'):
            return self.to_coo(*args, **kwargs)
        else:
            raise ValueError

    @classmethod
    def from_coo(cls, A, dense_index=False):
        "\n        Create a Series with sparse values from a scipy.sparse.coo_matrix.\n\n        Parameters\n        ----------\n        A : scipy.sparse.coo_matrix\n        dense_index : bool, default False\n            If False (default), the SparseSeries index consists of only the\n            coords of the non-null entries of the original coo_matrix.\n            If True, the SparseSeries index consists of the full sorted\n            (row, col) coordinates of the coo_matrix.\n\n        Returns\n        -------\n        s : Series\n            A Series with sparse values.\n\n        Examples\n        --------\n        >>> from scipy import sparse\n\n        >>> A = sparse.coo_matrix(\n        ...     ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(3, 4)\n        ... )\n        >>> A\n        <3x4 sparse matrix of type '<class 'numpy.float64'>'\n        with 3 stored elements in COOrdinate format>\n\n        >>> A.todense()\n        matrix([[0., 0., 1., 2.],\n        [3., 0., 0., 0.],\n        [0., 0., 0., 0.]])\n\n        >>> ss = pd.Series.sparse.from_coo(A)\n        >>> ss\n        0  2    1.0\n           3    2.0\n        1  0    3.0\n        dtype: Sparse[float64, nan]\n        "
        from pandas import Series
        from pandas.core.arrays.sparse.scipy_sparse import coo_to_sparse_series
        result = coo_to_sparse_series(A, dense_index=dense_index)
        result = Series(result.array, index=result.index, copy=False)
        return result

    def to_coo(self, row_levels=(0,), column_levels=(1,), sort_labels=False):
        '\n        Create a scipy.sparse.coo_matrix from a Series with MultiIndex.\n\n        Use row_levels and column_levels to determine the row and column\n        coordinates respectively. row_levels and column_levels are the names\n        (labels) or numbers of the levels. {row_levels, column_levels} must be\n        a partition of the MultiIndex level names (or numbers).\n\n        Parameters\n        ----------\n        row_levels : tuple/list\n        column_levels : tuple/list\n        sort_labels : bool, default False\n            Sort the row and column labels before forming the sparse matrix.\n\n        Returns\n        -------\n        y : scipy.sparse.coo_matrix\n        rows : list (row labels)\n        columns : list (column labels)\n\n        Examples\n        --------\n        >>> s = pd.Series([3.0, np.nan, 1.0, 3.0, np.nan, np.nan])\n        >>> s.index = pd.MultiIndex.from_tuples(\n        ...     [\n        ...         (1, 2, "a", 0),\n        ...         (1, 2, "a", 1),\n        ...         (1, 1, "b", 0),\n        ...         (1, 1, "b", 1),\n        ...         (2, 1, "b", 0),\n        ...         (2, 1, "b", 1)\n        ...     ],\n        ...     names=["A", "B", "C", "D"],\n        ... )\n        >>> s\n        A  B  C  D\n        1  2  a  0    3.0\n                 1    NaN\n           1  b  0    1.0\n                 1    3.0\n        2  1  b  0    NaN\n                 1    NaN\n        dtype: float64\n\n        >>> ss = s.astype("Sparse")\n        >>> ss\n        A  B  C  D\n        1  2  a  0    3.0\n                 1    NaN\n           1  b  0    1.0\n                 1    3.0\n        2  1  b  0    NaN\n                 1    NaN\n        dtype: Sparse[float64, nan]\n\n        >>> A, rows, columns = ss.sparse.to_coo(\n        ...     row_levels=["A", "B"], column_levels=["C", "D"], sort_labels=True\n        ... )\n        >>> A\n        <3x4 sparse matrix of type \'<class \'numpy.float64\'>\'\n        with 3 stored elements in COOrdinate format>\n        >>> A.todense()\n        matrix([[0., 0., 1., 3.],\n        [3., 0., 0., 0.],\n        [0., 0., 0., 0.]])\n\n        >>> rows\n        [(1, 1), (1, 2), (2, 1)]\n        >>> columns\n        [(\'a\', 0), (\'a\', 1), (\'b\', 0), (\'b\', 1)]\n        '
        from pandas.core.arrays.sparse.scipy_sparse import sparse_series_to_coo
        (A, rows, columns) = sparse_series_to_coo(self._parent, row_levels, column_levels, sort_labels=sort_labels)
        return (A, rows, columns)

    def to_dense(self):
        '\n        Convert a Series from sparse values to dense.\n\n        .. versionadded:: 0.25.0\n\n        Returns\n        -------\n        Series:\n            A Series with the same values, stored as a dense array.\n\n        Examples\n        --------\n        >>> series = pd.Series(pd.arrays.SparseArray([0, 1, 0]))\n        >>> series\n        0    0\n        1    1\n        2    0\n        dtype: Sparse[int64, 0]\n\n        >>> series.sparse.to_dense()\n        0    0\n        1    1\n        2    0\n        dtype: int64\n        '
        from pandas import Series
        return Series(self._parent.array.to_dense(), index=self._parent.index, name=self._parent.name)

class SparseFrameAccessor(BaseAccessor, PandasDelegate):
    '\n    DataFrame accessor for sparse data.\n\n    .. versionadded:: 0.25.0\n    '

    def _validate(self, data):
        dtypes = data.dtypes
        if (not all((isinstance(t, SparseDtype) for t in dtypes))):
            raise AttributeError(self._validation_msg)

    @classmethod
    def from_spmatrix(cls, data, index=None, columns=None):
        '\n        Create a new DataFrame from a scipy sparse matrix.\n\n        .. versionadded:: 0.25.0\n\n        Parameters\n        ----------\n        data : scipy.sparse.spmatrix\n            Must be convertible to csc format.\n        index, columns : Index, optional\n            Row and column labels to use for the resulting DataFrame.\n            Defaults to a RangeIndex.\n\n        Returns\n        -------\n        DataFrame\n            Each column of the DataFrame is stored as a\n            :class:`arrays.SparseArray`.\n\n        Examples\n        --------\n        >>> import scipy.sparse\n        >>> mat = scipy.sparse.eye(3)\n        >>> pd.DataFrame.sparse.from_spmatrix(mat)\n             0    1    2\n        0  1.0  0.0  0.0\n        1  0.0  1.0  0.0\n        2  0.0  0.0  1.0\n        '
        from pandas._libs.sparse import IntIndex
        from pandas import DataFrame
        data = data.tocsc()
        (index, columns) = cls._prep_index(data, index, columns)
        (n_rows, n_columns) = data.shape
        data.sort_indices()
        indices = data.indices
        indptr = data.indptr
        array_data = data.data
        dtype = SparseDtype(array_data.dtype, 0)
        arrays = []
        for i in range(n_columns):
            sl = slice(indptr[i], indptr[(i + 1)])
            idx = IntIndex(n_rows, indices[sl], check_integrity=False)
            arr = SparseArray._simple_new(array_data[sl], idx, dtype)
            arrays.append(arr)
        return DataFrame._from_arrays(arrays, columns=columns, index=index, verify_integrity=False)

    def to_dense(self):
        '\n        Convert a DataFrame with sparse values to dense.\n\n        .. versionadded:: 0.25.0\n\n        Returns\n        -------\n        DataFrame\n            A DataFrame with the same values stored as dense arrays.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0])})\n        >>> df.sparse.to_dense()\n           A\n        0  0\n        1  1\n        2  0\n        '
        from pandas import DataFrame
        data = {k: v.array.to_dense() for (k, v) in self._parent.items()}
        return DataFrame(data, index=self._parent.index, columns=self._parent.columns)

    def to_coo(self):
        '\n        Return the contents of the frame as a sparse SciPy COO matrix.\n\n        .. versionadded:: 0.25.0\n\n        Returns\n        -------\n        coo_matrix : scipy.sparse.spmatrix\n            If the caller is heterogeneous and contains booleans or objects,\n            the result will be of dtype=object. See Notes.\n\n        Notes\n        -----\n        The dtype will be the lowest-common-denominator type (implicit\n        upcasting); that is to say if the dtypes (even of numeric types)\n        are mixed, the one that accommodates all will be chosen.\n\n        e.g. If the dtypes are float16 and float32, dtype will be upcast to\n        float32. By numpy.find_common_type convention, mixing int64 and\n        and uint64 will result in a float64 dtype.\n        '
        import_optional_dependency('scipy')
        from scipy.sparse import coo_matrix
        dtype = find_common_type(self._parent.dtypes.to_list())
        if isinstance(dtype, SparseDtype):
            dtype = dtype.subtype
        (cols, rows, datas) = ([], [], [])
        for (col, name) in enumerate(self._parent):
            s = self._parent[name]
            row = s.array.sp_index.to_int_index().indices
            cols.append(np.repeat(col, len(row)))
            rows.append(row)
            datas.append(s.array.sp_values.astype(dtype, copy=False))
        cols = np.concatenate(cols)
        rows = np.concatenate(rows)
        datas = np.concatenate(datas)
        return coo_matrix((datas, (rows, cols)), shape=self._parent.shape)

    @property
    def density(self):
        '\n        Ratio of non-sparse points to total (dense) data points.\n        '
        return np.mean([column.array.density for (_, column) in self._parent.items()])

    @staticmethod
    def _prep_index(data, index, columns):
        from pandas.core.indexes.api import ensure_index
        import pandas.core.indexes.base as ibase
        (N, K) = data.shape
        if (index is None):
            index = ibase.default_index(N)
        else:
            index = ensure_index(index)
        if (columns is None):
            columns = ibase.default_index(K)
        else:
            columns = ensure_index(columns)
        if (len(columns) != K):
            raise ValueError(f'Column length mismatch: {len(columns)} vs. {K}')
        if (len(index) != N):
            raise ValueError(f'Index length mismatch: {len(index)} vs. {N}')
        return (index, columns)
