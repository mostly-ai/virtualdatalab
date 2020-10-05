from numpy cimport ndarray
cimport numpy as np
cimport cython
from libc.math cimport abs


@cython.boundscheck(False)
@cython.wraparound(False)
def mixed_distance(ndarray[np.float64_t, ndim=1] x,
                   ndarray[np.float64_t, ndim=1] y,
                   np.int_t categoric_slice):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i = 0
    cdef double res
    for i in range(categoric_slice):
        res += abs(x[i] != y[i])
    for i in range(categoric_slice, n):
        res += abs(x[i] - y[i])
    return res
