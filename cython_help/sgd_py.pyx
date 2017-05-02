import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
#cdef extern double c_multiply (int n_samples, double* v1, double* v2)
cdef extern double c_multiply(int n_samples,
                              long* u,
                              long* v,
                              double* r,
                              double* Umat,
                              double* Vmat,
                              int m,
                              int n,
                              int k,
                              double reg,
                              double eta
                              )
cdef extern double c_err(int n_samples,
                         long* u, 
                         long* v, 
                         double* r,
                         double* Umat,
                         double* Vmat,
                         int m,
                         int n,
                         int k
                         )

@cython.boundscheck(False)
@cython.wraparound(False)
def err (np.ndarray[long, ndim=1, mode="c"] users not None,
         np.ndarray[long, ndim=1, mode="c"] movies not None,
         np.ndarray[double, ndim=1, mode="c"] ratings not None,
         np.ndarray[double, ndim=2, mode="c"] Umatrix not None,
         np.ndarray[double, ndim=2, mode="c"] Vmatrix not None
        ):
    cdef int n_samples, m, n, k
    cdef double result
    n_samples = users.shape[0]
    m = Umatrix.shape[0] 
    n = Vmatrix.shape[0] 
    k = Umatrix.shape[1]
    assert(k == Vmatrix.shape[1])
    result = c_err(n_samples, &users[0], &movies[0], &ratings[0], &Umatrix[0,0],
                   &Vmatrix[0,0], m, n, k)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def multiply (np.ndarray[long, ndim=1, mode="c"] users not None, 
              np.ndarray[long, ndim=1, mode="c"] movies not None,
              np.ndarray[double, ndim=1, mode="c"] ratings not None,
              np.ndarray[double, ndim=2, mode="c"] Umatrix not None,
              np.ndarray[double, ndim=2, mode="c"] Vmatrix not None,
              double reg,
              double eta
             ):
    cdef int n_samples, m, n, k
    cdef double result
    n_samples = users.shape[0]
    m = Umatrix.shape[0] 
    n = Vmatrix.shape[0] 
    k = Umatrix.shape[1]
    assert(k == Vmatrix.shape[1])
    result = c_multiply(n_samples, &users[0], &movies[0], &ratings[0], &Umatrix[0,0],
                        &Vmatrix[0,0], m, n, k, reg, eta)
    return
