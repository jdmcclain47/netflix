import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


#cdef extern from 'mkl_cblas.h':
#    double ddot 'cblas_ddot'(int N,
#                             double* X, int incX,
#                             double* Y, int incY) nogil

# declare the interface to the C code
#cdef extern double c_multiply (int n_samples, double* v1, double* v2)
import multiprocessing

cdef extern void permutation(long* indices, long n, long blk_size)
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
cdef extern double c_multiply_hogwild(int n_samples,
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
cdef extern double c_multiply_minibatch(int n_samples, 
                  long* u, 
                  long* v, 
                  double* r,
                  double* Umat,
                  double* Vmat,
                  int m,
                  int n,
                  int k,
                  double reg,
                  double eta,
                  double* Umat_work,
                  double* Vmat_work,
                  long* Uindices,
                  long* Vindices,
                  int batch_size
                  )
cdef extern double c_multiply_hogbatch(int n_samples, 
                  long* u, 
                  long* v, 
                  double* r,
                  double* Umat,
                  double* Vmat,
                  int m,
                  int n,
                  int k,
                  double reg,
                  double eta,
                  double* Umat_work,
                  double* Vmat_work,
                  long* Uindices,
                  long* Vindices,
                  int batch_size
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

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def shuffle (np.ndarray[long, ndim=1, mode="c"] users not None,
#             np.ndarray[long, ndim=1, mode="c"] movies not None,
#             np.ndarray[double, ndim=1, mode="c"] ratings not None
#        ):
#    n = users.shape[0]
#    work1 = np.arange(n)
#    cdef long[:] cython_view1 = work1
#    cdef long *indices = &cython_view1[0]
#    cdef int blk_size = 100000
#    permutation(indices, n, blk_size)
#    n_blocks = n / blk_size
#    for j,i in enumerate(np.rand.permutation(n_blocks)):
#        tmp = indices[j*blk_size:(j+1)*blk_size]
#        indices[j*blk_size:(j+1)*blk_size] = indices[i*blk_size:(i+1)*blk_size]
#        indices[i*blk_size:(i+1)*blk_size] = tmp
#    return

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


@cython.boundscheck(False)
@cython.wraparound(False)
def multiply_parallel (np.ndarray[long, ndim=1, mode="c"] users not None, 
              np.ndarray[long, ndim=1, mode="c"] movies not None,
              np.ndarray[double, ndim=1, mode="c"] ratings not None,
              np.ndarray[double, ndim=2, mode="c"] Umatrix not None,
              np.ndarray[double, ndim=2, mode="c"] Vmatrix not None,
              double reg,
              double eta
             ):
    cdef int n_samples, m, n, k
    cdef double result
    import os
    cdef int n_threads = (int)(os.environ['OMP_NUM_THREADS'])
    n_samples = users.shape[0]
    m = Umatrix.shape[0] 
    n = Vmatrix.shape[0] 
    k = Umatrix.shape[1]
    assert(k == Vmatrix.shape[1])

    ####################################################################
    # The following was taken from  "High performance parallel sto-    #
    # chastic gradient descent in shared memory" by Sallinen et al.    #
    ####################################################################

    # METHOD: "Hogwild"

    # result = c_multiply_hogwild(n_samples, &users[0], &movies[0], &ratings[0], &Umatrix[0,0],
    #                     &Vmatrix[0,0], m, n, k, reg, eta)


    # METHOD: "Minibatch"
    # Minibatch isn't optimized since it's sort of the failure of the group (as it should be).
    # It was implemented for educational purposes only

    #fac = 1
    #batch_size = 900
    #work1 = np.zeros((fac * batch_size * k),dtype=np.float)
    #work2 = np.zeros((fac * batch_size * k),dtype=np.float)
    #work3 = np.zeros((fac * batch_size),dtype=np.int)
    #work4 = np.zeros((fac * batch_size),dtype=np.int)
    #cdef double[:] cython_view1 = work1
    #cdef double *UmatrixWork = &cython_view1[0]
    #cdef double[:] cython_view2 = work2
    #cdef double *VmatrixWork = &cython_view2[0]
    #cdef long[:] cython_view3 = work3
    #cdef long *Uindices = &cython_view3[0]
    #cdef long[:] cython_view4 = work4
    #cdef long *Vindices = &cython_view4[0]
    #result = c_multiply_minibatch(n_samples, &users[0], &movies[0], &ratings[0], &Umatrix[0,0],
    #                    &Vmatrix[0,0], m, n, k, reg, eta, &UmatrixWork[0], &VmatrixWork[0], &Uindices[0], &Vindices[0], batch_size)

    # METHOD: "Hogbatch"

    fac = n_threads
    batch_size = 30
    fac = n_threads
    work1 = np.zeros((fac * batch_size * k),dtype=np.float)
    work2 = np.zeros((fac * batch_size * k),dtype=np.float)
    work3 = np.zeros((fac * batch_size),dtype=np.int)
    work4 = np.zeros((fac * batch_size),dtype=np.int)
    cdef double[:] cython_view1 = work1
    cdef double *UmatrixWork = &cython_view1[0]
    cdef double[:] cython_view2 = work2
    cdef double *VmatrixWork = &cython_view2[0]
    cdef long[:] cython_view3 = work3
    cdef long *Uindices = &cython_view3[0]
    cdef long[:] cython_view4 = work4
    cdef long *Vindices = &cython_view4[0]
    result = c_multiply_hogbatch(n_samples, &users[0], &movies[0], &ratings[0], &Umatrix[0,0],
                        &Vmatrix[0,0], m, n, k, reg, eta, &UmatrixWork[0], &VmatrixWork[0], &Uindices[0], &Vindices[0], batch_size)

    return
