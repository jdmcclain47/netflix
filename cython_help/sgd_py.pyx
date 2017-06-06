import cython
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


cdef extern struct model_params:
    double* Vmat;
    double* Vmat_grad;
    double* Vfmat;
    double* Vfmat_grad;
    double* Umat;
    double* Umat_grad;
    long *user_indices_grad;
    long *movie_indices_grad;
    long *time_indices_grad;
    long *time_bin_indices_grad;
    long *user_date_indices_grad;
    long *freq_indices_grad;
    double *bvf;
    double *bvf_grad;
    double *bu;
    double *bu_grad;
    double *alpha_ku;
    double *alpha_ku_grad;
    double *yu;
    double *yu_grad;
    double *yv;
    double *yv_grad;
    double *cu;
    double *cu_grad;
    double *cu_t;
    double *cu_t_grad;
    double *bv;
    double *bv_grad;
    double* bu_t;
    double* bu_t_grad;
    double* bv_t_bin;
    double* bv_t_bin_grad;
    double* alpha_u;
    double* alpha_u_grad;

    double* reg;
    double* eta;
    double mu;
    double freq_base;

    int batch_size;
    long n_samples;
    long dates_per_bin, n_bins;
    long n_dpu;
    long n_freq;
    int m, n, k, t;
    
    long* csum_dpu;
    long* avg_dpu;
    long* dpu;
    long max_rpu;
    long* csum_mpu;
    long* csum_upm;
    long* mpu;
    long* upm;
    long* n_rpu;
    long* n_rpdpu;
    double* sum_rpu;

    long* u;
    long* v;
    long* d;
    double* r;

cdef extern long return_m2_(model_params* mpm);
cdef extern void c_predict(
        long n_samples,
        long* u, 
        long* v, 
        long* d,
        model_params* mpm,
        double* r_out
        )
cdef extern double c_rms_err(
        long n_samples,
        double* r_pred,
        double* r
        ) nogil


# Sed script to set getter/setters for a list of variables
# s/cdef \([a-z*]*\) \([a-z_0-9]*\)/cdef \1 \2;\r\tdef set_\2(self, in_\2):\r\t\tself.\2 = in_\2;\r\tdef get_\2(self):\r\t\treturn self.\2/
cdef class Model(object):
    cdef model_params parameters;

    def __cinit__(self, int m, int n, int k, int t, int batch_size, long n_tbins, int n_freq, long n_dpu, long max_rpu):
        import os
        cdef int n_threads = (int)(os.environ['OMP_NUM_THREADS'])
        self.parameters.m = m;
        self.parameters.n = n;
        self.parameters.k = k;
        self.parameters.t = t;
        self.parameters.dates_per_bin = int(np.ceil(t / float(n_tbins)));
        self.parameters.n_bins = n_tbins;
        self.parameters.n_dpu = n_dpu;
        self.parameters.n_freq = n_freq;
        self.parameters.batch_size = batch_size;
        self.parameters.max_rpu = max_rpu;

        self.parameters.reg = <double*> PyMem_Malloc(sizeof(double) * 30);
        self.parameters.eta = <double*> PyMem_Malloc(sizeof(double) * 30);

        self.parameters.Umat_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size * k);
        self.parameters.Vmat_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size * k);
        self.parameters.Vfmat_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size * k);
        self.parameters.user_indices_grad = <long*> PyMem_Malloc(sizeof(long) * n_threads * batch_size);
        self.parameters.movie_indices_grad = <long*> PyMem_Malloc(sizeof(long) * n_threads * batch_size);
        self.parameters.time_indices_grad = <long*> PyMem_Malloc(sizeof(long) * n_threads * batch_size);
        self.parameters.time_bin_indices_grad = <long*> PyMem_Malloc(sizeof(long) * n_threads * batch_size);
        self.parameters.user_date_indices_grad = <long*> PyMem_Malloc(sizeof(long) * n_threads * batch_size);
        self.parameters.freq_indices_grad = <long*> PyMem_Malloc(sizeof(long) * n_threads * batch_size);
        self.parameters.bvf_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);
        self.parameters.bu_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);
        self.parameters.alpha_ku_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size * k);
        self.parameters.yv_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size * k * max_rpu);
        self.parameters.cu_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);
        self.parameters.cu_t_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);
        self.parameters.bv_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);
        self.parameters.bv_t_bin_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);
        self.parameters.bu_t_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);
        self.parameters.alpha_u_grad = <double*> PyMem_Malloc(sizeof(double) * n_threads * batch_size);

        self.parameters.Umat = <double*> PyMem_Malloc(sizeof(double) * m * k);
        self.parameters.Vmat = <double*> PyMem_Malloc(sizeof(double) * n * k);
        self.parameters.Vfmat = <double*> PyMem_Malloc(sizeof(double) * n * n_freq * k);
        self.parameters.bvf = <double*> PyMem_Malloc(sizeof(double) * n * n_freq);
        self.parameters.bu = <double*> PyMem_Malloc(sizeof(double) * m);
        self.parameters.bv = <double*> PyMem_Malloc(sizeof(double) * n);
        self.parameters.bv_t_bin = <double*> PyMem_Malloc(sizeof(double) * n * n_tbins);
        self.parameters.alpha_ku = <double*> PyMem_Malloc(sizeof(double) * m * k);
        self.parameters.yu = <double*> PyMem_Malloc(sizeof(double) * m * k);
        self.parameters.yv = <double*> PyMem_Malloc(sizeof(double) * n * k);
        self.parameters.cu = <double*> PyMem_Malloc(sizeof(double) * m);
        self.parameters.alpha_u = <double*> PyMem_Malloc(sizeof(double) * m);
        self.parameters.cu_t = <double*> PyMem_Malloc(sizeof(double) * n_dpu)
        self.parameters.bu_t = <double*> PyMem_Malloc(sizeof(double) * n_dpu)

    def set_mu(self, double in_mu):
        self.parameters.mu = in_mu;
    def get_mu(self):
        return self.parameters.mu;
    def set_freq_base(self, double in_freq_base):
        self.parameters.freq_base = in_freq_base;
    def get_freq_base(self):
        return self.parameters.freq_base;
    def set_batch_size(self, int in_batch_size):
        self.parameters.batch_size = in_batch_size;
    def get_batch_size(self):
        return self.parameters.batch_size;
    def set_n_samples(self, long in_n_samples):
        self.parameters.n_samples = in_n_samples;
    def get_n_samples(self):
        return self.parameters.n_samples;
    def set_dates_per_bin(self, long in_dates_per_bin):
        self.parameters.dates_per_bin = in_dates_per_bin;
    def get_dates_per_bin(self):
        return self.parameters.dates_per_bin
    def set_n_bins(self, long in_n_bins):
        self.parameters.n_bins = in_n_bins;
    def get_n_bins(self):
        return self.parameters.n_bins;
    def set_m(self, int in_m):
        self.parameters.m = in_m;
    def get_m(self):
        return self.parameters.m;
    def set_n(self, int in_n):
        self.parameters.n = in_n;
    def get_n(self):
        return self.parameters.n;
    def set_k(self, int in_k):
        self.parameters.k = in_k;
    def get_k(self):
        return self.parameters.k;
    def set_t(self, int in_t):
        self.parameters.t = in_t;
    def get_t(self):
        return self.parameters.t;
    def set_max_rpu(self, long in_max_rpu):
        self.parameters.max_rpu = in_max_rpu;
    def get_max_rpu(self):
        return self.parameters.max_rpu;

    def set_reg(self, np.ndarray[double, ndim=1, mode="c"] in_reg):
        count = 0
        size1 = in_reg.shape[0];
        for i1 in range(size1):
            self.parameters.reg[count] = in_reg[i1]
            count += 1
    def set_eta(self, np.ndarray[double, ndim=1, mode="c"] in_eta):
        count = 0
        size1 = in_eta.shape[0];
        for i1 in range(size1):
            self.parameters.eta[count] = in_eta[i1]
            count += 1
    def set_Vmat(self, np.ndarray[double, ndim=2, mode="c"] in_Vmat):
        count = 0
        size1 = self.parameters.n;
        size2 = self.parameters.k;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.Vmat[count] = in_Vmat[i1,i2]
                count += 1
    def set_Vfmat(self, np.ndarray[double, ndim=2, mode="c"] in_Vfmat):
        count = 0
        size1 = self.parameters.n * self.parameters.n_freq;
        size2 = self.parameters.k;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.Vfmat[count] = in_Vfmat[i1,i2]
                count += 1
    def set_Umat(self, np.ndarray[double, ndim=2, mode="c"] in_Umat):
        count = 0
        size1 = self.parameters.m;
        size2 = self.parameters.k;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.Umat[count] = in_Umat[i1,i2]
                count += 1
    def set_bvf(self, np.ndarray[double, ndim=2, mode="c"] in_bvf):
        count = 0
        size1 = self.parameters.n;
        size2 = self.parameters.n_freq;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.bvf[count] = in_bvf[i1,i2]
                count += 1
    def set_bu(self, np.ndarray[double, ndim=1, mode="c"] in_bu):
        count = 0
        size1 = self.parameters.m;
        for i1 in range(size1):
            self.parameters.bu[count] = in_bu[i1]
            count += 1
    def set_alpha_ku(self, np.ndarray[double, ndim=2, mode="c"] in_alpha_ku):
        count = 0
        size1 = self.parameters.m;
        size2 = self.parameters.k;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.alpha_ku[count] = in_alpha_ku[i1,i2]
                count += 1
    def set_yu(self, np.ndarray[double, ndim=2, mode="c"] in_yu):
        count = 0
        size1 = self.parameters.m;
        size2 = self.parameters.k;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.yu[count] = in_yu[i1,i2]
                count += 1
    def set_yv(self, np.ndarray[double, ndim=2, mode="c"] in_yv):
        count = 0
        size1 = self.parameters.n;
        size2 = self.parameters.k;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.yv[count] = in_yv[i1,i2]
                count += 1
    def set_cu(self, np.ndarray[double, ndim=1, mode="c"] in_cu):
        count = 0
        size1 = self.parameters.m;
        for i1 in range(size1):
            self.parameters.cu[count] = in_cu[i1]
            count += 1
    def set_cu_t(self, np.ndarray[double, ndim=1, mode="c"] in_cu_t):
        count = 0
        size1 = self.parameters.n_dpu;
        for i1 in range(size1):
            self.parameters.cu_t[count] = in_cu_t[i1]
            count += 1
    def set_bv(self, np.ndarray[double, ndim=1, mode="c"] in_bv):
        count = 0
        size1 = self.parameters.n;
        for i1 in range(size1):
            self.parameters.bv[count] = in_bv[i1]
            count += 1
    def set_bu_t(self, np.ndarray[double, ndim=1, mode="c"] in_bu_t):
        count = 0
        size1 = self.parameters.n_dpu;
        for i1 in range(size1):
            self.parameters.bu_t[count] = in_bu_t[i1]
            count += 1
    def set_bv_t_bin(self, np.ndarray[double, ndim=2, mode="c"] in_bv_t_bin):
        count = 0
        size1 = self.parameters.n;
        size2 = self.parameters.n_bins;
        for i1 in range(size1):
            for i2 in range(size2):
                self.parameters.bv_t_bin[count] = in_bv_t_bin[i1,i2]
                count += 1
    def set_alpha_u(self, np.ndarray[double, ndim=1, mode="c"] in_alpha_u):
        count = 0
        size1 = self.parameters.m;
        for i1 in range(size1):
            self.parameters.alpha_u[count] = in_alpha_u[i1]
            count += 1
    def set_csum_dpu(self, np.ndarray[long, ndim=1, mode="c"] in_csum_dpu):
        self.parameters.csum_dpu = &in_csum_dpu[0];
    def set_avg_dpu(self, np.ndarray[long, ndim=1, mode="c"] in_avg_dpu):
        self.parameters.avg_dpu = &in_avg_dpu[0];
    def set_dpu(self, np.ndarray[long, ndim=1, mode="c"] in_dpu):
        self.parameters.dpu = &in_dpu[0];
    def set_csum_upm(self, np.ndarray[long, ndim=1, mode="c"] in_csum_upm):
        self.parameters.csum_upm = &in_csum_upm[0];
    def set_upm(self, np.ndarray[long, ndim=1, mode="c"] in_upm):
        self.parameters.upm = &in_upm[0];
    def set_csum_mpu(self, np.ndarray[long, ndim=1, mode="c"] in_csum_mpu):
        self.parameters.csum_mpu = &in_csum_mpu[0];
    def set_mpu(self, np.ndarray[long, ndim=1, mode="c"] in_mpu):
        self.parameters.mpu = &in_mpu[0];
    def set_n_rpu(self, np.ndarray[long, ndim=1, mode="c"] in_n_rpu):
        self.parameters.n_rpu = &in_n_rpu[0];
    def set_n_rpdpu(self, np.ndarray[long, ndim=1, mode="c"] in_n_rpdpu):
        self.parameters.n_rpdpu = &in_n_rpdpu[0];
    def set_sum_rpu(self, np.ndarray[double, ndim=1, mode="c"] in_sum_rpu):
        self.parameters.sum_rpu = &in_sum_rpu[0];
    #def set_Vmat_grad(self, np.ndarray[double, ndim=1, mode="c"] in_Vmat_grad):
    #    self.parameters.Vmat_grad = &in_Vmat_grad[0];
    #def set_Vfmat_grad(self, np.ndarray[double, ndim=1, mode="c"] in_Vfmat_grad):
    #    self.parameters.Vfmat_grad = &in_Vfmat_grad[0];
    #def set_Umat_grad(self, np.ndarray[double, ndim=1, mode="c"] in_Umat_grad):
    #    self.parameters.Umat_grad = &in_Umat_grad[0];
    #def set_user_indices_grad(self, np.ndarray[long, ndim=1, mode="c"] in_user_indices_grad):
    #    self.parameters.user_indices_grad = &in_user_indices_grad[0];
    #def set_movie_indices_grad(self, np.ndarray[long, ndim=1, mode="c"] in_movie_indices_grad):
    #    self.parameters.movie_indices_grad = &in_movie_indices_grad[0];
    #def set_time_indices_grad(self, np.ndarray[long, ndim=1, mode="c"] in_time_indices_grad):
    #    self.parameters.time_indices_grad = &in_time_indices_grad[0];
    #def set_time_bin_indices_grad(self, np.ndarray[long, ndim=1, mode="c"] in_time_bin_indices_grad):
    #    self.parameters.time_bin_indices_grad = &in_time_bin_indices_grad[0];
    #def set_user_date_indices_grad(self, np.ndarray[long, ndim=1, mode="c"] in_user_date_indices_grad):
    #    self.parameters.user_date_indices_grad = &in_user_date_indices_grad[0];
    #def set_freq_indices_grad(self, np.ndarray[long, ndim=1, mode="c"] in_freq_indices_grad):
    #    self.parameters.freq_indices_grad = &in_freq_indices_grad[0];
    #def set_bvf_grad(self, np.ndarray[double, ndim=1, mode="c"] in_bvf_grad):
    #    self.parameters.bvf_grad = &in_bvf_grad[0];
    #def set_bu_grad(self, np.ndarray[double, ndim=1, mode="c"] in_bu_grad):
    #    self.parameters.bu_grad = &in_bu_grad[0];
    #def set_alpha_ku_grad(self, np.ndarray[double, ndim=1, mode="c"] in_alpha_ku_grad):
    #    self.parameters.alpha_ku_grad = &in_alpha_ku_grad[0];
    #def set_yu_grad(self, np.ndarray[double, ndim=1, mode="c"] in_yu_grad):
    #    self.parameters.yu_grad = &in_yu_grad[0];
    #def set_yv_grad(self, np.ndarray[double, ndim=1, mode="c"] in_yv_grad):
    #    self.parameters.yv_grad = &in_yv_grad[0];
    #def set_cu_grad(self, np.ndarray[double, ndim=1, mode="c"] in_cu_grad):
    #    self.parameters.cu_grad = &in_cu_grad[0];
    #def set_cu_t_grad(self, np.ndarray[double, ndim=1, mode="c"] in_cu_t_grad):
    #    self.parameters.cu_t_grad = &in_cu_t_grad[0];
    #def set_bv_grad(self, np.ndarray[double, ndim=1, mode="c"] in_bv_grad):
    #    self.parameters.bv_grad = &in_bv_grad[0];
    #def set_bu_t_grad(self, np.ndarray[double, ndim=1, mode="c"] in_bu_t_grad):
    #    self.parameters.bu_t_grad = &in_bu_t_grad[0];
    #def set_bv_t_bin_grad(self, np.ndarray[double, ndim=1, mode="c"] in_bv_t_bin_grad):
    #    self.parameters.bv_t_bin_grad = &in_bv_t_bin_grad[0];
    #def set_alpha_u_grad(self, np.ndarray[double, ndim=1, mode="c"] in_alpha_u_grad):
    #    self.parameters.alpha_u_grad = &in_alpha_u_grad[0];

    def return_m2(self):
        return return_m2_(&self.parameters);

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mult_eta(self, fac):
        for i in range(30):
            self.parameters.eta[i] *= fac;
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(
            self,
            np.ndarray[long, ndim=1, mode="c"] users not None,
            np.ndarray[long, ndim=1, mode="c"] movies not None,
            np.ndarray[long, ndim=1, mode="c"] dates not None,
            np.ndarray[double, ndim=1, mode="c"] ratings not None
        ):
        cdef long n_samples;
        n_samples = users.shape[0];
        c_predict(n_samples, &users[0], &movies[0], &dates[0], &self.parameters, &ratings[0]);
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rms_err(
            self,
            np.ndarray[double, ndim=1, mode="c"] r_pred not None,
            np.ndarray[double, ndim=1, mode="c"] ratings not None
            ):
        cdef long n_samples;
        cdef double err;
        n_samples = r_pred.shape[0];
        err = c_rms_err(n_samples, &r_pred[0], &ratings[0]);
        return err;

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_training_error(
            self,
            np.ndarray[long, ndim=1, mode="c"] users not None,
            np.ndarray[long, ndim=1, mode="c"] movies not None,
            np.ndarray[long, ndim=1, mode="c"] dates not None,
            np.ndarray[double, ndim=1, mode="c"] ratings not None
            ):
        cdef long n_samples;
        cdef double err;
        n_samples = users.shape[0];
        err = c_training_err(n_samples, &users[0], &movies[0], &dates[0], &ratings[0], &self.parameters)
        return err

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_rms_error(
            self,
            np.ndarray[long, ndim=1, mode="c"] users not None,
            np.ndarray[long, ndim=1, mode="c"] movies not None,
            np.ndarray[long, ndim=1, mode="c"] dates not None,
            np.ndarray[double, ndim=1, mode="c"] ratings not None
            ):
        cdef long n_samples;
        cdef double err;
        n_samples = users.shape[0];
        r_pred = np.zeros((n_samples),dtype=np.float);
        self.predict(users, movies, dates, r_pred);
        err = self.rms_err(r_pred, ratings);
        return err

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def multiply_parallel(
            self,
            np.ndarray[long, ndim=1, mode="c"] users,
            np.ndarray[long, ndim=1, mode="c"] movies,
            np.ndarray[long, ndim=1, mode="c"] dates,
            np.ndarray[double, ndim=1, mode="c"] ratings
        ):
        self.parameters.u = &users[0];
        self.parameters.v = &movies[0];
        self.parameters.d = &dates[0];
        self.parameters.r = &ratings[0];

        c_multiply_hogbatch_with_baseline1(&self.parameters)

        return

    #def train(self, u, v, d, r):
    #    return train_(&self.parameters);
    #long* u;
    #long* v;
    #long* d;
    #double* r;

#cdef extern from 'mkl_cblas.h':
#    double ddot 'cblas_ddot'(int N,
#                             double* X, int incX,
#                             double* Y, int incY) nogil

# declare the interface to the C code
#cdef extern double c_multiply (int n_samples, double* v1, double* v2)
import multiprocessing


cdef extern double c_multiply_hogbatch_new(
        model_params* mpm
        ) nogil

cdef extern double c_multiply_hogbatch_with_baseline1(
        model_params* mpm
        ) nogil

cdef extern double c_training_err(
             long n_samples,
             long* u, 
             long* v, 
             long* d,
             double* r,
             model_params* mpm
        ) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def multiply_parallel (np.ndarray[long, ndim=1, mode="c"] users not None, 
              np.ndarray[long, ndim=1, mode="c"] movies not None,
              np.ndarray[long, ndim=1, mode="c"] dates not None,
              np.ndarray[double, ndim=1, mode="c"] ratings not None,
              int batch_size,
              np.ndarray[double, ndim=2, mode="c"] Umat not None,
              np.ndarray[double, ndim=2, mode="c"] Vmat not None,
              np.ndarray[double, ndim=2, mode="c"] Vfmat not None,
              np.ndarray[double, ndim=1, mode="c"] reg not None,
              np.ndarray[double, ndim=1, mode="c"] eta not None,
              double mu,
              double freq_base,
              np.ndarray[double, ndim=2, mode="c"] bvf,
              np.ndarray[double, ndim=1, mode="c"] bu,
              np.ndarray[double, ndim=1, mode="c"] bv,
              long dates_per_bin,
              long n_bins,
              np.ndarray[double, ndim=2, mode="c"] bv_t_bin,
              np.ndarray[long, ndim=1, mode="c"] csum_dpu,
              np.ndarray[long, ndim=1, mode="c"] dpu,
              np.ndarray[double, ndim=1, mode="c"] bu_t,
              np.ndarray[long, ndim=1, mode="c"] avg_dpu,
              np.ndarray[double, ndim=1, mode="c"] alpha_u,
              np.ndarray[double, ndim=1, mode="c"] sum_rpu,
              int max_rpu,
              np.ndarray[long, ndim=1, mode="c"] csum_mpu,
              np.ndarray[long, ndim=1, mode="c"] mpu,
              np.ndarray[long, ndim=1, mode="c"] n_rpu,
              np.ndarray[long, ndim=1, mode="c"] n_rpdpu,
              np.ndarray[double, ndim=2, mode="c"] alpha_ku,
              np.ndarray[double, ndim=2, mode="c"] yu,
              np.ndarray[double, ndim=2, mode="c"] yv,
              np.ndarray[double, ndim=1, mode="c"] cu,
              np.ndarray[double, ndim=1, mode="c"] cu_t
             ):
    cdef long n_samples
    cdef int m, n, k
    cdef double result
    import os
    cdef int n_threads = (int)(os.environ['OMP_NUM_THREADS'])
    n_samples = users.shape[0]
    m = Umat.shape[0] 
    n = Vmat.shape[0] 
    k = Umat.shape[1]
    assert(k == Vmat.shape[1])

    

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
    #grad1 = np.zeros((fac * batch_size * k),dtype=np.float)
    #grad2 = np.zeros((fac * batch_size * k),dtype=np.float)
    #grad3 = np.zeros((fac * batch_size),dtype=np.int)
    #grad4 = np.zeros((fac * batch_size),dtype=np.int)
    #cdef double[:] cython_view1 = grad1
    #cdef double *Umatrixgrad = &cython_view1[0]
    #cdef double[:] cython_view2 = grad2
    #cdef double *Vmatrixgrad = &cython_view2[0]
    #cdef long[:] cython_view3 = grad3
    #cdef long *Uindices = &cython_view3[0]
    #cdef long[:] cython_view4 = grad4
    #cdef long *Vindices = &cython_view4[0]
    #result = c_multiply_minibatch(n_samples, &users[0], &movies[0], &ratings[0], &Umatrix[0,0],
    #                    &Vmatrix[0,0], m, n, k, reg, eta, &Umatrixgrad[0], &Vmatrixgrad[0], &Uindices[0], &Vindices[0], batch_size)

    # METHOD: "Hogbatch"

    fac = n_threads
    
    grad1 = np.zeros((fac * batch_size * k),dtype=np.float)
    grad2 = np.zeros((fac * batch_size * k),dtype=np.float)
    grad3 = np.zeros((fac * batch_size),dtype=np.int)
    grad4 = np.zeros((fac * batch_size),dtype=np.int)
    grad5 = np.zeros((fac * batch_size),dtype=np.float)
    grad6 = np.zeros((fac * batch_size),dtype=np.float)
    grad6 = np.zeros((fac * batch_size),dtype=np.float)
    grad7 = np.zeros((fac * batch_size),dtype=np.int)
    grad8 = np.zeros((fac * batch_size),dtype=np.int)
    grad9 = np.zeros((fac * batch_size),dtype=np.float)
    grad10 = np.zeros((fac * batch_size),dtype=np.float)
    grad11 = np.zeros((fac * batch_size),dtype=np.int)
    grad12 = np.zeros((fac * batch_size),dtype=np.float)
    grad13 = np.zeros((fac * batch_size),dtype=np.float)
    grad14 = np.zeros((fac * batch_size),dtype=np.float)
    grad15 = np.zeros((fac * batch_size * k),dtype=np.float)
    #grad16 = np.zeros((fac * batch_size),dtype=np.float)
    #grad17 = np.zeros((fac * batch_size),dtype=np.float)
    grad18 = np.zeros((fac * batch_size * k),dtype=np.float)
    grad19 = np.zeros((fac * batch_size * k),dtype=np.float)
    grad20 = np.zeros((fac * batch_size),dtype=np.float)
    grad21 = np.zeros((fac * batch_size),dtype=np.int)
    grad22 = np.zeros((fac * batch_size * k),dtype=np.float)

    cdef double[:] cython_view1 = grad1
    cdef double *Umat_grad = &cython_view1[0]

    cdef double[:] cython_view2 = grad2
    cdef double *Vmat_grad = &cython_view2[0]

    cdef long[:] cython_view3 = grad3
    cdef long *Uindices = &cython_view3[0]

    cdef long[:] cython_view4 = grad4
    cdef long *Vindices = &cython_view4[0]

    cdef double[:] cython_view5 = grad5
    cdef double *bu_grad = &cython_view5[0]

    cdef double[:] cython_view6 = grad6
    cdef double *bv_grad = &cython_view6[0]

    cdef long[:] cython_view7 = grad7
    cdef long *Tindices = &cython_view7[0]

    cdef long[:] cython_view8 = grad8
    cdef long *Tbin_indices = &cython_view8[0]

    cdef double[:] cython_view9 = grad9
    cdef double *bv_t_bin_grad = &cython_view9[0]

    cdef double[:] cython_view10 = grad10
    cdef double *bu_t_grad = &cython_view10[0]

    cdef long[:] cython_view11 = grad11
    cdef long *Udate_indices = &cython_view11[0]

    cdef double[:] cython_view12 = grad12
    cdef double *alpha_u_grad = &cython_view12[0]

    cdef double[:] cython_view13 = grad13
    cdef double *cu_grad = &cython_view13[0]

    cdef double[:] cython_view14 = grad14
    cdef double *cu_t_grad = &cython_view14[0]

    cdef double[:] cython_view15 = grad15
    cdef double *yu_grad = &cython_view15[0]

    #cdef double[:] cython_view16 = grad16
    #cdef double *cu_grad = &cython_view16[0]

    #cdef double[:] cython_view17 = grad17
    #cdef double *cu_t_grad = &cython_view17[0]

    cdef double[:] cython_view18 = grad18
    cdef double *yv_grad = &cython_view18[0]

    cdef double[:] cython_view19 = grad19
    cdef double *alpha_ku_grad = &cython_view19[0]

    cdef double[:] cython_view20 = grad20
    cdef double *bvf_grad = &cython_view20[0]

    cdef long[:] cython_view21 = grad21
    cdef long *freq_indices = &cython_view21[0]

    cdef double[:] cython_view22 = grad22
    cdef double *Vfmat_grad = &cython_view22[0]

    cdef int t = 3
    cdef model_params parameters;
    parameters.Vmat = &Vmat[0,0];
    parameters.Vmat_grad = &Vmat_grad[0];
    parameters.Vfmat = &Vfmat[0,0];
    parameters.Vfmat_grad = &Vfmat_grad[0];
    parameters.Umat = &Umat[0,0];
    parameters.Umat_grad = &Umat_grad[0];
    parameters.user_indices_grad = &Uindices[0];
    parameters.movie_indices_grad = &Vindices[0];
    parameters.time_indices_grad = &Tindices[0];
    parameters.time_bin_indices_grad = &Tbin_indices[0];
    parameters.user_date_indices_grad = &Udate_indices[0];
    parameters.freq_indices_grad = &freq_indices[0];
    parameters.bvf = &bvf[0,0];
    parameters.bvf_grad = &bvf_grad[0];
    parameters.bu = &bu[0];
    parameters.bu_grad = &bu_grad[0];
    parameters.alpha_ku = &alpha_ku[0,0];
    parameters.alpha_ku_grad = &alpha_ku_grad[0];
    parameters.yu = &yu[0,0];
    parameters.yu_grad = &yu_grad[0];
    parameters.yv = &yv[0,0];
    parameters.yv_grad = &yv_grad[0];
    parameters.cu = &cu[0];
    parameters.cu_grad = &cu_grad[0];
    parameters.cu_t = &cu_t[0];
    parameters.cu_t_grad = &cu_t_grad[0];
    parameters.bv = &bv[0];
    parameters.bv_grad = &bv_grad[0];
    parameters.reg = &reg[0];
    parameters.eta = &eta[0];
    parameters.mu = mu;
    parameters.freq_base = freq_base;
    parameters.n_samples = n_samples;
    parameters.batch_size = batch_size;
    parameters.m = m;
    parameters.n = n;
    parameters.k = k;
    parameters.t = t;
    parameters.u = &users[0];
    parameters.v = &movies[0];
    parameters.d = &dates[0];
    parameters.r = &ratings[0];
    parameters.dates_per_bin = dates_per_bin;
    parameters.n_bins = n_bins;
    parameters.bv_t_bin = &bv_t_bin[0,0];
    parameters.bv_t_bin_grad = &bv_t_bin_grad[0];
    parameters.csum_dpu = &csum_dpu[0];
    parameters.dpu = &dpu[0];
    parameters.bu_t = &bu_t[0];
    parameters.bu_t_grad = &bu_t_grad[0];
    parameters.avg_dpu = &avg_dpu[0];
    parameters.alpha_u = &alpha_u[0];
    parameters.alpha_u_grad = &alpha_u_grad[0];
    parameters.sum_rpu = &sum_rpu[0];
    parameters.max_rpu = max_rpu;
    parameters.csum_mpu = &csum_mpu[0];
    parameters.mpu = &mpu[0];
    parameters.n_rpu = &n_rpu[0];
    parameters.n_rpdpu = &n_rpdpu[0];

    c_multiply_hogbatch_with_baseline1(
            &parameters)
    #c_multiply_hogbatch_with_baseline1(reg, eta, mu, n_samples, m, k, &users[0], n, &movies[0], t, &dates[0], &ratings[0], 
    #        batch_size, &Uindices[0], &Vindices[0], &Tindices[0], &Tbin_indices[0], &Udate_indices[0],
    #        Noob.Umat, Noob.Umat_grad, Noob.Vmat, Noob.Vmat_grad, &bu[0],
    #        &bugrad[0], &bv[0], &bvgrad[0], dates_per_bin, n_bins, &bv_t_bin[0,0], &bv_t_bin_grad[0], &csum_dates_per_user[0], &dates_per_user[0],
    #        &bu_t[0], &bu_t_grad[0], &avg_dpu[0], &alpha_u[0], &alpha_u_grad[0], &sum_rpu[0], &n_rpu[0])
    #c_multiply_hogbatch_with_baseline1(reg, eta, mu, n_samples, m, k, &users[0], n, &movies[0], t, &dates[0], &ratings[0], 
    #        batch_size, &Uindices[0], &Vindices[0], &Tindices[0], &Tbin_indices[0], &Udate_indices[0],
    #        &Umatrix[0,0], &Umatrixgrad[0], &Vmatrix[0,0], &Vmatrixgrad[0], &bu[0],
    #        &bugrad[0], &bv[0], &bvgrad[0], dates_per_bin, n_bins, &bv_t_bin[0,0], &bv_t_bin_grad[0], &csum_dates_per_user[0], &dates_per_user[0],
    #        &bu_t[0], &bu_t_grad[0], &avg_dpu[0], &alpha_u[0], &alpha_u_grad[0], &sum_rpu[0], &n_rpu[0])

    return
