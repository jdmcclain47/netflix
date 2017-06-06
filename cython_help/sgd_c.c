#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <assert.h>
#define ASSERT(EX) if(!(EX)) {fprintf(stdout,"%s:%i Initialization error (" #EX "): "  "\nAborting :-(\n",__FILE__,__LINE__);};

struct model_params {
    // Parameters to be optimized, their gradients,
    // and indices of their arrays
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

    // Regularization parameters
    double* reg;
    double* eta;
    double mu;
    double freq_base;

    // Size of various arrays
    int batch_size;
    long n_samples;
    long dates_per_bin, n_bins;
    long n_dpu;
    long n_freq;
    int m, n, k, t;
    
    // User-related data
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

    // Input data
    long* u;
    long* v;
    long* d;
    double* r;
};

long return_m2_(struct model_params* mpm){
    return pow(mpm->m,2);
}

struct lvw {
    int n_rows;
    int n_cols;
    long* v;
    long* grad;
};

void shuffle(long *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          long t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void permutation(long* indices, long n, long blk_size){
    int n_batches, mod, i;
    long offset, index;
    size_t size;
    n_batches = n / blk_size;
    mod = n - n_batches * blk_size;
#pragma omp parallel \
 private(offset)
{
    int n_threads = omp_get_num_threads();
    int my_thread = omp_get_thread_num();
    #pragma omp for schedule(static)
    for (i = 0; i < n_batches; ++i){
        offset = i*blk_size;
        size = blk_size;
        shuffle(&indices[offset],size);
    }
    offset = n_batches*blk_size;
    shuffle(&indices[offset],mod);
}
}

inline void grad_U_and_V_baseline1(
        long user, long movie, long date, float rating, long bdate, long user_date_id, long freq_id, long tid, struct model_params* mpm, 
        double* yv_grad, long* yv_indices_grad, long* c_idx, double* yu, long* nshuffle, double* tmp_k_work
        ){
    int k = (*mpm).k;
    int i;
    double vj, ui, vf;
    double diff;
    double* reg = (*mpm).reg;
    double* eta = (*mpm).eta;
    double mu = (*mpm).mu;
    double freq_base = (*mpm).freq_base;
    double inv_sqrt_n_rpu, aku;
    long upper_limit;
    long n_rpu = (*mpm).n_rpu[user];
    long n_r, i_rpu, tmp_mov;
    long max_rpu = (*mpm).max_rpu;
    long tmp_movie, offset;
    double rhs;
    long lmidx, umidx, midx;
    long n_freq = (*mpm).n_freq;
    double yv, re12;

    long sign = (date > (*mpm).avg_dpu[user]) - ((*mpm).avg_dpu[user] > date);
    double dev_u = sign * pow(sign*(date - (*mpm).avg_dpu[user]),0.4);

    double bv_t_bin = (*mpm).bv_t_bin[movie*(*mpm).n_bins + bdate];
    double bv =       (*mpm).bv[movie];
    double bvf = (*mpm).bvf[movie*n_freq + freq_id];
    double bu = (*mpm).bu[user];
    double cu_t = (*mpm).cu_t[user_date_id];
    double cu = (*mpm).cu[user];
    double bu_t = (*mpm).bu_t[user_date_id];
    double alpha_u = (*mpm).alpha_u[user];

    diff = rating - mu;
    diff += - bu;
    diff += - (bv + bv_t_bin) * (cu + cu_t);
    diff += - bu_t;
    diff += - alpha_u * dev_u;
    diff += - bvf;

    inv_sqrt_n_rpu = 1.;
    if (n_rpu > 0){ inv_sqrt_n_rpu = 1./pow(n_rpu,0.5);}

    lmidx = (*mpm).csum_mpu[user];
    umidx = (*mpm).csum_mpu[user+1];
    for (i = 0; i < k; ++i){
        yu[i] = 0.0;
    }
    for (midx = lmidx; midx < umidx; ++midx){
        tmp_movie = (*mpm).mpu[midx];
        offset = tmp_movie*k;
        for (i = 0; i < k; ++i){
            yu[i] += (*mpm).yv[offset + i];
        }
    }
    for (i = 0; i < k; ++i){
        yu[i] *= inv_sqrt_n_rpu;
    }

    for (i = 0; i < k; ++i){
        diff -= ( (*mpm).Umat[user*k + i] + yu[i] + (*mpm).alpha_ku[user*k + i]*dev_u) * ((*mpm).Vmat[movie*k + i] + (*mpm).Vfmat[movie*k*n_freq + k*freq_id + i]);
    }

    offset = tid*k;
    for (i = 0; i < k; ++i){
        vj = (*mpm).Vmat[movie*k + i];
        ui = (*mpm).Umat[user*k + i];
        aku = (*mpm).alpha_ku[user*k + i];
        vf = (*mpm).Vfmat[movie*k*n_freq + k*freq_id + i];

        (*mpm).Umat_grad    [offset + i] = eta[2]*(reg[2]*ui  - (vj + vf) * diff);
        (*mpm).alpha_ku_grad[offset + i] = eta[3]*(reg[3]*aku - dev_u * (vj + vf) * diff);
        (*mpm).Vmat_grad    [offset + i] = eta[0]*(reg[0]*vj  - (ui + yu[i] + aku*dev_u) * diff);
        (*mpm).Vfmat_grad   [offset + i] = eta[1]*(reg[1]*vf  - (ui + yu[i] + aku*dev_u) * diff);
    }
   
    // Beginning of baseline derivative
    (*mpm).bvf_grad[tid]  = eta[4]*(reg[4]*bvf - diff);
    (*mpm).bu_grad[tid]   = eta[5]*(reg[5]*bu  - diff);

    rhs = (bv + bv_t_bin) * diff;
    (*mpm).cu_grad[tid]   = eta[6]*(reg[6]*(cu-1.) - rhs);
    (*mpm).cu_t_grad[tid] = eta[7]*(reg[7]*cu_t    - rhs);

    rhs = (cu + cu_t) * diff;
    (*mpm).bv_grad[tid]       = eta[8]  * (reg[8]*bv       - rhs);
    (*mpm).bv_t_bin_grad[tid] = eta[9]  * (reg[9]*bv_t_bin - rhs);
    (*mpm).bu_t_grad[tid]     = eta[10] * (reg[10]*bu_t    - diff);
    (*mpm).alpha_u_grad[tid]  = eta[11] * (reg[11]*alpha_u - dev_u * diff);
    // End of baseline derivative


    lmidx = (*mpm).csum_mpu[user];
    umidx = (*mpm).csum_mpu[user+1];
    (*nshuffle) = umidx - lmidx;
    //if ((*nshuffle) > 40){
    //    (*nshuffle) = 40;
    //}
    upper_limit = lmidx + (*nshuffle);
    for (i = 0; i < k; ++i){
        vj = (*mpm).Vmat[movie*k + i];
        vf = (*mpm).Vfmat[movie*k*n_freq + k*freq_id + i];
        tmp_k_work[i] = eta[12] * (vj + vf) * inv_sqrt_n_rpu * diff;
    }
    re12 = eta[12] * reg[12];
    
    for (midx = lmidx; midx < upper_limit; ++midx){
        tmp_movie = (*mpm).mpu[midx];
        offset = tmp_movie*k;
        if (yv_indices_grad[tmp_movie] == 0){ // Do assignment
            for (i = 0; i < k; ++i){
                yv = (*mpm).yv[offset + i];
                yv_grad[offset + i] = re12 * yv - tmp_k_work[i];
            }
            yv_indices_grad[tmp_movie] += 1;
        }
        else{ // Do addition
            for (i = 0; i < k; ++i){
                yv = (*mpm).yv[offset + i];
                yv_grad[offset + i] += re12 * yv - tmp_k_work[i];
            }
            yv_indices_grad[tmp_movie] += 1;
        }
    }
}

long binary_search(long x, long* arr, long m, long n){
    long middle = (m+n)/2;
    if (arr[middle] == x){
        return middle;
    }else if(arr[middle] < x){
        return binary_search(x, arr, middle, n);
    }else if(arr[middle] > x){
        return binary_search(x, arr, m, middle);
    }
    printf("Not found! \n");
    return -1;
}

void reduce_list( int k, double* arr, long* arr_idx, long arr_size, double* out_arr, long* out_arr_idx, long* out_arr_size ){
    long i, j, m;
    long size = 0;
    for(i = 0; i < arr_size; i++) {
        for (j=0; j<i; j++){
            if (arr_idx[i] == arr_idx[j]){
                for (m=0; m < k; ++m){
                    arr[j*k + m] += arr[i*k + m];
                }
                break;
            }
        }
            
        if (i == j){
            /* No duplicate element found between index 0 to i */
            out_arr_idx[size] = arr_idx[j];
            for (m=0; m < k; ++m){
                out_arr[size*k + m] = arr[j*k + m];
            }
            size += 1;
        }
    }
    *out_arr_size = size;
}

void c_multiply_hogbatch_with_baseline1(
        struct model_params* mpm
        ){
    long user, movie, date, idx;
    double rating;
    long feature, ibatch;
    int i, j, index;
    
    // Declaring all variables from model_params
    double* Umat = (*mpm).Umat;
    double* Umat_grad = (*mpm).Umat_grad;
    double* Vmat = (*mpm).Vmat;
    double* Vmat_grad = (*mpm).Vmat_grad;
    double* Vfmat = (*mpm).Vfmat;
    double* Vfmat_grad = (*mpm).Vfmat_grad;
    long* user_indices_grad = (*mpm).user_indices_grad;
    long* movie_indices_grad = (*mpm).movie_indices_grad;
    long* time_indices_grad = (*mpm).time_indices_grad;
    long* time_bin_indices_grad = (*mpm).time_bin_indices_grad;
    long* user_date_indices_grad = (*mpm).user_date_indices_grad;
    long* freq_indices_grad = (*mpm).freq_indices_grad;
    double* bvf = (*mpm).bvf;
    double* bvf_grad = (*mpm).bvf_grad;
    double* bu = (*mpm).bu;
    double* bu_grad = (*mpm).bu_grad;
    double* alpha_ku = (*mpm).alpha_ku;
    double* alpha_ku_grad = (*mpm).alpha_ku_grad;
    double* yu = (*mpm).yu;
    double* yu_grad = (*mpm).yu_grad;
    double* yv = (*mpm).yv;
    //double* yv_grad = (*mpm).yv_grad;
    double* bv = (*mpm).bv;
    double* bv_grad = (*mpm).bv_grad;
    double* reg = (*mpm).reg;
    double* eta = (*mpm).eta;
    double mu = (*mpm).mu;
    double freq_base = (*mpm).freq_base;
    long n_samples = (*mpm).n_samples;
    int batch_size = (*mpm).batch_size;
    int m = (*mpm).m;
    int k = (*mpm).k;
    int n = (*mpm).n;
    int t = (*mpm).t;
    long* u = (*mpm).u;
    long* v = (*mpm).v;
    long* d = (*mpm).d;
    double* r = (*mpm).r;
    long dates_per_bin = (*mpm).dates_per_bin;
    long n_bins = (*mpm).n_bins;
    double* bv_t_bin = (*mpm).bv_t_bin;
    double* bv_t_bin_grad = (*mpm).bv_t_bin_grad;
    long* csum_dpu = (*mpm).csum_dpu;
    long* dpu = (*mpm).dpu;
    double* bu_t = (*mpm).bu_t;
    double* bu_t_grad = (*mpm).bu_t_grad;
    long* avg_dpu = (*mpm).avg_dpu;
    double* alpha_u = (*mpm).alpha_u;
    double* alpha_u_grad = (*mpm).alpha_u_grad;
    double* sum_rpu = (*mpm).sum_rpu;
    long max_rpu = (*mpm).max_rpu;
    long* csum_mpu = (*mpm).csum_mpu;
    long* mpu = (*mpm).mpu;
    long* n_rpu = (*mpm).n_rpu;
    long* n_rpdpu = (*mpm).n_rpdpu;
    double* cu = (*mpm).cu;
    double* cu_grad = (*mpm).cu_grad;
    double* cu_t = (*mpm).cu_t;
    double* cu_t_grad = (*mpm).cu_t_grad;
    int n_batches = n_samples/batch_size;

#pragma omp parallel \
 private(ibatch,user,movie,date,rating,idx,j,feature,index)
{

    int n_threads = omp_get_num_threads();
    int my_thread = omp_get_thread_num();
    long sample_offset, thread_offset, tid, utid, mtid, dtid, user_date_id;
    long user_unique, movie_unique, bdate;
    int found;
    long other_user, iuser, isample;
    long offset;
    long freq_id;
    long n_freq = (*mpm).n_freq;
    long lmidx, umidx, midx;
    long luidx, uuidx, uidx;
    long max_rpu_iter;
    double fac;
    double* yv_tmp;
    thread_offset = my_thread*batch_size;
    //double* yv_grad = calloc(batch_size * k * max_rpu, sizeof(double));
    double* yv_grad = calloc(n * k, sizeof(double));
    double* yv_grad_red = calloc(batch_size * k * max_rpu, sizeof(double));
    long* user_updated = calloc( m, sizeof(long));
    double* yu_tmp  = calloc(k * m, sizeof(double));
    //long* yv_indices_grad = calloc(batch_size * max_rpu, sizeof(long));
    long* yv_indices_grad = calloc(n, sizeof(long));
    long* yv_indices_grad_red = calloc(batch_size * max_rpu, sizeof(long));
    double* tmp_k_work = calloc(k, sizeof(double));
    double* yu = calloc(k, sizeof(double));
    long nshuffle = 0;
    long ii;

    //#pragma omp single
    //for (iuser = 0; iuser < m; ++iuser){
    //    for (j = 0; j < k; ++j){
    //        (*mpm).yu[iuser*k + j] = 0.0;
    //    }
    //}
    //#pragma omp single
    //for (isample = 0; isample < n_samples; ++isample){
    //    user = u[isample];
    //    movie = v[isample];
    //    for (j = 0; j < k; ++j){
    //        (*mpm).yu[user*k + j] += yv[movie*k + j];
    //    }
    //}
    //#pragma omp single
    //for (iuser = 0; iuser < m; ++iuser){
    //    fac = 1.;
    //    if (n_rpu[iuser] > 0){ fac = 1./pow(n_rpu[iuser],0.5);}
    //    for (j = 0; j < k; ++j){
    //        (*mpm).yu[iuser*k + j] *= fac;
    //    }
    //}

    max_rpu_iter = batch_size * max_rpu;
    long update_size = n_threads;
    #pragma omp for schedule(dynamic,6)
    for (ibatch = 0; ibatch < n_batches; ++ibatch){
        //if (ibatch % update_size == 0 && ibatch > 0){
        //    for (index = 0; index < m; ++index){
        //        user_updated[index] = 0;
        //    }
        //    for (iuser = 0; iuser < m; ++iuser){
        //        for (j = 0; j < k; ++j){
        //            yu_tmp[iuser*k + j] = 0.0;
        //        }
        //    }
        //    long lower_bound = ((ibatch/update_size) - 1) * batch_size;
        //    long upper_bound = lower_bound + batch_size*update_size;
        //    for (midx = lower_bound; midx < upper_bound; ++midx){
        //        movie = v[midx];
        //        luidx = (*mpm).csum_upm[movie];
        //        uuidx = (*mpm).csum_upm[movie+1];
        //        for (ii = luidx; ii < uuidx; ++ii){
        //            user = (*mpm).upm[ii];
        //            user_updated[user] += 1;
        //        }
        //        for (ii = luidx; ii < uuidx; ++ii){
        //            user = (*mpm).upm[ii];
        //            if (user_updated[user] > 0){
        //                fac = 1.;
        //                if (n_rpu[user] > 0){ fac = 1./pow(n_rpu[user],0.5);}
        //                for (j = 0; j < k; ++j){
        //                    yu_tmp[user*k + j] += yv[movie*k + j] * fac;
        //                }
        //                for (j = 0; j < k; ++j){
        //                    (*mpm).yu[user*k + j] = yu_tmp[user*k + j];
        //                }
        //            }
        //        }
        //    }
        //}

        for (index = 0; index < n; ++index){
            yv_indices_grad[index] = 0;
        }

        max_rpu_iter = 0;
        sample_offset = ibatch*batch_size;
        for (index = 0; index < batch_size; ++index){
            user   = u[sample_offset + index];
            movie  = v[sample_offset + index];
            date   = d[sample_offset + index];
            rating = r[sample_offset + index];
            bdate  = date / dates_per_bin;

            // Make an update at location 'tid'.  Note that if we have updated either
            // the user or movie in a previous iteration then utid or mtid may change.
            tid = thread_offset + index;

            // Instead of updating a sparse vector like the commented out lines above,
            // we just push user/movie onto our vector of indices.
            user_indices_grad[tid] = user;
            movie_indices_grad[tid] = movie;
            time_indices_grad[tid] = date;
            time_bin_indices_grad[tid] = bdate;

            user_date_id = binary_search(date, &dpu[0], csum_dpu[user], csum_dpu[user+1]);
            user_date_indices_grad[tid] = user_date_id;

            freq_id = n_rpdpu[user_date_id];
            freq_indices_grad[tid] = freq_id;

            grad_U_and_V_baseline1( user, movie, date, rating, bdate, user_date_id, freq_id, tid, mpm, &yv_grad[0], 
                    &yv_indices_grad[0], &max_rpu_iter, &yu[0], //&(*mpm).yu[user*k],
                    &nshuffle, &tmp_k_work[0]);
            //max_rpu_iter += npu[user];
        }

        // Add contribution of gradient from unique users
        for (index = 0; index < batch_size; ++index){
            utid = thread_offset + index;
            user = user_indices_grad[utid];
            for (j = 0; j < k; ++j ){
                Umat[user*k + j] -= Umat_grad[utid*k + j];
            }
        }
        for (index = 0; index < batch_size; ++index){
            utid = thread_offset + index;
            user = user_indices_grad[utid];
            bu[user] -= bu_grad[utid];
        }
        for (index = 0; index < batch_size; ++index){
            utid = thread_offset + index;
            user = user_indices_grad[utid];
            alpha_u[user] -= alpha_u_grad[utid];
            cu[user] -= cu_grad[utid];
            for (j = 0; j < k; ++j ){
                alpha_ku[user*k + j] -= alpha_ku_grad[utid*k + j];
            }
        }
        for (index = 0; index < batch_size; ++index){
            utid = thread_offset + index;
            user_date_id = user_date_indices_grad[utid];
            bu_t[user_date_id] -= bu_t_grad[utid];
            cu_t[user_date_id] -= cu_t_grad[utid];
        }
        
        // Add contribution of gradient from unique movies
        for (index = 0; index < batch_size; ++index){
            mtid = thread_offset + index;
            movie = movie_indices_grad[mtid];
            user = user_indices_grad[mtid];
            for (j = 0; j < k; ++j ){
                Vmat[movie*k + j] -= Vmat_grad[mtid*k + j];
            }
        }
        for (index = 0; index < batch_size; ++index){
            mtid = thread_offset + index;
            movie = movie_indices_grad[mtid];
            bv[movie] -= bv_grad[mtid];
            
            bdate  = time_bin_indices_grad[mtid];
            bv_t_bin[movie*n_bins + bdate] -= bv_t_bin_grad[mtid];

            freq_id = freq_indices_grad[mtid];
            bvf[movie*n_freq + freq_id] -= bvf_grad[mtid];

            for (j = 0; j < k; ++j ){
                Vfmat[movie*k*n_freq + k*freq_id + j] -= Vfmat_grad[mtid*k + j];
            }
        }
        for (midx = 0; midx < n; ++midx){
            offset = midx*k;
            if (yv_indices_grad[midx] > 0){
                for (j = 0; j < k; ++j){
                    yv[offset + j] -= yv_grad[offset + j];
                }
            }
        }
    }
}
return;
}

double c_rms_err(long n_samples,
                 double* r_pred,
                 double* r
        ){
    long isample;
    double uvdot;
    double err = 0.0;
#pragma omp parallel
{
    #pragma omp for schedule(dynamic) \
     private (uvdot) \
    reduction(+:err)
    for (isample = 0; isample < n_samples; ++isample){
        uvdot = r_pred[isample];
        uvdot -= r[isample];
        uvdot *= uvdot;
        err += uvdot;
    }
}
err /= n_samples;
err = sqrt(err);
return err;
}

double c_training_err(
             long n_samples,
             long* u, 
             long* v, 
             long* d,
             double* r,
             struct model_params* mpm
        ){
    double err = 0.0;
    int m = (*mpm).m;
    int n = (*mpm).n;
    int k = (*mpm).k;
#pragma omp parallel
{
    long isample;
    double uvdot;
    long user, movie, date, bdate, user_date_id, rating;
    double Yij;
    int i;
    int j;
    long sign, avg_d;
    long freq_id;
    long n_freq = (*mpm).n_freq;
    long iuser;
    double dev_u, fac;

    #pragma omp single
    for (iuser = 0; iuser < m; ++iuser){
        for (j = 0; j < k; ++j){
            (*mpm).yu[iuser*k + j] = 0.0;
        }
    }
    #pragma omp single
    for (isample = 0; isample < n_samples; ++isample){
        user = u[isample];
        movie = v[isample];
        for (j = 0; j < k; ++j){
            (*mpm).yu[user*k + j] += (*mpm).yv[movie*k + j];
        }
    }
    #pragma omp single
    for (iuser = 0; iuser < m; ++iuser){
        fac = 1.;
        if ((*mpm).n_rpu[iuser] > 0){ fac = 1./pow((*mpm).n_rpu[iuser],0.5);}
        for (j = 0; j < k; ++j){
            (*mpm).yu[iuser*k + j] *= fac;
        }
    }

    #pragma omp for schedule(static) \
     private (user, movie, date, bdate, Yij, uvdot) \
     reduction(+:err)
    for (i = 0; i < n_samples; ++i){
        user   = u[i];
        movie  = v[i];
        date   = d[i];
        rating = r[i];
        bdate  = date / (*mpm).dates_per_bin;
        user_date_id = binary_search(date, &((*mpm).dpu)[0], (*mpm).csum_dpu[user], (*mpm).csum_dpu[user+1]);
        freq_id = (*mpm).n_rpdpu[ user_date_id ];
        avg_d = (*mpm).avg_dpu[user];

        sign = (date > avg_d) - (avg_d > date);
        dev_u = sign * pow(sign*(date - avg_d),0.4);

        uvdot = - rating;
        uvdot += (*mpm).mu;
        uvdot += (*mpm).bu[user];
        uvdot += ((*mpm).bv[movie] + (*mpm).bv_t_bin[movie*(*mpm).n_bins + bdate]) * ((*mpm).cu[user] + (*mpm).cu_t[user_date_id]);
        uvdot += (*mpm).bu_t[user_date_id];
        uvdot += (*mpm).alpha_u[user] * dev_u;
        uvdot += (*mpm).bvf[movie*n_freq + freq_id];
        for (j = 0; j < k; ++j){
            uvdot += ((*mpm).Umat[user*k + j] + (*mpm).yu[user*k + j] + (*mpm).alpha_ku[user*k + j]*dev_u) * ((*mpm).Vmat[movie*k + j] + (*mpm).Vfmat[movie*k*n_freq + freq_id*k + j]);
        }
        uvdot *= uvdot;
        err += uvdot;
    }
}
err /= n_samples;
err = sqrt(err);
return err;
}

void c_predict(
             long n_samples,
             long* u, 
             long* v, 
             long* d,
             struct model_params* mpm,
             double* r_out
        ){
    double err = 0.0;
    int m = (*mpm).m;
    int n = (*mpm).n;
    int k = (*mpm).k;

#pragma omp parallel
{
    double uvdot;
    long user, movie, date, bdate, user_date_id;
    double Yij;
    int i;
    int j;
    long sign, avg_d;
    long freq_id;
    long n_freq = (*mpm).n_freq;
    long iuser, isample;
    double dev_u, fac;

    #pragma omp single
    for (iuser = 0; iuser < m; ++iuser){
        for (j = 0; j < k; ++j){
            (*mpm).yu[iuser*k + j] = 0.0;
        }
    }
    #pragma omp single
    for (isample = 0; isample < n_samples; ++isample){
        user = u[isample];
        movie = v[isample];
        for (j = 0; j < k; ++j){
            (*mpm).yu[user*k + j] += (*mpm).yv[movie*k + j];
        }
    }
    #pragma omp single
    for (iuser = 0; iuser < m; ++iuser){
        fac = 1.;
        if ((*mpm).n_rpu[iuser] > 0){ fac = 1./pow((*mpm).n_rpu[iuser],0.5);}
        for (j = 0; j < k; ++j){
            (*mpm).yu[iuser*k + j] *= fac;
        }
    }

    #pragma omp for schedule(static) \
     private (user, movie, date, bdate, Yij, uvdot) 
    for (i = 0; i < n_samples; ++i){
        user   = u[i];
        movie  = v[i];
        date   = d[i];
        bdate  = date / (*mpm).dates_per_bin;
        user_date_id = binary_search(date, &((*mpm).dpu)[0], (*mpm).csum_dpu[user], (*mpm).csum_dpu[user+1]);
        freq_id = (*mpm).n_rpdpu[ user_date_id ];
        avg_d = (*mpm).avg_dpu[user];

        sign = (date > avg_d) - (avg_d > date);
        dev_u = sign * pow(sign*(date - avg_d),0.4);

        uvdot = (*mpm).mu;
        uvdot += (*mpm).bu[user];
        uvdot += ((*mpm).bv[movie] + (*mpm).bv_t_bin[movie*(*mpm).n_bins + bdate]) * ((*mpm).cu[user] + (*mpm).cu_t[user_date_id]);
        uvdot += (*mpm).bu_t[user_date_id];
        uvdot += (*mpm).alpha_u[user] * dev_u;
        uvdot += (*mpm).bvf[movie*n_freq + freq_id];
        for (j = 0; j < k; ++j){
            uvdot += ((*mpm).Umat[user*k + j] + (*mpm).yu[user*k + j] + (*mpm).alpha_ku[user*k + j]*dev_u) * ((*mpm).Vmat[movie*k + j] + (*mpm).Vfmat[movie*k*n_freq + freq_id*k + j]);
        }
        r_out[i] = uvdot;
    }
}
    return;
}
