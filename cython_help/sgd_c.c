#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

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

inline void grad_U(int k, double* Ui, double Yij, double* Vj, double reg, double eta){
    int i, stride;
    double ui;
    double diff = 0.0;
    stride = 1;

    diff += Yij;
    //diff -= ddot(&k, &Ui, &stride, &Vj, &stride);
    for (i = 0; i < k; ++i){
        diff -= Ui[i]*Vj[i];
    }
    diff *= eta;

    for (i = 0; i < k; ++i){
        Ui[i] = (1-reg*eta)*Ui[i] + Vj[i]*diff;
    }
}

inline void grad_V(int k, double* Ui, double Yij, double* Vj, double reg, double eta){
    int i, stride;
    double vj, ui;
    double diff = 0.0;
    stride = 1;

    diff += Yij;
    //diff -= ddot(&k, &Ui, &stride, &Vj, &stride);
    for (i = 0; i < k; ++i){
        diff -= Ui[i]*Vj[i];
    }
    diff *= eta;

    for (i = 0; i < k; ++i){
        Vj[i] = (1-reg*eta)*Vj[i] + Ui[i]*diff;
    }
}

// This does something slightly different from just grad_U or grad_V alone.  Before,
// we were, for a single sample, updating the user matrix (given the user matrix, movie matrix, 
// rating) and then updating the movie matrix (given the *new* user matrix, movie matrix, rating).
// This of course means we have to calculate two similar dot products two separate times.
// Here we now update the movie matrix in a new manner (given the *old* user matrix, movie matrix,
// rating).
inline void update_U_and_V(int k, double* Ui, double Yij, double* Vj, double reg, double eta){
    int i, stride;
    double vj, ui;
    double diff = 0.0;
    stride = 1;

    diff += Yij;
    //diff -= ddot(&k, &Ui, &stride, &Vj, &stride);
    for (i = 0; i < k; ++i){
        diff -= Ui[i]*Vj[i];
    }
    diff *= eta;

    for (i = 0; i < k; ++i){
        vj = Vj[i];
        ui = Ui[i];
        Vj[i] = (1-reg*eta)*vj + ui*diff;
        Ui[i] = (1-reg*eta)*ui + vj*diff;
    }
}

inline void grad_U_and_V(double* out_grad_Ui, double* out_grad_Vj, int k, double* Ui, double Yij, double* Vj, double reg, double eta){
    int i, stride;
    double vj, ui;
    double diff = 0.0;
    stride = 1;

    diff += Yij;
    //diff -= ddot(&k, &Ui, &stride, &Vj, &stride);
    for (i = 0; i < k; ++i){
        diff -= Ui[i]*Vj[i];
    }
    diff *= eta;

    for (i = 0; i < k; ++i){
        vj = Vj[i];
        ui = Ui[i];
        out_grad_Vj[i] += reg*eta*vj - ui*diff;
        out_grad_Ui[i] += reg*eta*ui - vj*diff;
    }
}

double c_multiply(int n_samples, 
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
                  ){
    double result;
    long user, movie;
    double Yij;
    result = 0.0;
    int i;

    for (i = 0; i < n_samples; ++i){
        user   = u[i];
        movie  = v[i];
        Yij = r[i];
        //Ui = &Umat[user*k];
        //Vj = &Vmat[movie*k];
        update_U_and_V(k,&Umat[user*k],Yij,&Vmat[movie*k],reg,eta);
    }

    return result;
}

double c_multiply_hogwild(int n_samples, 
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
                  ){
    double result;
    long user, movie;
    double Yij;
    int i, idx;
    result = 0.0;
    long u_, m_;

#pragma omp parallel
{
    // Technically, since we are doing work on the same
    // Umat and Vmat over multiple threads, there is some
    // race condition.  But I just chalk this up to the
    // stochastic nature of the SGD.  Sometimes there are 
    // updates, and sometimes there aren't.
    //
    // This method it referred to as "Hogwild SGD" (I
    // didn't make up the name for it...)
    #pragma omp for schedule(dynamic) \
     private(user,movie,Yij)
    //#pragma omp for schedule(static) \
    // private(user,movie,Yij)
    for (i = 0; i < n_samples; ++i){
        // New Method
        user   = u[i];
        movie  = v[i];
        Yij = r[i];
        update_U_and_V(k,&Umat[user*k],Yij,&Vmat[movie*k],reg,eta);
    }
}
    return result;
}

// **WARNING** This is super-slow.  Rather than try and speed it
// up I just used a different algorithm...
double c_multiply_minibatch(int n_samples, 
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
                  double* Umat_work, // work space for U gradients
                  double* Vmat_work, // work space for V gradients
                  long* Umat_indices, // list of indices for non-zero elements
                  long* Vmat_indices, // list of indices for non-zero elements
                  int size
                  ){
    double result;
    long user, movie, idx;
    double Yij;
    long feature, st;
    int i, j, index;
    int thread;
    int n_batches = n_samples/size;
    result = 0.0;

#pragma omp parallel \
 private(st,user,movie,Yij,idx,j,feature,index)
{
    int n_threads = omp_get_num_threads();
    int my_thread = omp_get_thread_num();
    long offset;
    long rel_idx;

    for (st = 0; st < n_batches; ++st){
        offset = st*size;
        #pragma omp for schedule(dynamic)
        for (index = 0; index < size; ++index){
            //printf("Index , n_batches= %d %d \n", index, n_batches);
            user   = u[offset + index];
            movie  = v[offset + index];
            Yij    = r[offset + index];

            // Get index of work space and put in the 'um' index for
            // where the non-zero value occurs and the value inside
            // Umat_work/Vmat_work arrays
            Umat_indices[index] = user;
            Vmat_indices[index] = movie;
            grad_U_and_V(&Umat_work[index*k],&Vmat_work[index*k],k,&Umat[user*k],Yij,&Vmat[movie*k],reg,eta);
        }

        // Look at all possible users
        #pragma omp for schedule(static)
        for (user = 0; user < m; ++user){
            // Now loop over all updates of 'size'
            for (index = 0; index < size; ++index){
                if (Umat_indices[index] == user){ // We found a match! Update.
                    for (j = 0; j < k; ++j ){
                        Umat[user*k + j] -= Umat_work[index*k + j];
                    }
                }
            }
        }

        // Look at all possible movies
        #pragma omp for schedule(static)
        for (movie = 0; movie < n; ++movie){
            // Now loop over all updates of 'size'
            for (index = 0; index < size; ++index){
                if (Vmat_indices[index] == movie){ // We found a match! Update.
                    for (j = 0; j < k; ++j ){
                        Vmat[movie*k + j] -= Vmat_work[index*k + j];
                    }
                }
            }
        }
    }
}
    return result;
}

double c_multiply_hogbatch(int n_samples, 
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
                  double* Umat_work, // work space for U gradients
                  double* Vmat_work, // work space for V gradients
                  long* Umat_indices, // list of indices for non-zero elements
                  long* Vmat_indices, // list of indices for non-zero elements
                  int size
                  ){
    double result;
    long user, movie, idx;
    double Yij;
    long feature, st;
    int i, j, index;
    int thread;
    int n_batches = n_samples/size;
    result = 0.0;

#pragma omp parallel \
 private(st,user,movie,Yij,idx,j,feature,index)
{
    int n_threads = omp_get_num_threads();
    int my_thread = omp_get_thread_num();
    long sample_offset, thread_offset, tid, utid, mtid;
    long user_unique, movie_unique;
    int found;
    thread_offset = my_thread*size;

    #pragma omp for schedule(dynamic)
    for (st = 0; st < n_batches; ++st){
        sample_offset = st*size;
        for (index = 0; index < size; ++index){
            tid = thread_offset + index;
            for (j = 0; j < k; ++j){
                Umat_work[tid*k + j] = 0.0;
            }
            for (j = 0; j < k; ++j){
                Vmat_work[tid*k + j] = 0.0;
            }
            Umat_indices[tid] = -1;
            Vmat_indices[tid] = -1;
        }
        user_unique = size;
        movie_unique = size;
        // Uncomment these lines if you wish to do the 'sparse updates' defined below
        //user_unique = 0;
        //movie_unique = 0;
        for (index = 0; index < size; ++index){
            user   = u[sample_offset + index];
            movie  = v[sample_offset + index];
            Yij    = r[sample_offset + index];

            // Make an update at location 'tid'.  Note that if we have updated either
            // the user or movie in a previous iteration then utid or mtid may change.
            tid = thread_offset + index;
            utid = tid;
            mtid = tid;

            // Instead of updating a sparse vector like the commented out lines above,
            // we just push user/movie onto our vector of indices.
            Umat_indices[thread_offset + index] = user;
            Vmat_indices[thread_offset + index] = movie;

            // The following didn't really affect the performance too much. Tried to
            // take advantage of sparsity: if out of 100 samples only 70 users were
            // unique, then you would only have to update 70 entries instead of 100.
            // Also this cut down on the number of false cache line sharing - instances
            // where one thread updates a cache line that another has stored in memory.
            // But the optimal batch size of around 30 for large k meant that 
            // sparsity couldn't really be leveraged, so the following only served to slow
            // things down (though negligibly).
            //
            // Find if we have updated this user.  Note it's faster to not use break.
            //found = 0;
            //for (j = 0; j < user_unique; ++j)
            //    if (Umat_indices[thread_offset + j] == user){
            //        utid = thread_offset + j;
            //        found = 1;
            //        //break;
            //    }
            //if (!found){
            //    user_unique += 1;
            //    Umat_indices[thread_offset + j] = user;
            //}
            //// Find if we have updated this movie.  Note it's faster to not use break.
            //found = 0; 
            //for (j = 0; j < movie_unique; ++j)
            //    if (Vmat_indices[thread_offset + j] == movie){ 
            //        mtid = thread_offset + j;
            //        found = 1;
            //        //break;
            //    }
            //if (!found){
            //    movie_unique += 1;
            //    Vmat_indices[thread_offset + j] = movie;
            //}
            grad_U_and_V(&Umat_work[utid*k],&Vmat_work[mtid*k],k,&Umat[user*k],Yij,&Vmat[movie*k],reg,eta);
        }
        // Add contribution of gradient from unique users
        for (index = 0; index < user_unique; ++index){
            utid = thread_offset + index;
            user = Umat_indices[utid];
            for (j = 0; j < k; ++j ){
                Umat[user*k + j]  -= Umat_work[utid*k + j];
            }
        }
        // Add contribution of gradient from unique movies
        for (index = 0; index < movie_unique; ++index){
            mtid = thread_offset + index;
            movie = Vmat_indices[mtid];
            for (j = 0; j < k; ++j ){
                Vmat[movie*k + j]  -= Vmat_work[mtid*k + j];
            }
        }
    }
}
    return result;
}

double c_err(int n_samples,
             long* u, 
             long* v, 
             double* r,
             double* Umat,
             double* Vmat,
             int m,
             int n,
             int k
        ){
    double err = 0.0;
    double uvdot;
    long user, movie;
    double Yij;
    int i;

#pragma omp parallel
{
    int j;
    #pragma omp for schedule(static) \
     private (user, movie, Yij, uvdot) \
     reduction(+:err)
    for (i = 0; i < n_samples; ++i){
        user   = u[i];
        movie  = v[i];
        Yij = r[i];
        uvdot = 0.0;
        for (j = 0; j < k; ++j){
            uvdot += Umat[user*k + j]*Vmat[movie*k + j];
        }
        uvdot -= Yij;
        uvdot *= uvdot;
        err += uvdot;
    }
}
    err *= 0.5;
    return err;
}

