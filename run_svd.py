import time
import numpy as np
from cython_help import sgd_py
np.random.seed(0)

def Yij_diff(Ui, Yij, Vj, mu=0.0, ai=0.0, bj=0.0):
    return ((Yij - mu) - (np.dot(Ui,Vj) + ai + bj))

def grad_U(Ui, Yij, Vj, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*Ui + eta * Vj * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

def grad_V(Ui, Yij, Vj, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*Vj + eta * Ui * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

def grad_a(Ui, Yij, Vj, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*ai + eta * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

def grad_b(Ui, Yij, Vj, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*bj + eta * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

def get_err(U, V, user_id, movie_id, rating):
    t0 = time.time()
    batch_size = int(1e7)
    n_samples = user_id.shape[0]
    err = 0.0

    # Method 1
    #for u,m,r in zip(user_id,movie_id,rating):
    #    Ui, Vj, Yij = U[u,:], V[m,:], r
    #    err += 0.5 * Yij_diff(Ui,Yij,Vj)**2
    #print "*** Time ('get err') = ", time.time() - t0, " seconds ***"

    # Method 2
    #t0 = time.time()
    #n_batch = int( np.ceil(n_samples / batch_size) )
    #n_batch = max(n_batch,1)
    #print "Looping over %3d batches..." % n_batch
    #for ibatch in range(n_batch):
    #    sls = slice(ibatch*batch_size,min((ibatch+1)*batch_size,n_samples))
    #    u_batch, m_batch, r_batch = user_id[sls], movie_id[sls], rating[sls]
    #    r_pred = np.einsum('ik,ik->i',U[u_batch,:],V[m_batch,:])
    #    r_pred -= r_batch
    #    r_pred = r_pred**2
    #    err += 0.5*np.sum(r_pred)
    #print "*** Time ('get err') = ", time.time() - t0, " seconds ***"

    # Method 3
    t0 = time.time()
    err = sgd_py.err(user_id,
                     movie_id,
                     rating,
                     U,
                     V)
    print "*** Time ('get err') = ", time.time() - t0, " seconds ***"
    return err / n_samples

def run( size_M, size_N, size_K, eta, reg, user_id, movie_id, rating, eps, max_epochs, mu = 0.0):
    '''Runs a standard SGD

    size_M     = number of users
    size_N     = number of movies
    size_K     = size of hidden dimension
    eta        = controls step size for SGD
    reg        = regularization parameter
    user_id    = the user id for a given set of data
    movie_id   = the movie id for a given set of data
    rating     = the rating given for user_i, movie_j
    eps        = controls convergence
    max_epochs = max number of epochs you wish to run
    '''

    # Initialize variables
    U = np.random.random((size_M,size_K))
    V = np.random.random((size_N,size_K))
    U /= size_K
    V /= size_K

    old_U = U.copy()
    old_V = V.copy()

    n_samples = rating.shape[0]
    n_epochs = max_epochs

    _user_id, _movie_id, _rating = [user_id.copy(), movie_id.copy(), rating.copy()]

    print "Starting..."
    err0 = get_err(U,V,user_id,movie_id,rating)
    old_err = err0
    print "Initial error = %.15g" % err0

    # Shuffle after 'n' iterations
    n_shuffle = 10

    for i_epoch in range(n_epochs):

        #eta = eta / (1. + eta*i_epoch)**(2./3)

        # Shuffle data
        if i_epoch % n_shuffle == 0:
            print "Shuffling data..."
            t0 = time.time()
            shuffle = np.random.permutation(n_samples)
            _user_id, _movie_id, _rating = [x[shuffle] for x in _user_id, _movie_id, _rating]
            print "*** Time ('shuffling data') = ", time.time() - t0, " seconds ***"

        #sgd_py.shuffle(_user_id, _movie_id, _rating)

        t0 = time.time()
        # Loop over shuffled data
        sgd_py.multiply_parallel(_user_id,
                        _movie_id,
                        _rating,
                        U,
                        V,
                        reg,
                        eta)

        #for _u, _m, _r in zip(_user_id, _movie_id, _rating):
        #    Ui, Vj, Yij = U[_u,:], V[_m,:], _r
        #    # Move this user by this gradient amount
        #    U[_u,:] = grad_U(Ui, Yij, Vj, reg, eta)

        #    Ui, Vj, Yij = U[_u,:], V[_m,:], _r
        #    # Move this movie by this gradient amount
        #    V[_m,:] = grad_V(Ui, Yij, Vj, reg, eta)
        print "*** Time ('epoch time') = ", time.time() - t0, " seconds ***"

        # Getting error
        err = get_err(U,V,user_id,movie_id,rating)
        if i_epoch == 0:
            err1 = err
            print "Stopping condition err - err0 < (eps)*(err0 - err1) [eps,err0,err1] = %.12g %.12g %.12g" % (
                    eps, err0, err1 )
        print "Current error = %.12g" % err
        if abs(err-old_err)/abs(err0-err1) < eps:
            print "Stopping criterion met! Exiting"
            return U, V

        # If we don't decrease the error (i.e. our new error is higher),
        # then only move 20% in that direction
        if old_err < err:
            U = 0.2*U + 0.8*old_U
            V = 0.2*V + 0.8*old_V
        # Otherwise, we decrease the eta (stepsize) by 10%
        else:
            eta = 0.9*eta

        old_err = err
    print "Failed to converge! Max iterations reached."
    return U, V
