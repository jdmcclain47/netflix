import time
import numpy as np

def Yij_diff(Ui, Yij, Vj, mu=0.0, ai=0.0, bj=0.0):
    return ((Yij - mu) - (np.dot(Ui,Vj) + ai + bj))

def grad_U(Ui, Yij, Vj, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*Ui + eta * Vj * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

def grad_V(Vj, Yij, Ui, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*Vj + eta * Ui * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

def grad_a(Ui, Yij, Vj, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*ai + eta * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

def grad_b(Ui, Yij, Vj, reg, eta, mu=0.0, ai=0.0, bj=0.0):
    return (1-reg*eta)*bj + eta * Yij_diff(Ui,Yij,Vj,mu,ai,bj)

@profile
def get_err(U, V, user_id, movie_id, rating):
    #for u,m,r in zip(user_id,movie_id,rating):
    #    Ui, Vj, Yij = U[u,:], V[:,m], r
    #    err += 0.5 * Yij_diff(Ui,Yij,Vj)**2
    t0 = time.time()
    batch_size = int(1e6)
    n_samples = user_id.shape[0]
    n_batch = int( np.ceil(n_samples / batch_size) )
    print "Looping over %3d batches..." % n_batch
    err = 0.0
    for ibatch in range(n_batch):
        sls = slice(ibatch*batch_size,min((ibatch+1)*batch_size,n_samples))
        u_batch, m_batch, r_batch = user_id[sls], movie_id[sls], rating[sls]
        r_pred = np.einsum('ik,ki->i',U[u_batch,:],V[:,m_batch])
        r_batch -= r_pred
        r_batch = r_batch**2
        err += 0.5*np.sum(r_batch)
    print "*** Time ('get err') = ", time.time() - t0, " seconds ***"
    return err / n_samples

def run( size_M, size_N, size_K, eta, reg, user_id, movie_id, rating, eps, max_epochs):
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
    V = np.random.random((size_K,size_N))

    n_samples = rating.shape[0]
    n_epochs = max_epochs

    #data_triple = np.dstack((user_id, movie_id, rating))
    #data_triple = np.reshape(data_triple, (n_samples,3))
    _user_id, _movie_id, _rating = [user_id.copy(), movie_id.copy(), rating.copy()]

    print "Starting..."
    err0 = get_err(U,V,_user_id,_movie_id,_rating)
    print "Initial error = %.15g" % err0

    # Shuffle after 'n' iterations
    n_shuffle = 10

    for i_epoch in range(n_epochs):
        # Shuffle data
        if i_epoch % n_shuffle == 0:
            print "Shuffling data..."
            t0 = time.time()
            [np.random.shuffle(x) for x in _user_id, _movie_id, _rating]
            print "*** Time ('shuffing data') = ", time.time() - t0, " seconds ***"

        # Loop over shuffled data
        t0 = time.time()
        for _u, _m, _r in zip(_user_id, _movie_id, _rating):
            Ui, Vj, Yij = U[_u,:], V[:,_m], _r
            # Move this user by this gradient amount
            U[_u,:] = grad_U(Ui, Yij, Vj, reg, eta)
            # Move this movie by this gradient amount
            V[:,_m] = grad_V(Ui, Yij, Vj, reg, eta)
        print "*** Time ('epoch time') = ", time.time() - t0, " seconds ***"

        # Getting error
        err = get_err(U,V,_user_id,_movie_id,_rating)
        if i_epoch == 0:
            err1 = err
            print "Stopping condition err - err0 < (eps)*(err0 - err1) [eps,err0,err1] = %.12g %.12g %.12g" % (
                    eps, err0, err1 )
        print "Current error = %.12g" % err
        if abs(err-err0)/abs(err0-err1) < eps:
            print "Stopping criterion met! Exiting"
            return
