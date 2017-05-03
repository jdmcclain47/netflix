import reader
import numpy as np
import run_svd
import time
import sys

# User arguments for dimension K and regularizer 10**(reg)
K = int(sys.argv[1])
reg = int(sys.argv[2])

data = reader.read_data()
n_samples = data.shape[0]
data_idx = reader.read_data_idx()
data_idx = data_idx[:n_samples]
data_idx = np.squeeze(data_idx)

# Extract user, movie, etc. data from dataframe
user_id, movie_id, date_id, rating = data[:,0], data[:,1], data[:,2], data[:,3]
user_id  -= 1
movie_id -= 1
# Easier to use rating as a float to avoid implicit conversion
rating = rating.astype(np.float)
data = None

t0 = time.time()
base_idx = data_idx == reader.BASE
base_user_id, base_movie_id, base_rating = [x[base_idx] for x in user_id, movie_id, rating]
print "*** Time ('get base data') = ", time.time() - t0, " seconds ***"

print "Number of samples for BASE case: ", base_user_id.shape
# Could just set this manually given the number of users/movies rather than looping...
# Add 1 since we made the id's have 0-indexing
M, N = [np.amax(x)+1 for x in base_user_id, base_movie_id]
print "M, N, K = ", M, N, K

U, V = \
run_svd.run( M, N, K,
             eta=0.003,
             reg=10**(reg),
             user_id  = base_user_id,
             movie_id = base_movie_id,
             rating   = base_rating,
             eps=1e-6,
             max_epochs = 50)

suffix = "_K%d_reg%d.dat" % (M,N,K,reg)
U.tofile("U" + suffix)
V.tofile("V" + suffix)

# TODO : the following should probably go elsewhere... but I'll leave it here for now
# Make qual set
t0 = time.time()
qual_idx = data_idx == reader.QUAL
qual_user_id, qual_movie_id, qual_rating = [x[qual_idx] for x in user_id, movie_id, rating]
print "*** Time ('get qual data') = ", time.time() - t0, " seconds ***"

print "Number of samples for qual case: ", qual_user_id.shape
#suffix = "_K%d_reg%d.dat" % (M,N,K,reg)
U = np.fromfile("U" + suffix)
V = np.fromfile("V" + suffix)
U = np.reshape(U, (M,K))
V = np.reshape(V, (N,K))

def predict(user_id, movie_id, U, V):
    n_samples = user_id.shape[0]
    r = np.empty((n_samples,))
    for i in range(n_samples):
        u, m = user_id[i], movie_id[i]
        # TODO: define the following in maybe some sort of model class so we
        # don't have to set r_pred for every new model used...
        r_pred = np.dot(U[u,:],V[m,:])
        r[i] = r_pred
    return r

def rms_error(pred_r, r):
    n_samples = r.shape[0]
    return np.sum((pred_r-r)**2) / n_samples

# Trim predictions above 5.0 to be 5
# and those below 1.0 to be just 1
def trim_pred(pred_r):
    rating = pred_r.copy()
    below_one = rating < 1.0
    rating[below_one] = 1.0
    above_five = pred_r > 5.0
    rating[above_five] = 5.0
    return rating

pred_rating = predict(qual_user_id, qual_movie_id, U, V)
pred_rating = trim_pred(pred_rating)
outfile = "qual" + suffix
np.savetxt(outfile, pred_rating, fmt='%.3f')
