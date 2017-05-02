import reader
import numpy as np
import run_svd
import time

data = reader.read_data()
n_samples = data.shape[0]
data_idx = reader.read_data_idx()
data_idx = data_idx[:n_samples]

# Extract user, movie, etc. data from dataframe
user_id, movie_id, date_id, rating = data[:,0], data[:,1], data[:,2], data[:,3]
# Easier to use rating as a float to avoid implicit conversion
rating = rating.astype(np.float)
data = None

t0 = time.time()
base_idx = data_idx == reader.BASE
base_user_id, base_movie_id, base_rating = [x[base_idx] for x in user_id, movie_id, rating]
print "*** Time ('get base data') = ", time.time() - t0, " seconds ***"

print "Number of samples for BASE case: ", base_user_id.shape
# Could just set this manually given the number of users/movies rather than looping...
M, N = [max(x)+1 for x in base_user_id, base_movie_id]
K = 20
print "M, N, K = ", M, N, K

U, V = \
run_svd.run( M, N, K,
             eta=0.002,
             reg=1e-10,
             user_id  = base_user_id,
             movie_id = base_movie_id,
             rating   = base_rating,
             eps=1e-5,
             max_epochs = 50)

suffix = "_M%d_N%d_K%d" % (M,N,K)
U.tofile("U" + suffix)
V.tofile("V" + suffix)

# Find metadata (could store this). Takes a while to compute.
#n_users = len(set(user_id))
#n_movies = len(set(movie_id))
#n_dates = len(set(date_id))
#print " :: Number of data points   = %3d" % n_samples
#print " :: Number of unique users  = %3d" % n_users
#print " :: Number of unique movies = %3d" % n_movies
#print " :: Number of unique dates  = %3d" % n_dates


