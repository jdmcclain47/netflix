import reader
import numpy as np

data = reader.read_data()
n_samples = data.shape[0]
data_idx = reader.read_data_idx()
data_idx = data_idx[:n_samples]

# Extract user, movie, etc. data from dataframe
user_id, movie_id, date_id, rating = data[:,0], data[:,1], data[:,2], data[:,3]

#t0 = time.time()
#valid_data = get_valid_data(data,data_idx)
#print "*** Time ('get valid data') = ", time.time() - t0, " seconds ***"

# Find metadata (could store this). Takes a while to compute.
#n_users = len(set(user_id))
#n_movies = len(set(movie_id))
#n_dates = len(set(date_id))
#print " :: Number of data points   = %3d" % n_samples
#print " :: Number of unique users  = %3d" % n_users
#print " :: Number of unique movies = %3d" % n_movies
#print " :: Number of unique dates  = %3d" % n_dates


