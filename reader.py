import numpy as np
import time
import os.path

# Location and name of file to read
root="/scratch/gpfs/jmcclain/netflix/data/"
#root="./data/"
fileroot="um/"

def read_data(filebase="all"):
    print "Reading in data..."
    filename = filebase + ".dta"
    # If binary numpy file exists, read it as int32; otherwise
    # read as numpy csv file and convert to np.int
    if os.path.isfile(root + fileroot + filename + "_np"):
        t0 = time.time()
        data = np.fromfile(root + fileroot + filename + "_np", dtype=np.int32)
        data = np.reshape(data, (-1,4))
        print "*** Time ('read data non-pandas') = ", time.time() - t0, " seconds ***"
    else:
        import pandas as pd
        t0 = time.time()
        data = pd.read_csv(root + fileroot + filename, delimiter=' ', dtype=np.int32, header=None)
        print "*** Time ('read data pandas') = ", time.time() - t0, " seconds ***"
        data = np.asarray(data, dtype=np.int32)
        data.tofile(root + fileroot + filename + "_np")
    t0 = time.time()
    data = data.astype(np.int)
    print "*** Time ('convert array type') = ", time.time() - t0, " seconds ***"
    return data

def read_data_idx(filebase="all"):
    # The indices for each training set are in the README:
    #    1: base (96% of the training set picked at random, use freely)
    #    2: valid (2% of the training set picked at random, use freely)
    #    3: hidden (2% of the training set picked at random, use freely)
    #    4: probe (1/3 of the test set picked at random, use freely but carefully)
    #    5: qual (2/3 of the test set picked at random, for testing the results)
    t0 = time.time()
    filename = filebase + ".idx"
    read_success = False
    # If binary numpy file exists, read it as int8; otherwise
    # read as numpy csv file and convert to np.int
    if os.path.isfile(root + fileroot + filename + "_np"):
        read_success = True
        t0 = time.time()
        data_idx = np.fromfile(root + fileroot + filename + "_np", dtype=np.int8)
        print "*** Time ('idx read non-pandas') = ", time.time() - t0, " seconds ***"
    else:
        import pandas as pd
        t0 = time.time()
        data_idx = pd.read_csv(root + fileroot + filename, delimiter=' ', dtype=np.int8, header=None)
        print "*** Time ('idx read pandas') = ", time.time() - t0, " seconds ***"
        data_idx = np.asarray(data_idx, dtype=np.int8)
        data_idx.tofile(root + fileroot + filename + "_np")
    t0 = time.time()
    data_idx = data_idx.astype(np.int)
    print "*** Time ('convert array type') = ", time.time() - t0, " seconds ***"
    return data_idx

# The ratings are given for the probe (idx=4) set but are given
# a rating of 0 for the qual (idx=5) set.
BASE   = 1
VALID  = 2
HIDDEN = 3
PROBE  = 4
QUAL   = 5

# Get user, movie, date, rating
def get_umdr_and_idx(filebase="all"):
    data = read_data(filebase)
    n_samples = data.shape[0]
    data_idx = read_data_idx(filebase)
    data_idx = data_idx[:n_samples]
    data_idx = np.squeeze(data_idx)

    # 0-indexing for user, movie
    data[:,:2] -= 1
    # Extract user, movie, etc. data from dataframe
    user_id, movie_id, date_id, rating = [data[:,x] for x in range(4)]
    # Easier to use rating as a float to avoid implicit conversion
    rating = rating.astype(np.float)
    data = None
    return user_id, movie_id, date_id, rating, data_idx

def get_base_umdr(u, m, d, r, data_idx):
    return get_umdr_from_id(BASE, u, m, d, r, data_idx)

def get_valid_umdr(u, m, d, r, data_idx):
    return get_umdr_from_id(VALID, u, m, d, r, data_idx)

def get_hidden_umdr(u, m, d, r, data_idx):
    return get_umdr_from_id(HIDDEN, u, m, d, r, data_idx)

def get_probe_umdr(u, m, d, r, data_idx):
    return get_umdr_from_id(PROBE, u, m, d, r, data_idx)

def get_qual_umdr(u, m, d, r, data_idx):
    return get_umdr_from_id(QUAL, u, m, d, r, data_idx)

def get_umdr_from_id(id_, u, m, d, r, data_idx):
    idx = data_idx == id_
    out = [x[idx] for x in u, m, d, r]
    return out
