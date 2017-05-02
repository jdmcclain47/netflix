import numpy as np
import time
import os.path

# Location and name of file to read
root="./data/"
fileroot="um/"

def read_data():
    print "Reading in data..."
    filename="all.dta"
    #filename="10000.dta"
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

def read_data_idx():
    # The indices for each training set are in the README:
    #    1: base (96% of the training set picked at random, use freely)
    #    2: valid (2% of the training set picked at random, use freely)
    #    3: hidden (2% of the training set picked at random, use freely)
    #    4: probe (1/3 of the test set picked at random, use freely but carefully)
    #    5: qual (2/3 of the test set picked at random, for testing the results)
    t0 = time.time()
    filename="all.idx"
    #filename="10000.idx"
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

def get_base_data(data=None, data_idx=None):
    return get_data_from_id(BASE, data, data_idx)

def get_valid_data(data=None, data_idx=None):
    return get_data_from_id(VALID, data, data_idx)

def get_hidden_data(data=None, data_idx=None):
    return get_data_from_id(HIDDEN, data, data_idx)

def get_probe_data(data=None, data_idx=None):
    return get_data_from_id(PROBE, data, data_idx)

def get_qual_data(data=None, data_idx=None):
    return get_data_from_id(QUAL, data, data_idx)

def get_data_from_id(id_, data=None, data_idx=None):
    if data is None:
        data = read_data()
    n_samples = data.shape[0]
    if data_idx is None:
        data_idx = read_data_idx()
    data_idx = data_idx[:n_samples]
    return data[data_idx==id_]

