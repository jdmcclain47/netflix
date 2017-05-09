import reader
import numpy as np
import run_svd
import model_helper as mh
import time
import sys

# User arguments for dimension K and regularizer 10**(reg)
K = int(sys.argv[1])
reg = int(sys.argv[2])

# Load full data (for filebase: use "all" for all data, "10000000" for a smaller reduced sample)
#filebase="10000000"
filebase="all"
user_id, movie_id, date_id, rating, data_idx = reader.get_umdr_and_idx(filebase)
print "Number of total samples: ", user_id.shape
M, N = [np.amax(x)+1 for x in user_id, movie_id]
print "M, N = ", M, N

# Define for saving/loading data
suffix = "_K%d_reg%d.dat" % (K,reg)
########################################
#                                      #
# Train data                           #
#                                      #
########################################

# Get training data
base_user_id, base_movie_id, base_data_id, base_rating = \
        reader.get_base_umdr(user_id, movie_id, date_id, rating, data_idx)
print "Number of samples for BASE case: ", base_user_id.shape

# Train model
U, V = \
run_svd.run( M, N, K,
             eta=0.004,
             reg=10**(reg),
             user_id  = base_user_id,
             movie_id = base_movie_id,
             rating   = base_rating,
             eps=1e-6,
             max_epochs = 50)

# Save model from run
mh.save_um_data(suffix, U, V)

########################################
#                                      #
# Get validation error                 #
#                                      #
########################################

# Make validation
valid_user_id, valid_movie_id, valid_data_id, valid_rating = \
        reader.get_valid_umdr(user_id, movie_id, date_id, rating, data_idx)
print "Number of samples for valid case: ", valid_user_id.shape

# Load data in
U, V = mh.load_um_data(suffix = "_K%d_reg%d.dat" % (K,reg), U_shape=(M,K), V_shape=(N,K))

# Run on validation set
pred_rating = mh.predict(valid_user_id, valid_movie_id, U, V)
pred_rating = mh.trim_pred(pred_rating)
rms_error = mh.rms_err(pred_rating, valid_rating)
print "validation error = %.15g" % rms_error

########################################
#                                      #
# Get probe error                      #
# NOTE: validation error seems to be   #
#       very low, whereas probe seems  #
#       to be around the quiz error..  #
#       Maybe use probe for choosing   #
#       models?                        #
########################################

# Make probe
probe_user_id, probe_movie_id, probe_data_id, probe_rating = \
        reader.get_probe_umdr(user_id, movie_id, date_id, rating, data_idx)
print "Number of samples for probe case: ", probe_user_id.shape

# Load data in
U, V = mh.load_um_data(suffix = "_K%d_reg%d.dat" % (K,reg), U_shape=(M,K), V_shape=(N,K))

# Run on probeation set
pred_rating = mh.predict(probe_user_id, probe_movie_id, U, V)
pred_rating = mh.trim_pred(pred_rating)
rms_error = mh.rms_err(pred_rating, probe_rating)
print "probe error = %.15g" % rms_error

########################################
#                                      #
# Make qual set                        #
#                                      #
########################################

## Make qual set
#qual_user_id, qual_movie_id, qual_data_id, qual_rating = \
#        reader.get_qual_umdr(user_id, movie_id, date_id, rating, data_idx)
#print "Number of samples for qual case: ", qual_user_id.shape
#
## Load data in
#U, V = mh.load_um_data(suffix, U_shape=(M,K), V_shape=(N,K))
#
## Run on qual set
#pred_rating = mh.predict(qual_user_id, qual_movie_id, U, V)
#pred_rating = mh.trim_pred(pred_rating)
#outfile = "qual" + suffix
#np.savetxt(outfile, pred_rating, fmt='%.3f')
