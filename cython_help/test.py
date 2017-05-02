import numpy as np
import sgd

m = 20
n = 20
k = 10
users = np.arange(20)
movies = np.arange(20)
ratings = np.arange(20)
ratings = ratings.astype(np.float)
U = np.random.random((m,k))
V = np.random.random((n,k))
reg = 1.0
eta = 1.0

print U
sgd.multiply(users,movies,ratings,U,V,reg,eta)
print U
#def multiply (np.ndarray[double, ndim=1, mode="c"] users not None,
#              np.ndarray[double, ndim=1, mode="c"] movies not None,
#              np.ndarray[double, ndim=1, mode="c"] ratings not None,
#              np.ndarray[double, ndim=2, mode="c"] Umatrix not None,
#              np.ndarray[double, ndim=2, mode="c"] Vmatrix not None,
#              double reg,
#              double eta
#             ):
