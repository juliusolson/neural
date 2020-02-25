import numpy as np
import h5py
import csv

f = h5py.File("MNISTdata_1.hdf5", "r")
xtrain = f["x_train"][:]
ytrain = f["y_train"][:]
xtest = f["x_test"][:]
ytest = f["y_test"][:]

np.savetxt("xtrain.csv", xtrain, delimiter=",")
np.savetxt("ytrain.csv", ytrain, delimiter=",")
np.savetxt("xtest.csv", xtest, delimiter=",")
np.savetxt("ytest.csv", ytest, delimiter=",")

