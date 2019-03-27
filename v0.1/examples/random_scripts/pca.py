"""
Takes files multiple files of specified format and performs a PCA analysis. Intended use is to perform PCA on the PN's from the antennal lobe.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import itertools
import glob


'''
These functions can be used if one wants to bin the spike numbers first. Suggested time window for bin is 1/LFP freq: 50 ms or so.

"""
Counts the number of spikes in a time series.  Works well
for data that isn't very noisy and consists of Na spikes
"""
def num_spikes(V, spike_thresh = 0):
    return np.sum(sp.logical_and(V[:-1] <
                  spike_thresh, V[1:]>= spike_thresh))

"""
Returns array of network activity defined by number of spikes
in a time dt = bin_size.
"""
def bins(data, bin_size):
    num_points = len(data)
    num_bins = num_points/bin_size
    split = np.split(data, num_bins)
    return np.apply_along_axis(num_spikes, 1, split)

'''

#run command

#folder with exported data
folder_prefix = ''
num_odors = 2  # Different spatial injections
num_conc = 1  # Different current amplitudes
num_trials = 1  # trials will get averaged -- useful in stochastic network

# load data
tot_data = []
# Loads all files
for i in sorted(glob.glob('{0}PN*od*inj*'.format(folder_prefix))):
    name = i
    tot_data.append(np.load(name))

# print(np.shape(tot_data))
single_ts = len(tot_data[0][0])


# Average multiple trials of same odor/conc value
#if num_trials > 1:

avg_data = []
for k in range(int(len(tot_data)/num_trials)):
    tmp = tot_data[k*num_trials:num_trials + k*num_trials]
    avg_data.append(np.mean(tmp,axis=0))
# print(np.shape(avg_data))
data = np.hstack(avg_data)
# print(data.shape)
num_neurons, num_points = np.shape(data)

#average over 50 ms of network activity
#bin_size = 50 #ms
dt = 0.02 # ms
#bin_size_pts = bin_size/dt

t = np.arange(0., num_points*dt, dt)

# Get the binned data from the time series
# dataBinned = np.apply_along_axis(bins, 1, data, bin_size_pts)

k = 3 #three principle components
# X = dataBinned.T
# print(X.shape)

# svd decomposition and extract eigen-values/vectors
pca = PCA(n_components=k)
pca.fit(data.T)
w = pca.explained_variance_
v = pca.components_

# Save the pca data into each odor/conc
Xk = pca.transform(data.T)
# Xorg = pca.transform(np.asarray(avg_data).T)
for i in range(num_odors):
	for j in range(num_conc):
		np.savetxt('{2}odor{0}_conc{1}.txt'.format(i,j,folder_prefix), Xk[(i*num_conc+j)*single_ts:(i*num_conc+j+1)*single_ts])
# print(Xk.shape)
