# might be useful for mac user, uncommnet below if needed
# import matplotlib
# matplotlib.use("TKAgg")

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
'''Counts the number of spikes in a time series.  Works well
for data that isn't very noisy and consists of Na spikes'''
def num_spikes(V, spike_thresh = 0):
    return np.sum(sp.logical_and(V[:-1] <
                  spike_thresh, V[1:]>= spike_thresh))

'''Returns array of network activity defined by number of spikes
in a time dt = bin_size.'''
def bins(data, bin_size):
    num_points = len(data)
    num_bins = num_points/bin_size
    split = np.split(data, num_bins)
    return np.apply_along_axis(num_spikes, 1, split)

#run command
#python pca.py Data/filenames.txt

#import filenames by loading in file
#filenames = np.loadtxt(sys.argv[1], dtype = str)

#folder with exported data
prefix = 'results/'
num_odors = 2
num_conc = 4
num_trials = 10
#load data
tot_data = []
for i in sorted(glob.glob('results/AL*_od*_inj*')):
    print(i)
    name = i
    tot_data.append(np.load(name))

print(np.shape(tot_data))
print(len(tot_data))
avg_data = []
for k in range(int(len(tot_data)/num_trials)):
    tmp = tot_data[k*num_trials:num_trials + k*num_trials]
    avg_data.append(np.mean(tmp,axis=0))
print(np.shape(avg_data))
data = np.hstack(avg_data)
print(data.shape)
num_neurons, num_points = np.shape(data)

#average over 50 ms of network activity
#bin_size = 50 #ms
dt = 0.02
#bin_size_pts = bin_size/dt

t = np.arange(0., num_points*dt, dt)

#get the binned data from the time series
#dataBinned = np.apply_along_axis(bins, 1, data, bin_size_pts)

k = 3 #three principle components
#X = dataBinned.T
#print(X.shape)
#svd decomposition and extract eigen-values/vectors
pca = PCA(n_components=k)
pca.fit(data.T)
w = pca.explained_variance_
v = pca.components_

Xk = pca.transform(data.T)
#Xorg = pca.transform(np.asarray(avg_data).T)

np.savetxt('projected_AL_30-90.txt', Xk)
print(Xk.shape)
