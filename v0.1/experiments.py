"""
experiments.py

A collection of various experiments that you can do without killing an animal.
Feel free to put your experiment design here!

TODO: jitcode requires symbolic inputs to integrate. How to we generate a series
of input currents (stored as data elsewhere), and use that with jitcode?

"""
import numpy as np
import math
import scipy as sp
import pickle
import os.path
import time
from struct import unpack

import electrodes
from jitcode import t # symbolic time varibale, useful for defining currents




"""
a simple experiment playing with calcium-based STDP in a 2-layer FC network
"""

def pulse_on_layer(net, layer_idx, t0=50., i_max=50.,w=1.):
    for (i,neuron) in enumerate(net.layers[layer_idx].nodes()):
        neuron.i_inj = i_max*electrodes.unit_pulse(t,t0,w) # the jitcode t

def pulse_train_on_layer(net, layer_idx, t0s, i_max=50.,w=1.):
    i_inj = i_max*sum(electrodes.unit_pulse(t,t0,w) for t0 in t0s)

    for (i,neuron) in enumerate(net.layers[layer_idx].nodes()):
        neuron.i_inj = i_inj # the jitcode t

def delay_pulses_on_layer_0_and_1(net, t0s=[0., 20], i_max=55., w = 1.0):
    #i_max = 50. #5. # (some unit)
    # t0=50. # ms
    #dts = 10.
    # w = 1. #ms
    for (i,neuron) in enumerate(net.layers[0].nodes()):
        neuron.i_inj = i_max*electrodes.unit_pulse(t,t0s[0],w) # the jitcode t
    for neuron in net.layers[1].nodes():
        neuron.i_inj = i_max*electrodes.unit_pulse(t,t0s[1],w)

def constant_current_on_top_layer(net, i_max=50.,w=1.):
    #i_max = 50. #5. # (some unit)
    t0 = 50. # ms
    w = 1. #ms
    for neuron in net.layers[0].nodes():
        neuron.i_inj = i_max

def get_gaussian_clusters(num_cluster=2, num_dim=2, num_sample=100):
    if num_cluster > 2:
        print("not yet done")
        pass
    x_means = [0.8, 0.2]
    y_means = [0.2, 0.8]
    cov = 0.01*np.diag([1., 1.])  # diagonal covariance
    # gaussian 0
    i = 0
    mean = [x_means[i], y_means[i]]
    g0 = np.random.multivariate_normal(mean, cov, num_sample)
    g0 = np.clip(g0, 0.1, 1.)
    # gaussian 1
    i = 1
    mean = [x_means[i], y_means[i]]
    g1 = np.random.multivariate_normal(mean, cov, num_sample)
    g1 = np.clip(g1, 0.1, 1.)
    return g0, g1

def draw_from_gaussian_clusters(i, num_sample=1):
    x_means = [0.8, 0.2]
    y_means = [0.2, 0.8]
    cov = 0.01*np.diag([1., 1.])  # diagonal covariance
    mean = [x_means[i], y_means[i]]
    return np.clip(np.random.multivariate_normal(mean, cov, num_sample), 0.1, 1.)

def poisson_train(rates, time_total=100.):
    trains = []
    for rate in rates:
        intervals = np.random.exponential(1./rate, math.floor(time_total*rate))
        train = np.cumsum(intervals)
        # strictly enforcing the last spike has to be within time_total
        train = train[train <= time_total]
        trains.append(train)
    return trains

"""
Helper fuction to feed_gaussian_rate_poisson_spikes()
"""
def get_poisson_spike_train(rates, t0=0., time_total=100., i_max=50., w=1.):
    #w = 1. #pules width ms
    i_injs = []
    trains = poisson_train(rates, time_total)
    for train in trains:
        train = train[train <= time_total]
        i_inj = sum(i_max*electrodes.unit_pulse(t,t0+t_spike,w) for t_spike in train)
        i_injs.append(i_inj)
    return i_injs, trains

# rates = [0.1, 0.01]
# time_sampled_range = np.arange(0., 100, 0.1)
# trains = poisson_train(rates, time_total=100.)
# len(trains[0])
# i_injs = get_poisson_spike_train(rates)
# i_injs[0]
# import matplotlib.pyplot as plt
# for i_inj in i_injs:
#     i_inj = electrodes.sym2num(t, i_inj)
#     i_inj = i_inj(time_sampled_range)

"""
Adds constant current to specified neurons
"""
def const_current(net, num_layers, neuron_inds, current_vals):
    for l in range(num_layers):
        layer = net.layers[l].nodes()
        layer_list = list(layer)
        for i in range(len(neuron_inds[l])):
            layer_list[neuron_inds[l][i]].i_inj = current_vals[l][i]

def gradual_current_increase(net, num_layers, neuron_inds, max_current,tr=20, tf=50, w=100):
    # Defining the stimuli structure. tr is rise time, tf is fall time, w is time
    # for constant current
    def gradual_stimulus(t,tr,tf,w):
        return 2*(electrodes.sigmoid(t*7.0/tr) - 0.5)*electrodes.sigmoid((w+tr-t)*7/tf)

    for l in range(num_layers):
        layer = net.layers[l].nodes()
        layer_list = list(layer)
        for i in range(len(neuron_inds[l])):
            layer_list[neuron_inds[l][i]].i_inj = gradual_stimulus(t,tr,tf,w)*max_current[l][i]



'''
Feeds in different classes of poisson spike trains

net: the network
base_rate: draw rates for the classes from gaussian around the mean 'base rate'
i_max: maximum current of the poisson spike
num_sniffs: number of times we feed in a random class
time_per_sniff: length of the spike train per class
'''
def feed_gaussian_rate_poisson_spikes(
    net, base_rate, i_max=50., num_sniffs=5, time_per_sniff=100.):
    # i_max = 50. #5. # (some unit)
    # if len(net.layers[0].nodes()) != 2:
    #     print("not yet done")
    #     return
    # will draw one class at a time
    num_class = len(net.layers[0].nodes())
    #Will return vector of classes. Feed in classes in random order
    classes = np.random.randint(num_class, size=num_sniffs)
    t0 = 0.
    w = 1.
    for i in range(num_sniffs):
        c = classes[i]
        rates = base_rate*draw_from_gaussian_clusters(c)[0]
        i_injs, _ = get_poisson_spike_train(rates, t0=t0, time_total=time_per_sniff,i_max=i_max)
        t0 += time_per_sniff
        if len(i_injs) != len(net.layers[0].nodes()):
            print("Input dimension does not match!")
        for (n, neuron) in enumerate(net.layers[0].nodes()):
            neuron.i_inj += i_injs[n]

def feed_gaussian_rate_poisson_spikes_DL(
    net, base_rate, i_max=50., num_sniffs=5, time_per_sniff=100.):
    # i_max = 50. #5. # (some unit)
    # if len(net.layers[0].nodes()) != 2:
    #     print("not yet done")
    #     return
    # will draw one class at a time
    num_class = 2
    num_input = len(net.layers[0].nodes())
    #Will return vector of classes. Feed in classes in random order
    classes = np.random.randint(num_class, size=num_sniffs)
    t0 = 0.
    w = 1.
    spatial_idex = np.random.randint(num_class, size=num_input)
    for i in range(num_sniffs):
        c = classes[i]
        rates = base_rate*draw_from_gaussian_clusters(c)[0]
        i_injs, _ = get_poisson_spike_train(rates, t0=t0, time_total=time_per_sniff,i_max=i_max)
        t0 += time_per_sniff
        for (n, neuron) in enumerate(net.layers[0].nodes()):
            class_num = spatial_idex[n]
            neuron.i_inj += i_injs[class_num]



#--------------------------------------------------------------------
#Inputs for MNIST dataset


def get_labeled_data(picklename, MNIST_data_path, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'), encoding='utf-8')
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"), -1)
    return data

# '''
# produce a binary spike train
# fr = frquency: Hz
# time_tot = total time: seconds
# dt: seconds
# nTrains =  number of spike trains
# '''
# def poisson(fr, time_tot, dt = 1e-3, nTrains = 1):
#     nBins = int(np.floor(time_tot/dt))
#     spikeMat = (np.random.rand(nTrains, nBins) < fr*dt).astype(int)
#     tVec = np.arange(0,time_tot, dt)
#     return tVec, spikeMat
# '''
# N = number of spiking neurons
# mu = average firing rate-Hz
# time_len = length of the input sequence-s
# dt = delta t - s
# '''
# def poisson_spike_sum(N, mu, time_len):
#     time, trains = poisson(mu, time_len, nTrains = N)
#     cum = np.sum(trains, axis = 0)
#     return time, cum


# '''rates: 28 x 28 dimensional input current'''
# def input_curr(rates, time, dt):
#     I = []
#     t_interp = np.arange(0, time, dt)
#     #number of RN per odour
#     nRNs = 200
#     for r in rates:
#         t, cum = poisson_spike_sum(nRNs, r, time, )
#         interp = np.interp(t_interp, t, cum)
#         I.append(interp)
#     return np.reshape(I, (28,28, len(t_interp)))
