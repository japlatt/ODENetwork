"""
experiments.py

A collection of various experiments that you can do without killing an animal.
Feel free to put your experiment design here!

"""
import electrodes
from jitcode import t # symbolic time varibale, useful for defining currents
import numpy as np
import math

"""
a simple experiment playing with calcium-based STDP in a 2-layer FC network
"""
def delay_pulses_on_layer_0_and_1(net, i_max=50.):
    #i_max = 50. #5. # (some unit)
    t0 = 50. # ms
    dt = 10.
    w = 1. #ms
    for neuron in net.layers[0].nodes():
        neuron.i_inj = i_max*electrodes.unit_pulse(t,t0,w) # the jitcode t
    for neuron in net.layers[1].nodes():
        neuron.i_inj = i_max*electrodes.unit_pulse(t,t0+dt,w)

def constant_current_on_top_layer(net, i_max=50.):
    #i_max = 50. #5. # (some unit)
    t0 = 50. # ms
    dt = 10.
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
    g0 = np.clip(g0, 0., 1.)
    # gaussian 1
    i = 1
    mean = [x_means[i], y_means[i]]
    g1 = np.random.multivariate_normal(mean, cov, num_sample)
    g1 = np.clip(g1, 0., 1.)
    return g0, g1

def draw_from_gaussian_clusters(i, num_sample=1):
    x_means = [0.8, 0.2]
    y_means = [0.2, 0.8]
    cov = 0.01*np.diag([1., 1.])  # diagonal covariance
    mean = [x_means[i], y_means[i]]
    return np.clip(np.random.multivariate_normal(mean, cov, num_sample), 0., 1.)

def poisson_train(rates, time_total=100.):
    trains = []
    for rate in rates:
        intervals = np.random.exponential(1./rate, math.floor(time_total*rate))
        train = np.cumsum(intervals)
        # strictly enforcing the last spike has to be within time_total
        train = train[train <= time_total]
        trains.append(train)
    return trains

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

def feed_gaussian_rate_poisson_spikes(
    net, base_rate, i_max=50., num_sniffs=10, time_per_sniff=100.):
    #i_max = 50. #5. # (some unit)
    if len(net.layers[0].nodes()) != 2:
        print("not yet done")
        return
    # will draw one class at a time
    num_class = 2
    classes = np.random.randint(num_class, size=num_sniffs)
    t0 = 0.
    w = 1.
    for i in range(num_sniffs):
        c = classes[i]
        rates = base_rate*draw_from_gaussian_clusters(c)[0]
        i_injs, _ = get_poisson_spike_train(rates, t0=t0, time_total=time_per_sniff)
        t0 += time_per_sniff
        if len(i_injs) != len(net.layers[0].nodes()):
            print("Input dimension does not match!")
        for (n, neuron) in enumerate(net.layers[0].nodes()):
            neuron.i_inj += i_injs[n]
