"""
used to test stdp learning behavior
"""
%matplotlib inline
# begin boiler plate for compatibility
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import sys

# This is a janky solution to import the modules. We will need to go back
# and make this a proper package
sys.path.append('..')

import numpy as np
import networks as ns
import neuron_models as nm
import experiments as ex
import lab_manager as lm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

from importlib import reload
reload(nm)
reload(lm)
reload(ex)

# Step 1: Pick a network
neuron_nums = [1,1] # number of neurons in each layer
NUM_NEURON = np.sum(neuron_nums)
NUM_SYN = np.prod(neuron_nums)
neuron_type = nm.Soma
NUM_DIM_NEURON = neuron_type.DIM
##### change your synapse class here
synapse_type = nm.SynapseWithDendrite
NUM_DIM_SYN = synapse_type.DIM

net = ns.get_multilayer_fc(neuron_type, synapse_type, neuron_nums)
total_time = 1000.
time_sampled_range = np.arange(0., total_time, 0.1)

def get_data(delta_time):

    # step 2: design an experiment
    T0 = 500.
    TIME_DELAY = delta_time
    ex.delay_pulses_on_layer_0_and_1(net, t0s=[T0, T0+TIME_DELAY], i_max=160.)
    # step 3: ask our lab manager to set up the lab for the experiment
    f, initial_conditions, neuron_inds = lm.set_up_lab(net)

    # step 4: run the lab and gather data

    data = lm.run_lab(f, initial_conditions, time_sampled_range)

    return data

# step 5: plot

delta_time = 100.
data = get_data(delta_time)
#dimension index of calcium and stdp weight
ca_index = 12
data_ca = data[:,NUM_NEURON*NUM_DIM_NEURON + ca_index]
p0_index = 13
p1_index = 14
data_p0 = data[:,NUM_NEURON*NUM_DIM_NEURON + p0_index]
data_p1 = data[:,NUM_NEURON*NUM_DIM_NEURON + p1_index]
data_p2 = 1 - data_p0 - data_p1
data_w = synapse_type.G0*data_p0 + synapse_type.G1*data_p1 + synapse_type.G2*data_p2


fig, axes = plt.subplots(2, 1, sharex = 'col', figsize = (8,5))
#axes = axes.reshape(2, 1)
fig.suptitle(r'$\Delta t =  {}(ms)$'.format(delta_time), fontsize = 22)
axes[0].plot(time_sampled_range, data_ca, color="blue", label="Ca", linewidth = 2)
#axes[0].axhline(nm.StdpSynapse.THETA_D, color="orange", label=r"$\theta_d$", linewidth = 2)
#axes[0].axhline(nm.StdpSynapse.THETA_P, color="green", label=r"$\theta_p$", linewidth = 2)
axes[0].legend(loc = 'best', frameon = False)
axes[1].plot(time_sampled_range, data_w, label="Connection Weight", linewidth = 2)
axes[1].legend(loc = 'best', frameon = False)
axes[1].set_xlabel("time [ms]")

plt.show()

# stdp profile

delta_time = 1.
DT = np.linspace(-50,50,10)
DW = np.zeros(len(DT))

for (i,dt) in enumerate(DT):
    data = get_data(dt)
    #dimension index of calcium and stdp weight
    data_ca = data[:,NUM_NEURON*NUM_DIM_NEURON + ca_index]
    data_p0 = data[:,NUM_NEURON*NUM_DIM_NEURON + p0_index]
    data_p1 = data[:,NUM_NEURON*NUM_DIM_NEURON + p1_index]
    data_p2 = 1 - data_p0 - data_p1
    data_w = synapse_type.G0*data_p0 + synapse_type.G1*data_p1 + synapse_type.G2*data_p2
    DW[i] = (data_w[-1] - data_w[0])

plt.figure()
plt.plot(DT,DW,marker=".",color="black")
plt.ylabel(r"$\Delta W$")
#plt.yticks([])
plt.xlabel(r"$\Delta t$[ms]")
plt.show()
# plt.savefig("stdp.png",dpi=500)

for layer_idx in range(len(net.layers)):
    lm.show_all_neuron_in_layer(
        time_sampled_range, data, net, layer_idx)
    lm.show_all_synaspe_onto_layer(
        time_sampled_range, data, net, layer_idx)
