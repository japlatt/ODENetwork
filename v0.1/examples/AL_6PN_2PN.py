'''
This is a test example of a small antennal lobe. It consists of two
inhibitory local neurons and two projection neurons. The archtecture
of the network can be seen in Bazhenov's 2001 paper. This small network
demonstrates key characteristics in a large antennal lobe: (1) competition
between mutually inhibited local neurons and (2) transient synchrony of
projection neurons with the local field potential. Connections here are
made manually as random connections on such a small network cause issue
'''

# begin boiler plate for compatibility
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import sys

# This is a janky solution to import the modules. We will need to go back
# and make this a proper package
sys.path.append('..')

import lab_manager as lm

import scipy as sp
import numpy as np
from jitcode import jitcode, y, t
try:
    import symengine as sym_backend
except:
    import sympy as sym_backend
import networkx as nx
import random
import networks as net
import neuron_models as nm
import experiments as ex
from itertools import chain

#First number is #LNs, second is #PNs
neuron_nums = [2, 6]

AL = net.create_AL_man(nm.LN, nm.PN_2, nm.Synapse_gaba_LN, nm.Synapse_nAch_PN_2,gPNLN=400,gLN=110,gLNPN = 200,gPN=300)

#AL = net.create_AL_man(nm.LN, nm.PN_2, nm.Synapse_gaba_LN, nm.Synapse_nAch_PN_2)
#Set up the experiment
num_layers = 2

#The value of the input current is 400 pA
val = 250 #nA
#val = 400
#These are the neuron indicies within each layer which receive the current
neuron_inds=[[0,1],[1,3]]
current_vals = [[val,val],[val,val]]

# Set up the experiment with a constant input current
ex.const_current(AL, num_layers, neuron_inds, current_vals)

#set up the lab
f, initial_conditions, neuron_inds  = lm.set_up_lab(AL)

#run for specified time with dt
time_len = 1000.0 #Run for 600 ms
dt = 0.02 # Integration time step 0.02 ms
time_sampled_range = np.arange(0., time_len, dt)

# Run the lab and get output
data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5',compile=True)

#save PN data for analysis
V_vec = []
all_neurons = AL.layers[1].nodes()
for (n, neuron) in enumerate(all_neurons):
    ii = neuron.ii
    V_vec.append(data[:,ii])
#np.save('AL_data_62', V_vec)


# This is code generated to plot this specific example. There are other plotting
# functions within the lab_manager file.
#lm.show_all_neuron_in_layer(time_sampled_range,data,AL,1)
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
cycol = cycle(['C0','C1'])

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.style.use('ggplot')
n0 = AL.layers[0].nodes()
node_list = list(AL.layers[1].nodes())
node_list = node_list[1:5:2] #Only use first two neurons
fig, axes = plt.subplots(4,1,sharex='all',figsize=(8,10))
for (n, neuron) in enumerate(n0):
    ii = neuron.ii
    v_m = data[:,ii]
    axes[n].plot(time_sampled_range, v_m, label="Local Neuron %d"%(neuron.ni+1),color=next(cycol))
    axes[n].set_ylabel(r"$V_m$ [mV]")
    axes[-1].set_xlabel("Time [ms]")
    axes[n].set_xlim(time_sampled_range[0],time_sampled_range[-1])
    axes[n].set_title("Local Neuron %d"%(neuron.ni+1))

for (n, neuron) in enumerate(node_list):
    ii = neuron.ii
    v_m = data[:,ii]
    axes[n+2].plot(time_sampled_range, v_m, label=r"Projection Neuron %d"%(neuron.ni-1),color=next(cycol))
    axes[n+2].set_ylabel(r"$V_m$ [mV]")
    axes[n+2].set_title("Projection Neuron %d"%(neuron.ni-1))

plt.show()
