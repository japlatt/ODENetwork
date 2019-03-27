import sys
sys.path.append('..')
import lab_manager as lm

import scipy as sp
import numpy as np
from jitcode import jitcode, y, t
import networkx as nx
import random
import networks as net
import neuron_models as nm
import experiments as ex
from itertools import chain
import pickle

# These need to be set manually
num_odors = 2
num_per_od = 1
num_trials = 1

folder_prefix = 'results/'
# This a function where you define how you want to scale the amplitude of constant current
def stim_amp(c_num):
	odors = [500,1000]
	return odors[c_num]

# Specify run time
time_len = 300.0
dt = 0.02
time_sampled_range = np.arange(0., time_len, dt)


# This is the bash t, it ranges from 1 - whatever
t = int(sys.argv[1]) - 1

# Identifies value for each
odor = int(np.floor(t/(num_per_od*num_trials)))
conc = int(np.floor(t/num_trials)%num_per_od)
trial = t % num_trials


# Load the previously created network and stimuli
network_pickle = open('{0}network.pickle'.format(folder_prefix),'rb')
AL = pickle.load(network_pickle)
network_pickle.close()


stimuli_pickle = open('{0}stimuli.pickle'.format(folder_prefix),'rb')
stim = pickle.load(stimuli_pickle)
stimuli_pickle.close()


# Set the current value -- assumes a dc current for all neurons
current_amp = stim_amp(conc)


# Numer of glomeruli
num_layers = len(AL.layers)



#adj_m = net.get_edges_data(AL, "weight")


curr_inds =  stim[0][odor]
curr_vals = current_amp*np.asarray(stim[1][odor])
stim = None
print('Odor: {0}, Concentration {1}, Trial {2}'.format(odor,conc,trial))
ex.const_current(AL, num_layers, curr_inds, curr_vals)

# set up the lab
f, initial_conditions, all_neuron_inds = lm.set_up_lab(AL)

# run the lab
data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5',compile=True)

data = np.transpose(data)

# Isolate Projection Neuron and Local Neuron indicies for use elsewhere
pn_inds = np.concatenate(np.array([[n.ii for n in AL.layers[j].layers[0].nodes()] for j in range(np.shape(AL.layers)[0])]),axis=0)
ln_inds = np.concatenate(np.array([[n.ii for n in AL.layers[j].layers[1].nodes()] for j in range(np.shape(AL.layers)[0])]),axis=0)

# Save projection neuron voltage data
np.save('{3}PN_od{0}_inj{1}_t{2}'.format(odor,conc,trial,folder_prefix),data[pn_inds])
np.save('{3}LN_od{0}_inj{1}_t{2}'.format(odor,conc,trial,folder_prefix),data[ln_inds])




# This is code to export an adjacency matrix
# import networkx as nx
# import matplotlib.pyplot as plt
# np.savetxt('adj_mat.dat',nx.to_numpy_matrix(AL))

# lm.show_random_neuron_in_layer(time_sampled_range,data,AL,0,2)
# lm.show_random_neuron_in_layer(time_sampled_range,data,AL,1,6)
# lm.plot_LFP(time_sampled_range, data, AL, layer_pn = 1)
