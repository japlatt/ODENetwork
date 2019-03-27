'''
A script to create the stimuli for a given network. It requires that
create_network.py has already been run and sources the network.pickle file.

It then creates all a list of neurons indicies which will receive current, along with
a multiplier for how much they should receive. This does NOT indicate how many
concentrations each odor will receive. That must be specified in run_script.py

The stimuli are saved as stimuli.pickle.
'''

import sys
sys.path.append('..')

import scipy as sp
import numpy as np
import random
import pickle

folder_prefix = 'results/'

network_pickle = open('{0}network.pickle'.format(folder_prefix),'rb')
AL = pickle.load(network_pickle)

# Numer of glomeruli
num_layers = len(AL.layers)
num_odors = 2
#num_per_od = 2


p = 0.33 # probability of current injected into neuron

# all_stimuli will be a list of neurons that receive current
all_stimuli = []
# all_iext will be the fraction of current they should receive
all_iext = []
for number in range(num_odors):
	# Each odour is encoded by injected current in different subsets of neurons
	I_ext = []
	# Establishes 1 for receives current and 0 for not receiving current
	for i in range(num_layers):
		# neurons per glomeruli
		num_neur = len(AL.layers[i].nodes)
    	# array of length num_neur with random vals between 0-1
		I_ext.append(np.random.rand(num_neur))
    	# find neurons with input current
		I_ext[i][np.nonzero(I_ext[i] >= (1-p))] = 1.0
    	# set all non-input neurons curr to 0
		I_ext[i] = np.floor(I_ext[i]).tolist()
	all_iext.append(I_ext)
	# Records neuron indicies to be injected with current
	curr_inds = [np.nonzero(I_ext[k])[0].tolist() for k in range(num_layers)]
	all_stimuli.append(curr_inds)

# Comment out if no odor mixtures
def mixture(a, odor1,odor2):
	'''
	Mixture of odors, optional

	Will take a*(odor #1) + (1-a)*(odor #2) and add that as a third odor
	'''
	return a*np.asarray(odor1) + (1-a)*np.asarray(odor2)

new_odor = mixture(0.4,all_iext[0],all_iext[1]).tolist()
curr_inds = [np.nonzero(new_odor[k])[0].tolist() for k in range(num_layers)]

all_iext.append(new_odor)
all_stimuli.append(curr_inds)
# End mixture section

# Get rid of zeros
all_iext = [[np.take(all_iext[l][k],np.nonzero(all_iext[l][k])[0]) for k in range(num_layers)] for l in range(len(all_iext))]

final_stim = []
final_stim.append(all_stimuli)
final_stim.append(all_iext)

# all_iext holds multiplier for current
# all_stimuli holds the indices
stimuli_pickle = open('{0}stimuli.pickle'.format(folder_prefix),'wb')
pickle.dump(final_stim,stimuli_pickle)
stimuli_pickle.close()
