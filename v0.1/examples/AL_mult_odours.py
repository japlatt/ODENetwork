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





#Layer 1: Antennal Lobe (AL) [first stage in separation]
glo_para = dict(num_pn=6, num_ln=2, # flies: 3 and 30
    PNClass=nm.PN_2, LNClass=nm.LN,
    PNSynapseClass=nm.Synapse_nAch_PN_2, LNSynapseClass=nm.Synapse_gaba_LN_with_slow)


al_prob_para = {}
al_cond_para = dict(gLN = 110, gPN = -1.0, gLNPN=200, gPNLN=600)
al_para = dict(num_glo=15, glo_para=glo_para,al_prob_para=al_prob_para, al_cond_para=al_cond_para) # flies: 54


AL = net.get_antennal_lobe(**al_para)


folder_prefix = 'results/test_script/'
# net.draw_colored_layered_digraph(AL)

num_layers = len(AL.layers)
Ibase = 400 # nA
p = 0.33 # probability of injecting

#run for specified time with dt
time_len = 300.0
dt = 0.02
time_sampled_range = np.arange(0., time_len, dt)

num_odors = int(input('Enter number of odours: '))
num_per_od = int(input('Enter # different concentrations per od: '))
adj_m = net.get_edges_data(AL, "weight")
num_trials = int(input('Enter number of runs to be completed for each odor/concentration: '))

for number in range(num_odors):
	# Each odour is encoded by injected current in different subsets of neurons
	I_ext = []
	# Establishes 1 for receives current and 0 for not receiving current
	for i in range(num_layers):
		# neurons per glomeruli
		num_neur = len(AL.layers[i].nodes)
    	#array of length num_neur with random vals between 0-1
		I_ext.append(np.random.rand(num_neur))
    	#find neurons with input current
		I_ext[i][(np.nonzero(I_ext[i] >= (1-p)))] = 1.0
    	# set all non-input neurons curr to 0
		I_ext[i] = np.floor(I_ext[i])

	# Records neuron indicies to be injected with current
	curr_inds = [np.nonzero(I_ext[k])[0].tolist() for k in range(num_layers)]

	for j in range(num_per_od):
		# Varies the amplitude of injected current
		Iscale = j*Ibase + Ibase


		current_vals = Iscale*np.asarray([I_ext[k][np.nonzero(I_ext[k])] for k in range(num_layers)])

		# If we want repeat trials (for example, in the event that we add noise to the system)
		for trial in range(num_trials):
			print('Odor: {0}, Concentration {1}, Trial {2}'.format(number,j,trial))
			ex.const_current(AL, num_layers, curr_inds, current_vals)

			# set up the lab
			f, initial_conditions, all_neuron_inds = lm.set_up_lab(AL)
			# run the lab
			data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5',compile=True)

			data = np.transpose(data)
			pn_inds = np.concatenate(np.array([[n.ii for n in AL.layers[j].layers[0].nodes()] for j in range(np.shape(AL.layers)[0])]),axis=0)
			ln_inds = np.concatenate(np.array([[n.ii for n in AL.layers[j].layers[1].nodes()] for j in range(np.shape(AL.layers)[0])]),axis=0)
			np.save('{3}PN_od{0}_inj{1}_t{2}'.format(number,Iscale,trial,folder_prefix),data[pn_inds])
			np.save('{3}LN_od{0}_inj{1}_t{2}'.format(number,Iscale,trial,folder_prefix),data[ln_inds])
			data = None



# This is code to export an adjacency matrix
# import networkx as nx
# import matplotlib.pyplot as plt
# np.savetxt('adj_mat.dat',nx.to_numpy_matrix(AL))

# lm.show_random_neuron_in_layer(time_sampled_range,data,AL,0,2)
# lm.show_random_neuron_in_layer(time_sampled_range,data,AL,1,6)
# lm.plot_LFP(time_sampled_range, data, AL, layer_pn = 1)
