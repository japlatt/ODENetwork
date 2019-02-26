import networks as net
import lab_manager as lm
import neuron_models as nm
import experiments as ex


import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt



#build network

"""
General TODO list:

TODO: Fix the network. Currently, the KC's are not firing sparsely enough,
which is causing the beta lobe neurons to fire in the same pattern.
"""


#Layer 1: Antennal Lobe (AL) [first stage in separation]
glo_para = dict(num_pn=6, num_ln=2, # flies: 3 and 30
    PNClass=nm.PNRescaled, LNClass=nm.LNRescaled,
    PNSynapseClass=nm.Synapse_PN_Rescaled, LNSynapseClass=nm.Synapse_LNSI_Rescaled)

al_para = dict(num_glo=15, glo_para=glo_para) # flies: 54
al_cond_para = {}
al_prob_para = {}

# AL = net.get_antennal_lobe(**al_para)
# net.draw_colored_layered_digraph(AL)

# #Layer 2: Mushroom Body (MB) [second stage in separation]
mb_para = dict(num_kc=1200, # flies: 2500
    KCClass = nm.HHNeuronWithCaJL, GGNClass = nm.LNRescaled,
    KCSynapseClass = nm.Synapse_glu_HH, GGNSynapseClass = nm.Synapse_LN_Rescaled, gGNNKC=0.5)

# mb = net.get_mushroom_body(**mb_para)
# net.draw_colored_layered_digraph(mb)

# #Layer 3: Beta-lobe (BL) [read-out]
bl_para = dict(num_bl=10, #flies: 34
    BLClass=nm.HHNeuronWithCaJL, BLSynapseClass=nm.Synapse_gaba_HH)

# bl = net.get_beta_lobe(**bl_para)
# net.draw_colored_layered_digraph(bl)

# Added in option to use different connecting synapse.
# Play with the last 2 conductance values.
other_para = dict(prob_a2k=0.3, prob_k2b=0.5, al_to_mb=nm.Synapse_PN_Rescaled,
    mb_to_bl=nm.StdpSynapse,gALMB=0.2,gKCGNN=0.15,gKCBL=3.5)

full = net.get_olfaction_net(al_para=al_para,
    mb_para=mb_para, al_cond_para=al_cond_para, al_prob_para=al_prob_para, bl_para=bl_para, other_para=other_para)

#------------------------------------------------------------------------------

#run network

# Isolate the antennal lobe and inject current into 1/3 of neurons randomly
# Should we inject into glomeruli specifically?
AL = full.layers[0]

num_layers = len(AL.layers)
Iscale = 4
p = 0.33 #probability
I_ext = []
for i in range(num_layers):
    num_neur = len(AL.layers[i].nodes)
    #array of length num_neur with random vals between 0-1
    I_ext.append(np.random.rand(num_neur))
    #find neurons with input current
    I_ext[i][(np.nonzero(I_ext[i] >= (1-p)))] = 1.0
    #set all non-input neurons curr to 0
    I_ext[i] = np.floor(I_ext[i])
    #scale
    I_ext[i][np.nonzero(I_ext[i])] = Iscale*np.asarray(I_ext[i][np.nonzero(I_ext[i])])
current_vals = [I_ext[j][np.nonzero(I_ext[j])] for j in range(num_layers)]
neuron_inds = [np.nonzero(I_ext[j])[0].tolist() for j in range(num_layers)]
ex.const_current(AL, num_layers, neuron_inds, current_vals)

#set up the lab
f, initial_conditions, neuron_inds  = lm.set_up_lab(full)

#run for specified time with dt
time_len = 400.0
dt = 0.02
time_sampled_range = np.arange(0., time_len, dt)

data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5',compile=True)

# AL indicies
pn_inds = np.concatenate(np.array([[n.ii for n in AL.layers[j].layers[0].nodes()] for j in range(np.shape(AL.layers)[0])]),axis=0)
ln_inds = np.concatenate(np.array([[n.ii for n in AL.layers[j].layers[1].nodes()] for j in range(np.shape(AL.layers)[0])]),axis=0)
al_inds = np.array([n.ii for n in AL.nodes()])


# As it stands, too much firing in MB
MB = full.layers[1]
# MB indicies
kc_inds = np.array([n.ii for n in MB.layers[0].nodes()])
ggn_ind = np.array([n.ii for n in MB.layers[1].nodes()])
mb_inds = np.array([n.ii for n in MB.nodes()])

# BL indicies
BL = full.layers[2]
bl_inds = np.array([n.ii for n in BL.nodes()])


# np.save('large_network.npy',sol[inds])

# This is code to export an adjacency matrix
# import networkx as nx
# import matplotlib.pyplot as plt
# np.savetxt('adj_mat.dat',nx.to_numpy_matrix(AL))

#lm.show_random_neuron_in_layer(time_sampled_range,data,AL,0,8)
#lm.show_random_neuron_in_layer(time_sampled_range,data,AL,1,8)
# input IS transposed before being put in
# input is NOT transponsed before being put in

# KC
lm.show_random_neuron_in_layer(time_sampled_range,data,MB,0,10)
# GGN
lm.show_random_neuron_in_layer(time_sampled_range,data,MB,1,1)
#BL
lm.show_random_neuron_in_layer(time_sampled_range,data,BL,0,10)
data = np.transpose(data)
lm.plot_LFP(time_sampled_range, data, AL, pn_inds)
np.save('solution', data[pn_inds,:])
