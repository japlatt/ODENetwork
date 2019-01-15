import networks as net
import lab_manager as lm
import neuron_models as nm
import experiments as ex

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

#build network


#Layer 1: Antennal Lobe (AL) [first stage in separation]
glo_para = dict(num_pn=6, num_ln=2, # flies: 3 and 30
    PNClass=nm.PN_2, LNClass=nm.LN,
    PNSynapseClass=nm.Synapse_nAch_PN_2, LNSynapseClass=nm.Synapse_gaba_LN_with_slow)

al_para = dict(num_glo=15, glo_para=glo_para) # flies: 54

# AL = net.get_antennal_lobe(**al_para)
# net.draw_colored_layered_digraph(AL)

# #Layer 2: Mushroom Body (MB) [second stage in separation]
mb_para = dict(num_kc=1200, # flies: 2500
    KCClass = nm.HHNeuronWithCaJL, GGNClass = nm.LN,
    KCSynapseClass = nm.PlasticNMDASynapseWithCaJL, GGNSynapseClass = nm.Synapse_gaba_LN)

# mb = net.get_mushroom_body(**mb_para)
# net.draw_colored_layered_digraph(mb)

# #Layer 3: Beta-lobe (BL) [read-out]
bl_para = dict(num_bl=10, #flies: 34
    BLClass=nm.HHNeuronWithCaJL, BLSynapseClass=nm.Synapse_gaba_HH)

# bl = net.get_beta_lobe(**bl_para)
# net.draw_colored_layered_digraph(bl)


other_para = dict(prob_a2k=0.5, prob_k2b=0.5)
full = net.get_olfaction_net(al_para=al_para,
    mb_para=mb_para, bl_para=bl_para, other_para=other_para)

#------------------------------------------------------------------------------

#run network


AL = full.layers[0]

num_layers = len(AL.layers)
Iscale = 500
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
time_len = 100.0
dt = 0.02
time_sampled_range = np.arange(0., time_len, dt)

data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5',compile=True)

# pn_inds = np.array([n.ii for n in AL.layers[1].nodes()])
# ln_inds = np.array([n.ii for n in AL.layers[0].nodes()])
# inds = np.append(np.asarray(ln_inds),np.asarray(pn_inds))

sol = np.transpose(data)
np.save('solution', sol)