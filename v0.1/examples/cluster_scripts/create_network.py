'''
This is a script intended to be used on a cluster. This should be used in
conjunction with create_stimuli.py and run_script.py, along with whatever
bash script is necessary to submit jobs.

This script in particular uses scripts from ODENetwork to create the network
we wish to run on, and saves it as a pickle. This network must be created
before running create_stimuli.py

All network settings must be specified here.
'''

# Janky way to access ODENetwork scripts. Move to examples/ folder for use.
import sys
sys.path.append('..')

import scipy as sp
import numpy as np
import random
import networks as net
import neuron_models as nm
import pickle

# Where you want to save network.pickle, to be sourced by create_stimuli.py
folder_prefix = 'results/'

#Layer 1: Antennal Lobe (AL) [first stage in separation]
glo_para = dict(num_pn=6, num_ln=2, # flies: 3 and 30
    PNClass=nm.PN_2, LNClass=nm.LN,
    PNSynapseClass=nm.Synapse_nAch_PN_2, LNSynapseClass=nm.Synapse_gaba_LN_with_slow)

al_cond_para = dict(gLN = 110, gPN = -1, gLNPN = 300.0, gPNLN = 600.0)
al_prob_para = {}

al_para = dict(num_glo=15, glo_para=glo_para,al_cond_para=al_cond_para,al_prob_para=al_prob_para) # flies: 54


AL = net.get_antennal_lobe(**al_para)

network_pickle = open('{0}network.pickle'.format(folder_prefix),'wb')
pickle.dump(AL, network_pickle)
network_pickle.close()
