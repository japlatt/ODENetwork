import numpy as np
import scipy as sp
import pickle
import os.path
import time
from struct import unpack
import math
import skimage

import networks as net
import experiments as ex
import lab_manager as lm
import neuron_models as nm

#function to turn 
def add_current(AL, curr, Nglo, Npn):
    I_ext = np.zeros((Nglo, Npn))
    i = 0
    j = -1
    for i in range(len(linear)):
        k = i%Npn
        if k == 0:
            j+=1
        I_ext[j][k] = curr[i]

    neuron_inds = [np.nonzero(I_ext[j])[0].tolist() for j in range(Nglo)]
    current_vals = [I_ext[j][np.nonzero(I_ext[j])] for j in range(Nglo)]
    return neuron_inds, current_vals

MNIST_data_path = '/Users/Jason/Desktop/Mothnet/MNIST_data/'


#build antenna lobe
Npn = 6
Nln = 2
Nglo = 17

tot_pns = Npn*Nglo
glo_para = dict(num_pn=Npn, num_ln=Nln, # flies: 3 and 30
    PNClass=nm.PN_2, LNClass=nm.LN,
    PNSynapseClass=nm.Synapse_nAch_PN_2, LNSynapseClass=nm.Synapse_gaba_LN_with_slow)

al_para = dict(num_glo=Nglo, glo_para=glo_para) # flies: 54

AL = net.get_antennal_lobe(**al_para)

#load MNIST data
start = time.time()
training = ex.get_labeled_data(MNIST_data_path + 'training', MNIST_data_path)
end = time.time()
print('time needed to load training set:', end - start)

start = time.time()
testing = ex.get_labeled_data(MNIST_data_path + 'testing', MNIST_data_path, bTrain = False)
end = time.time()
print('time needed to load test set:', end - start)

n_input = training['rows']*training['cols'] #28x28=784
input_intensity = 60. #scale input
time_per_image = .2 #seconds
dt = 2e-5

resolution = np.round(np.sqrt(tot_pns))
scalex = int(np.round(training['rows']/resolution))
scaley = int(np.round(training['cols']/resolution))

print(resolution)

num_examples = 4.0 #len(training)
j = 0
dat_arr = []
while j < (int(num_examples)):
	print('image: ' + str(j))
	rates = training['x'][j%60000,:,:] / 8.*input_intensity
	downsample = skimage.transform.downscale_local_mean(rates, (scalex, scaley))
	print('downsampled image from (28, 28) to ' + str(downsample.shape))
	linear = np.ravel(downsample)
	neuron_inds, current_vals = add_current(AL, linear, Nglo, Npn)

	ex.const_current(AL, Nglo, neuron_inds, current_vals)
	#set up the lab
	f, initial_conditions, neuron_inds  = lm.set_up_lab(AL)
	time_sampled_range = np.arange(0., time_per_image*1000, dt*1000)

	data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5',compile=True)
	dat_arr.append(data)

	lm.show_random_neuron_in_layer(time_sampled_range,data,AL,1,2)
	lm.show_random_neuron_in_layer(time_sampled_range,data,AL,3,2)
	lm.show_random_neuron_in_layer(time_sampled_range,data,AL,5,2)

	j+=1
np.save('MNIST_AL_data.npy', dat_arr)

