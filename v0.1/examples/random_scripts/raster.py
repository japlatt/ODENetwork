'''
This is a script intended to make a raster plot of PN and LN activity.
'''
import numpy as np
import glob
import scipy as sp
import pickle
import sys

#sys.path.append('..')
#import networks as net

folder_prefix = ''
# Works for single voltage trace input
def get_spikes_PN(time_sampled_range,v_m):
    spike_thresh = -5

    spike_bool = sp.logical_and(v_m[:-1] < spike_thresh, v_m[1:] >= spike_thresh)
    spike_idx = [idx for idx, x in enumerate(spike_bool) if x]
    time_spikes = time_sampled_range[spike_idx] # in ms

    dt = np.diff(time_spikes)
    #isi_mean = np.mean(dt)
    #isi_dev = np.std(dt)
    return time_spikes

def get_spikes_LN(time_sampled_range,v_m):
    spike_thresh = -10

    spike_bool = sp.logical_and(v_m[:-1] < spike_thresh, v_m[1:] >= spike_thresh)
    spike_idx = [idx for idx, x in enumerate(spike_bool) if x]
    time_spikes = time_sampled_range[spike_idx] # in ms

    return time_spikes
num_odors = 3
'''
odor_file = open('stimuli.pickle','rb')
odors = pickle.load(odor_file)
odor_file.close()
# Loaded to check stuff -- unnecessary for function
network_file = open('network.pickle','rb')
network = pickle.load(network_file)
print(np.shape(odors))

outF = open('stimulus.txt','w')
'''
PN_odor = []
LN_odor = []
for odor in range(num_odors):
    '''
    pn_stim = []
    ln_stim = []
    for j in range(15):
        pn_stim.append(list(np.asarray([int(k) for k in odors[0][odor][j] if k <6]) + j*6))
        ln_stim.append(list(np.asarray([int(k) for k in odors[0][odor][j] if k >5]) - 6 + j*2))
    pn_stim = list(np.hstack(pn_stim))
    ln_stim = list(np.hstack(ln_stim))

    outF.write('PN stimulus: {0}\n'.format(pn_stim))
    outF.write('LN stimulus: {0}\n'.format(ln_stim))
    '''
    # Load PN data
    for filename in sorted(glob.glob('{1}PN*_od{0}_inj*'.format(odor,folder_prefix))):
        data = np.load(filename)
        pn_num=data.shape[0]
        time = np.arange(np.shape(data)[1])*0.02
        PN_times = []
        for i in range(np.shape(data)[0]):
            spike_times = get_spikes_PN(time,data[i,:])
            #if not len(spike_times) == 0:
            #    spiked.append(i)
            PN_times.append(spike_times)
        PN_odor.append(PN_times)
    # Load LN data
    for filename in sorted(glob.glob('{1}LN_od{0}_inj*'.format(odor,folder_prefix))):
        data = np.load(filename)
        ln_num = data.shape[0]
        time = np.arange(np.shape(data)[1])*0.02
        LN_times = []
        for i in range(np.shape(data)[0]):
            spike_times = get_spikes_LN(time,data[i,:])
            #if not len(spike_times) == 0:
            #    spiked.append(i)
            LN_times.append(spike_times)
        LN_odor.append(LN_times)

import matplotlib
import matplotlib.pyplot as plt
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.style.use('ggplot')

fig, axs = plt.subplots(figsize=(20,16))
axs.set_ylim(0,pn_num)
axs.set_ylabel('PNs')
axs.set_xlabel('Time (ms)')
axs.eventplot(PN_odor[0],orientation='horizontal',linelengths=1.5,linewidths=3.5)
axs.set_title('PN Raster Odor 0')
plt.savefig('{0}PNraster.png'.format(folder_prefix))
plt.show()
plt.close()

fig, axs = plt.subplots(figsize=(20,16))
axs.set_ylim(0,ln_num)
axs.set_ylabel('LNs')
axs.set_xlabel('Time (ms)')
axs.eventplot(LN_odor[0],orientation='horizontal',linelengths=1.5,linewidths=3.5)
axs.set_title('LN Raster Odor 0')
plt.savefig('{0}LNraster.png'.format(folder_prefix))
plt.show()
plt.close()

#outF.close()
