'''
This is a script intended to take the output of the antennal lobe projection neurons
and compare that to the neurons which received an input stimulus. To be used for
troubleshooting whether the activity is moving from saddle point to saddle point.
'''
import numpy as np
import glob
import scipy as sp
import pickle
import sys

sys.path.append('../../..')
import networks as net


# Works for single voltage trace input
def get_spikes(time_sampled_range,v_m):
    spike_thresh = 0

    spike_bool = sp.logical_and(v_m[:-1] < spike_thresh, v_m[1:] >= spike_thresh)
    spike_idx = [idx for idx, x in enumerate(spike_bool) if x]
    time_spikes = time_sampled_range[spike_idx] # in ms


    dt = np.diff(time_spikes)
    #isi_mean = np.mean(dt)
    #isi_dev = np.std(dt)
    return time_spikes


num_odors = 3
odor_file = open('stimuli.pickle','rb')
odors = pickle.load(odor_file)
odor_file.close()
# Loaded to check stuff -- unnecessary for function
network_file = open('network.pickle','rb')
network = pickle.load(network_file)
print(np.shape(odors))

outF = open('stimulus.txt','w')

# k is a variable which numbers the figures saved. The only reason to do this is
# to get a properly working ffmpeg video of all these images in order
# k = 1
for odor in range(num_odors):
    pn_stim = []
    ln_stim = []
    for j in range(15):
        pn_stim.append(list(np.asarray([int(k) for k in odors[0][odor][j] if k <6]) + j*6))
        ln_stim.append(list(np.asarray([int(k) for k in odors[0][odor][j] if k >5]) - 6 + j*2))
    pn_stim = list(np.hstack(pn_stim))
    ln_stim = list(np.hstack(ln_stim))

    outF.write('PN stimulus: {0}\n'.format(pn_stim))
    outF.write('LN stimulus: {0}\n'.format(ln_stim))

    # Load PN data
    for filename in sorted(glob.glob('AL_3090_od{0}_inj*'.format(odor))):
        data = np.load(filename)
        time = np.arange(np.shape(data)[1])*0.02
        all_times = []
        spiked = []
        for i in range(np.shape(data)[0]):
            spike_times = get_spikes(time,data[i,:])
            if not len(spike_times) == 0:
                spiked.append(i)
            all_times.append(spike_times)
        outF.write('{0}'.format(spiked))
        outF.write('\n')

	#all_isi = np.hstack(all_isi)
        #counts, bins = np.histogram(all_isi,bins=10, range=(0,100))

outF.close()
