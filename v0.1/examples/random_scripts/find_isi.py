import numpy as np
import glob
import scipy as sp


# Works for single voltage trace input
def get_isi(time_sampled_range,v_m):
    spike_thresh = 0

    spike_bool = sp.logical_and(v_m[:-1] < spike_thresh, v_m[1:] >= spike_thresh)
    spike_idx = [idx for idx, x in enumerate(spike_bool) if x]
    time_spikes = time_sampled_range[spike_idx] # in ms

    dt = np.diff(time_spikes)
    #isi_mean = np.mean(dt)
    #isi_dev = np.std(dt)
    return dt, time_spikes


odors = 3
# k is a variable which numbers the figures saved. The only reason to do this is
# to get a properly working ffmpeg video of all these images in order
k = 1
for odor in range(odors):
    for filename in sorted(glob.glob('AL_3090_od{0}_inj???_*'.format(odor))):
        data = np.load(filename)
        time = np.arange(np.shape(data)[1])*0.02
        all_isi = []
        for i in range(np.shape(data)[0]):
            isi,spike_times = get_isi(time,data[i,:])
            all_isi.append(isi)
        all_isi = np.hstack(all_isi)
        counts, bins = np.histogram(all_isi,bins=10, range=(0,100))

	# Make a plot for a given odor isi of all PN's
        import matplotlib.pyplot as plt
        import seaborn
        plt.figure(figsize=(8,4))
        n, bins, patches = plt.hist(x=all_isi,bins=range(0,1000,5),alpha=0.7,rwidth=0.9)
        plt.xlim([0,100])
        plt.xlabel('Interspike Interval (ms)')
        plt.ylabel('Count')
        plt.title('Hist for odor {1} inj {0}'.format(filename[15:-7],odor))
        #plt.savefig('isi_odor{0}_inj{1}'.format(odor,filename[15:-7]))
        #plt.savefig('image{0}'.format(k))
        plt.show()
        plt.close()
        k = k + 1

