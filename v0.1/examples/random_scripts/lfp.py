import matplotlib.pyplot as plt
import numpy as np
import glob

for filename in glob.glob('PN*od*inj*'):
    data = np.load(filename)
    print(data.shape)

time = np.arange(15000)*0.02

def plot_LFP(time_sampled_range, data, transpose=False, smooth=False):
    t = time_sampled_range
    fig = plt.figure(figsize = (8,5))
    plt.title('Local Field Potential')
    plt.ylabel('LFP (mV)')
    plt.xlabel('time (ms)')
    if transpose:
        data = np.transpose(data)
    lfp = np.mean(data,axis=0)
    print(lfp.shape)
    if smooth:
        lfp = ewma_pd(t,lfp)
    plt.xlim(t[0],t[-1])
    plt.plot(t, lfp,linewidth=2)
    plt.savefig('lfp.png')
    plt.show()

plot_LFP(time,data)
