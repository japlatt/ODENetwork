import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import glob
from itertools import cycle
import seaborn as sns

sns.set_style('white')

# color order: r,b,g,p,o,y
cycol = cycle(['#f10c45','#069af3','#02590f','#ab33ff','#ff8c00','#ffd700'])
marker = cycle(['^','o','s','p'])

# These settings need to be changes each time!
num_odor = 2
num_conc = 1
total_paths = num_odor
folder_prefix = ''

# load pca data
# TODO: Use glob to pull files and count odor and conc to automatically set the above
data = []
for i in range(num_odor):
	for j in range(num_conc):
		x = np.loadtxt('{2}odor{0}_conc{1}.txt'.format(i,j,folder_prefix))
		data.append(x)

# print(np.shape(data))
# TODO: automate finding number of timesteps
nt = 150000
fig = plt.figure().gca(projection='3d')
c = next(cycol)
m = next(marker)
# print(data[0].shape)

# Bounds on what is shown - eliminate the first 500 time steps, or 10 ms since each neuron starts at the exact same value, weird things happen before 10 ms
start = 0
end = 10000

for k in range(num_odor*num_conc):
	fig.scatter(data[k][start:end:1,0],data[k][start:end:1,1],data[k][start:end:1,2],color=c,s=10,label='odor{0} inj{1}'.format(int(np.floor(k/num_conc)),k%num_conc))
	c = next(cycol)
fig.view_init(azim=-10,elev=15)
plt.legend()
plt.savefig('{0}pca.png'.format(folder_prefix))
plt.show()
