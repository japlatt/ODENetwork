"""
lab_manager.py

It does what a lab manager should be doing. i.e
1. set_up_lab()
2. run_lab()

TODO: Create a raster plot function.
"""

import sys
import pandas as pd
import numpy as np
import random
# might be useful for mac user, uncommnet below if needed
import matplotlib
matplotlib.use("TKAgg")

import matplotlib.pyplot as plt

from jitcode import jitcode, y, t # this "y" will now allow symbolic tracking
from jitcode import integrator_tools
import networks #; reload(networks)
import electrodes#; reload(electrodes)
import neuron_models as nm

"""
set_up_lab(net):

Prepare all the ODEs and impose initial coonditions.
"""
def set_up_lab(net):
    neurons = net.nodes()
    neuron_inds = []
    # step 3a: fix the integration indices sequencially
    ii = 0 # integration index
    for (n, pos_neuron) in enumerate(neurons):
        pos_neuron.set_neuron_index(n) # maybe it will be usefull?
        neuron_inds.append(ii)
        if pos_neuron.DIM: # same as if pos_neuron.DIM > 0
            pos_neuron.set_integration_index(ii)
            ii += pos_neuron.DIM
            pre_synapses =  (
                net[pre_neuron][pos_neuron]["synapse"] # order matter!
                for pre_neuron in net.predecessors(pos_neuron))
        for pre_neuron in net.predecessors(pos_neuron):
            synapse = net[pre_neuron][pos_neuron]["synapse"]
            if synapse.DIM:
                synapse.set_integration_index(ii)
                ii += synapse.DIM
    # Have the ODEs ready construct a generator for that
    def f():
        # adja_list = net.adja_list # the list of lists of pre-synaptic neurons
        # synapse_lists = net.edges_list # the list of lists of pre-synapses
        # step 3b: must yield the derivatives in the exact same order in step 3a
        for (n, pos_neuron) in enumerate(neurons):
            pre_neurons = [neuron for neuron in net.predecessors(pos_neuron)] # can try replace [] -> ()
            pre_synapses = [
                net[pre_neuron][pos_neuron]["synapse"]
                for pre_neuron in net.predecessors(pos_neuron)]
            yield from pos_neuron.dydt(pre_synapses, pre_neurons)
            for pre_neuron in net.predecessors(pos_neuron):
                synapse = net[pre_neuron][pos_neuron]["synapse"]
                if synapse.dydt(pre_neuron,pos_neuron) is not None:
                    yield from synapse.dydt(pre_neuron, pos_neuron)
    #for debug use only
    #for dydt in f():
    #    print(dydt)
    #end debug
    # Impose initial conditions
    initial_conditions = []
    #neurons = net.vertexs # the list of all neruons
    #synapse_lists = net.edges_list # the list of lists of pre-synapses
    # Must follow the same order in the appearance in f()
    for (n, pos_neuron) in enumerate(neurons):
        if pos_neuron.DIM:
            initial_conditions += pos_neuron.get_initial_condition()
        pre_synapses = (
            net[pre_neuron][pos_neuron]["synapse"]
            for pre_neuron in net.predecessors(pos_neuron))
        for synapse in pre_synapses:
            if synapse.DIM:
                initial_conditions += synapse.get_initial_condition()
    initial_conditions = np.array(initial_conditions)
    return f, initial_conditions, neuron_inds

"""
run_lab(f, initial_conditions, time_sampled_range, integrator='dopri5'):

Run the lab.
"""
def run_lab(f, initial_conditions, time_sampled_range, integrator='dopri5',
    compile=False):
    dim_total = len(initial_conditions)
    ODE = jitcode(f, n=dim_total)
    if compile:
        ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
    else:
        ODE.generate_lambdas()
    ODE.set_integrator(integrator)# ,nsteps=10000000)
    ODE.set_initial_value(initial_conditions, 0.0)
    data = np.zeros((len(time_sampled_range), dim_total)) # will set it to np.empty
    for (i,T) in enumerate(time_sampled_range):
        try:
            data[i,:] = ODE.integrate(T)
        except integrator_tools.UnsuccessfulIntegration:
            print("gotcha")
            return data
    return data


"""
Reset all the input currents to be zero.
"""
def reset_lab(net):
    for neuron in net.nodes:
        neuron.i_inj = 0

"""
sample_plot(data, net):

Just a demo. Nothing special really.
"""
def sample_plot(time_sampled_range, data, net):
    neuron_1 = list(net.layers[0].nodes)[0] # just pick one neuron from each layer
    neuron_2 = list(net.layers[1].nodes)[0]
    syn = net[neuron_1][neuron_2]["synapse"]
    THETA_D = syn.THETA_D
    THETA_P = syn.THETA_P

    for (n, neuron) in enumerate([neuron_1, neuron_2]):
        ii = neuron.ii
        v_m = data[:,ii]
        ca = data[:,ii+6]
        i_inj = electrodes.sym2num(t, neuron.i_inj)
        i_inj = i_inj(time_sampled_range)
        fig, axes = plt.subplots(3,1,sharex=True)
        axes[0].plot(time_sampled_range, v_m, label="V_m")
        axes[0].set_ylabel("V_m [mV]")
        axes[0].legend()
        axes[1].plot(time_sampled_range, ca, label="[Ca]")
        axes[1].set_ylabel("Calcium [a.u.]")
        axes[1].axhline(THETA_D, color="orange", label="theta_d")
        axes[1].axhline(THETA_P, color="green", label="theta_p")
        axes[1].legend()
        axes[2].plot(time_sampled_range, i_inj, label="i_inj")
        axes[2].set_ylabel(" Injected Current [some unit]")
        axes[2].legend()
        axes[-1].set_xlabel("time [ms]")
        plt.suptitle("Neuron {}".format(n))

    ii = syn.ii
    plt.figure()
    syn_weight = data[:,ii]
    plt.plot(time_sampled_range, syn_weight, label="w_ij")
    plt.xlabel("time [ms]")
    plt.legend()
    plt.show()

"""
show_layer(time_sampled_range, data, net, layer_idx):

Show a neuron in layer_idx and the neruon it synapses onto, and the synapse.
"""
# def show_layer(time_sampled_range, data, net, layer_idx):
#     pre_neuron = list(net.layers[layer_idx].nodes)[0] # just pick one neuron from each layer
#     pos_neuron = list(net.successors(pre_neuron))[0]
#     synapse = net[pre_neuron][pos_neuron]["synapse"]
#     THETA_D = synapse.THETA_D
#     THETA_P = synapse.THETA_P
#     labels= ["Pre-synaptic", "Post-synaptic"]
#     for (n, neuron) in enumerate([pre_neuron, pos_neuron]):
#         ii = neuron.ii
#         v_m = data[:,ii]
#         ca = data[:,ii+6]
#         if neuron.i_inj is None:
#             fig, axes = plt.subplots(2,1,sharex=True)
#         else:
#             fig, axes = plt.subplots(3,1,sharex=True)
#             i_inj = electrodes.sym2num(t, neuron.i_inj)
#             i_inj = i_inj(time_sampled_range)
#             axes[2].plot(time_sampled_range, i_inj, label="i_inj")
#             axes[2].set_ylabel("[some unit]")
#             axes[2].legend()
#         axes[0].plot(time_sampled_range, v_m, label="V_m")
#         axes[0].set_ylabel("V_m [mV]")
#         axes[0].legend()
#         axes[1].plot(time_sampled_range, ca, label="[Ca]")
#         axes[1].set_ylabel("[a.u.]")
#         axes[1].axhline(THETA_D, color="orange", label="theta_d")
#         axes[1].axhline(THETA_P, color="green", label="theta_p")
#         axes[1].legend()
#         axes[-1].set_xlabel("time [ms]")
#         plt.suptitle(labels[n])
#     if synapse is not None:
#         ii = synapse.ii
#         plt.figure()
#         syn_weight = data[:,ii]
#         plt.plot(time_sampled_range, syn_weight, label="w_ij")
#         plt.xlabel("time [ms]")
#         plt.legend()
#         plt.show()

"""
show_layer(time_sampled_range, data, net, layer_idx):

Show a neuron in layer_idx and the neruon it synapses onto, and the synapse.
"""
def show_all_neuron_in_layer(time_sampled_range, data, net, layer_idx):

    pre_neurons = net.layers[layer_idx].nodes()
    for (n, pre_neuron) in enumerate(pre_neurons):
        ii = pre_neuron.ii
        v_m = data[:,ii]
        ca = data[:,ii+6]
        if pre_neuron.i_inj is None:
            fig, axes = plt.subplots(2,1,sharex=True)
        else:
            fig, axes = plt.subplots(2,1,sharex=True)
            i_inj = electrodes.sym2num(t, pre_neuron.i_inj)
            i_inj = i_inj(time_sampled_range)
            axes[1].plot(time_sampled_range, i_inj, label="i_inj")
            axes[1].set_ylabel("I [some unit]")
            axes[1].legend()
        axes[0].plot(time_sampled_range, v_m, label="V_m")
        axes[0].set_ylabel("V_m [mV]")
        axes[0].legend()
        # axes[1].plot(time_sampled_range, ca, label="[Ca]")
        # axes[1].set_ylabel(" Ca [a.u.]")
        # axes[1].axhline(THETA_D, color="orange", label="theta_d")
        # axes[1].axhline(THETA_P, color="green", label="theta_p")
        # axes[1].legend()
        axes[-1].set_xlabel("time [ms]")
        plt.suptitle("Neuron {} in layer {}".format(pre_neuron.ni, layer_idx))
    plt.show()

def show_all_synaspe_onto_layer(time_sampled_range, data, net, layer_idx):
    def sigmoid(x):
        return 1./(1.+ np.exp(-x))
    pos_neurons = net.layers[layer_idx].nodes()
    for pos_neuron in pos_neurons:
        pre_neurons = list(net.predecessors(pos_neuron))
        for pre_neuron in pre_neurons:
            synapse = net[pre_neuron][pos_neuron]["synapse"]
            #THETA_D = synapse.THETA_D
            #THETA_P = synapse.THETA_P
            ii = synapse.ii
            fig, axes = plt.subplots(3,1,sharex=True)
            red_syn_weight = data[:,ii]
            ca = data[:,ii+2]
            axes[0].plot(time_sampled_range, red_syn_weight, label="reduced synaptic weight")
            axes[0].legend()
            axes[1].plot(time_sampled_range, sigmoid(red_syn_weight), label="synaptic weight")
            axes[1].legend()
            axes[2].plot(time_sampled_range, ca, label="Ca")
            #axes[2].axhline(THETA_D, color="orange", label="theta_d")
            #axes[2].axhline(THETA_P, color="green", label="theta_p")
            axes[2].legend()
            plt.suptitle("w_{}{}".format(pre_neuron.ni, pos_neuron.ni))
            plt.show()

def show_all_dendrite_onto_layer_old(time_sampled_range, data, net, layer_idx):
    pos_neurons = net.layers[layer_idx].nodes()
    for pos_neuron in pos_neurons:
        pre_neurons = list(net.predecessors(pos_neuron))
        for pre_neuron in pre_neurons:
            synapse = net[pre_neuron][pos_neuron]["synapse"]
            ii = synapse.ii
            fig, axes = plt.subplots(6,1,sharex=True)
            #Dim=17
            v = data[:,ii]
            m = data[:,ii+1]
            h = data[:,ii+2]
            n = data[:,ii+3]
            a = data[:,ii+4]
            b = data[:,ii+5]
            u = data[:,ii+6]
            ma = data[:,ii+7]
            ha = data[:,ii+8]
            ngf = data[:,ii+9]
            ngs = data[:,ii+10]
            ng = synapse.PROP_FAST_NMDA*ngf+(1-synapse.PROP_FAST_NMDA)*ngs
            ag = data[:,ii+11]
            ca = data[:,ii+12]
            p0 = data[:,ii+13]
            p1 = data[:,ii+14]
            p2 = 1 - p0 - p1
            P = data[:,ii+15]
            D = data[:,ii+16]
            w = synapse.G0*p0 + synapse.G1*p1 + synapse.G2*p2
            i_a = -synapse.COND_A*ma*ha*(v - synapse.RE_PO_K)
            i_ampa = -synapse.initial_cond*w*ag*(v-synapse.RE_PO_EX)
            B = 1./(1.+0.288*synapse.MG*np.exp(-0.062*v))
            i_nmda = -synapse.COND_NMDA*ng*B*(v-synapse.RE_PO_EX)
            i_ca = -synapse.COND_CA*(v/synapse.CA_EQM) \
                *(ca - 15000*np.exp(-2*v*synapse.F/(synapse.R*synapse.T))) \
                /(1 - np.exp(-2*v*synapse.F/(synapse.R*synapse.T))) \
                *a**2*b
            ca_vgcc = synapse.ICA_TO_CA*i_ca
            ca_nmda = synapse.INMDA_TO_CA*i_nmda
            axes[0].plot(time_sampled_range, v, label="v")
            axes[0].legend()
            axes[1].plot(time_sampled_range, B, label="B")
            axes[1].legend()
            axes[2].plot(time_sampled_range, i_ca, label="i_ca")
            #axes[2].plot(time_sampled_range, i_ampa, label="i_ampa")
            axes[2].plot(time_sampled_range, i_nmda, label="i_nmda")
            axes[2].legend()
            axes[3].plot(time_sampled_range, ca, label="ca")
            axes[3].legend()
            axes[4].plot(time_sampled_range, ca_vgcc, label="ica_vgcc")
            axes[4].plot(time_sampled_range, ca_nmda, label="ica_nmda")
            axes[4].legend()
            axes[5].plot(time_sampled_range, w, label="w")
            axes[5].legend()
            plt.suptitle("dendrite_{}{}".format(pre_neuron.ni, pos_neuron.ni))
            plt.show()
            fig.savefig("dendrite.png",dpi=500, bbox_inches = 'tight')

def show_all_dendrite_onto_layer(time_sampled_range, data, net, layer_idx, delta_time = None):
    pos_neurons = net.layers[layer_idx].nodes()
    for pos_neuron in pos_neurons:
        pre_neurons = list(net.predecessors(pos_neuron))
        for pre_neuron in pre_neurons:
            synapse = net[pre_neuron][pos_neuron]["synapse"]
            ii = synapse.ii
            v = data[:,ii]
            m = data[:,ii+1]
            h = data[:,ii+2]
            n = data[:,ii+3]
            a = data[:,ii+4]
            a_eq = data[-1,ii+4]
            print(a_eq)
            b = data[:,ii+5]
            b_eq = data[-1,ii+5]
            print(b_eq)
            ng = data[:,ii+6]
            ag = data[:,ii+7]
            ca = data[:,ii+8]
            print(max(ca))
            p0 = data[:,ii+9]
            print(p0)
            p1 = data[:,ii+10]
            print(p1)
            p2 = 1 - p0 - p1
            P = data[:,ii+11]
            D = data[:,ii+12]
            w = synapse.G0*p0 + synapse.G1*p1 + synapse.G2*p2
            print(w)
            #i_a = -synapse.COND_A*ma*ha*(v - synapse.RE_PO_K)
            i_ampa = -synapse.initial_cond*w*ag*(v-synapse.RE_PO_EX)
            B_nmda = 1./(1.+(1./3.57)*synapse.MG*np.exp(-0.062*v))
            i_nmda = -synapse.COND_NMDA*ng*B_nmda*(v-synapse.RE_PO_EX)
            i_vgcc = -synapse.COND_CA*(v/synapse.CA_EQM) \
                *(ca - synapse.CA_EX*np.exp(-2*v*synapse.FRT)) \
                /(1 - np.exp(-2*v*synapse.FRT)) \
                *a**2*b
            ca_vgcc = synapse.ICA_TO_CA*i_vgcc
            ca_nmda = synapse.INMDA_TO_CA*i_nmda
            DT = time_sampled_range
            fig, axes = plt.subplots(8,1,sharex=True,figsize=(7.5,15))
            plt.suptitle("dendrite_{}{}, dt={}ms".format(pre_neuron.ni, pos_neuron.ni, delta_time), fontsize=20)
            axes[0].plot(time_sampled_range, v, label="V")
            #np.savetxt('V.txt', np.transpose([DT, v]))
            axes[0].legend()
            #axes[1].plot(DT, B_nmda, label="B_nmda")
            #np.savetxt('B_nmda.txt', np.transpose([DT, B_nmda]))
            axes[1].plot(DT, m, label="m")
            #np.savetxt('m.txt', np.transpose([DT, m]))
            axes[1].plot(DT, h, label="h")
            #np.savetxt('h.txt', np.transpose([DT, h]))
            axes[1].plot(DT, n, label="n")
            #np.savetxt('n.txt', np.transpose([DT, n]))
            axes[1].plot(DT, a, label="a")
            #np.savetxt('a.txt', np.transpose([DT, a]))
            axes[1].plot(DT, b, label="b")
            #np.savetxt('b.txt', np.transpose([DT, b]))
            axes[1].legend()
            axes[2].plot(DT, i_vgcc, label="I_VGCC")
            #np.savetxt('i_vgcc.txt', np.transpose([DT, i_vgcc]))
            axes[2].plot(DT, i_nmda, label="I_NMDA")
            #np.savetxt('i_nmda.txt', np.transpose([DT, i_nmda]))
            axes[2].legend()
            axes[3].plot(DT, ca_vgcc, label="C_VGCC")
            #np.savetxt('ca_vgcc.txt', np.transpose([DT, ca_vgcc]))
            axes[3].plot(DT, ca_nmda, label="C_NMDA")
            #np.savetxt('ca_nmda.txt', np.transpose([DT, ca_nmda]))
            axes[3].legend()
            axes[4].plot(DT, ca, label="Ca")
            #np.savetxt('ca.txt', np.transpose([DT, ca]))
            axes[4].legend()
            axes[5].plot(DT, P, label="P")
            #np.savetxt('P.txt', np.transpose([DT, P]))
            axes[5].plot(DT, D, label="D")
            #np.savetxt('D.txt', np.transpose([DT, D]))
            axes[5].legend()
            axes[6].plot(DT, p0, label="p0")
            #np.savetxt('p0.txt', np.transpose([DT, p0]))
            axes[6].plot(DT, p1, label="p1")
            #np.savetxt('p1.txt', np.transpose([DT, p1]))
            axes[6].plot(DT, p2, label="p2")
            #np.savetxt('p2.txt', np.transpose([DT, p2]))
            axes[6].legend()
            axes[7].plot(DT, w, label="W")
            #np.savetxt('w.txt', np.transpose([DT, w]))
            axes[7].legend()
            plt.show()
            fig.savefig("dendrite.png",dpi=500, bbox_inches = 'tight')

"""
Helper function to smooth data using exponential weighted moving weighted moving average
"""
def ewma_pd(x,y):
    ewma = pd.Series.ewm
    df = pd.Series(y)
    return ewma(df,span=100).mean()

"""
Plot the local field potential for AL.py

Args:
    time_sampled_range:
        A time array to be used for plotting
    data:
        The output of integration.
    net:
        The network structure
    pn_inds:
        The integration indices of projection neurons
    transpose (optional) default = False:
        An option to take the transpose of data. In order to extract data using
        data[pn_inds], the data must be transposed from the output of lm.run().
        transpose = False means that the transpose has already been taken and
        does not need to be taken now.
    smooth (optional) default=False:
        An option to smooth the local field potential. Useful to see qualitative
        properties in small networks.
"""
def plot_LFP(time_sampled_range, data, net, pn_inds,transpose=False, smooth=False):
    t = time_sampled_range
    fig = plt.figure(figsize = (8,5))
    plt.title('Local Field Potential')
    plt.ylabel('LFP (mV)')
    plt.xlabel('time (ms)')
    if transpose:
        data = np.transpose(data)
    lfp = np.mean(data[pn_inds],axis=0)
    if smooth:
        lfp = ewma_pd(t,lfp)

    plt.plot(t, lfp,linewidth=2)
    plt.show()

"""
This function randomly plots a specified number of neurons in a layer.

Args:
    time_sampled_range:
        A time array to be used for plotting
    data:
        The output of integration.
    net:
        The network structure
    layder_idx:
        The layer that you want to display neurons from
    num_neurons (optional) default=1:
        The number of neurons you want to display from the given layer
"""
def show_random_neuron_in_layer(time_sampled_range, data, net, layer_idx, num_neurons=1):
    THETA_D = nm.PlasticNMDASynapse.THETA_D
    THETA_P = nm.PlasticNMDASynapse.THETA_P

    pre_neurons = net.layers[layer_idx].nodes()
    display_neurons = random.sample(pre_neurons,num_neurons)
    fig, axes = plt.subplots(2,1,sharex=True)
    for (n, neuron) in enumerate(display_neurons):
        ii = neuron.ii
        v_m = data[:,ii]
        i_inj = electrodes.sym2num(t, neuron.i_inj)
        i_inj = i_inj(time_sampled_range)
        axes[1].plot(time_sampled_range, i_inj, label=r"$I_{inj}$ Neuron %d"%neuron.ni)
        axes[1].set_ylabel("I [pA]")
        axes[1].legend()
        axes[0].plot(time_sampled_range, v_m, label="Neuron %d"%neuron.ni)
        axes[0].set_ylabel(r"$V_m$ [mV]")
        axes[0].legend()
        axes[-1].set_xlabel("time [ms]")
        plt.suptitle("Random Neuron in layer {}".format(layer_idx))
    plt.show()

def show_random_neuron_sv_in_layer(time_sampled_range, data, net, layer_idx, num_neurons=1):
    pre_neurons = net.layers[layer_idx].nodes()
    display_neurons = random.sample(pre_neurons,num_neurons)
    #fig, axes = plt.subplots(2,1,sharex=True)

    ## Fix for DIM = 1
    for (n, neuron) in enumerate(display_neurons):
        dim = neuron.DIM
        ii = neuron.ii
        fig, axes = plt.subplots(dim,1,sharex=True)

        if dim > 1:

            for j in range(dim):
                tmp = data[:,ii+j]
                #i_inj = electrodes.sym2num(t, neuron.i_inj)
                #i_inj = i_inj(time_sampled_range)
                axes[j].plot(time_sampled_range, tmp, label=r"$SV %d"%j)
                axes[j].set_ylabel("SV")
                axes[j].legend()
            axes[-1].set_xlabel("time [ms]")
        else:
                tmp = data[:,ii]
                axes.plot(time_sampled_range, tmp, label=r"$SV %d"%1)
                axes.set_ylabel("SV")
                axes.legend()
                axes.set_xlabel('time [ms]')
        plt.suptitle("Random Neuron State Variables in layer {}".format(layer_idx))
    plt.show()

def interspike_interval(time_sampled_range,data,net,layer_idx,num_neurons=1):
        pre_neurons=net.layers[layer_idx].nodes()
        display_neurons = random.sample(pre_neurons,num_neurons)

        spike_thresh = 0

        for (n,neuron) in enumerate(display_neurons):
            ii = neuron.ii
            v_m = data[:,ii]

            spike_bool = sp.logical_and(v_m[:-1] < spike_thresh, v_m[1:] >= spike_thresh)
            spike_idx = [idx for idx, x in enumerate(spikes) if x]
            time_spikes = time_sampled_range[spike_idx] # in ms
            dt = np.diff(time_spikes)
            isi_mean = np.mean(dt)
            isi_dev = np.std(dt)

            print('{} {}'.format(isi_mean,isi_dev))
