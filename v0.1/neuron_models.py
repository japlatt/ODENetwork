"""
neuron_models.py
A module that contains all the neuron and synapse model classes.
To-dos:
1. make method/function to check dimensions consistency across DIM, ic,...etc
2. introduce delay, related to choices for rho_gate placement
3. too many "self"... can improve readability?
4. Get Antennal Lobe neurons (dimension pA) on the same scale as HH neurons
    (dim uA/cm^2). If we assume that the surface area of a neuron is ~ 100 um^2
    (roughly 3 um radius), then we don't have to do any conversion. For example,
    if we do g/C for the local neuron leak current, we have 21 nS/142 pF = 0.15.
    If we look at the same for HH neuron, it's 0.3 mS/1 uF = 0.3. Same order of
    magnitude.
"""

import numpy as np

from jitcode import jitcode, y, t

try:
    import symengine as sym_backend
except:
    import sympy as sym_backend
# "Global" constants, if any

# Some very common helper functions
def sigmoid(x):
    return 1./(1.+ sym_backend.exp(-x))

def heaviside(x):
    K = 1e5 # some big number
    return sigmoid(K*x)

def step(x):
    return 0.5*(1+sym_backend.tanh(120*(x-0.1)))

def pulse(t,t0,w):
    return heaviside(t-t0)*heaviside(w+t0-t)


class StaticExcitatorySynapse:
    """
    A static excitatory synapse.

    TODO: Does not work.
    """
    COND_SYN = 1.
    RE_PO_SYN = 0.

    # Dimension
    DIM = 0
    def __init__(self, syn_weight=1.):
        """
        Args:
            syn_weight (:obj: 'float', optional): Scales synaptic weight.
        """
        self.syn_weight = syn_weight

    def i_syn_ij(self, v_pos):
        rho = self.COND_SYN
        wij = self.syn_weight
        return rho*wij*(v_pos - self.RE_PO_SYN)
        # We should not need the followings for static object:
    def fix_integration_index(self, i):
        pass
    def dydt(self, pre_neuron, pos_neuron):
        pass
    def get_initial_condition():
        pass

class StaticInhibitorySynapse:
    """
    A static inhibitory synapse.

    TODO: Does not work!
    """
    COND_SYN = 1.
    RE_PO_SYN = -90.
    #v_pre = -60.0
    # Dimension
    DIM = 1
    def __init__(self, syn_weight=1.):
        """
        Args:
            syn_weight (:obj: 'float', optional): Scales synaptic weight.
        """
        self.syn_weight = syn_weight
        self.r = None
        #self.v_pre = None

    def i_syn_ij(self, v_pos):
        rho = self.COND_SYN
        wij = self.syn_weight
        # Make as a function for v_pre
        a = -rho*wij*(v_pos - self.v_pre)
        print(a)
        return a

    # We should not need the followings for static object:
    def set_integration_index(self, i):
        self.ii = i
        self.r = y(i)
    def dydt(self, pre_neuron, pos_neuron):
        # This is symbolic assignment...
        self.v_pre = pre_neuron.v_mem
        yield 0
    def get_initial_condition(self):
        return [1.0]
    #

class PlasticNMDASynapse:
    """
    A plastic synaspe.

    This synapse only works when the post-synaptic cell has calcium.
    """

    COND_SYN = 3.
    RE_PO_SYN = 0.
    # Nernst/reversal potentials
    HF_PO_NMDA = 20 # NMDA half potential, unit: mV
    # Transmitter shit
    MAX_NMDA = 1. # it is always one!!! don't chnage it
    ALPHA_NMDA = 1.
    BETA_NMDA = 5.
    # Voltage response width (sigma)
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # time constants
    TAU_CA = 5.
    TAU_W = 1000.
    # stdp stuff
    THETA_P = 1.3
    THETA_D = 1.
    GAMMA_P = 220
    GAMMA_D = 100
    W_STAR = 0.5

    # Dimension
    DIM = 2
    def __init__(self,para=None):
        """
        Args:
            para: list of instance specific parameters
        """
        # self.rho_gate = y(i)
        # self.syn_weight = y(i+1)
        self.ii = None # integration index
        self.syn_weight = None #y(i)
        self.rho_gate = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i # integration index
        self.syn_weight = y(i)
        self.rho_gate = y(i+1)

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        ca_pos = pos_neuron.ca
        rho = self.rho_gate # some choice has to be made here
        #rho = pre_neuron.rho_gate
        wij = self.syn_weight
        t_conc = self.MAX_NMDA*sigmoid((v_pre-self.HF_PO_NMDA)/self.V_REW_NMDA)
        yield 1./self.TAU_W*(
            - wij*(1-wij)*(self.W_STAR-wij)
            + self.GAMMA_P*(1-wij)*heaviside(ca_pos - self.THETA_P)
            - self.GAMMA_D*wij*heaviside(ca_pos - self.THETA_D) )
        yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho

    def get_initial_condition(self):
        return [np.random.rand(), 0.]

    def i_syn_ij(self, v_pos):
        rho = self.COND_SYN*self.rho_gate
        wij = self.syn_weight
        return rho*wij*(v_pos - self.RE_PO_SYN)

class PlasticNMDASynapseWithCa:
    """
    A plastic synaspe

    TODO: Define i_syn_ij function and i_syn_ca_ij
    """

    COND_SYN = 3.
    COND_CA_SYN = 1.5

    RE_PO_SYN = 0.
    RE_PO_CA = 140.
    # Nernst/reversal potentials
    HF_PO_NMDA = 20 # NMDA half potential, unit: mV
    # Transmitter shit
    MAX_NMDA = 1. # it is always one!!! don't chnage it
    ALPHA_NMDA = 1.
    BETA_NMDA = 5.
    # Voltage response width (sigma)
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # CALCIUM
    CA_EQM = 0.
    AVO_CONST = 0.03 # "Avogadros" constant, relate calcium concentraion and current
    # time constants
    TAU_CA = 5.
    TAU_W = 1000.
    # stdp stuff
    THETA_P = 1.3
    THETA_D = 1.
    GAMMA_P = 220
    GAMMA_D = 100
    W_STAR = 0.5

    # Dimension
    #DIM = 2
    DIM = 3
    def __init__(self,para=None):
        """
        Args:
            para: list of instance specific parameters
        """
        # self.rho_gate = y(i)
        # self.syn_weight = y(i+1)
        self.ii = None # integration index
        self.syn_weight = None #y(i)
        self.rho_gate = None
        self.ca = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i # integration index
        self.syn_weight = y(i)
        self.rho_gate = y(i+1)
        self.ca = y(i+2) ###

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        v_pos = pos_neuron.v_mem ###
        a_pos = pos_neuron.a_gate ###
        b_pos = pos_neuron.b_gate ###
        i_syn_ca = self.i_syn_ca_ij(v_pos) # negated in post synaptic cell to state how much goes into it, if we leave this as positive, it will describe how much is leaving the synapse
        #ca_pos = pos_neuron.calcium
        rho = self.rho_gate # some choice has to be made here
        #rho = pre_neuron.rho_gate
        wij = self.syn_weight
        t_conc = self.MAX_NMDA*sigmoid((v_pre-self.HF_PO_NMDA)/self.V_REW_NMDA)
        yield 1./self.TAU_W*(
            - wij*(1-wij)*(self.W_STAR-wij)
            + self.GAMMA_P*(1-wij)*heaviside(self.ca - self.THETA_P)
            - self.GAMMA_D*wij*heaviside(self.ca - self.THETA_D) )
        yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho
        yield self.AVO_CONST*(-pos_neuron.i_ca(v_pos, a_pos, b_pos) - i_syn_ca) + (self.CA_EQM-self.ca)/self.TAU_CA
        #This is just the calcium concentration of the post-synaptic cell
    def get_initial_condition(self):
        return [0.5+ 0.1*np.random.rand(), 0., 0.] ###

    def i_syn_ij(self, v_pos):
        rho = self.COND_SYN*self.rho_gate
        wij = self.syn_weight
        return rho*wij*(v_pos - self.RE_PO_SYN)

    def i_syn_ca_ij(self, v_pos):
        #current set such that weight = 0.5??
        rho = self.rho_gate*self.COND_CA_SYN
        return rho*0.5*(v_pos - self.RE_PO_CA)

class HHNeuronWithCa:
    """
    A slight variations of the canonical Hodgkin-Huxley (NaKL) neuron with the
    addition of a small calcium ion channel.
    """
    # Class Parameters:

    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2

    # Conductances
    COND_LEAK = 0.3 # Max. leak conductance, unit: mScm^-2
    COND_NA = 120 # Max. Na conductance, unit: mScm^-2
    COND_K = 36 # Max. K conductance, unit: mScm^-2
    COND_CA = 1. # Max. Ca conductance, unit: mScm^-2
    COND_SYN = 3. #?
    COND_CA_SYN = 1.5

    # Nernst/reversal potentials
    RE_PO_LEAK = -70 # Leak Nernst potential, unit: mV
    RE_PO_NA = 50 # Na Nernst potential, unit: mV
    RE_PO_K = -95 # K Nernst potential, unit: mV
    RE_PO_CA = 140 # K Nernst potential, unit: mV


    # Half potentials of gating variables
    HF_PO_M = -40 # m half potential, unit: mV
    HF_PO_H = -60 # h half potential, unit: mV
    HF_PO_N = -55 # n half potential, unit: mV
    HF_PO_A = -20#-70 # a half potential, unit: mV
    HF_PO_B = -25 #-65 # b half potential, unit: mV

    # Voltage response width (sigma)
    V_REW_M = 16 # m voltage response width, unit: mV
    V_REW_H = -16 # m voltage response width, unit: mV
    V_REW_N = 25 # m voltage response width, unit: mV
    V_REW_A = 13 #10 # m voltage response width, unit: mV
    V_REW_B = -24#-10 # m voltage response width, unit: mV
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV

    # Time constants
    TAU_0_M = 0.1 # unit ms
    TAU_1_M = 0.4
    TAU_0_H = 1.
    TAU_1_H = 7.
    TAU_0_N = 1.
    TAU_1_N = 5.
    TAU_0_A = 0.1
    TAU_1_A = 0.2
    TAU_0_B = 1.
    TAU_1_B = 5.
    TAU_CA = 5.

    # CALCIUM
    CA_EQM = 0.
    AVO_CONST = 0.014085831147459489 # DONT CHANGE IT # "Avogadros" constant, relate calcium concentraion and current

    # Transmitter
    MAX_NMDA = 1.
    ALPHA_NMDA = 1.
    BETA_NMDA = 5.

    # Dimension
    DIM = 7 #8 if exclude rho gate
    def __init__(self,para=None):
        """
        Put all the internal variables and instance specific constants here
        Examples of varibales include Vm, gating variables, calcium ...etc
        Constants can be various conductances, which can vary across
        instances.
        Args:
            para: list of instance specific parameters
        """
        self.i_inj = 0 # injected currents
        self.ii = None # integration index
        self.ni = None # neuron index
        self.v_mem = None #y(i) # membrane potential
        self.m_gate = None #y(i+1)
        self.n_gate = None #y(i+2)
        self.h_gate = None #y(i+3)
        self.a_gate = None #y(i+4)
        self.b_gate = None #y(i+5)
        self.ca = None #y(i+6)

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i # integration index
        self.v_mem = y(i) # membrane potential
        self.m_gate = y(i+1)
        self.n_gate = y(i+2)
        self.h_gate = y(i+3)
        self.a_gate = y(i+4)
        self.b_gate = y(i+5)
        self.ca = y(i+6)

    def set_neuron_index(self, ni):
        """
        Sets the neuron number.
        Args:
            ni (int): neuron index
        """
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected
                pre-synaptically to this neuron.
            pre_neurons: A list of all neuron objects connected
                pre-synaptically to this neuron.

        TODO: Figure out calcium dynamics stuff
        """
        # define how neurons are coupled here
        v = self.v_mem
        m = self.m_gate
        n = self.n_gate
        h = self.h_gate
        a = self.a_gate
        b = self.b_gate
        ca = self.ca
        #rho = self.rho_gate
        i_inj = self.i_inj
        i_leak = self.i_leak
        i_na = self.i_na
        # i_syn = sum(
        #     self.i_syn_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        # i_syn_ca = sum(
        #     self.i_syn_ca_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        i_syn = sum(
            synapse.i_syn_ij(v)
            for (i,synapse) in enumerate(pre_synapses) )
        i_syn_ca = sum(
            synapse.i_syn_ca_ij(v, synapse.rho_gate, synapse.syn_weight)
            for (i,synapse) in enumerate(pre_synapses) )
        # ignores synapse calcium current??
        i_base = (
            i_syn + self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n)
            + self.i_ca(v,a,b))

        yield -1./self.CAP_MEM*(i_base - i_inj)
        yield 1./self.tau_x(
            v, self.HF_PO_M, self.V_REW_M, self.TAU_0_M, self.TAU_1_M
            )*(self.x_eqm(v, self.HF_PO_M, self.V_REW_M) - m)
        yield 1/self.tau_x(
            v, self.HF_PO_N, self.V_REW_N, self.TAU_0_N, self.TAU_1_N
            )*(self.x_eqm(v, self.HF_PO_N, self.V_REW_N) - n)
        yield 1/self.tau_x(
            v, self.HF_PO_H, self.V_REW_H, self.TAU_0_H, self.TAU_1_H
            )*(self.x_eqm(v, self.HF_PO_H, self.V_REW_H) - h)
        yield 1/self.tau_x(
            v, self.HF_PO_A, self.V_REW_A, self.TAU_0_A, self.TAU_1_A
            )*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        yield 1/self.tau_x(
            v, self.HF_PO_B, self.V_REW_B, self.TAU_0_B, self.TAU_1_B
            )*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        yield self.AVO_CONST*(
            -self.i_ca(v,m,h) - i_syn_ca) + (self.CA_EQM-ca)/self.TAU_CA
        #yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho

    def get_initial_condition(self):
        return [-73.,0.2,0.8,0.2,0.2,0.8,0.]

    # some helper functions for dydt
    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    # def t_eqm(self, T):
    #     return ALPHA_NMDA*T/(ALPHA_NMDA*T + BETA_NMDA)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    # def tau_syn(T):
    #     return 1./(ALPHA_NMDA*T + BETA_NMDA)
    #@staticmethod
    def i_leak(self, Vm):
        return self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return self.COND_K*n**4*(Vm - self.RE_PO_K)

    def i_ca(self, Vm, a, b):
        return self.COND_CA*a**2*b*(Vm - self.RE_PO_CA)

    def i_syn_ij(self, Vm_po, rho_ij, W_ij):
        return self.COND_SYN*W_ij*rho_ij*(Vm_po - self.RE_PO_SYN)

    def i_syn_ca_ij(self, Vm_po, rho_ij, W_ij):
        return self.COND_CA_SYN*0.5*rho_ij*(Vm_po - self.RE_PO_CA)

class PlasticNMDASynapseWithCaJL:
    """
    A plastic synaspe inspired by Graupner and Brunel (2012).
    The model used by them has a limited dynamical range of synaptic weight
    fixed the ratio GAMMA_P/(GAMMA_D + GAMMA_P). We relaxed that by a
    modification to the eom of synaptic weight.

    This current only works when the CaJL neuron is the post-synaptic neuron!!!
    """
    COND_SYN = .5 # have to be fiine tuned according to each network
    #COND_CA_SYN = 1.5

    RE_PO_SYN = 0.

    # Nernst/reversal potentials
    HF_PO_NMDA = 20 # NMDA half potential, unit: mV
    RE_PO_CA = 140 # K Nernst potential, unit: mV treating the same as neuron
    # Transmitter shit
    ALPHA_NMDA = 10. # just make it as big as possible so that rho_max is one
    # Voltage response width (sigma)
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # CALCIUM
    CA_EQM = 0.
    RELATIVE_COND_CA_SYN = 1.
    # This choices normalizes both synapse- and voltage-gated calcium peak to 1.
    AVO_CONST_SYN = 0.00095#0.0002 # ~"Avogadros constant", relate calcium concentraion and current
    AVO_CONST_POS = 0.013972995788339456
    #COND_CA_SYN = RELATIVE_COND_CA_SYN*5.119453924914676#1.5
    # time constants
    TAU_CA = 5.
    #TAU_W = 10000.
    TAU_RHO = 1.5*TAU_CA
    # stdp stuff
    # THETA_* are measured in unit of voltage-gated calcium peak = 1
    THETA_P = 0.85
    THETA_D = 0.4
    GAMMA = 0.1
    # if has zero rise time
    # GAMMA_D = 1*np.log(THETA_P)/np.log(THETA_D) # have to be calibtated
    # finite rise time: need calibration
    GAMMA_D = 0.23389830508474577
    # Dimension
    #DIM = 2
    DIM = 3
    def __init__(self,para=None):
        """
        Args:
            para: list of instance specific parameters
        """
        # self.rho_gate = y(i)
        # self.syn_weight = y(i+1)
        self.ii = None # integration index
        self.reduced_weight = None
        self.rho_gate = None
        self.ca = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i # integration index
        #self.syn_weight = y(i)
        self.reduced_weight = y(i)
        self.rho_gate = y(i+1)
        self.ca = y(i+2) ###

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # gating varibales
        v_pre = pre_neuron.v_mem
        v_pos = pos_neuron.v_mem ###
        # a_pos = pos_neuron.a_gate ###
        # b_pos = pos_neuron.b_gate ###
        rw = self.reduced_weight
        #wij = self.syn_weight()
        rho = self.rho_gate
        ca = self.ca
        # calcium currents
        i_syn_ca = self.AVO_CONST_SYN*self.i_syn_ca_ij(v_pos) #leaving synapse
        i_pos_ca = self.AVO_CONST_POS*pos_neuron.i_ca() # why do I care about this??
        i_leak_ca = (self.ca-self.CA_EQM)/self.TAU_CA
        # transmitter (only NMDA here)
        t_conc = sigmoid((v_pre-self.HF_PO_NMDA)/self.V_REW_NMDA)
        # derivatives
        yield self.GAMMA*(
            heaviside(ca- self.THETA_P)
            -self.GAMMA_D*heaviside(ca - self.THETA_D))
        yield self.ALPHA_NMDA*t_conc*(1-rho) - rho/self.TAU_RHO
        #Is this supposed to include post-synaptic cell current?????
        yield - i_syn_ca - i_pos_ca - i_leak_ca
        # yield self.AVO_CONST*( pos_neuron.i_ca(-70., a_pos, b_pos)
        #     + i_syn_ca) + (self.CA_EQM-self.ca)/self.TAU_CA ###
    # helper functions
    # The synaptic current at this particular dendrite/synapse
    # It should depends only on the pos-synaptic voltage
    def i_syn_ca_ij(self, v_pos):
        rho = self.rho_gate
        wij = self.syn_weight()
        #return - self.COND_CA_SYN*rho_ij*(Vm_po - self.RE_PO_CA)
        return wij*rho*(v_pos - self.RE_PO_CA)

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        # No conductance value??
        rho = self.rho_gate*self.COND_SYN
        wij = self.syn_weight()
        return wij*rho*(v_pos - self.RE_PO_SYN)

    def syn_weight(self):
        return sigmoid(self.reduced_weight)

    def get_initial_condition(self):
        return [0.5, 0.5, 0.1] ###

class StdpSynapse:
    """
    A STDP plastic synaspe
    with dynamics of both NMDA and AMPA receptors

    paramters taken from Abarbanel, Henry DI, et al.
    "Synaptic plasticity with discrete state synapses."
    Physical Review E 72.3 (2005): 031914.
    """

    # parameters for synaptic gating variables
    PROP_FAST_NMDA = 0.81
    TAU_NMDA_FAST = 67.5
    PARA_NMDA_FAST = 70./67.5
    TAU_NMDA_SLOW = 245.
    PARA_NMDA_SLOW = 250./245.
    TAU_AMPA = 1.4
    PARA_AMPA = 1.5/1.4

    # reversal potential for excititory synapse
    RE_PO_EX = 0.

    # related conductances
    COND_NMDA = 0.05
    # COND_AMPA = 1.75
    INMDA_TO_CA = 0.15/0.05
    # IAMPA_TO_CA = 1.5e-5/1.75
    ICA_TO_CA = 3.5/0.1
    # G_NMDA = 0.05
    # G_AMPA = 1.75
    # G_C = 1.0e-6
    # G_NC = 0.15
    # G_AC = 1.5e-5
    # G_CC = 3.5e-5

    #magnesium concentration
    MG = 1.
    #calcium
    TAU_CA = 30.
    CA_EQM = 1.

    # brunel model, sensitive parameters
    TAU_W = 150000.
    THETA_P = 1.3
    THETA_D = 1.
    GAMMA_P = 322
    GAMMA_D = 200
    W_STAR = 0.

    # Dimension
    DIM = 5

    def __init__(self, initial_cond=3.5):
        # integration index
        self.ii = None
        # Plasticity: changable maximum conductance of AMPA receptor
        self.stdp_weight = None
        # gating variables of synaptic currents
        self.nmda_gate_fast = None
        self.nmda_gate_slow = None
        self.ampa_gate = None
        # post-synaptic calcium concentration
        self.ca = None
        # initial AMPA conductances before stdp learning
        self.initial_cond = initial_cond

    def set_integration_index(self, i):
        self.ii = i
        self.stdp_weight = y(i)
        self.nmda_gate_fast = y(i+1)
        self.nmda_gate_slow = y(i+2)
        self.ampa_gate = y(i+3)
        self.ca = y(i+4)

    def get_gating_dynamics(self, time_const, v_pre, gating_var, control_para):
        return (1./time_const)*((step(v_pre)-gating_var)/(control_para-step(v_pre)))

    def get_nmda_current(self, v_pos):

        nmda_gate = self.PROP_FAST_NMDA*self.nmda_gate_fast+(1-self.PROP_FAST_NMDA)*self.nmda_gate_slow
        magnesium_control = 1./(1.+0.288*self.MG*sym_backend.exp(-0.062*v_pos))
        return self.COND_NMDA*nmda_gate*magnesium_control*(v_pos-self.RE_PO_EX)

    def get_ampa_current(self, v_pos):
        return self.initial_cond*self.stdp_weight*self.ampa_gate*(v_pos-self.RE_PO_EX)

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        v_pos = pos_neuron.v_mem
        a_pos = pos_neuron.a_gate
        b_pos = pos_neuron.b_gate
        i_nmda = self.get_nmda_current(v_pos)
        i_ampa = self.get_ampa_current(v_pos)

        ca_nmda = self.INMDA_TO_CA*i_nmda
        #ca_ampa = self.IAMPA_TO_CA*i_ampa
        ca_vgcc = self.ICA_TO_CA*pos_neuron.i_ca()

        wij = self.stdp_weight
        yield 1./self.TAU_W*(
            - wij*(1-wij)*(self.W_STAR-wij)
            + self.GAMMA_P*(1-wij)*heaviside(self.ca - self.THETA_P)
            - self.GAMMA_D*wij*heaviside(self.ca - self.THETA_D) )
        yield self.get_gating_dynamics(self.TAU_NMDA_FAST, v_pre, self.nmda_gate_fast, self.PARA_NMDA_FAST)
        yield self.get_gating_dynamics(self.TAU_NMDA_SLOW, v_pre, self.nmda_gate_slow, self.PARA_NMDA_SLOW)
        yield self.get_gating_dynamics(self.TAU_AMPA, v_pre, self.ampa_gate, self.PARA_AMPA)
        #This is the calcium concentration of the post-synaptic cell, ca_ampa is ignored
        yield - ca_nmda - ca_vgcc - (self.ca-self.CA_EQM)/self.TAU_CA

    def i_syn_ij(self, v_pos):
        return self.get_nmda_current(v_pos) + self.get_ampa_current(v_pos)

    def get_initial_condition(self):
        return [0.5, 0.5, 0.5, 0.5, 1.]

class StdpSynapse2:
    """
    A stdp plastic synaspe
    with dynamics of both NMDA and AMPA receptors

    paramters taken from Abarbanel, Henry DI, et al.
    "Synaptic plasticity with discrete state synapses."
    Physical Review E 72.3 (2005): 031914.

    Different from 1 by implementing master equation
    form for updating weight parameters.
    """

    # parameters for synaptic gating variables
    PROP_FAST_NMDA = 0.81
    TAU_NMDA_FAST = 67.5
    PARA_NMDA_FAST = 70./67.5
    TAU_NMDA_SLOW = 245.
    PARA_NMDA_SLOW = 250./245.
    TAU_AMPA = 1.4
    PARA_AMPA = 1.5/1.4

    # reversal potential for excititory synapse
    RE_PO_EX = 0.

    # related conductances
    COND_NMDA = 0.05
    # COND_AMPA = 1.75
    INMDA_TO_CA = 0.15/0.05
    # IAMPA_TO_CA = 1.5e-5/1.75
    ICA_TO_CA = 3.5/0.1
    # G_NMDA = 0.05
    # G_AMPA = 1.75
    # G_C = 1.0e-6
    # G_NC = 0.15
    # G_AC = 1.5e-5
    # G_CC = 3.5e-5

    #bounds on synaptic weight
    G0 = 0.
    G1 = 0.5
    G2 = 1.

    #master equation transitions
    A = 1.0
    B = 1.0

    #magnesium concentration
    MG = 1.
    #calcium
    TAU_CA = 30.
    CA_EQM = 1.

    # Dimension
    DIM = 9


    def __init__(self, initial_cond):
        # integration index
        self.ii = None
        # Plasticity: changable maximum conductance of AMPA receptor
        self.stdp_weight = None
        # gating variables of synaptic currents
        self.nmda_gate_fast = None
        self.nmda_gate_slow = None
        self.ampa_gate = None
        # post-synaptic calcium concentration
        self.ca = None
        # initial AMPA conductances before stdp learning
        self.initial_cond = initial_cond

        self.p0 = None
        self.p1 = None

        self.P = None
        self.D = None

    def set_integration_index(self, i):
        self.ii = i
        self.stdp_weight = y(i)
        self.nmda_gate_fast = y(i+1)
        self.nmda_gate_slow = y(i+2)
        self.ampa_gate = y(i+3)
        self.ca = y(i+4)
        self.p0 = y(i+5)
        self.p1 = y(i+6)
        self.P = y(i+7)
        self.D = y(i+8)

    def get_gating_dynamics(self, time_const, v_pre, gating_var, control_para):
        return (1./time_const)*((step(v_pre)-gating_var)/(control_para-step(v_pre)))

    def get_nmda_current(self, v_pos):

        nmda_gate = self.PROP_FAST_NMDA*self.nmda_gate_fast+(1-self.PROP_FAST_NMDA)*self.nmda_gate_slow
        magnesium_control = 1./(1.+0.288*self.MG*sym_backend.exp(-0.062*v_pos))
        return self.COND_NMDA*nmda_gate*magnesium_control*(v_pos-self.RE_PO_EX)

    def get_ampa_current(self, v_pos):
        return self.initial_cond*self.stdp_weight*self.ampa_gate*(v_pos-self.RE_PO_EX)

    def gamma01(self, P, D):
        return P*D**4
    def gamma10(self, P, D):
        return D*P**4

    def Fp(self, x):
        return x**10.5/(6.7**10.5+x**10.5)

    def Fd(self, x):
        return 1.25*x**4.75/(13.5**4.75+x**4.75)


    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        v_pos = pos_neuron.v_mem
        # a_pos = pos_neuron.a_gate
        # b_pos = pos_neuron.b_gate
        i_nmda = self.get_nmda_current(v_pos)
        # i_ampa = self.get_ampa_current(v_pos)

        ca_nmda = self.INMDA_TO_CA*i_nmda
        #ca_ampa = self.IAMPA_TO_CA*i_ampa
        ca_vgcc = self.ICA_TO_CA*pos_neuron.i_ca()

        f = self.gamma01(self.P, self.D)
        g = self.gamma10(self.P, self.D)

        dca = (self.CA_EQM-self.ca)/self.CA_EQM


        p2 = 1-self.p0-self.p1
        #weight
        yield self.G0*self.p0+self.G1*self.p1+self.G2*p2
        #nmda gate fast
        yield self.get_gating_dynamics(self.TAU_NMDA_FAST, v_pre, self.nmda_gate_fast, self.PARA_NMDA_FAST)
        #nmda gate slow
        yield self.get_gating_dynamics(self.TAU_NMDA_SLOW, v_pre, self.nmda_gate_slow, self.PARA_NMDA_SLOW)
        #ampa gate
        yield self.get_gating_dynamics(self.TAU_AMPA, v_pre, self.ampa_gate, self.PARA_AMPA)
        #This is the calcium concentration of the post-synaptic cell, ca_ampa is ignored
        yield - ca_nmda - ca_vgcc + dca/self.TAU_CA
        #p0
        yield -f*self.p0 + g*self.p1
        #p1
        yield f*self.p0 - g*self.p1 + self.A*f*p2 - self.B*f*self.p1
        #P
        yield self.Fp(dca)*(1-self.P)-self.P/10.
        #D
        yield self.Fd(dca)*(1-self.D)-self.D/30.

    def i_syn_ij(self, v_pos):
        return self.get_nmda_current(v_pos) + self.get_ampa_current(v_pos)

    def get_initial_condition(self):
        return [0.5, 0.5, 0.5, 0.5, 1.]

class SynapseWithDendrite_old:
    """
    A stdp plastic synaspe
    with dynamics of both NMDA and AMPA receptors

    paramters taken from Abarbanel, Henry DI, et al.
    "Synaptic plasticity with discrete state synapses."
    Physical Review E 72.3 (2005): 031914.

    Different from 1 by implementing master equation
    form for updating weight parameters.
    """
    # Parameters:
    # reducing_factor = 0.5
    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2
    # Conductances
    COND_LEAK = 0.813 # Max. leak conductance, unit: mScm^-2
    COND_NA = 215 # Max. Na conductance, unit: mScm^-2
    COND_K = 43 # Max. K conductance, unit: mScm^-2
    COND_CA = 1e-6 # Max. Ca conductance, unit: mScm^-2
    COND_M = 6.7
    COND_A = 100
    # Nernst/reversal potentials
    RE_PO_LEAK = -64 # Leak Nernst potential, unit: mV
    RE_PO_NA = 50 # Na Nernst potential, unit: mV
    RE_PO_K = -95 # K Nernst potential, unit: mV
    # parameters of gating variables
    V_TH = -48
    HF_PO_A = -52 # a half potential, unit: mV
    HF_PO_B = -72 # b half potential, unit: mV
    V_REW_A = 12.4
    V_REW_B = -8
    #ghk
    F = 96.485
    R = 8.314
    T = 298
    # parameters for synaptic gating variables
    PROP_FAST_NMDA = 0.81
    TAU_NMDA_FAST = 67.5
    PARA_NMDA_FAST = 70./67.5
    TAU_NMDA_SLOW = 245.
    PARA_NMDA_SLOW = 250./245.
    TAU_AMPA = 1.4
    PARA_AMPA = 1.5/1.4

    # reversal potential for excititory synapse
    RE_PO_EX = 0.

    # related conductances
    COND_NMDA = 0.05
    # COND_AMPA = 1.75
    # conductance from soma to dendrite
    COND_SOMA_DEND = 1
    COND_DEND_SOMA = 3.5
    ICA_TO_CA = 3.5e-5/1e-6
    INMDA_TO_CA = 0.15/0.05
    G_AMPA_CA = 1.5e-5
    #IAMPA_TO_CA = 1.5e-5/1.75

    # G_NMDA = 0.05
    # G_AMPA = 1.75
    # G_C = 1.0e-6
    # G_NC = 0.15
    # G_AC = 1.5e-5
    # G_CC = 3.5e-5

    #bounds on synaptic weight
    G0 = 2./3.
    G1 = 2
    G2 = 2

    #master equation transitions
    A = 1.0
    B = 1.0

    #magnesium concentration
    MG = 1.
    #calcium
    TAU_CA = 30.
    CA_EQM = 1.

    # Dimension
    DIM = 17


    def __init__(self, initial_cond = 1.75):
        # integration index
        self.ii = None
        # membrane potential and gating variables of dendrite
        self.v_mem = None
        self.m_gate = None
        self.h_gate = None
        self.n_gate = None
        self.a_gate = None
        self.b_gate = None
        self.u_gate = None
        self.ma_gate = None
        self.ha_gate = None
        # Plasticity: changable maximum conductance of AMPA receptor
        self.stdp_weight = None #not a state variable
        # gating variables of synaptic currents
        self.nmda_gate_fast = None
        self.nmda_gate_slow = None
        self.ampa_gate = None

        # post-synaptic calcium concentration
        self.ca = None
        # initial AMPA conductances before stdp learning
        self.initial_cond = initial_cond
        # variables of master equation
        self.p0 = None
        self.p1 = None
        self.P = None
        self.D = None

    def set_integration_index(self, i):
        #DIM = 17
        self.ii = i
        self.v_mem = y(i)
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)
        self.a_gate = y(i+4)
        self.b_gate = y(i+5)
        self.u_gate = y(i+6)
        self.ma_gate = y(i+7)
        self.ha_gate = y(i+8)
        self.nmda_gate_fast = y(i+9)
        self.nmda_gate_slow = y(i+10)
        self.ampa_gate = y(i+11)
        self.ca = y(i+12)
        self.p0 = y(i+13)
        self.p1 = y(i+14)
        self.P = y(i+15)
        self.D = y(i+16)

    def ghk(self, Vm, ca):
        return -(Vm/self.CA_EQM)*(ca - 15000*sym_backend.exp(-2*Vm*self.F/(self.R*self.T)))/(1 - sym_backend.exp(-2*Vm*self.F/(self.R*self.T)))

    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    def tau_a(self, Vm):
        return 0.204 + 0.333/(sym_backend.exp(-(131+Vm)/16.7)+sym_backend.exp((15+Vm)/18.2))

    def tau_b(self, Vm):
        if Vm <= -81:
            return 0.333*(sym_backend.exp((466+Vm)/66.6))
        else:
            return 9.32 + 0.333*(sym_backend.exp(-(21+Vm)/10.5))

    def get_initial_condition(self):
        return [-73.,0.05,0.05,0.9,0.2,0.8]
    def alpha_m(self, Vm):
        return 0.32*(13 - (Vm-self.V_TH))/(sym_backend.exp((13-(Vm-self.V_TH))/4) - 1)

    def beta_m(self, Vm):
        return 0.28*((Vm-self.V_TH) - 40)/(sym_backend.exp(((Vm-self.V_TH)-40)/5) - 1)

    def alpha_h(self, Vm):
        return 0.128*sym_backend.exp((17 - (Vm-self.V_TH))/18)

    def beta_h(self, Vm):
        return 4/(sym_backend.exp((40-(Vm-self.V_TH))/5) + 1)

    def alpha_n(self, Vm):
        return 0.032*(15 - (Vm-self.V_TH))/(sym_backend.exp((15-(Vm-self.V_TH))/5) - 1)

    def beta_n(self, Vm):
        return 0.5/sym_backend.exp(((Vm-self.V_TH)-10)/40)

    def alpha_u(self, Vm):
        return 0.016/sym_backend.exp(-(Vm+52.7)/23)

    def beta_u(self, Vm):
        return 0.016/sym_backend.exp((Vm+52.7)/18.8)

    def alpha_ma(self, Vm):
        return -0.05*(Vm + 20)/(sym_backend.exp(-(Vm+20)/15) - 1)

    def beta_ma(self, Vm):
        return 0.1*(Vm + 10)/(sym_backend.exp((Vm+10)/8) - 1)

    def alpha_ha(self, Vm):
        return 0.00015/sym_backend.exp((Vm+18)/15)

    def beta_ha(self, Vm):
        return 0.06/(sym_backend.exp(-(Vm+73)/12) + 1)

    def i_leak(self, Vm):
        return -self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return -self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return -self.COND_K*n**4*(Vm - self.RE_PO_K)

    def i_ca(self, Vm, a, b):
        return self.COND_CA*self.ghk(Vm,self.ca)*a**2*b

    def i_m(self, Vm, u):
        return -self.COND_M*u**2*(Vm - self.RE_PO_K)

    def i_a(self, Vm, ma, ha):
        return -self.COND_A*ma*ha*(Vm - self.RE_PO_K)

    def get_gating_dynamics(self, time_const, v_pre, gating_var, control_para):
        return (1./time_const)*((step(v_pre)-gating_var)/(control_para-step(v_pre)))

    def get_nmda_current(self, v):
        nmda_gate = self.PROP_FAST_NMDA*self.nmda_gate_fast+(1-self.PROP_FAST_NMDA)*self.nmda_gate_slow
        magnesium_control = 1./(1.+0.288*self.MG*sym_backend.exp(-0.062*v))
        return -self.COND_NMDA*nmda_gate*magnesium_control*(v-self.RE_PO_EX)

    def get_ampa_current(self, v):
        return -self.initial_cond*self.stdp_weight*self.ampa_gate*(v-self.RE_PO_EX)

    def i_syn(self, v):
        return self.get_nmda_current(v) + self.get_ampa_current(v)

    def gamma01(self, P, D):
        return P*D**4

    def gamma10(self, P, D):
        return D*P**4

    def Fp(self, x):
        return x**10/(6.7**10+x**10)

    def Fd(self, x):
        return 1.25*x**5/(13.5**5+x**5)

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        v = self.v_mem
        m = self.m_gate
        h = self.h_gate
        n = self.n_gate
        a = self.a_gate
        b = self.b_gate
        u = self.u_gate
        ma = self.ma_gate
        ha = self.ha_gate

        dca = (self.ca - self.CA_EQM)/self.CA_EQM
        p2 = 1 - self.p0 - self.p1
        self.stdp_weight = self.G0*self.p0+self.G1*self.p1+self.G2*p2
        f = self.gamma01(self.P, self.D)
        g = self.gamma10(self.P, self.D)

        i_base = self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n) + self.i_ca(v,a,b) + self.i_m(v,u) + self.i_a(v,ma,ha)
        i_nmda = self.get_nmda_current(v)
        i_ampa = self.get_ampa_current(v)
        i_sd = self.COND_SOMA_DEND*(pos_neuron.v_mem - v)

        ca_nmda = self.INMDA_TO_CA*i_nmda
        ca_ampa = -self.G_AMPA_CA*self.ampa_gate*(v-self.RE_PO_EX)
        ca_vgcc = self.ICA_TO_CA*self.i_ca(v,a,b)

        # membrane potential and gating variables of dendrite
        yield 1/self.CAP_MEM*(i_base + self.i_syn(v) + i_sd - 7.0)
        # yield 1/self.tau_x(
        #     v, self.HF_PO_M, self.V_REW_M, self.TAU_0_M, self.TAU_1_M
        #     )*(self.x_eqm(v, self.HF_PO_M, self.V_REW_M) - m)
        # yield 1/self.tau_x(
        #     v, self.HF_PO_N, self.V_REW_N, self.TAU_0_N, self.TAU_1_N
        #     )*(self.x_eqm(v, self.HF_PO_N, self.V_REW_N) - n)
        # yield 1/self.tau_x(
        #     v, self.HF_PO_H, self.V_REW_H, self.TAU_0_H, self.TAU_1_H
        #     )*(self.x_eqm(v, self.HF_PO_H, self.V_REW_H) - h)
        yield self.alpha_m(v)*(1-m) - self.beta_m(v)*m
        yield self.alpha_h(v)*(1-h) - self.beta_h(v)*h
        yield self.alpha_n(v)*(1-n) - self.beta_n(v)*n
        # yield 1/self.tau_x(
        #     v, self.HF_PO_A, self.V_REW_A, self.TAU_0_A, self.TAU_1_A
        #     )*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        # yield 1/self.tau_x(
        #     v, self.HF_PO_B, self.V_REW_B, self.TAU_0_B, self.TAU_1_B
        #     )*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        yield 1/self.tau_a(v)*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        yield 1/self.tau_b(v)*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        yield self.alpha_u(v)*(1-u) - self.beta_u(v)*u
        yield self.alpha_ma(v)*(1-ma) - self.beta_ma(v)*ma
        yield self.alpha_ha(v)*(1-ha) - self.beta_ha(v)*ha
        #nmda gate fast
        yield self.get_gating_dynamics(self.TAU_NMDA_FAST, v_pre, self.nmda_gate_fast, self.PARA_NMDA_FAST)
        #nmda gate slow
        yield self.get_gating_dynamics(self.TAU_NMDA_SLOW, v_pre, self.nmda_gate_slow, self.PARA_NMDA_SLOW)
        #ampa gate
        yield self.get_gating_dynamics(self.TAU_AMPA, v_pre, self.ampa_gate, self.PARA_AMPA)
        #This is the calcium concentration of the post-synaptic cell, ca_ampa is ignored
        yield ca_nmda + ca_vgcc - dca/self.TAU_CA
        #p0
        yield -f*self.p0 + g*self.p1
        #p1
        yield f*self.p0 - g*self.p1 + self.A*f*p2 - self.B*f*self.p1
        #P
        yield self.Fp(dca)*(1-self.P) - self.P/10.
        #D
        yield self.Fd(dca)*(1-self.D) - self.D/30.

    def get_initial_condition(self):
        #DIM = 17
        temp = []
        temp.append(-75)
        temp.append(0.1) # m
        temp.append(0.7) # h
        temp.append(0.3) # n
        temp.append(0.01) # a
        temp.append(0.01) # b
        temp.append(0.01) # u
        temp.append(0.01) # ma
        temp.append(0.01) # ha
        temp.append(0.01) # nmda_gate_fast
        temp.append(0.01) # nmda_gate_slow
        temp.append(0.01) # ampa_gate
        temp.append(1.0) # ca
        temp.append(0.75) # p0
        temp.append(0.25) # p1
        temp.append(0.01) # P
        temp.append(0.01) # D
        return temp

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the toal synaptic current, used by the post-synaptic cell
        """
        return self.COND_DEND_SOMA*(self.v_mem - v_pos)

class SynapseWithDendrite_old2:
    """
    A stdp plastic synaspe
    with dynamics of both NMDA and AMPA receptors

    paramters taken from Abarbanel, Henry DI, et al.
    "Synaptic plasticity with discrete state synapses."
    Physical Review E 72.3 (2005): 031914.

    Different from 1 by implementing master equation
    form for updating weight parameters.
    """
    # Parameters:
    COND_CA = 0 #4e-6 * 0.09 # Max. Ca conductance, unit: mScm^-2
    COND_NMDA = 0.005 * 10 #amplitude 4
    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2
    # Conductances
    COND_LEAK = 0.03 # Max. leak conductance, unit: mScm^-2
    COND_NA = 20 # Max. Na conductance, unit: mScm^-2
    COND_K = 5.2 # Max. K conductance, unit: mScm^-2
    # Nernst/reversal potentials
    RE_PO_LEAK = -49.4 # Leak Nernst potential, unit: mV
    RE_PO_NA = 55 # Na Nernst potential, unit: mV
    RE_PO_K = -72 # K Nernst potential, unit: mV
    # parameters of gating variables
    HF_PO_A = -56 # a half potential, unit: mV
    HF_PO_B = -80 # b half potential, unit: mV
    V_REW_A = 12.4
    V_REW_B = -8
    #ghk
    CA_EX = 15000
    FRT = 0.039
    # parameters for synaptic gating variables
    #PROP_FAST_NMDA = 0.81
    TAU_NMDA = 37.
    PARA_NMDA = 40./37.
    TAU_AMPA = 1.4
    PARA_AMPA = 1.5/1.4

    # reversal potential for excititory synapse
    RE_PO_EX = 0.

    # conductance from soma to dendrite
    COND_SOMA_DEND = 1
    COND_DEND_SOMA = 3.5

    INMDA_TO_CA = 0.0298/0.005
    G_AMPA_CA = 1.5e-4
    ICA_TO_CA = 7e-5/4e-6

    #bounds on synaptic weight
    G0 = 2./3.
    G1 = 2
    G2 = 2

    #master equation transitions
    A = 1.0
    B = 1.0

    #magnesium concentration
    MG = 1.2
    #calcium
    TAU_CA = 30.
    CA_EQM = 1.

    ##########
    # Dimension
    DIM = 13

    def __init__(self, initial_cond = 0.06):
        # integration index
        self.ii = None
        # membrane potential and gating variables of dendrite
        self.v_mem = None
        self.m_gate = None
        self.h_gate = None
        self.n_gate = None
        self.a_gate = None
        self.b_gate = None
        # Plasticity: changable maximum conductance of AMPA receptor
        self.stdp_weight = None #not a state variable
        # gating variables of synaptic currents
        self.nmda_gate = None
        self.ampa_gate = None
        # post-synaptic calcium concentration
        self.ca = None
        # initial AMPA conductances before stdp learning
        self.initial_cond = initial_cond
        # variables of master equation
        self.p0 = None
        self.p1 = None
        self.P = None
        self.D = None

    def set_integration_index(self, i):
        self.ii = i
        self.v_mem = y(i)
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)
        self.a_gate = y(i+4)
        self.b_gate = y(i+5)
        self.nmda_gate = y(i+6)
        self.ampa_gate = y(i+7)
        self.ca = y(i+8)
        self.p0 = y(i+9)
        self.p1 = y(i+10)
        self.P = y(i+11)
        self.D = y(i+12)

    def get_initial_condition(self):
        temp = []
        temp.append(-77)
        temp.append(0.1) # m
        temp.append(0.4) # h
        temp.append(0.4) # n
        temp.append(0.) # a
        #temp.append(0.4357) # a
        temp.append(0.) # a
        #temp.append(0.0014) # b
        temp.append(0.01) # nmda_gate
        temp.append(0.01) # ampa_gate
        temp.append(1.0) # ca
        temp.append(0.75) # p0
        temp.append(0.25) # p1
        temp.append(0.01) # P
        temp.append(0.01) # D
        return temp

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        v = self.v_mem
        m = self.m_gate
        h = self.h_gate
        n = self.n_gate
        a = self.a_gate
        b = self.b_gate

        dca = (self.ca - self.CA_EQM)/self.CA_EQM
        p2 = 1 - self.p0 - self.p1
        self.stdp_weight = self.G0*self.p0+self.G1*self.p1+self.G2*p2
        f = self.gamma01(self.P, self.D)
        g = self.gamma10(self.P, self.D)

        i_base = self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n) + self.i_ca(v,a,b)
        i_nmda = self.get_nmda_current(v)
        i_ampa = self.get_ampa_current(v)
        i_sd = self.COND_SOMA_DEND*(pos_neuron.v_mem - v)

        ca_nmda = self.INMDA_TO_CA*i_nmda
        ca_ampa = -self.G_AMPA_CA*self.ampa_gate*(v-self.RE_PO_EX)
        ca_vgcc = self.ICA_TO_CA*self.i_ca(v,a,b)

        # membrane potential and gating variables of dendrite
        yield 1/self.CAP_MEM*(i_base + self.i_syn(v) + i_sd - 0.85)
        yield self.alpha_m(v)*(1-m) - self.beta_m(v)*m
        yield self.alpha_h(v)*(1-h) - self.beta_h(v)*h
        yield self.alpha_n(v)*(1-n) - self.beta_n(v)*n
        yield 1/self.tau_a(v)*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        yield 1/self.tau_b(v)*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        #nmda gate
        yield self.get_gating_dynamics(self.TAU_NMDA, v_pre, self.nmda_gate, self.PARA_NMDA)
        #ampa gate
        yield self.get_gating_dynamics(self.TAU_AMPA, v_pre, self.ampa_gate, self.PARA_AMPA)
        #This is the calcium concentration of the post-synaptic cell, ca_ampa is ignored
        yield ca_nmda + ca_vgcc - dca/self.TAU_CA
        #p0
        yield -f*self.p0 + g*self.p1
        #p1
        yield f*self.p0 - g*self.p1 + self.A*f*p2 - self.B*f*self.p1
        #P
        yield self.Fp(dca)*(1-self.P) - self.P/10.
        #D
        yield self.Fd(dca)*(1-self.D) - self.D/30.

    def ghk(self, Vm, ca):
        return -(Vm/self.CA_EQM)*(ca - self.CA_EX*sym_backend.exp(-2*Vm*self.FRT))/(1 - sym_backend.exp(-2*Vm*self.FRT))

    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    def tau_a(self, Vm):
        return 0.204 + 0.333/(sym_backend.exp(-(131+Vm)/16.7)+sym_backend.exp((15+Vm)/18.2))

    def tau_b(self, Vm):
        if Vm <= -81:
            return 0.333*(sym_backend.exp((466+Vm)/66.6))
        else:
            return 9.32 + 0.333*(sym_backend.exp(-(21+Vm)/10.5))

    def alpha_m(self, Vm):
        return -0.1*(35 + Vm)/(sym_backend.exp(-(35+Vm)/10) - 1)

    def beta_m(self, Vm):
        return 4*sym_backend.exp(-(60+Vm)/18)

    def alpha_h(self, Vm):
        return 0.07*sym_backend.exp(-(60+Vm)/20)

    def beta_h(self, Vm):
        return 1/(sym_backend.exp(-(30+Vm)/10) + 1)

    def alpha_n(self, Vm):
        return -0.01*(50 + Vm)/(sym_backend.exp(-(50+Vm)/10) - 1)

    def beta_n(self, Vm):
        return 0.125*sym_backend.exp(-(60+Vm)/80)

    def i_leak(self, Vm):
        return -self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return -self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return -self.COND_K*n**4*(Vm - self.RE_PO_K)

    def i_ca(self, Vm, a, b):
        return self.COND_CA*self.ghk(Vm,self.ca)*a**2*b

    def get_gating_dynamics(self, time_const, v_pre, gating_var, control_para):
        return (1./time_const)*((step(v_pre)-gating_var)/(control_para-step(v_pre)))

    def get_nmda_current(self, v):
        magnesium_control = 1./(1.+(1./3.57)*self.MG*sym_backend.exp(-0.062*v))
        return -self.COND_NMDA*self.nmda_gate*magnesium_control*(v-self.RE_PO_EX)

    def get_ampa_current(self, v):
        return -self.initial_cond*self.stdp_weight*self.ampa_gate*(v-self.RE_PO_EX)

    def i_syn(self, v):
        return self.get_nmda_current(v) + self.get_ampa_current(v)

    def gamma01(self, P, D):
        return P*D**4

    def gamma10(self, P, D):
        return D*P**4

    def Fp(self, x):
        return x**10/(6.7**10+x**10)

    def Fd(self, x):
        return 1.25*x**5/(13.5**5+x**5)

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the toal synaptic current, used by the post-synaptic cell
        """
        return self.COND_DEND_SOMA*(self.v_mem - v_pos)

class SynapseWithDendrite:
    """
    A stdp plastic synaspe
    with dynamics of both NMDA and AMPA receptors

    paramters taken from Abarbanel, Henry DI, et al.
    "Synaptic plasticity with discrete state synapses."
    Physical Review E 72.3 (2005): 031914.

    Different from 1 by implementing master equation
    form for updating weight parameters.
    """
    # Parameters:
    I_DC = 0 # -0.85 #modify the level of rest potential
    COND_CA = 4e-6 * 0.18 # Max. Ca conductance, unit: mScm^-2
    COND_NMDA = 0.005 * 5.5 #amplitude 4
    reducing_factor = 1
    # conductance from soma to dendrite
    COND_SOMA_DEND = 1.0
    COND_DEND_SOMA = 3.5

    INMDA_TO_CA = 0.0298/0.005 * reducing_factor
    ICA_TO_CA = 7e-5/4e-6 * reducing_factor
    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2
    # Conductances
    COND_LEAK = 0.03 # Max. leak conductance, unit: mScm^-2
    COND_NA = 20 # Max. Na conductance, unit: mScm^-2
    COND_K = 5.2 # Max. K conductance, unit: mScm^-2
    # Nernst/reversal potentials
    RE_PO_LEAK = -49.4 # Leak Nernst potential, unit: mV
    RE_PO_NA = 55 # Na Nernst potential, unit: mV
    RE_PO_K = -72 # K Nernst potential, unit: mV
    # parameters of gating variables
    HF_PO_A = -56 # a half potential, unit: mV
    HF_PO_B = -80 # b half potential, unit: mV
    V_REW_A = 12.4
    V_REW_B = -8
    #ghk
    CA_EX = 15000
    FRT = 0.039
    # parameters for synaptic gating variables
    #PROP_FAST_NMDA = 0.81
    TAU_NMDA = 37.
    PARA_NMDA = 40./37.
    TAU_AMPA = 1.4
    PARA_AMPA = 1.5/1.4

    # reversal potential for excititory synapse
    RE_PO_EX = 0.

    #bounds on synaptic weight
    G0 = 2./3.
    G1 = 2
    G2 = 2

    #master equation transitions
    A = 1.0
    B = 1.0

    #magnesium concentration
    MG = 1.2
    #calcium
    TAU_CA = 30.
    CA_EQM = 1.

    ##########
    # Dimension
    DIM = 13

    def __init__(self, initial_cond = 0.06):
        # integration index
        self.ii = None
        # membrane potential and gating variables of dendrite
        self.v_mem = None
        self.m_gate = None
        self.h_gate = None
        self.n_gate = None
        self.a_gate = None
        self.b_gate = None
        # Plasticity: changable maximum conductance of AMPA receptor
        self.stdp_weight = None #not a state variable
        # gating variables of synaptic currents
        self.nmda_gate = None
        self.ampa_gate = None
        # post-synaptic calcium concentration
        self.ca = None
        # initial AMPA conductances before stdp learning
        self.initial_cond = initial_cond
        # variables of master equation
        self.p0 = None
        self.p1 = None
        self.P = None
        self.D = None

    def set_integration_index(self, i):
        self.ii = i
        self.v_mem = y(i)
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)
        self.a_gate = y(i+4)
        self.b_gate = y(i+5)
        self.nmda_gate = y(i+6)
        self.ampa_gate = y(i+7)
        self.ca = y(i+8)
        self.p0 = y(i+9)
        self.p1 = y(i+10)
        self.P = y(i+11)
        self.D = y(i+12)

    def get_initial_condition(self):
        temp = []
        temp.append(-77)
        temp.append(0.1) # m
        temp.append(0.4) # h
        temp.append(0.4) # n
        temp.append(0.) # a
        #temp.append(0.4357) # a
        temp.append(0.) # a
        #temp.append(0.0014) # b
        temp.append(0.01) # nmda_gate
        temp.append(0.01) # ampa_gate
        temp.append(1.0) # ca
        temp.append(0.75) # p0
        temp.append(0.25) # p1
        temp.append(0.01) # P
        temp.append(0.01) # D
        return temp

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        v = self.v_mem
        m = self.m_gate
        h = self.h_gate
        n = self.n_gate
        a = self.a_gate
        b = self.b_gate

        dca = (self.ca - self.CA_EQM)/self.CA_EQM
        p2 = 1 - self.p0 - self.p1
        self.stdp_weight = self.G0*self.p0+self.G1*self.p1+self.G2*p2
        f = self.gamma01(self.P, self.D)
        g = self.gamma10(self.P, self.D)

        i_base = self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n) + self.i_ca(v,a,b)
        i_nmda = self.get_nmda_current(v)
        i_ampa = self.get_ampa_current(v)
        i_sd = self.COND_SOMA_DEND*(pos_neuron.v_mem - v)

        ca_nmda = self.INMDA_TO_CA*i_nmda
        ca_vgcc = self.ICA_TO_CA*self.i_ca(v,a,b)

        # membrane potential and gating variables of dendrite
        yield 1/self.CAP_MEM*(i_base + self.i_syn(v) + i_sd + self.I_DC)
        yield self.alpha_m(v)*(1-m) - self.beta_m(v)*m
        yield self.alpha_h(v)*(1-h) - self.beta_h(v)*h
        yield self.alpha_n(v)*(1-n) - self.beta_n(v)*n
        yield 1/self.tau_a(v)*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        yield 1/self.tau_b(v)*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        #nmda gate
        yield self.get_gating_dynamics(self.TAU_NMDA, v_pre, self.nmda_gate, self.PARA_NMDA)
        #ampa gate
        yield self.get_gating_dynamics(self.TAU_AMPA, v_pre, self.ampa_gate, self.PARA_AMPA)
        #This is the calcium concentration of the post-synaptic cell, ca_ampa is ignored
        yield ca_nmda + ca_vgcc - dca/self.TAU_CA
        #p0
        yield -f*self.p0 + g*self.p1
        #p1
        yield f*self.p0 - g*self.p1 + self.A*f*p2 - self.B*f*self.p1
        #P
        yield self.Fp(dca)*(1-self.P) - self.P/10.
        #D
        yield self.Fd(dca)*(1-self.D) - self.D/30.

    def ghk(self, Vm, ca):
        return -(Vm/self.CA_EQM)*(ca - self.CA_EX*sym_backend.exp(-2*Vm*self.FRT))/(1 - sym_backend.exp(-2*Vm*self.FRT))

    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    def tau_a(self, Vm):
        return 0.204 + 0.333/(sym_backend.exp(-(131+Vm)/16.7)+sym_backend.exp((15+Vm)/18.2))

    def tau_b(self, Vm):
        if Vm <= -81:
            return 0.333*(sym_backend.exp((466+Vm)/66.6))
        else:
            return 9.32 + 0.333*(sym_backend.exp(-(21+Vm)/10.5))

    def alpha_m(self, Vm):
        return -0.1*(35 + Vm)/(sym_backend.exp(-(35+Vm)/10) - 1)

    def beta_m(self, Vm):
        return 4*sym_backend.exp(-(60+Vm)/18)

    def alpha_h(self, Vm):
        return 0.07*sym_backend.exp(-(60+Vm)/20)

    def beta_h(self, Vm):
        return 1/(sym_backend.exp(-(30+Vm)/10) + 1)

    def alpha_n(self, Vm):
        return -0.01*(50 + Vm)/(sym_backend.exp(-(50+Vm)/10) - 1)

    def beta_n(self, Vm):
        return 0.125*sym_backend.exp(-(60+Vm)/80)

    def i_leak(self, Vm):
        return -self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return -self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return -self.COND_K*n**4*(Vm - self.RE_PO_K)

    def i_ca(self, Vm, a, b):
        return self.COND_CA*self.ghk(Vm,self.ca)*a**2*b

    def get_gating_dynamics(self, time_const, v_pre, gating_var, control_para):
        return (1./time_const)*((step(v_pre)-gating_var)/(control_para-step(v_pre)))

    def get_nmda_current(self, v):
        magnesium_control = 1./(1.+(1./3.57)*self.MG*sym_backend.exp(-0.062*v))
        return -self.COND_NMDA*self.nmda_gate*magnesium_control*(v-self.RE_PO_EX)

    def get_ampa_current(self, v):
        return -self.initial_cond*self.stdp_weight*self.ampa_gate*(v-self.RE_PO_EX)

    def i_syn(self, v):
        return self.get_nmda_current(v) + self.get_ampa_current(v)

    def gamma01(self, P, D):
        return P*D**4

    def gamma10(self, P, D):
        return D*P**4

    def Fp(self, x):
        return x**10/(6.7**10+x**10)

    def Fd(self, x):
        return 1.25*x**5/(13.5**5+x**5)

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the toal synaptic current, used by the post-synaptic cell
        """
        return self.COND_DEND_SOMA*(self.v_mem - v_pos)

class Soma:
    # Parameters:
    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2
    # Conductances
    COND_LEAK = 0.03 # Max. leak conductance, unit: mScm^-2
    COND_NA = 20 # Max. Na conductance, unit: mScm^-2
    COND_K = 5.2 # Max. K conductance, unit: mScm^-2
    # Nernst/reversal potentials
    RE_PO_LEAK = -49.4 # Leak Nernst potential, unit: mV
    RE_PO_NA = 55 # Na Nernst potential, unit: mV
    RE_PO_K = -72 # K Nernst potential, unit: mV
    # paramters of gating variables
    #V_TH = -65
    # # old model
    # HF_PO_M = -40 # m half potential, unit: mV
    # HF_PO_H = -60 # h half potential, unit: mV
    # HF_PO_N = -55 # n half potential, unit: mV
    # V_REW_M = 16 # m voltage response width, unit: mV
    # V_REW_H = -16 # m voltage response width, unit: mV
    # V_REW_N = 25 # m voltage response width, unit: mV
    # TAU_0_M = 0.1 # unit ms
    # TAU_1_M = 0.4
    # TAU_0_H = 1.
    # TAU_1_H = 7.
    # TAU_0_N = 1.
    # TAU_1_N = 5.

    COND_DEND_SOMA = 3.5

    # Dimension
    DIM = 4
    def __init__(self,para=None):
        """
        Put all the internal variables and instance specific constants here
        Examples of varibales include Vm, gating variables, calcium ...etc
        Constants can be various conductances, which can vary across
        instances.
        Args:
            para: list of instance specific parameters
            i_inj: injected current
        """
        self.i_inj = 0 # injected currents
        self.ii = None # integration index
        self.ni = None # neruon index
        self.v_mem = None #y(i) # membrane potential
        self.m_gate = None #y(i+1)
        self.h_gate = None #y(i+2)
        self.n_gate = None #y(i+3)

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i # integration index
        self.v_mem = y(i) # membrane potential
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        v = self.v_mem
        m = self.m_gate
        h = self.h_gate
        n = self.n_gate
        i_inj = self.i_inj

        i_ds = sum([synapse.i_syn_ij(v) for (i,synapse) in enumerate(pre_synapses)])

        # i_ds = sum(self.COND_DEND_SOMA*(synapse.v_mem - v)
        #     for (i,synapse) in enumerate(pre_synapses))

        i_base = self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n)

        yield 1/self.CAP_MEM*(i_base + i_ds + i_inj - 0.85)
        yield self.alpha_m(v)*(1-m) - self.beta_m(v)*m
        yield self.alpha_h(v)*(1-h) - self.beta_h(v)*h
        yield self.alpha_n(v)*(1-n) - self.beta_n(v)*n
        # yield 1/self.tau_x(
        #     v, self.HF_PO_M, self.V_REW_M, self.TAU_0_M, self.TAU_1_M
        #     )*(self.x_eqm(v, self.HF_PO_M, self.V_REW_M) - m)
        # yield 1/self.tau_x(
        #     v, self.HF_PO_N, self.V_REW_N, self.TAU_0_N, self.TAU_1_N
        #     )*(self.x_eqm(v, self.HF_PO_N, self.V_REW_N) - n)
        # yield 1/self.tau_x(
        #     v, self.HF_PO_H, self.V_REW_H, self.TAU_0_H, self.TAU_1_H
        #     )*(self.x_eqm(v, self.HF_PO_H, self.V_REW_H) - h)

    def get_initial_condition(self):
        return [-77.,0.1,0.4,0.4]

    # some helper functions for dydt
    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    def alpha_m(self, Vm):
        return -0.1*(35 + Vm)/(sym_backend.exp(-(35+Vm)/10) - 1)

    def beta_m(self, Vm):
        return 4*sym_backend.exp(-(60+Vm)/18)

    def alpha_h(self, Vm):
        return 0.07*sym_backend.exp(-(60+Vm)/20)

    def beta_h(self, Vm):
        return 1/(sym_backend.exp(-(30+Vm)/10) + 1)

    def alpha_n(self, Vm):
        return -0.01*(50 + Vm)/(sym_backend.exp(-(50+Vm)/10) - 1)

    def beta_n(self, Vm):
        return 0.125*sym_backend.exp(-(60+Vm)/80)

    def i_leak(self, Vm):
        return -self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return -self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return -self.COND_K*n**4*(Vm - self.RE_PO_K)


class HHNeuronWithCaJL:
    """
    Actually a slight variations of the canonical Hodgkin-Huxley neuron.
    Originally we added calcium as a perturbation, which is not important
    in the neuron dynamics anyway. Here the ca is just tagging along, so are
    the relevant gating varibales. They are important in the synaptic activity
    but not neuron activity.
    Also the original motivation was to treat all gating variables on equal
    footing so that their taus and x_eqm have the same functional form. It
    probably does not matter much...?

    TODO: Figure out what the heck this neuron does... It seems like calcium is
    all commented out. There is something with the synapse where there is some
    form of reduced weight, but it is very unclear.
    """
    # Parameters:
    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2
    # Conductances
    COND_LEAK = 0.3 # Max. leak conductance, unit: mScm^-2
    COND_NA = 120 # Max. Na conductance, unit: mScm^-2
    COND_K = 36 # Max. K conductance, unit: mScm^-2
    COND_CA = 1. # Max. Ca conductance, unit: mScm^-2
    COND_SYN = .5 # have to be fiine tuned according to each network
    #COND_CA_SYN = 1.5
    # Nernst/reversal potentials
    RE_PO_LEAK = -70 # Leak Nernst potential, unit: mV
    RE_PO_NA = 50 # Na Nernst potential, unit: mV
    RE_PO_K = -95 # K Nernst potential, unit: mV
    RE_PO_CA = 140 # K Nernst potential, unit: mV
    RE_PO_SYN = 0.
    # Half potentials of gating variables
    HF_PO_M = -40 # m half potential, unit: mV
    HF_PO_H = -60 # h half potential, unit: mV
    HF_PO_N = -55 # n half potential, unit: mV
    HF_PO_A = -20#-70 # a half potential, unit: mV
    HF_PO_B = -25 #-65 # b half potential, unit: mV
    # Voltage response width (sigma)
    V_REW_M = 16 # m voltage response width, unit: mV
    V_REW_H = -16 # m voltage response width, unit: mV
    V_REW_N = 25 # m voltage response width, unit: mV
    V_REW_A = 13 #10 # m voltage response width, unit: mV
    V_REW_B = -24#-10 # m voltage response width, unit: mV
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # time constants
    TAU_0_M = 0.1 # unit ms
    TAU_1_M = 0.4
    TAU_0_H = 1.
    TAU_1_H = 7.
    TAU_0_N = 1.
    TAU_1_N = 5.
    TAU_0_A = 0.1
    TAU_1_A = 0.2
    TAU_0_B = 1.
    TAU_1_B = 5.
    TAU_CA = 5.
    # CALCIUM
    #CA_EQM = 0.
    #AVO_CONST = 0.014085831147459489 # DONT CHANGE IT # "Avogadros" constant, relate calcium concentraion and current

    # Dimension
    DIM = 6  # 8 if exclude rho gate
    def __init__(self,para=None):
        """
        Put all the internal variables and instance specific constants here
        Examples of varibales include Vm, gating variables, calcium ...etc
        Constants can be various conductances, which can vary across
        instances.
        Args:
            para: list of instance specific parameters
            i_inj: injected current
        """
        self.i_inj = 0 # injected currents
        self.ii = None # integration index
        self.ni = None # neruon index
        self.v_mem = None #y(i) # membrane potential
        self.m_gate = None #y(i+1)
        self.n_gate = None #y(i+2)
        self.h_gate = None #y(i+3)
        self.a_gate = None #y(i+4)
        self.b_gate = None #y(i+5)
        #self.calcium = None #y(i+6)
        #self.rho_gate = None #y(i+7)
        # may put para here

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i # integration index
        self.v_mem = y(i) # membrane potential
        self.m_gate = y(i+1)
        self.n_gate = y(i+2)
        self.h_gate = y(i+3)
        self.a_gate = y(i+4)
        self.b_gate = y(i+5)
        #self.calcium = y(i+6)
        #self.rho_gate = y(i+7)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        v = self.v_mem
        m = self.m_gate
        n = self.n_gate
        h = self.h_gate
        a = self.a_gate
        b = self.b_gate
        #ca = self.calcium
        #rho = self.rho_gate
        i_inj = self.i_inj
        i_leak = self.i_leak
        i_na = self.i_na
        # i_syn = sum(
        #     self.i_syn_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        # i_syn_ca = sum(
        #     self.i_syn_ca_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        i_syn = sum(synapse.i_syn_ij(v)
            for (i,synapse) in enumerate(pre_synapses))
        # i_syn_ca = sum(
        #     self.i_syn_ca_ij(v, synapse.rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        i_base = i_syn + self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n)
            #+ self.i_ca(v,a,b) )

        yield -1/self.CAP_MEM*(i_base - i_inj)
        yield 1/self.tau_x(
            v, self.HF_PO_M, self.V_REW_M, self.TAU_0_M, self.TAU_1_M
            )*(self.x_eqm(v, self.HF_PO_M, self.V_REW_M) - m)
        yield 1/self.tau_x(
            v, self.HF_PO_N, self.V_REW_N, self.TAU_0_N, self.TAU_1_N
            )*(self.x_eqm(v, self.HF_PO_N, self.V_REW_N) - n)
        yield 1/self.tau_x(
            v, self.HF_PO_H, self.V_REW_H, self.TAU_0_H, self.TAU_1_H
            )*(self.x_eqm(v, self.HF_PO_H, self.V_REW_H) - h)
        yield 1/self.tau_x(
            v, self.HF_PO_A, self.V_REW_A, self.TAU_0_A, self.TAU_1_A
            )*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        yield 1/self.tau_x(
            v, self.HF_PO_B, self.V_REW_B, self.TAU_0_B, self.TAU_1_B
            )*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        # yield self.AVO_CONST*(
        #     self.i_ca(v,m,h) + i_syn_ca) + (self.CA_EQM-ca)/self.TAU_CA
        #yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho

    def get_initial_condition(self):
        return [-73.,0.2,0.8,0.2,0.2,0.8]

    # some helper functions for dydt
    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    # def t_eqm(self, T):
    #     return ALPHA_NMDA*T/(ALPHA_NMDA*T + BETA_NMDA)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    # def tau_syn(T):
    #     return 1./(ALPHA_NMDA*T + BETA_NMDA)
    #@staticmethod
    def i_leak(self, Vm):
        return self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return self.COND_K*n**4*(Vm - self.RE_PO_K)

    # def i_ca(self, Vm, a, b):
    #     return -self.COND_CA*a**2*b*(Vm - self.RE_PO_CA)
    def i_ca(self):
        v = self.v_mem
        a = self.a_gate
        b = self.b_gate
        return self.COND_CA*a**2*b*(v-self.RE_PO_CA)

    # def i_syn_ij(self, Vm_po, rho_ij, W_ij):
    #     return - self.COND_SYN*W_ij*rho_ij*(Vm_po - self.RE_PO_SYN)

    # def i_syn_ca_ij(self, Vm_po, rho_ij, W_ij):
    #     return - self.COND_CA_SYN*0.5*rho_ij*(Vm_po - self.RE_PO_CA)

"Hodgkin-Huxley Neuron with paramaters from Henry\
Defined uA/cm^2"
class HHNeuron:
    # Constants
    CAP_MEM  =   1.0 # membrane capacitance, in uF/cm^2

    # maximum conducances, in mS/cm^2
    COND_NA =   120.0
    COND_K  =   20.0
    COND_LEAK  =   0.3

    # Nernst reversal potentials, in mV
    RE_PO_NA = 50.0
    RE_PO_K  = -77.0
    RE_PO_LEAK  = -54.4

    # kinetics, mv
    HF_PO_M = -40.0 # m half potential
    HF_PO_N = -55.0
    HF_PO_H = -60.0

    V_REW_M = 15.0
    V_REW_N = 30.0
    V_REW_H = -15.0

    HF_PO_MT = -40.0
    HF_PO_NT = -55.0
    HF_PO_HT = -60.0

    V_REW_MT = 15.0
    V_REW_NT = 30.0
    V_REW_HT = -15.0

    #ms
    TAU_0_M = 0.1
    TAU_1_M = 0.4

    TAU_0_N = 1.0
    TAU_1_N = 5.0

    TAU_0_H = 1.0
    TAU_1_H = 7.0

    DIM = 4

    def __init__(self, para = None):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        self.i_inj = 0.0 # injected currents
        self.v_mem = None
        self.m_gate = None
        self.h_gate = None
        self.n_gate = None

    #H-H model
    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem = y(i)
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        VV = self.v_mem
        mm = self.m_gate
        hh = self.h_gate
        nn = self.n_gate
        i_inj = self.i_inj
        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])

        i_base = (self.i_na(VV, mm, hh) + self.i_k(VV, nn) +
                            self.i_leak(VV) + i_syn)

        yield -1/self.CAP_MEM*(-i_inj+i_base)
        yield self.dm_dt(VV, mm)
        yield self.dh_dt(VV, hh)
        yield self.dn_dt(VV, nn)


    def get_initial_condition(self):
        return [-65.0, 0.05, 0.6, 0.32]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def m0(self, V): return 0.5*(1+sym_backend.tanh((V - self.HF_PO_M)/self.V_REW_M))
    def n0(self, V): return 0.5*(1+sym_backend.tanh((V - self.HF_PO_N)/self.V_REW_N))
    def h0(self, V): return 0.5*(1+sym_backend.tanh((V - self.HF_PO_H)/self.V_REW_H))

    def tau_m(self, V): return self.TAU_0_M+self.TAU_1_M*(1-sym_backend.tanh((V - self.HF_PO_MT)/self.V_REW_MT)**2)
    def tau_n(self, V): return self.TAU_0_N+self.TAU_1_N*(1-sym_backend.tanh((V - self.HF_PO_NT)/self.V_REW_NT)**2)
    def tau_h(self, V): return self.TAU_0_H+self.TAU_1_H*(1-sym_backend.tanh((V - self.HF_PO_HT)/self.V_REW_HT)**2)

    def i_na(self, V, m, h): return self.COND_NA*m**3*h*(V - self.RE_PO_NA) #mS*mV = uA
    def i_k(self, V, n): return self.COND_K*n**4*(V - self.RE_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.RE_PO_LEAK)

    def dm_dt(self, V, m): return (self.m0(V) - m)/self.tau_m(V)
    def dh_dt(self, V, h): return (self.h0(V) - h)/self.tau_h(V)
    def dn_dt(self, V, n): return (self.n0(V) - n)/self.tau_n(V)


class Synapse_glu_HH:
    """
    A standard excitory synapse for the Hodgkin Huxley neuron.
    """
    #Excitation
    REV_PO_CL = -38.0  #mV
    ALPHA_R = 2.4
    BETA_R = 0.56
    MAX_CONC = 1.0  # maximum neurotransmitter concentration


    V_REW_R = 5.0
    HF_PO_R = 7.0

    DIM = 1
    def __init__(self, g=0.4,para = None):
        self.r_gate = None
        self.syn_weight = 1.0
        self.cond_glu = g

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.HF_PO_R)/self.V_REW_R)))*(1-r) - self.BETA_R*r
    def get_params(self):
        return [self.cond_glu, self.REV_PO_CL]

    def get_ind(self):
        return self.ii

    def get_initial_condition(self):
        return [0.1]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.cond_glu*self.r_gate
        wij = self.syn_weight
        return rho*wij*(v_pos - self.REV_PO_CL)

class Synapse_gaba_HH:
    """
    A standard inhibitory synapse for the hodgin huxley neuron.
    """
    #inhibition
    COND_GABA = 1.0
    REV_PO_GABA = -80.0 #mV
    ALPHA_R = 5.0
    BETA_R = 0.18
    MAX_CONC = 1.5


    V_REW_R = 5.0
    HF_PO_R = 7.0

    DIM = 1
    def __init__(self, para = None):
        self.r_gate = None
        self.syn_weight = 1.0

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse
                objects connected pre-synaptically to this synapse
            pre_neurons: A list of all neuron
                objects connected pre-synaptically to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.HF_PO_R)/self.V_REW_R)))*(1-r) - self.BETA_R*r

    def get_params(self):
        return [self.COND_GABA, self.REV_PO_GABA]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.r_gate
        wij = self.syn_weight
        return wij*rho*(v_pos - self.REV_PO_GABA)

    def get_initial_condition(self):
        return [0.1]


class SynapseGabaBl:
    """
    Synapse class for the beta lobe neuron mutual inhibition.
    """
    #inhibition
    COND_GABA = 1.0
    REV_PO_GABA = -80.0 #mV
    ALPHA_R = 5.0
    BETA_R = 0.18
    MAX_CONC = 1.5


    V_REW_R = 5.0
    HF_PO_R = 7.0

    DIM = 1
    CURRENTS = 1
    def __init__(self, weight = 1.0):
        self.r_gate = None
        self.syn_weight = weight

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.HF_PO_R)/self.V_REW_R)))*(1-r) - self.BETA_R*r

    def get_params(self):
        return [self.COND_GABA, self.REV_PO_GABA]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.r_gate
        wij = self.syn_weight
        return -wij*rho*(v_pos - self.REV_PO_GABA)

    def get_initial_condition(self):
        return [0.1]

"""
Antennal Lobe Neurons
"""
class PN_2:
    """
    Fitted Model of Projection Neurons from the Bazhenov Papers. The main difference
    from PN class is that the time constants for the h gating variable are altered.
    This seems to produce the dynamics we want in the antennal lobe.
    """
    # Constants for PNs
    CAP_MEM  =   142.0 # membrane capacitance, in pF

    # maximum conducances, in nS
    COND_NA =   7150.0
    COND_K  =   1430.0
    COND_LEAK  =   21.0
    COND_K_LEAK=   5.72
    COND_A  =   1430.0

    # Nernst reversal potentials, in mV
    RE_PO_NA = 50.0
    RE_PO_K  = -95.0
    RE_PO_LEAK  = -55.0
    RE_PO_K_LEAK = -95.0


    # Gating Variable m parameters
    HF_PO_M = -43.9
    V_REW_M = -7.4
    HF_PO_MT = -47.5
    V_REW_MT = 40.0
    TAU_0_M = 0.024
    TAU_1_M = 0.093

    # Gating Variable h Parameters
    HF_PO_H = -48.3
    V_REW_H = 4.0
    HF_PO_HT = -56.8
    V_REW_HT = 16.9
    TAU_0_H = 0.0
    TAU_1_H = 5.6

    shift = 70.0

    DIM = 6

    def __init__(self, para = None):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        self.i_inj = 0.0 # injected currents
        self.v_mem = None
        self.m_gate = None
        self.h_gate = None
        self.n_gate = None
        self.z_gate = None
        self.u_gate = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem = y(i)
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)
        self.z_gate = y(i+4)
        self.u_gate = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        VV = self.v_mem
        mm = self.m_gate
        hh = self.h_gate
        nn = self.n_gate
        zz = self.z_gate
        uu = self.u_gate
        i_inj = self.i_inj

        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])

        i_base = (self.i_na(VV, mm, hh) + self.i_k(VV, nn) +
                        self.i_leak(VV) + self.i_a(VV,zz,uu) + self.i_k_leak(VV)
                        + i_syn)

        yield -1/self.CAP_MEM*(i_base-i_inj)
        yield self.dm_dt(VV, mm)
        yield self.dh_dt(VV, hh)
        yield self.dn_dt(VV, nn)
        yield self.dz_dt(VV, zz)
        yield self.du_dt(VV, uu)


    def get_initial_condition(self):
        return [-65.0+np.random.normal(0,2.0), 0.05+np.random.uniform()*0.1, 0.8 + np.random.uniform()*0.2, 0.2+np.random.uniform()*0.3, 0.1+np.random.uniform()*0.1, 0.8+np.random.uniform()*0.2]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def x_eqm(self,V,theta,sigma): return 0.5*(1.0 - sym_backend.tanh(0.5*(V-theta)/sigma))
    def tau_x(self,V,theta,sigma,t0,t1): return t0 + t1*(1.0-sym_backend.tanh((V-theta)/sigma)**2)

    def dm_dt(self,V,m): return (self.m0(V)-m)/self.tm(V)
    def dh_dt(self,V,h): return (self.h0(V)-h)/self.th(V)
    def dn_dt(self, V, n): return self.a_n(V)*(1-n)-self.b_n(V)*n
    def dz_dt(self, V, z): return (self.z0(V)-z)/self.tz(V)
    def du_dt(self, V, u): return (self.u0(V)-u)/self.tu(V)

    def m0(self,V): return self.x_eqm(V,self.HF_PO_M,self.V_REW_M)
    def tm(self,V): return self.tau_x(V,self.HF_PO_MT,self.V_REW_MT,self.TAU_0_M,self.TAU_1_M)

    def h0(self,V): return self.x_eqm(V,self.HF_PO_H,self.V_REW_H)
    def th(self,V): return self.tau_x(V,self.HF_PO_HT,self.V_REW_HT,self.TAU_0_H,self.TAU_1_H)

    def a_n(self, V): return 0.016*(V-35.1+self.shift)/(1-sym_backend.exp(-(V-35.1+self.shift)/5.0))
    def b_n(self, V): return 0.25*sym_backend.exp(-(V-20+self.shift)/40.0)

    def z0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+60)/8.5))
    def tz(self, V): return 1.0/(sym_backend.exp((V+35.8)/19.7)+sym_backend.exp(-(V+79.7)/12.7)+0.37)

    def u0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+78)/6.0))

    #adapted from bazhenov
    def tu(self, V):
        return 0.27/(sym_backend.exp((V+46)/5.0)+sym_backend.exp(-(V+238)/37.5)) \
                    +5.1/2*(1+sym_backend.tanh((V+57)/3))

    def i_na(self, V, m, h): return self.COND_NA*m**3*h*(V - self.RE_PO_NA) #nS*mV = pA
    def i_k(self, V, n): return self.COND_K*n*(V - self.RE_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.RE_PO_LEAK)
    def i_a(self, V, z, u): return self.COND_A*z**4*u*(V - self.RE_PO_K)
    def i_k_leak(self, V): return self.COND_K_LEAK*(V - self.RE_PO_K_LEAK)

class LN:
    """
    Model of Local Neurons. Use Synapse_gaba_LN or Synapse_gaba_LN_with_slow.
    Defined in pico amps
    """
    #Constants for LN

    CAP_MEM  =   142.0 # membrane capacitance, in pF
    # maximum conducances, in nS
    COND_K  =   1000.0
    COND_LEAK  =   21.5
    COND_K_LEAK =   1.43
    COND_CA =   290.0
    COND_KCA=   35.8

    # Nernst reversal potentials, in mV
    REV_PO_NA = 50.0
    REV_PO_K  = -95.0
    REV_PO_LEAK  = -50.0
    REV_PO_K_LEAK = -95.0
    REV_PO_CA = 140.0

    DIM = 6

    def __init__(self, para = None):
        self.i_inj = 0 # injected currents
        self.v_mem = None
        self.n_gate = None
        self.q_gate = None
        self.s_gate = None
        self.v_gate = None
        self.ca = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem  = y(i)
        self.n_gate  = y(i+1)
        self.s_gate  = y(i+2)
        self.v_gate  = y(i+3)
        self.q_gate  = y(i+4)
        self.ca = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        VV = self.v_mem
        nn = self.n_gate
        qq = self.q_gate
        ss = self.s_gate
        vv = self.v_gate
        Ca = self.ca
        i_inj = self.i_inj

        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])
        i_base = (self.i_k(VV, nn) + self.i_leak(VV) + self.i_kca(VV, qq) + \
                        self.i_ca(VV, ss, vv) + self.i_k_leak(VV) + i_syn)


        yield -1/self.CAP_MEM*(i_base - i_inj)
        yield self.dnl_dt(VV, nn)
        yield self.ds_dt(VV, ss)
        yield self.dv_dt(VV, vv)
        yield self.dq_dt(Ca, qq)
        yield self.dCa_dt(VV, ss, vv, Ca)

    def get_initial_condition(self):
        return [-60.0 + np.random.normal(0,2), 0.0+np.random.uniform()*0.1, 0.0+np.random.uniform()*0.1, 0.8+np.random.uniform()*0.2, 0.0+np.random.uniform()*0.1, 0.2+np.random.uniform()*0.3]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def a_nl(self, V): return 0.02*(-(35.0+V)/(sym_backend.exp(-(35.0+V)/5.0)-1.0))
    def b_nl(self, V): return 0.5*sym_backend.exp((-(40.0+V)/40.0))

    def nl0(self, V): return self.a_nl(V)/(self.a_nl(V)+self.b_nl(V))
    def tnl(self, V): return 4.65/(self.a_nl(V)+self.b_nl(V))

    def s0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+20.0)/6.5))
    #def ts(self, V): return 1+(V+30)*0.014
    def ts(self,V): return 1.5

    def v0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+25.0)/12.0))
    def tv(self, V): return 0.3*sym_backend.exp((V-40)/13.0)+0.002*sym_backend.exp(-(V-60.0)/29.0)

    def q0(self, Ca): return Ca/(Ca+2.0)
    def tq(self, Ca): return 100.0/(Ca+2.0)

    def i_ca(self, V, s, v): return self.COND_CA*s**2*v*(V-self.REV_PO_CA)
    def i_kca(self, V, q):   return self.COND_KCA*q*(V-self.REV_PO_K)
    def i_k_leak(self, V): return self.COND_K_LEAK*(V - self.REV_PO_K_LEAK)
    def i_k(self, V, nl): return  self.COND_K*nl**4*(V - self.REV_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.REV_PO_LEAK)



    def dCa_dt(self, V, s, v, Ca): return -2.86e-6*self.i_ca(V, s, v)-(Ca-0.24)/150.0
    def ds_dt(self, V, s): return (self.s0(V)-s)/self.ts(V)
    def dv_dt(self, V, v): return (self.v0(V)-v)/self.tv(V)
    def dq_dt(self, Ca, q): return (self.q0(Ca)-q)/self.tq(Ca)
    def dnl_dt(self, V, nl): return (self.nl0(V)-nl)/self.tnl(V)


"""
Antennal Lobe Synapse Functions:
"""

class Synapse_gaba_LN:
    """
    A synapse class for the antennal lobe local neuron. This synapse only includes
    dynaics for fast inhibition.
    """
    #inhibition
    RE_PO_GABA = -70.0
    ALPHA_R = 10.0
    BETA_R = 0.16
    MAX_CONC = 1.0


    V_REW_R = 1.5
    HF_PO_R = -20.0

    DIM = 1
    def __init__(self, gGABA = 800.0):
        self.r_gate = None
        self.syn_weight = 1.0
        self.COND_GABA = gGABA

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.HF_PO_R)/self.V_REW_R)))*(1-r) - self.BETA_R*r

    def get_params(self):
        return [self.COND_GABA, self.REV_PO_GABA]

    def get_initial_condition(self):
        return [0.0]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.COND_GABA*self.r_gate
        wij = self.syn_weight
        return wij*rho*(v_pos - self.RE_PO_GABA)

"""
This is synapse is an inhibitory synapse based on the 2nd of Bazhenov's 2001 papers.

It is used within the antennal lobe structure, when local neurons are the pre-synaptic
neuron.

There are two types of inhibitory current coming from this synapse.
(1) Fast inhibition
(2) Slow inhibition - Slow inhibition functions to create periods of bursting and
quiescence within projection neurons during odor stimulation. This is very important
to the overall dynamics of the antennal lobe.
"""
class Synapse_gaba_LN_with_slow:
    #inhibition
    REV_PO_GABA = -70.0
    ALPHA_R = 10.0
    BETA_R = 0.16
    MAX_CONC = 1.0


    V_REW_R = 1.5
    HF_PO_R = -20.0

    #s1 = 0.001 # uM^{-1}ms^{-1} check units?
    s1 = 1.0 #realistically should be 1e-3 ish
    s2 = 0.0025 # ms^{-1}
    s3 = 0.1 # ms^{-1}
    s4 = 0.06 # ms^{-1}
    K = 100.0 # uM^4
    REV_PO_K = -95.0 # mV

    DIM = 3
    def __init__(self, gGABA = 400.0, gSI = 125.0):
        self.r_gate = None
        self.s_gate = None
        self.g_gate = None
        self.syn_weight = 1.0
        self.COND_GABA = gGABA
        self.COND_SI = gSI


    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)
        self.s_gate = y(i+1)
        self.g_gate = y(i+2)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate #This corresponds to fast GABA
        s = self.s_gate # This is the fraction of activated receptors for SI
        G = self.g_gate # This is the concentration of receptor coupled G proteins
        yield self.ALPHA_R*self.MAX_CONC*self.t_gaba_a(Vpre)*(1-r) - self.BETA_R*r
        yield self.s1*(1.0 - s)*self.t_gaba_b(Vpre) - self.s2*s
        yield self.s3*s - self.s4*G

    def t_gaba_a(self,V): return 1.0/(1.0+sym_backend.exp(-(V - self.HF_PO_R)/self.V_REW_R))
    def t_gaba_b(self,V): return 1.0/(1.0+sym_backend.exp(-(V - 2.0)/5.0))

    def get_params(self):
        return [self.COND_GABA, self.REV_PO_GABA, self.COND_SI, self.REV_PO_K, self.K]

    def get_initial_condition(self):
        return [0.0,0.0,0.0]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho0 = self.COND_GABA*self.r_gate
        wij = self.syn_weight
        rho1 = self.COND_SI*self.g_gate**4/(self.g_gate**4 + self.K)
        return wij*rho0*(v_pos - self.REV_PO_GABA) + wij*rho1*(v_pos - self.REV_PO_K)

"""
This is a different version of Bazhenov 2001 projection neuron synapse. Version 1
alters only the equation dictating the release of acetylcholine (nAch) into the synapse,
but it does not alter any of the dynamics. This version of synapse both alters the
above and changes the differential equation describing the fraction of open ion channels.

This synapse is used within the antennal lobe, and is the synapse when the pre-synaptic
neuron is an excitatory projection neuron. The current neuron class to be used with this
synapse is PN_2.
"""
class Synapse_nAch_PN_2:

    RE_PO_NACH = 0.0
    r1 = 1.5 #1.5
    tau = 1.0 #1
    Kp = 1.5
    Vp = 0.0 # -20 for gaba

    DIM = 1
    def __init__(self, g = 300.0):
        self.r_gate = None
        self.syn_weight = 1.0
        self.COND_NACH = g


    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.r_inf(Vpre) - r)/(self.tau*(self.r1-self.r_inf(Vpre)))

    def r_inf(self,V): return 0.5*(1.0-sym_backend.tanh(-0.5*(V - self.Vp)/self.Kp))

    def get_params(self):
        return [self.COND_NACH, self.RE_PO_NACH]

    def get_initial_condition(self):
        return [0.0]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.COND_NACH*self.r_gate
        wij = self.syn_weight
        return wij*rho*(v_pos - self.RE_PO_NACH)


"""
Trial Models: Models we think may work better/make connecting multiple parts easier

Capacitance set to 1.0, all conductance values divied by 100. Should we divide
by 142 instead?
"""
class LNRescaled:
    """
    Model of Local Neurons above, but rescaled.


    Taken from:
     Kee T., et.al, "Feed-Forward versus Feedback Inhibition in a Basic Olfactory
     Circuit", PLOS Comput Biol, 2015.
    """
    #Constants for LN

    CAP_MEM  =   1.0 # membrane capacitance, in uF (really shoud be nF based on dimension)
    # maximum conducances, in uS
    COND_K  =   10.0 # too high?
    COND_LEAK  =   0.2 #
    COND_K_LEAK =   0.02 #
    COND_CA =   2.9
    COND_KCA=   0.36

    # Nernst reversal potentials, in mV
    REV_PO_K  = -95.0 #
    REV_PO_LEAK  = -50.0 #
    REV_PO_K_LEAK = -90.0 #
    REV_PO_CA = 140.0

    PHI = 1. # A temperature dependent constant of the form 3^((22-T)/10)

    DIM = 6

    def __init__(self, para = None):
        self.i_inj = 0 # injected currents
        self.v_mem = None
        self.n_gate = None
        self.q_gate = None
        self.s_gate = None
        self.v_gate = None
        self.ca = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem  = y(i)
        self.n_gate  = y(i+1)
        self.s_gate  = y(i+2)
        self.v_gate  = y(i+3)
        self.q_gate  = y(i+4)
        self.ca = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        VV = self.v_mem
        nn = self.n_gate
        qq = self.q_gate
        ss = self.s_gate
        vv = self.v_gate
        Ca = self.ca
        i_inj = self.i_inj

        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])
        i_base = (self.i_k(VV, nn) + self.i_leak(VV) + self.i_kca(VV, qq) + \
                        self.i_ca(VV, ss, vv) + self.i_k_leak(VV) + i_syn)


        yield -1/self.CAP_MEM*(i_base - i_inj)
        yield self.dnl_dt(VV, nn)
        yield self.ds_dt(VV, ss)
        yield self.dv_dt(VV, vv)
        yield self.dq_dt(Ca, qq)
        yield self.dCa_dt(VV, ss, vv, Ca)

    def get_initial_condition(self):
        return [-60.0+ np.random.normal(0,0.6), 0.0+np.random.uniform()*0.01,
            0.0+np.random.uniform()*0.01, 0.8+np.random.uniform()*0.01,
            0.0+np.random.uniform()*0.01, 0.2+np.random.uniform()*0.01]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def a_nl(self, V): return 0.02*(-(35.0+V)/(sym_backend.exp(-(35.0+V)/5.0)-1.0))
    def b_nl(self, V): return 0.5*sym_backend.exp((-(40.0+V)/40.0))

    def nl0(self, V): return self.a_nl(V)/(self.a_nl(V)+self.b_nl(V))
    def tnl(self, V): return 4.65/(self.a_nl(V)+self.b_nl(V))

    def s0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+20.0)/6.5))
    #def ts(self, V): return 1+(V+30)*0.014
    def ts(self,V): return 1.5

    def v0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+25.0)/12.0))
    def tv(self, V): return 0.3*sym_backend.exp((V-40)/13.0)+0.002*sym_backend.exp(-(V-60.0)/29.0)

    def q0(self, Ca): return Ca/(Ca+2.0)
    def tq(self, Ca): return 100.0/(Ca+2.0)

    def i_ca(self, V, s, v): return self.COND_CA*s**2*v*(V-self.REV_PO_CA)
    def i_kca(self, V, q):   return self.COND_KCA*q*(V-self.REV_PO_K)
    def i_k_leak(self, V): return self.COND_K_LEAK*(V - self.REV_PO_K_LEAK)
    def i_k(self, V, nl): return  self.COND_K*nl**4*(V - self.REV_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.REV_PO_LEAK)



    def dCa_dt(self, V, s, v, Ca): return -2.86e-4*self.i_ca(V, s, v)-(Ca-0.24)/150.0
    def ds_dt(self, V, s): return (self.s0(V)-s)/self.ts(V)
    def dv_dt(self, V, v): return (self.v0(V)-v)/self.tv(V)
    def dq_dt(self, Ca, q): return (self.q0(Ca)-q)/self.tq(Ca)
    def dnl_dt(self, V, nl): return (self.nl0(V)-nl)/self.tnl(V)

class Synapse_LNSI_Rescaled:
    #inhibition
    REV_PO_GABA = -70.0
    ALPHA_R = 10.0
    BETA_R = 0.16
    MAX_CONC = 1.0


    V_REW_R = 1.5
    HF_PO_R = -20.0

    #s1 = 0.001 # uM^{-1}ms^{-1} check units?
    s1 = 1.0 #realistically should be 1e-3 ish
    s2 = 0.0025 # ms^{-1}
    s3 = 0.1 # ms^{-1}
    s4 = 0.06 # ms^{-1}
    K = 100.0 # uM^4
    REV_PO_K = -95.0 # mV

    DIM = 3
    def __init__(self, gGABA = 4.0, gSI = 4.0):
        self.r_gate = None
        self.s_gate = None
        self.g_gate = None
        self.syn_weight = 1.0
        self.COND_GABA = gGABA
        self.COND_SI = gSI


    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)
        self.s_gate = y(i+1)
        self.g_gate = y(i+2)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate #This corresponds to fast GABA
        s = self.s_gate # This is the fraction of activated receptors for SI
        G = self.g_gate # This is the concentration of receptor coupled G proteins
        yield self.ALPHA_R*self.MAX_CONC*self.t_gaba_a(Vpre)*(1-r) - self.BETA_R*r
        yield self.s1*(1.0 - s)*self.t_gaba_b(Vpre) - self.s2*s
        yield self.s3*s - self.s4*G

    def t_gaba_a(self,V): return 1.0/(1.0+sym_backend.exp(-(V - self.HF_PO_R)/self.V_REW_R))
    def t_gaba_b(self,V): return 1.0/(1.0+sym_backend.exp(-(V - 2.0)/5.0))

    def get_params(self):
        return [self.COND_GABA, self.REV_PO_GABA, self.COND_SI, self.REV_PO_K, self.K]

    def get_initial_condition(self):
        return [0.0+np.random.uniform()*0.01,0.0+np.random.uniform()*0.01,0.0]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho0 = self.COND_GABA*self.r_gate
        wij = self.syn_weight
        rho1 = self.COND_SI*self.g_gate**4/(self.g_gate**4 + self.K)
        return wij*rho0*(v_pos - self.REV_PO_GABA) + wij*rho1*(v_pos - self.REV_PO_K)

class Synapse_LN_Rescaled:
    #inhibition
    RE_PO_GABA = -70.0
    ALPHA_R = 10.0
    BETA_R = 0.16
    MAX_CONC = 1.0


    V_REW_R = 1.5
    HF_PO_R = -20.0


    DIM = 1
    def __init__(self, gGABA = 8.0):
        self.r_gate = None
        self.syn_weight = 1.0
        self.COND_GABA = gGABA



    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)


    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate #This corresponds to fast GABA
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.HF_PO_R)/self.V_REW_R)))*(1-r) - self.BETA_R*r


    def get_params(self):
        return [self.COND_GABA, self.REV_PO_GABA]

    def get_initial_condition(self):
        return [0.0+np.random.uniform()*0.01]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.COND_GABA*self.r_gate
        wij = self.syn_weight
        return wij*rho*(v_pos - self.RE_PO_GABA)

class PNRescaled:
    # Constants for PNs
    CAP_MEM  =   1.0 # membrane capacitance, in pF

    # maximum conducances, in uS
    COND_NA =   71.5
    COND_K  =   14.3
    COND_LEAK  =   0.15
    COND_K_LEAK=   0.05
    COND_A  =   10.0

    # Nernst reversal potentials, in mV
    RE_PO_NA = 50.0
    RE_PO_K  = -95.0
    RE_PO_LEAK  = -55.0
    RE_PO_K_LEAK = -95.0


    # Gating Variable m parameters
    HF_PO_M = -43.9
    V_REW_M = -7.4
    HF_PO_MT = -47.5
    V_REW_MT = 40.0
    TAU_0_M = 0.024
    TAU_1_M = 0.093

    # Gating Variable h Parameters
    HF_PO_H = -48.3
    V_REW_H = 4.0
    HF_PO_HT = -56.8
    V_REW_HT = 16.9
    TAU_0_H = 0.0
    TAU_1_H = 5.6

    shift = 70.0

    DIM = 6

    def __init__(self, para = None):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        self.i_inj = 0.0 # injected currents
        self.v_mem = None
        self.m_gate = None
        self.h_gate = None
        self.n_gate = None
        self.z_gate = None
        self.u_gate = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem = y(i)
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)
        self.z_gate = y(i+4)
        self.u_gate = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        VV = self.v_mem
        mm = self.m_gate
        hh = self.h_gate
        nn = self.n_gate
        zz = self.z_gate
        uu = self.u_gate
        i_inj = self.i_inj

        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])

        i_base = (self.i_na(VV, mm, hh) + self.i_k(VV, nn) +
                        self.i_leak(VV) + self.i_a(VV,zz,uu) + self.i_k_leak(VV)
                        + i_syn)

        yield -1/self.CAP_MEM*(i_base-i_inj)
        yield self.dm_dt(VV, mm)
        yield self.dh_dt(VV, hh)
        yield self.dn_dt(VV, nn)
        yield self.dz_dt(VV, zz)
        yield self.du_dt(VV, uu)


    def get_initial_condition(self):
        return [-65.0+ np.random.normal(0,0.6), 0.05+np.random.uniform()*0.01, 0.6+np.random.uniform()*0.01, 0.32+np.random.uniform()*0.01, 0.6+np.random.uniform()*0.01, 0.6+np.random.uniform()*0.01]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def x_eqm(self,V,theta,sigma): return 0.5*(1.0 - sym_backend.tanh(0.5*(V-theta)/sigma))
    def tau_x(self,V,theta,sigma,t0,t1): return t0 + t1*(1.0-sym_backend.tanh((V-theta)/sigma)**2)

    def dm_dt(self,V,m): return (self.m0(V)-m)/self.tm(V)
    def dh_dt(self,V,h): return (self.h0(V)-h)/self.th(V)
    def dn_dt(self, V, n): return self.a_n(V)*(1-n)-self.b_n(V)*n
    def dz_dt(self, V, z): return (self.z0(V)-z)/self.tz(V)
    def du_dt(self, V, u): return (self.u0(V)-u)/self.tu(V)

    def m0(self,V): return self.x_eqm(V,self.HF_PO_M,self.V_REW_M)
    def tm(self,V): return self.tau_x(V,self.HF_PO_MT,self.V_REW_MT,self.TAU_0_M,self.TAU_1_M)

    def h0(self,V): return self.x_eqm(V,self.HF_PO_H,self.V_REW_H)
    def th(self,V): return self.tau_x(V,self.HF_PO_HT,self.V_REW_HT,self.TAU_0_H,self.TAU_1_H)

    def a_n(self, V): return 0.016*(V-35.1+self.shift)/(1-sym_backend.exp(-(V-35.1+self.shift)/5.0))
    def b_n(self, V): return 0.25*sym_backend.exp(-(V-20+self.shift)/40.0)

    def z0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+60)/8.5))
    def tz(self, V): return 1.0/(sym_backend.exp((V+35.8)/19.7)+sym_backend.exp(-(V+79.7)/12.7)+0.37)

    def u0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+78)/6.0))

    #adapted from bazhenov
    def tu(self, V):
        return 0.27/(sym_backend.exp((V+46)/5.0)+sym_backend.exp(-(V+238)/37.5)) \
                    +5.1/2*(1+sym_backend.tanh((V+57)/3))

    def i_na(self, V, m, h): return self.COND_NA*m**3*h*(V - self.RE_PO_NA) #nS*mV = pA
    def i_k(self, V, n): return self.COND_K*n*(V - self.RE_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.RE_PO_LEAK)
    def i_a(self, V, z, u): return self.COND_A*z**4*u*(V - self.RE_PO_K)
    def i_k_leak(self, V): return self.COND_K_LEAK*(V - self.RE_PO_K_LEAK)


class Synapse_PN_Rescaled:

    RE_PO_NACH = 0.0
    r1 = 1.5 #1.5
    tau = 1.0 #1
    Kp = 1.5
    Vp = 0.0 # -20 for gaba

    DIM = 1
    def __init__(self, g = 3.0):
        self.r_gate = None
        self.syn_weight = 1.0
        self.COND_NACH = g


    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.r_inf(Vpre) - r)/(self.tau*(self.r1-self.r_inf(Vpre)))

    def r_inf(self,V): return 0.5*(1.0-sym_backend.tanh(-0.5*(V - self.Vp)/self.Kp))

    def get_params(self):
        return [self.COND_NACH, self.RE_PO_NACH]

    def get_initial_condition(self):
        return [0.0+np.random.uniform()*0.01]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.COND_NACH*self.r_gate
        wij = self.syn_weight
        return wij*rho*(v_pos - self.RE_PO_NACH)


class GGNTrial:
    """
    Model of Giant Gabaergic Neuron of the Mushroom Body. Very similar to the
    local neuron in the antennal lobe.

    The GGN enforces sparsity of Kenyon Cell firing within the antennal lobe.
    All Kenyon Cells connect to this and this connects to all Kenyon Cells.

    TODO: Explore altering the time constant of Calcium decay, should be ~20 rather
    than ~150 ms.
    """
    #Constants for LN

    CAP_MEM  =   1.0 # membrane capacitance, in uF (really shoud be nF based on dimension)
    # maximum conducances, in uS
    COND_K  =   10.0 # too high?
    COND_LEAK  =   0.25 #
    COND_K_LEAK =   0.02 #
    COND_CA =   2.9
    COND_KCA=   0.36

    # Nernst reversal potentials, in mV
    REV_PO_K  = -95.0 #
    REV_PO_LEAK  = -50.0 #
    REV_PO_K_LEAK = -90.0 #
    REV_PO_CA = 140.0

    PHI = 1. # A temperature dependent constant of the form 3^((22-T)/10)

    DIM = 6

    def __init__(self, para = None):
        self.i_inj = 0 # injected currents
        self.v_mem = None
        self.n_gate = None
        self.q_gate = None
        self.s_gate = None
        self.v_gate = None
        self.ca = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem  = y(i)
        self.n_gate  = y(i+1)
        self.s_gate  = y(i+2)
        self.v_gate  = y(i+3)
        self.q_gate  = y(i+4)
        self.ca = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        VV = self.v_mem
        nn = self.n_gate
        qq = self.q_gate
        ss = self.s_gate
        vv = self.v_gate
        Ca = self.ca
        i_inj = self.i_inj

        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])
        i_base = (self.i_k(VV, nn) + self.i_leak(VV) + \
                        self.i_kca(VV, qq) + \
                        self.i_ca(VV, ss, vv) + self.i_k_leak(VV) + i_syn)
        #i_base = i_syn + self.i_k_leak(VV) + self.i_ca(VV,ss,vv)+self.i_k(VV,nn)

        yield -1/self.CAP_MEM*(i_base - i_inj)
        yield self.dnl_dt(VV, nn)
        yield self.ds_dt(VV, ss)
        yield self.dv_dt(VV, vv)
        yield self.dq_dt(Ca, qq)
        yield self.dCa_dt(VV, ss, vv, Ca)

    def get_initial_condition(self):
        return [-60.0+ np.random.normal(0,0.6), 0.0+np.random.uniform()*0.01,
            0.0+np.random.uniform()*0.01, 0.8+np.random.uniform()*0.01,
            0.0+np.random.uniform()*0.01, 0.2+np.random.uniform()*0.01]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def a_nl(self, V): return 0.02*(-(35.0+V)/(sym_backend.exp(-(35.0+V)/5.0)-1.0))
    def b_nl(self, V): return 0.5*sym_backend.exp((-(40.0+V)/40.0))

    def nl0(self, V): return self.a_nl(V)/(self.a_nl(V)+self.b_nl(V))
    def tnl(self, V): return 4.65/(self.a_nl(V)+self.b_nl(V))

    def s0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+20.0)/6.5))
    #def ts(self, V): return 1+(V+30)*0.014
    def ts(self,V): return 1.5

    def v0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+25.0)/12.0))
    def tv(self, V): return 0.3*sym_backend.exp((V-40)/13.0)+0.002*sym_backend.exp(-(V-60.0)/29.0)

    def q0(self, Ca): return Ca/(Ca+2.0)
    def tq(self, Ca): return 100.0/(Ca+2.0)

    def i_ca(self, V, s, v): return self.COND_CA*s**2*v*(V-self.REV_PO_CA)
    def i_kca(self, V, q):   return self.COND_KCA*q*(V-self.REV_PO_K)
    def i_k_leak(self, V): return self.COND_K_LEAK*(V - self.REV_PO_K_LEAK)
    def i_k(self, V, nl): return  self.COND_K*nl**4*(V - self.REV_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.REV_PO_LEAK)



    def dCa_dt(self, V, s, v, Ca): return -2.86e-4*self.i_ca(V, s, v)-(Ca-0.24)/150.0
    def ds_dt(self, V, s): return (self.s0(V)-s)/self.ts(V)
    def dv_dt(self, V, v): return (self.v0(V)-v)/self.tv(V)
    def dq_dt(self, Ca, q): return (self.q0(Ca)-q)/self.tq(Ca)
    def dnl_dt(self, V, nl): return (self.nl0(V)-nl)/self.tnl(V)

class GGNTrial2:
    """
    A simplified model of the Giant Gabaergic Neuron in the Mushroom Body.

    This is a one dimensional neuron with two current inputs:
        (1) synapse current
        (2) A leak current
    """
    #Constants for LN

    CAP_MEM  =   1.0 # membrane capacitance, in uF (really shoud be nF based on dimension)
    # maximum conducances, in uS
    COND_K  =   10.0 # too high?
    COND_LEAK  =   0.25 #
    COND_K_LEAK =   0.02 #
    COND_CA =   2.9
    COND_KCA=   0.36

    # Nernst reversal potentials, in mV
    REV_PO_K  = -95.0 #
    REV_PO_LEAK  = -50.0 #
    REV_PO_K_LEAK = -90.0 #
    REV_PO_CA = 140.0

    PHI = 1. # A temperature dependent constant of the form 3^((22-T)/10)

    DIM = 1

    def __init__(self, para = None):
        self.i_inj = 0 # injected currents
        self.v_mem = None
        self.n_gate = None
        self.q_gate = None
        self.s_gate = None
        self.v_gate = None
        self.ca = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem  = y(i)
        self.n_gate  = y(i+1)
        self.s_gate  = y(i+2)
        self.v_gate  = y(i+3)
        self.q_gate  = y(i+4)
        self.ca = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
        # define how neurons are coupled here
        VV = self.v_mem
        nn = self.n_gate
        qq = self.q_gate
        ss = self.s_gate
        vv = self.v_gate
        Ca = self.ca
        i_inj = self.i_inj

        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])
        #i_base = (self.i_k(VV, nn) + self.i_leak(VV) + \
        #                self.i_kca(VV, qq) + \
        #                self.i_ca(VV, ss, vv) + self.i_k_leak(VV) + i_syn)
        i_base = i_syn + self.i_k_leak(VV) #+ self.i_ca(VV,ss,vv)+self.i_k(VV,nn)

        yield -1/self.CAP_MEM*(i_base - i_inj)
        #yield self.dnl_dt(VV, nn)
        #yield self.ds_dt(VV, ss)
        #yield self.dv_dt(VV, vv)
        #yield self.dq_dt(Ca, qq)
        #yield self.dCa_dt(VV, ss, vv, Ca)

    def get_initial_condition(self):
        return [-90.0+ np.random.normal(0,0.6), 0.0+np.random.uniform()*0.01,
            0.0+np.random.uniform()*0.01, 0.8+np.random.uniform()*0.01,
            0.0+np.random.uniform()*0.01, 0.2+np.random.uniform()*0.01]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def a_nl(self, V): return 0.02*(-(35.0+V)/(sym_backend.exp(-(35.0+V)/5.0)-1.0))
    def b_nl(self, V): return 0.5*sym_backend.exp((-(40.0+V)/40.0))

    def nl0(self, V): return self.a_nl(V)/(self.a_nl(V)+self.b_nl(V))
    def tnl(self, V): return 4.65/(self.a_nl(V)+self.b_nl(V))

    def s0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+20.0)/6.5))
    #def ts(self, V): return 1+(V+30)*0.014
    def ts(self,V): return 1.5

    def v0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+25.0)/12.0))
    def tv(self, V): return 0.3*sym_backend.exp((V-40)/13.0)+0.002*sym_backend.exp(-(V-60.0)/29.0)

    def q0(self, Ca): return Ca/(Ca+2.0)
    def tq(self, Ca): return 100.0/(Ca+2.0)

    def i_ca(self, V, s, v): return self.COND_CA*s**2*v*(V-self.REV_PO_CA)
    def i_kca(self, V, q):   return self.COND_KCA*q*(V-self.REV_PO_K)
    def i_k_leak(self, V): return self.COND_K_LEAK*(V - self.REV_PO_K_LEAK)
    def i_k(self, V, nl): return  self.COND_K*nl**4*(V - self.REV_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.REV_PO_LEAK)



    def dCa_dt(self, V, s, v, Ca): return -2.86e-4*self.i_ca(V, s, v)-(Ca-0.24)/150.0
    def ds_dt(self, V, s): return (self.s0(V)-s)/self.ts(V)
    def dv_dt(self, V, v): return (self.v0(V)-v)/self.tv(V)
    def dq_dt(self, Ca, q): return (self.q0(Ca)-q)/self.tq(Ca)
    def dnl_dt(self, V, nl): return (self.nl0(V)-nl)/self.tnl(V)

class Synapse_GGNTrial:
    #inhibition
    RE_PO_GABA = -80.0
    ALPHA_R = 10.0
    BETA_R = 0.16
    MAX_CONC = 1.0


    V_REW_R = 1.5
    HF_PO_R = -50


    DIM = 1
    def __init__(self, gGABA = 8.0):
        self.r_gate = None
        self.syn_weight = 1.0
        self.COND_GABA = gGABA



    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)


    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate #This corresponds to fast GABA
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.HF_PO_R)/self.V_REW_R)))*(1-r) - self.BETA_R*r


    def get_params(self):
        return [self.COND_GABA, self.REV_PO_GABA]

    def get_initial_condition(self):
        return [0.0+np.random.uniform()*0.01]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.COND_GABA*self.r_gate
        wij = self.syn_weight
        return wij*rho*(v_pos - self.RE_PO_GABA)


class Synapse_KCGNN:
    """
    An excitory synapse based on the glutamate synapse.
    """
    #Excitation
    REV_PO_CL = -38.0  #mV
    ALPHA_R = 2.4
    #ALPHA_R = 1
    BETA_R = 0.56
    MAX_CONC = 1.0  # maximum neurotransmitter concentration


    V_REW_R = 5.0
    HF_PO_R = 7.0

    DIM = 1
    def __init__(self, g=0.4,para = None):
        self.r_gate = None
        self.syn_weight = 1.0
        self.cond_glu = g

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.HF_PO_R)/self.V_REW_R)))*(1-r) - self.BETA_R*r
    def get_params(self):
        return [self.cond_glu, self.REV_PO_CL]

    def get_ind(self):
        return self.ii

    def get_initial_condition(self):
        return [0.1]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.cond_glu*self.r_gate
        wij = self.syn_weight
        a = rho*wij*(v_pos - self.REV_PO_CL)
        print(a)
        return a

class Synapse_nAch_TD:

    RE_PO_NACH = 0.0
    r1 = 1.5 #1.5
    tau = 1.0 #1
    Kp = 1.5
    Vp = 0.0 # -20 for gaba

    DIM = 1
    def __init__(self, g = 3.0):
        self.r_gate = None
        self.syn_weight = 1.0
        self.COND_NACH = g


    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.r_inf(Vpre) - r)/(self.tau*(self.r1-self.r_inf(Vpre)))
        pulse(t,t0,0.3)
    def r_inf(self,V): return 0.5*(1.0-sym_backend.tanh(-0.5*(V - self.Vp)/self.Kp))

    def get_params(self):
        return [self.COND_NACH, self.RE_PO_NACH]

    def get_initial_condition(self):
        return [0.0+np.random.uniform()*0.01]

    def i_syn_ij(self, v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.COND_NACH*self.r_gate
        wij = self.syn_weight
        return wij*rho*(v_pos - self.RE_PO_NACH)

"""
We do not current use the models below, but will be kept below.
-------------------------------------------------------------------------------
"""
"""
This is the excitatory synapse model as seen in Bazhenov 2001, except that the
time dependent square wave pulse function has been replaced by a voltage dependent
sigmoid function. The differential equation for fraction of open ion channels
remains unchanged.
"""
class Synapse_nAch_PN:
    #Excitation
    RE_PO_NACH = 0.0
    ALPHA_R = 10.0
    BETA_R = 0.2
    MAX_CONC = 0.5
    SI = False

    Kp = 1.5
    Vp = -20.0

    DIM = 1
    def __init__(self, gnAch = 300.0):
        self.r_gate = None
        self.syn_weight = 1.0
        self.COND_NACH = gnAch


    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.r_gate = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.syn_weight = w

    def dydt(self, pre_neuron, pos_neuron):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this synapse
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this synapse
        """
        Vpre = pre_neuron.v_mem
        r = self.r_gate
        yield (self.ALPHA_R*self.MAX_CONC/(1+sym_backend.exp(-(Vpre - self.Vp)/self.Kp)))*(1-r) - self.BETA_R*r

    def get_params(self):
        return [self.COND_NACH, self.RE_PO_NACH]

    def get_initial_condition(self):
        return [0.0+np.random.uniform()*0.01]

    def i_syn_ij(self,v_pos):
        """
        A function which calculates the total synaptic current
        Args:
            v_pos (float): The membrane potential of the post synaptic neuron
        Returns:
            A value for the total synaptic current, used by the post-synaptic cell
        """
        rho = self.COND_NACH*self.r_gate
        wij=self.syn_weight
        return rho*wij*(v_pos - self.RE_PO_NACH)


"""
The exact model of projection neurons from Bazhenov's 2001 paper. The model does
not give us the dynamics we want when testing the antennal lobe, possibly because
we've altered the synapse dynamics to remove time dependence. Use PN_2 class instead.
"""
class PN:
        # Constants for PNs
    CAP_MEM  =   290.0 # membrane capacitance, in pF

    # maximum conducances, in nS
    COND_NA =   7150.0
    COND_K  =   1430.0
    COND_LEAK  =   21.0
    COND_K_LEAK =   5.72
    COND_A  =   1430.0

    # Nernst reversal potentials, in mV
    REV_PO_NA = 50.0
    REV_PO_K  = -95.0
    REV_PO_LEAK  = -55.0
    REV_PO_K_LEAK = -95.0

    shift = 65.0
    DIM = 6

    def __init__(self, para = None):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        self.i_inj = 0.0 # injected currents
        self.v_mem = None
        self.m_gate = None
        self.h_gate = None
        self.n_gate = None
        self.z_gate = None
        self.u_gate = None

    def set_integration_index(self, i):
        """
        Sets the integration index and state variable indicies.

        Args:
            i (int): integration variable index
        """
        self.ii = i
        self.v_mem = y(i)
        self.m_gate = y(i+1)
        self.h_gate = y(i+2)
        self.n_gate = y(i+3)
        self.z_gate = y(i+4)
        self.u_gate = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        """
        A function that will be used for integration. Necessary for jitcode.

        Args:
            pre_synapses: A list of all synapse objects connected pre-synaptically
                to this neuron
            pre_neurons: A list of all neuron objectes connected pre-synaptically
                to this neuron
        """
    # define how neurons are coupled here
        VV = self.v_mem
        mm = self.m_gate
        hh = self.h_gate
        nn = self.n_gate
        zz = self.z_gate
        uu = self.u_gate
        i_inj = self.i_inj

        i_syn = sum([synapse.i_syn_ij(VV) for (i,synapse) in enumerate(pre_synapses)])

        i_base = (self.i_na(VV, mm, hh) + self.i_k(VV, nn) +
                    self.i_leak(VV) + self.i_a(VV,zz,uu) + self.i_k_leak(VV)
                    + i_syn)

        yield -1/self.CAP_MEM*(i_base-i_inj)
        yield self.dm_dt(VV, mm)
        yield self.dh_dt(VV, hh)
        yield self.dn_dt(VV, nn)
        yield self.dz_dt(VV, zz)
        yield self.du_dt(VV, uu)


    def get_initial_condition(self):
        return [-65.0, 0.05, 0.6, 0.32, 0.6, 0.6]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.v_mem

    def dm_dt(self, V, m): return self.a_m(V)*(1-m)-self.b_m(V)*m
    def dh_dt(self, V, h): return self.a_h(V)*(1-h)-self.b_h(V)*h
    def dn_dt(self, V, n): return self.a_n(V)*(1-n)-self.b_n(V)*n
    def dz_dt(self, V, z): return (self.z0(V)-z)/self.tz(V)
    def du_dt(self, V, u): return (self.u0(V)-u)/self.tu(V)

    #def a_m(self, V): return 0.32*(V + 37)/(1 - sym_backend.exp(-(V + 37)/4.0))
    def a_m(self, V): return 0.32*(V - 13.1+self.shift)/(1 - sym_backend.exp(-(V - 13.1+self.shift)/4.0))
    #def b_m(self, V): return 0.28*(V + 10)/(sym_backend.exp((V+10.0)/5.0)-1)
    def b_m(self, V): return 0.28*(V - 40.1+self.shift)/(sym_backend.exp((V-40.1+self.shift)/5.0)-1)

    #def a_h(self, V): return 0.128*sym_backend.exp(-(V+33.0)/18.0)
    def a_h(self, V): return 0.128*sym_backend.exp(-(V-17.0+self.shift)/18.0)
    #def b_h(self, V): return 4.0/(1+sym_backend.exp(-(V+10.0)/5.0))
    def b_h(self, V): return 4.0/(1+sym_backend.exp(-(V-40+self.shift)/5.0))

    #def a_n(self, V): return 0.032*(V+35.0)/(1.0-sym_backend.exp(-(V+35.0)/5.0))
    def a_n(self, V): return 0.016*(V-35.1+self.shift)/(1-sym_backend.exp(-(V-35.1+self.shift)/5.0))
    #def b_n(self, V): return 0.5*sym_backend.exp(-(V+40.0)/40.0)
    def b_n(self, V): return 0.25*sym_backend.exp(-(V-20+self.shift)/40.0)

    def z0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+60)/8.5))
    #def tz(self, V): return 0.25/(sym_backend.exp((V+35.8)/19.7)+sym_backend.exp(-(V+79.7)/12.7)+0.09)
    def tz(self, V): return 0.25/(sym_backend.exp((V+35.8)/19.7)+sym_backend.exp(-(V+79.7)/12.7)+0.37)

    def u0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+78.0)/6.0))

    #adapted from bazhenov
    def tu(self, V):
        return 0.25/(sym_backend.exp((V+46)/5.0)+sym_backend.exp(-(V+238)/37.5)) \
                    +4.8/2*(1+sym_backend.tanh((V+57)/3))

    def i_na(self, V, m, h): return self.COND_NA*m**3*h*(V - self.REV_PO_NA) #nS*mV = pA
    def i_k(self, V, n): return self.COND_K*n**4*(V - self.REV_PO_K)
    def i_leak(self, V): return self.COND_LEAK*(V - self.REV_PO_LEAK)
    def i_a(self, V, z, u): return self.COND_A*z**4*u*(V - self.REV_PO_K)
    def i_k_leak(self, V): return self.COND_K_LEAK*(V - self.REV_PO_K_LEAK)
