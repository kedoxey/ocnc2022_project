import random

import scipy as sp
import scipy.signal
import numpy as np
import brian2
from brian2.units import ms, mV, second
from brian2 import check_units

# Parameters
N_E = 40
N_I = 10
tau_E = 20*ms  # ms
tau_I = 10*ms  # ms
V_rest = -70*mV  # mV
V_0 = -70*mV  # mV
n = 2
k = 0.3*(mV**-n)/2  # mV^(-n)*s^(-1)
W_EE = 1.25*mV*second  # mV*s, E -> E
W_IE = 1.2*mV*second  # mV*s, E -> I
W_EI = 0.65*mV*second  # mV*s, I -> E
W_II = 0.5*mV*second  # mV*s, I -> I
tau_noise = 50*ms  # ms
sigma_0E = 1*mV  # mV
sigma_0I = 0.5*mV  # mV
P_E = 0.1
P_I = 0.4
tau_syn = 2*ms  # ms
axon_delay = 0.5 * ms  # ms
l_syn = 45  # deg
l_noise = 60  # deg
l_stim = 60  # deg
b = 2*mV  # mV
A_max = 20*mV  # mV
theta_stim = 0  # deg
dt = 0.1*ms  # ms
c = 1  # contrast

# Synaptic Weights (J_ij) for j to i
J_EE = W_EE/(tau_syn*P_E*N_E)  # E -> E
J_EI = W_EI/(tau_syn*P_I*N_I)  # I -> E
J_IE = W_IE/(tau_syn*P_E*N_E)  # E -> I
J_II = W_II/(tau_syn*P_I*N_I)  # I -> I

# Pre-synaptic Connections for each neuron i
E_neurons = np.random.random_integers(0, N_E, int(P_E*N_E))
I_neurons = np.random.random_integers(0, N_I, int(P_I*N_I))


# Momentary Spiking Rate
@check_units(v=mV)
def r(v):
    return (k*(np.floor((v-V_0)/mV)*mV)**n) if v-V_0 > 0 else 0


# Mean External Input
@check_units(t=ms)
def h_i(t):
    return b + c*A_max*np.exp((np.cos(t)-1)/l_stim**2)


# Delta Function
@check_units(t=ms)
def delta(t):
    t_j = 'firing times of neuron j'
    return scipy.signal.unit_impulse(len(t_j), int(t/ms)-1)


# Sum of delta(t - t_j - ax_delay)
def sum_delta(t_j):
    summed = 0
    for t in t_j:
        summed = summed + delta(t)
    return summed


# Membrane Potential
Vm_eqs = '''
dv/dt = -v/tau_noise + V_rest/tau_noise + h_i/tau_noise + sigma*sqrt(2/tau_noise)*xi + : 1
a_tot : 1

'''

# Post-synaptic Current
syn_eqs = '''
da/dt = -a/tau_syn + sum_delta(t_j-axon_delay) : 1
a_tot_post = J_ij*a : 1 (summed)
'''

test_rate = r(30*mV)/dt
test_delta = axon_delay(5 * ms)

temp = 5

