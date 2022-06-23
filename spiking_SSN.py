import random

import scipy as sp
import scipy.signal
import numpy as np
import pickle
from brian2 import *
import brian2
from brian2.units import ms, mV, second, Hz, kHz
from brian2 import check_units
import matplotlib.pyplot as plt

brian2.seed(74)

prefs.codegen.target = "numpy"

# Parameters
N_E = 40
N_I = 10
N = N_E + N_I
tau_E = 20*ms  # ms
tau_I = 10*ms  # ms
V_rest = -70*mV  # mV
V_0 = -70*mV  # mV
n = 2
k = 0.3*(mV**-n)/second  # mV^(-n)*s^(-1)
W_EE = 1.25*mV*second  # mV*s, E -> E
W_IE = 1.2*mV*second  # mV*s, E -> I
W_EI = -0.65*mV*second  # mV*s, I -> E
W_II = -0.5*mV*second  # mV*s, I -> I
tau_noise = 50*ms  # ms
sigma_0E = 1*mV  # mV
sigma_0I = 0.5*mV  # mV
P_E = 0.1
P_I = 0.4
tau_syn = 2*ms  # ms
axon_delay = 0.5*ms  # ms
l_syn = 45  # deg
l_noise = 60  # deg
l_stim = 60  # deg
b = 2*mV  # mV
A_max = 20*mV  # mV
theta_stim = 0  # deg
time_step = 0.1*ms  # ms
c = 1  # contrast


# Pre-synaptic Connections for each neuron i
def get_pre_syn_conns(neuron_type):
    if neuron_type == 'E':
        return np.random.randint(0, N, int(P_E*N_E))
    else:
        return np.random.randint(0, N, int(P_I*N_I))


C = np.zeros((N, N))
for i in range(0, N):
    pre_syn_E = get_pre_syn_conns('E')
    pre_syn_I = get_pre_syn_conns('I')

    temp = 5
    for e_idx in pre_syn_E:
        C[e_idx, i] = 1
    for i_idx in pre_syn_I:
        C[i_idx, i] = 1

C_sources, C_targets = C.nonzero()

# Synaptic Weights (J_ij) for j to i
J_EE = W_EE/(tau_syn*P_E*N_E)  # E -> E
J_EI = W_EI/(tau_syn*P_I*N_I)  # I -> E
J_IE = W_IE/(tau_syn*P_E*N_E)  # E -> I
J_II = W_II/(tau_syn*P_I*N_I)  # I -> I

J = np.zeros((N,N))
J[0:N_E,0:N_E] = J_EE*C[0:N_E,0:N_E]
J[0:N_E,N_E:N] = J_EI*C[0:N_E,N_E:N]
J[N_E:N,0:N_E] = J_IE*C[N_E:N,0:N_E]
J[N_E:N,N_E:N] = J_II*C[N_E:N,N_E:N]

J_sources, J_targets = J.nonzero()
weights = J.flatten()


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

# Stochastic Firing Probability
threshold = 'rand() < time_step*r(v)'


# Momentary Spiking Rate
@check_units(v=mV,result=Hz)
def r(v):
    rate = (k*((np.floor((v-V_0)/mV)*mV)**n))

    if v.size > 1:
        for rate_idx, rate_idv in enumerate(rate):
            rate[rate_idx] = rate_idv if v[rate_idx]-V_0 > 0 else 0*Hz
    else:
        rate = rate if v-V_0 > 0 else 0*Hz

    return rate


# Validate Firing Rates
fig, axs = plt.subplots(1,2,figsize=(10,5),sharey=True)
axs = axs.ravel()

low_voltage = -80
high_voltage = -50
num_neurons = 30

test_voltages = np.linspace(low_voltage,high_voltage,num_neurons)  #
test_rates = [r(test_voltage*mV)/Hz for test_voltage in test_voltages]

axs[0].plot(test_voltages,test_rates)
axs[0].set_xlabel('membrane potential (mV)')
axs[0].set_ylabel('instantaneous firing rate (Hz)')
axs[0].set_xticks(np.arange(low_voltage,high_voltage+10,10))
axs[0].set_title('Calculated')

fixed_Vm_eqs = '''
v : volt
'''

fixed_G = brian2.NeuronGroup(len(test_voltages), fixed_Vm_eqs, threshold='rand() < time_step*r(v)', dt=time_step)
fixed_G.v = test_voltages*mV
fixed_M = brian2.SpikeMonitor(fixed_G, variables='v', record=True)

duration = 9*second

brian2.run(duration, report="text")

voltages = fixed_M.v
spike_trains = fixed_M.spike_trains()
spike_counts = fixed_M.count
spike_rates = spike_counts/duration
all_values = fixed_M.all_values()

axs[1].plot(test_voltages, list(spike_rates))
axs[1].set_xlabel('membrane potential (mV)')
axs[1].set_ylabel('firing rate (Hz)')
axs[1].set_xticks(np.arange(low_voltage,high_voltage+10,10))
axs[1].set_title('Stimulated')

fig.tight_layout()
fig.show()
fig.savefig(f'output/spiking_SSN-{num_neurons}-{int(duration/second)}s.jpg',bbox_inches='tight')

temp = 5

