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

seed(74)

# prefs.codegen.target = "numpy"

# Parameters
N_E = 1
N_I = 1
N = N_E + N_I
tau_E = 20*ms  # ms
tau_I = 10*ms  # ms
V_rest = -70*mV  # mV
V_0 = -70*mV  # mV
n = 2
k = 0.3*(mV**-n)/second  # mV^(-n)*s^(-1)
weight_scale = 5
W_EE = 1.25*mV*weight_scale  # mV, E -> E
W_IE = 1.2*mV*weight_scale  # mV, E -> I
W_EI = -0.65*mV*weight_scale  # mV, I -> E
W_II = -0.5*mV*weight_scale  # mV, I -> I
tau_noise = 50*ms  # ms
sigma_0E = 0.2*mV  # mV
sigma_0I = 0.1*mV  # mV
time_step = 0.1*ms  # ms
c = 1  # contrast
# P_E = 0.1
# P_I = 0.4
# tau_syn = 2*ms  # ms
# axon_delay = 0.5*ms  # ms
# l_syn = 45  # deg
# l_noise = 60  # deg
# l_stim = 60  # deg
# b = 2*mV  # mV
# A_max = 20*mV  # mV
# theta_stim = 0  # deg


@check_units(v=mV,result=Hz)
def r(v):
    rate = (k*((np.floor((v-V_0)/mV)*mV)**n))

    if v.size > 1:
        for rate_idx, rate_idv in enumerate(rate):
            rate[rate_idx] = rate_idv if v[rate_idx]-V_0 > 0 else 0*Hz
    else:
        rate = rate if v-V_0 > 0 else 0*Hz

    return rate


# h = TimedArray([0*mV, 2*mV, 2*mV, 15*mV, 15*mV], dt=2*second)
input_voltages = np.arange(0,20,1)
input_dur = 1*second
h = TimedArray(input_voltages*mV, dt=input_dur)

noise_E = '''
sqrt(2*tau_noise*(sigma_0E*sqrt(1+(tau_E/tau_noise)))**2)*xi

(sigma_0E*sqrt(1+(tau_E/tau_noise))*sqrt(2/tau_E))*xi
'''

exc_V_eqs = '''
dv/dt = (-v + V_rest + h(t) + sqrt(2*tau_noise*(sigma_0E*sqrt(1+(tau_E/tau_noise)))**2)*xi)/tau_E : volt
'''

inh_V_eqs = '''
dv/dt = (-v + V_rest + h(t)+ sqrt(2*tau_noise*(sigma_0I*sqrt(1+(tau_I/tau_noise)))**2)*xi)/tau_I : volt
'''

exc_G = NeuronGroup(N_E, exc_V_eqs, threshold='rand() < int(v>V_0)*time_step*(k*((floor((v-V_0)/mV)*mV)**n))', method='euler', dt=time_step)
inh_G = NeuronGroup(N_I, inh_V_eqs, threshold='rand() < int(v>V_0)*time_step*(k*((floor((v-V_0)/mV)*mV)**n))', method='euler', dt=time_step)

exc_G.v = V_rest
inh_G.v = V_rest

S_EE = Synapses(exc_G, exc_G, on_pre='v+=W_EE')
S_IE = Synapses(exc_G, inh_G, on_pre='v+=W_IE')
S_EI = Synapses(inh_G, exc_G, on_pre='v+=W_EI')
S_II = Synapses(inh_G, inh_G, on_pre='v+=W_II')

S_EE.connect(i=0,j=0)
S_IE.connect(i=0,j=0)
S_EI.connect(i=0,j=0)
S_II.connect(i=0,j=0)

exc_M = StateMonitor(exc_G, 'v', record=True)
inh_M = StateMonitor(inh_G, 'v', record=True)

exc_spike_M = SpikeMonitor(exc_G, record=True)
inh_spike_M = SpikeMonitor(inh_G, record=True)

duration = len(input_voltages)*input_dur

run(duration, report='text')

exc_spike_trains = exc_spike_M.spike_trains()
inh_spike_trains = inh_spike_M.spike_trains()

exc_spike_counts = exc_spike_M.count
inh_spike_counts = inh_spike_M.count

exc_spikes = exc_spike_M.i
exc_spike_times = exc_spike_M.t
inh_spikes = inh_spike_M.i
inh_spike_times = inh_spike_M.t

np.save('output/data/debug/rate_SSN-exc_spikes.npy', exc_spikes)
np.save('output/data/debug/rate_SSN-exc_spike_times.npy', exc_spike_times)
np.save('output/data/debug/rate_SSN-inh_spikes.npy', inh_spikes)
np.save('output/data/debug/rate_SSN-inh_spike_times.npy', inh_spike_times)

# Spike Times
plt.figure(1)
plt.scatter(list(exc_spike_M.t/ms), list(exc_spike_M.i), label='E')
plt.scatter(list(inh_spike_M.t/ms), list(inh_spike_M.i+1), label='I')
plt.legend()
plt.xlabel('time (ms)')
plt.ylabel('neuron (E/I)')

# V_E/I
exc_times = exc_M.t
exc_V = exc_M.v[0]

inh_times = inh_M.t
inh_V = inh_M.v[0]

np.save('output/data/debug/rate_SSN-exc_times.npy', exc_times)
np.save('output/data/debug/rate_SSN-exc_V.npy', exc_V)
np.save('output/data/debug/rate_SSN-inh_times.npy', inh_times)
np.save('output/data/debug/rate_SSN-inh_V.npy', inh_V)

plt.figure(2)
plt.plot(list(exc_times/ms), list(exc_V/mV), label='E')
plt.plot(list(inh_times/ms), list(inh_V/mV), label='I')
plt.legend()
plt.xlabel('time (ms)')
plt.ylabel('membrane potential (mV)')
# plt.ylim((-71,-55))
# plt.xticks(np.arange(-70,-50,10))
# plt.xticks(np.arange(0,15,5))
plt.show()

temp = 5
