import numpy as np
import matplotlib.pyplot as plt

from brian2.units import mV, ms

input_voltages = np.arange(0,20,1)
input_dur = 10000  # ms

# Excitatory
exc_spikes = np.load('output/data/debug/rate_SSN-exc_spikes.npy')
exc_spike_times = np.load('output/data/debug/rate_SSN-exc_spike_times.npy')

exc_times = np.load('output/data/debug/rate_SSN-exc_times.npy')
exc_V = np.load('output/data/debug/rate_SSN-exc_V.npy')

exc_times_sep = exc_times.reshape((20,10000))
exc_V_sep = exc_V.reshape((20,10000))
exc_V_avg = np.mean(exc_V_sep, axis=1)
exc_V_std = np.std(exc_V_sep, axis=1)

# Inhibitory
inh_spikes = np.load('output/data/debug/rate_SSN-inh_spikes.npy')
inh_spike_times = np.load('output/data/debug/rate_SSN-inh_spike_times.npy')

inh_times = np.load('output/data/debug/rate_SSN-inh_times.npy')
inh_V = np.load('output/data/debug/rate_SSN-inh_V.npy')

inh_times_sep = inh_times.reshape((20,10000))
inh_V_sep = inh_V.reshape((20,10000))
inh_V_avg = np.mean(inh_V_sep, axis=1)
inh_V_std = np.std(inh_V_sep,axis=1)

exc_spikes_sep = np.zeros((1,20))
for exc_spike_time in exc_spike_times:
    for input_voltage in input_voltages:
        if exc_spike_time in exc_times_sep[input_voltage,:]:
            exc_spikes_sep[0,input_voltage] = exc_spikes_sep[0,input_voltage] + 1

inh_spikes_sep = np.zeros((1,20))
for inh_spike_time in inh_spike_times:
    for input_voltage in input_voltages:
        if inh_spike_time in inh_times_sep[input_voltage,:]:
            inh_spikes_sep[0,input_voltage] = inh_spikes_sep[0,input_voltage] + 1

# Figures
plt.figure(1)
plt.plot(exc_times/ms, exc_V/mV, label='E')
plt.plot(inh_times/ms, inh_V/mV, label='I')
plt.legend()
plt.xlabel('time (ms)')
plt.ylabel('membrane potential (mV)')
plt.title('V_E/I')

plt.figure(2)
plt.plot(input_voltages, exc_V_avg/mV, '.-', label='E')
plt.plot(input_voltages, inh_V_avg/mV, '.-', label='I')
plt.legend()
plt.xlabel('input (mV)')
plt.ylabel('avg membrane potential (mV)')
plt.title('avg(V_E/I)')

plt.figure(3)
plt.plot(input_voltages, exc_V_std, '.-', label='E')
plt.plot(input_voltages, inh_V_std, '.-', label='I')
plt.legend()
plt.xlabel('input (mV)')
plt.ylabel('std membrane potential (mV)')
plt.title('std(V_E/I)')

plt.figure(4)
plt.scatter(exc_spike_times/ms, exc_spikes, label='E')
plt.scatter(inh_spike_times/ms, inh_spikes + 1, label='I')
plt.legend()
plt.xlabel('time (ms)')
plt.ylabel('neuron (E/I)')
plt.title('spike times')

plt.figure(5)
plt.plot(input_voltages, exc_spikes_sep[0,:], '.-', label='E')
plt.plot(input_voltages, inh_spikes_sep[0,:], '.-', label='I')
plt.legend()
plt.xlabel('input (mV)')
plt.ylabel('mean rate (Hz)')
plt.title('mean firing rate')

plt.tight_layout()
plt.show()

# Rate Validation
spike_rates = np.load('output/data/rate_validation-spike_rates.npy')
# test_rates = np.load('output/data/rate_validation-test_rates.npy')
# test_voltages = np.load('rate_validation-test_voltages.npy')
#
# fig, axs = plt.subplots(1,2,figsize=(10,5),sharey=True)
# axs = axs.ravel()
#
# low_voltage = -80
# high_voltage = -50
#
# axs[0].plot(test_voltages,test_rates,'k')
# axs[0].set_xlabel('membrane potential (mV)')
# axs[0].set_ylabel('firing rate (Hz)')
# axs[0].set_xticks(np.arange(low_voltage,high_voltage+10,10))
# axs[0].set_title('Calculated')
#
# axs[1].plot(test_voltages, list(spike_rates),'k')
# axs[1].set_xlabel('membrane potential (mV)')
# axs[1].set_xticks(np.arange(low_voltage,high_voltage+10,10))
# axs[1].set_title('Stimulated')
#
# fig.tight_layout()
# fig.savefig(f'output/figures/rate_validation.jpg',bbox_inches='tight')
# fig.savefig(f'output/figures/rate_validation.pdf',bbox_inches='tight')
#
# # Rate SSN
# exc_times = np.load('output/data/rate_SSN-exc_times.npy')
# exc_V = np.load('output/data/rate_SSN-exc_V.npy')
#
# inh_times = np.load('output/data/rate_SSN-inh_times.npy')
# inh_V = np.load('output/data/rate_SSN-inh_v.npy')
#
# plt.figure(2)
# plt.plot(exc_times, exc_V/mV)
# plt.plot(inh_times, inh_V/mV)
# plt.legend(['E', 'I'])
# plt.xlabel('time (ms)')
# plt.ylabel('membrane potential (mV)')
# # plt.ylim((-71,-50))
# # plt.xticks(np.arange(-70,-50,10))
# # plt.xticks(np.arange(0,15,5))
# plt.show()

temp = 5
