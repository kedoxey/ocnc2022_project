import numpy as np
import matplotlib.pyplot as plt

from brian2.units import mV, ms

spike_counts = np.load('output/data/variance_validation-spike_counts.npy')

bin_width = 2
start = np.floor(np.min(spike_counts))
end = np.ceil(np.max(spike_counts))
bins = np.arange(start, end+bin_width, bin_width)
counts, edges = np.histogram(spike_counts, bins)
centers = edges[:-1] + np.diff(bins)/2

plt.figure(1)
plt.bar(centers,counts,bin_width,edgecolor='k')
plt.xlabel('spike count')

# plt.savefig(f'output/figures/variance_validation-{bin_width}.jpg',bbox_inches='tight')
# plt.savefig(f'output/figures/variance_validation-{bin_width}.pdf',bbox_inches='tight')

input_voltages = np.load('output/data/rate_SSN-input_voltages.npy')
input_voltages = input_voltages.astype('int')
input_dur = 10000

# Excitatory
# exc_color = [0,151/255,252/255]

num_rows = len(input_voltages)
num_cols = input_dur

exc_spikes = np.load('output/data/rate_SSN-exc_spikes.npy')
exc_spike_times = np.load('output/data/rate_SSN-exc_spike_times.npy')

exc_times = np.load('output/data/rate_SSN-exc_times.npy')
exc_V = np.load('output/data/rate_SSN-exc_V.npy')

print('load exc files')

exc_times_sep = exc_times.reshape((num_rows, num_cols))
exc_V_sep = exc_V.reshape((num_rows, num_cols))
exc_V_avg = np.mean(exc_V_sep, axis=1)
exc_V_std = np.std(exc_V_sep, axis=1)
exc_V_var = np.var(exc_V_sep, axis=1)

print('calc exc stats')

# Inhibitory
# inh_color = [255/255,58/255,20/255]

inh_spikes = np.load('output/data/rate_SSN-inh_spikes.npy')
inh_spike_times = np.load('output/data/rate_SSN-inh_spike_times.npy')

inh_times = np.load('output/data/rate_SSN-inh_times.npy')
inh_V = np.load('output/data/rate_SSN-inh_V.npy')

print('load inh files')

inh_times_sep = inh_times.reshape((num_rows, num_cols))
inh_V_sep = inh_V.reshape((num_rows, num_cols))
inh_V_avg = np.mean(inh_V_sep, axis=1)
inh_V_std = np.std(inh_V_sep, axis=1)
inh_V_var = np.var(inh_V_sep, axis=1)

print('calc inh stats')

exc_spikes_sep = np.zeros((1,num_rows))
for exc_spike_time in exc_spike_times:
    for input_voltage in input_voltages:
        if exc_spike_time in exc_times_sep[input_voltage,:]:
            exc_spikes_sep[0,input_voltage] = exc_spikes_sep[0,input_voltage] + 1

print('calc exc spike counts')

inh_spikes_sep = np.zeros((1,num_rows))
for inh_spike_time in inh_spike_times:
    for input_voltage in input_voltages:
        if inh_spike_time in inh_times_sep[input_voltage,:]:
            inh_spikes_sep[0,input_voltage] = inh_spikes_sep[0,input_voltage] + 1

print('calc inh spike counts')

exc_spike_var = np.var(exc_spikes_sep[0,:])
inh_spike_var = np.var(inh_spikes_sep[0,:])

# Figures
# plt.figure(2)
# plt.plot(exc_times/ms, exc_V/mV, label='E', color=exc_color)
# plt.plot(inh_times/ms, inh_V/mV, label='I', color=inh_color)
# plt.legend()
# plt.xlabel('time (ms)')
# plt.ylabel('membrane potential (mV)')
# plt.title('V_E/I')

exc_color = [255/255,0/255,0/255]
inh_color = [0/255,130/255,255/255]

fig, axs = plt.subplots(3,1,figsize=(6,5),sharex=True)
axs = axs.ravel()

axs[0].plot(input_voltages, exc_spikes_sep[0,:], '.-', label='E', color=exc_color, zorder=12)
axs[0].plot(input_voltages, inh_spikes_sep[0,:], '.-', label='I', color=inh_color, zorder=12)
axs[0].axvline(2, linestyle='dashed', color='grey', alpha=0.4, zorder=0)
axs[0].axvline(15, linestyle='dashed', color='grey', alpha=0.4, zorder=0)
axs[0].set_ylabel(r'$\rm{mean\ rate\ [Hz]}$')
axs[0].legend()

axs[1].plot(input_voltages, exc_V_avg/mV, '.-', label='E', color=exc_color, zorder=12)
axs[1].plot(input_voltages, inh_V_avg/mV, '.-', label='I', color=inh_color, zorder=12)
axs[1].axvline(2, linestyle='dashed', color='grey', alpha=0.4, zorder=0)
axs[1].axvline(15, linestyle='dashed', color='grey', alpha=0.4, zorder=0)
axs[1].set_ylabel(r'$\rm{\overline{V}_{E/I}\ [mV]}$')
axs[1].set_yticks(np.arange(-70,-45,5))
axs[1].set_ylim((-72,-47))


axs[2].plot(input_voltages, exc_V_std/mV, '.-', label='E', color=exc_color, zorder=12)
axs[2].plot(input_voltages, inh_V_std/mV, '.-', label='I', color=inh_color, zorder=12)
axs[2].axvline(2, linestyle='dashed', color='grey', alpha=0.4, zorder=0)
axs[2].axvline(15, linestyle='dashed', color='grey', alpha=0.4, zorder=0)
axs[2].set_ylabel(r'$\rm{V_{E/I}\ std.\ [mV]}$')
axs[2].set_yticks(np.arange(0,3,0.5))
axs[2].set_ylim((0,2.5))
axs[2].set_xticks(np.arange(0,25,5))
axs[2].set_xlim((-0.2,20.2))
axs[2].set_xlabel(r'$\rm{input\ [mV]}$')

fig.tight_layout()
fig.show()

fig.savefig('output/figures/rate_SSN-cont_input.jpg', bbox_inches='tight')
fig.savefig('output/figures/rate_SSN-cont_input.pdf', bbox_inches='tight')

temp = 5

#
# plt.figure(3)
# plt.plot(input_voltages, exc_V_avg/mV, '.-', label='E')
# plt.plot(input_voltages, inh_V_avg/mV, '.-', label='I')
# plt.legend()
# plt.xlabel('input (mV)')
# plt.ylabel('avg membrane potential (mV)')
# plt.title('avg(V_E/I)')
#
# plt.figure(4)
# plt.plot(input_voltages, exc_V_std, '.-', label='E')
# plt.plot(input_voltages, inh_V_std, '.-', label='I')
# plt.legend()
# plt.xlabel('input (mV)')
# plt.ylabel('std membrane potential (mV)')
# plt.title('std(V_E/I)')

# plt.figure(5)
# plt.scatter(exc_spike_times/ms, exc_spikes, label='E')
# plt.scatter(inh_spike_times/ms, inh_spikes + 1, label='I')
# plt.legend()
# plt.xlabel('time (ms)')
# plt.ylabel('neuron (E/I)')
# plt.title('spike times')

# plt.figure(6)
# plt.plot(input_voltages, exc_spikes_sep[0,:], '.-', label='E')
# plt.plot(input_voltages, inh_spikes_sep[0,:], '.-', label='I')
# plt.legend()
# plt.xlabel('input (mV)')
# plt.ylabel('mean rate (Hz)')
# plt.title('mean firing rate')

# plt.figure(7)
# plt.plot(input_voltages, exc_V_var, '.-', label='E')
# plt.plot(input_voltages, inh_V_var, '.-', label='I')
# plt.legend()
# plt.xlabel('input (mV)')
# plt.ylabel('var membrane potential')
# plt.title('var(V_E/I)')
