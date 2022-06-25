import numpy as np
# from brian2 import *
import brian2
from brian2.units import ms, mV, second, Hz, kHz
from brian2 import check_units
import matplotlib.pyplot as plt

brian2.seed(74)

time_step = 0.1*ms

n = 2
k = 0.3*(mV**-n)/second
V_0 = -70*mV


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


fixed_Vm_eqs = '''
v : volt
'''

duration = 1*second

num_tests = 500
spike_rates = np.zeros((1,num_tests))
spike_counts = np.zeros((1,num_tests))

for i in range(0, num_tests):

    fixed_G = brian2.NeuronGroup(1, fixed_Vm_eqs, threshold='rand() < int(v>V_0)*time_step*(k*(v-V_0)**n)', dt=time_step)
    fixed_G.v = -60*mV
    fixed_M = brian2.SpikeMonitor(fixed_G, variables='v', record=True)

    brian2.run(duration, report='text')

    spike_count = fixed_M.count
    spike_rate = spike_count/duration

    spike_counts[0,i] = spike_count[0]
    spike_rates[0,i] = spike_rate

spike_counts.flatten()
spike_rates.flatten()

np.save(f'output/data/variance_validation-spike_counts.npy', spike_counts)

bin_width = 0.3
start = np.floor(np.min(spike_counts))
end = np.ceil(np.max(spike_counts))
bins = np.arange(start, end+bin_width, bin_width)
counts, edges = np.histogram(spike_counts, bins)
centers = edges[:-1] + np.diff(bins)/2

variance = np.var(spike_counts)
mean = np.mean(spike_counts)

plt.bar(centers,counts,bin_width,edgecolor='k')
plt.xlabel('spike count')
plt.show()

plt.savefig(f'output/figures/variance_validation-{num_tests}.jpg',bbox_inches='tight')
plt.savefig(f'output/figures/variance_validation-{num_tests}.pdf',bbox_inches='tight')

temp = 5
