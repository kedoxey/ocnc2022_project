import numpy as np
from brian2 import *
import brian2
from brian2.units import ms, mV, second, Hz, kHz
from brian2 import check_units
import matplotlib.pyplot as plt

brian2.seed(74)

# prefs.codegen.target = "numpy"

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


# Validate Firing Rates
fig, axs = plt.subplots(1,2,figsize=(10,5),sharey=True)
axs = axs.ravel()

low_voltage = -80
high_voltage = -50
num_neurons = 30

test_voltages = np.linspace(low_voltage,high_voltage,num_neurons)  #
test_rates = [r(test_voltage*mV)/Hz for test_voltage in test_voltages]

axs[0].plot(test_voltages,test_rates,'k')
axs[0].set_xlabel('membrane potential (mV)')
axs[0].set_ylabel('firing rate (Hz)')
axs[0].set_xticks(np.arange(low_voltage,high_voltage+10,10))
axs[0].set_title('Calculated')

fixed_Vm_eqs = '''
v : volt
'''

fixed_G = brian2.NeuronGroup(len(test_voltages), fixed_Vm_eqs, threshold='rand() < int(v>V_0)*time_step*(k*((floor((v-V_0)/mV)*mV)**n))', dt=time_step)
fixed_G.v = test_voltages*mV
fixed_M = brian2.SpikeMonitor(fixed_G, variables='v', record=True)

duration = 9*second

brian2.run(duration, report="text")

voltages = fixed_M.v
spike_trains = fixed_M.spike_trains()
spike_counts = fixed_M.count
spike_rates = spike_counts/duration
all_values = fixed_M.all_values()

np.save('output/data/rate_validation-test_voltages.npy', test_voltages)
np.save('output/data/rate_validation-spike_rates.npy', spike_rates)
np.save('output/data/rate_validation-test_rates.npy', test_rates)


# with open('rate_validation-{num_neurons}-{int(duration/second)}s.npy', 'w') as f:
#     np.save(f, test_voltages)
#     np.save(f, spike_rates)
#     np.save(f, test_rates)

axs[1].plot(test_voltages, list(spike_rates),'k')
axs[1].set_xlabel('membrane potential (mV)')
axs[1].set_xticks(np.arange(low_voltage,high_voltage+10,10))
axs[1].set_title('Stimulated')

fig.tight_layout()
fig.show()
fig.savefig(f'output/figures/rate_validation-{num_neurons}-{int(duration/second)}s.jpg',bbox_inches='tight')
fig.savefig(f'output/figures/rate_validation-{num_neurons}-{int(duration/second)}s.pdf',bbox_inches='tight')

temp = 5
