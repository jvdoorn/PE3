import matplotlib.pyplot as plt
from spectral.aquisition.nimydaq import NIMyDAQInterface
from spectral.utils import timestamp

from tools import TemperatureConverter

temperature_channel = 'myDAQ1/AI0'
sample_rate = 1000
samples_time = 20
samples = samples_time * sample_rate

daq = NIMyDAQInterface(sample_rate)
converter = TemperatureConverter()

data = daq.read([temperature_channel], samples)

plt.plot(daq.calculate_time_array(samples), converter.T(data))
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')

plt.savefig(f'figures/temperature-{timestamp()}.svg')
plt.show()
