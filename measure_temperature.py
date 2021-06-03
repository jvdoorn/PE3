import matplotlib.pyplot as plt
import numpy as np
from specc.aquisition.nimydaq import NIMyDAQInterface
from specc.data.signal import Signal
from specc.utils import timestamp

from tools import TemperatureConverter, to_celsius

time = timestamp()
print(f'TIME: {time}')

temperature_channel = 'myDAQ1/AI1'
sample_rate = 100
samples_time = 60
samples = samples_time * sample_rate

daq = NIMyDAQInterface(sample_rate)
converter = TemperatureConverter()

data = np.asarray(daq.read([temperature_channel], samples))
Signal(sample_rate, data).save(f'data/{time}.npz')
temperature = to_celsius(converter.T(data))

print(f'MEAN: {np.mean(temperature)}')
print(f'STD:  {np.std(temperature)}')

plt.plot(daq.calculate_time_array(samples), temperature)
plt.xlabel('Time [s]')
plt.ylabel('Temperature [$\\degree$C]')

plt.savefig(f'figures/temperature-{time}.svg')
plt.show()
