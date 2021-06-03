import matplotlib.pyplot as plt
import numpy as np
from specc.aquisition.nimydaq import NIMyDAQInterface
from specc.data.signal import Signal
from specc.utils import timestamp

from tools import TemperatureConverter, to_celsius

time = timestamp()
print(f'TIME: {time}')

filter_frequency = 1
resistor = 500e3

temperature_channel = 'myDAQ1/AI1'
diode_channel = 'myDAQ1/AI0'

sample_rate = 500
sample_time = 30
samples = int(sample_time * sample_rate)

daq = NIMyDAQInterface(sample_rate)
converter = TemperatureConverter()

data = np.asarray(daq.read([temperature_channel, diode_channel], samples))

temperature_signal = Signal(sample_rate, data[0])
diode_signal = Signal(sample_rate, data[1])
# temperature_signal = Signal.load('data/temperature-1622623979.npz')
# diode_signal = Signal.load('data/current-1622623979.npz')
current_signal = Signal(sample_rate, diode_signal.samples / resistor)

filtered_current_fft = current_signal.fft
filtered_current_fft[np.abs(diode_signal.frequencies) >= filter_frequency] = 0
filtered_current_signal = Signal(sample_rate, np.fft.ifft(filtered_current_fft))

temperature_signal.save(f'data/temperature-{time}.npz')
diode_signal.save(f'data/current-{time}.npz')

timestamps = daq.calculate_time_array(samples)
temperature = to_celsius(converter.T(temperature_signal.samples))

plt.plot(timestamps, temperature)
plt.xlabel('Time [s]')
plt.ylabel('Temperature [$\\degree$C]')

plt.savefig(f'figures/temperature-{time}.svg')
plt.show()

plt.plot(timestamps, filtered_current_signal.samples)
plt.xlabel('Time [s]')
plt.ylabel('Filtered diode current [A]')

plt.savefig(f'figures/current-{time}.svg')
plt.show()

plt.scatter(temperature, filtered_current_signal.samples)
plt.xlabel('Temperature [$\\degree$C]')
plt.ylabel('Filtered diode current [A]')

plt.ylim(np.min(filtered_current_signal.samples), np.max(filtered_current_signal.samples))

plt.savefig(f'figures/temperature-current-{time}.svg')
plt.show()
