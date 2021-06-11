from specc.aquisition.nimydaq import NIMyDAQInterface
from specc.data.signal import Signal
from specc.utils import timestamp

from tools import TemperatureConverter

time = timestamp()
print(f'TIME: {time}')

temperature_channel = 'myDAQ1/AI1'
diode_channel = 'myDAQ1/AI0'

sample_rate = 500
sample_time = 4 * 60
samples = int(sample_time * sample_rate)

daq = NIMyDAQInterface(sample_rate)
converter = TemperatureConverter()

data = daq.read([temperature_channel, diode_channel], samples)

temperature_signal = Signal(sample_rate, data[0])
diode_signal = Signal(sample_rate, data[1])

temperature_signal.save(f'data/temperature-{time}.npz')
diode_signal.save(f'data/current-{time}.npz')
