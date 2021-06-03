import numpy as np
from specc.data.signal import Signal
from specc.plotting import plot
from tabulate import tabulate

from tools import DiodeConverter, TemperatureConverter

target_timestamp = '1622712949'
filter_frequency = 1
resistor = 0.511e6

temperature_file = f'data/temperature-{target_timestamp}.npz'
diode_file = f'data/current-{target_timestamp}.npz'

temperature_converter = TemperatureConverter()
diode_converter = DiodeConverter(resistor)

temperature_signal = Signal.load(temperature_file, converter=temperature_converter)
diode_signal = Signal.load(diode_file)

filtered_diode_fft = diode_signal.fft
filtered_diode_fft[np.abs(diode_signal.frequencies) >= filter_frequency] = 0
filtered_diode_signal = Signal(diode_signal.sample_rate, np.fft.ifft(filtered_diode_fft), converter=diode_converter)

table_headers = ['Observable', 'Unit', 'Mean', 'STD']
table_content = [
    ['Temperature', temperature_signal.converter.unit, np.mean(temperature_signal.csamples), temperature_signal.error],
    # ['Diode current', diode_signal.converter.unit, np.mean(diode_signal.csamples), diode_signal.error],
    ['Filtered diode current', filtered_diode_signal.converter.unit, np.mean(filtered_diode_signal.csamples),
     filtered_diode_signal.error]
]

print(tabulate(table_content, table_headers))

plot(temperature_signal).show()
# plot(diode_signal).show()
plot(filtered_diode_signal).show()
