import numpy as np
from matplotlib import pyplot as plt
from specc.data.signal import Signal
from tabulate import tabulate

from tools import DiodeConverter, TemperatureConverter

filter_frequency = 1
resistor = 0.511e6

timestamps = [
    '1622627302',
    '1622629593',
]

for timestamp in timestamps:
    temperature_file = f'data/temperature-{timestamp}.npz'
    diode_file = f'data/current-{timestamp}.npz'

    temperature_converter = TemperatureConverter()
    diode_converter = DiodeConverter(resistor)

    temperature_signal = Signal.load(temperature_file, converter=temperature_converter)
    diode_signal = Signal.load(diode_file)

    filtered_diode_fft = diode_signal.fft
    filtered_diode_fft[np.abs(diode_signal.frequencies) >= filter_frequency] = 0
    filtered_diode_signal = Signal(diode_signal.sample_rate, np.fft.ifft(filtered_diode_fft), converter=diode_converter)

    table_headers = ['Observable', 'Unit', 'Mean', 'STD']
    table_content = [
        ['Temperature', temperature_signal.converter.unit, np.mean(temperature_signal.csamples),
         temperature_signal.error],
        # ['Diode current', diode_signal.converter.unit, np.mean(diode_signal.csamples), diode_signal.error],
        ['Filtered diode current', filtered_diode_signal.converter.unit, np.mean(filtered_diode_signal.csamples),
         filtered_diode_signal.error]
    ]

    print(tabulate(table_content, table_headers))

    # plot(temperature_signal).show()
    # plot(diode_signal).show()
    # plot(filtered_diode_signal).show()

    plt.scatter(temperature_signal.csamples, filtered_diode_signal.samples)
    plt.xlabel(f'Temperature [{temperature_converter.unit}]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')
    plt.tight_layout()
    plt.show()
