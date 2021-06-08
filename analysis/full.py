import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import k
from specc.data.signal import Signal
from tabulate import tabulate

from tools import DiodeConverter, TemperatureConverter, arrhenius_equation, fit_arrhenius_equation

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
    diode_signal = Signal.load(diode_file, converter=diode_converter)

    filtered_diode_fft = diode_signal.fft
    filtered_diode_fft[np.abs(diode_signal.frequencies) >= filter_frequency] = 0
    filtered_diode_signal = Signal(diode_signal.sample_rate, np.real(np.fft.ifft(filtered_diode_fft)),
                                   converter=diode_converter)

    table_headers = ['Observable', 'Unit', 'Mean', 'STD']
    table_content = [
        ['Temperature', temperature_signal.converter.unit, np.mean(temperature_signal.csamples),
         temperature_signal.error],
        ['Diode current', diode_signal.converter.unit, np.mean(diode_signal.csamples), diode_signal.error],
        ['Filtered diode current', filtered_diode_signal.converter.unit, np.mean(filtered_diode_signal.csamples),
         filtered_diode_signal.error]
    ]

    print(tabulate(table_content, table_headers))

    # plot(temperature_signal).show()
    # plot(diode_signal).show()
    # plot(filtered_diode_signal).show()

    temperature = temperature_signal.csamples
    beta = 1 / (k * temperature)
    current = filtered_diode_signal.csamples

    sort_indices = np.argsort(temperature)
    temperature = np.take_along_axis(temperature, sort_indices, axis=0)
    beta = np.take_along_axis(beta, sort_indices, axis=0)
    current = np.take_along_axis(current, sort_indices, axis=0)

    temperature, unique_indices = np.unique(temperature, return_index=True)
    beta = np.take_along_axis(beta, unique_indices, axis=0)

    averaged_current = np.empty((len(unique_indices)))
    std_current = np.empty((len(unique_indices)))

    unique_indices = np.append(unique_indices, -1)
    for i, index_pair in enumerate(zip(unique_indices[0:-1], unique_indices[1:])):
        start_index, end_index = index_pair
        if start_index == end_index or start_index == len(current) - 1:
            averaged_current[i] = current[-1]
            std_current[i] = filtered_diode_signal.error
        else:
            averaged_current[i] = np.mean(current[start_index:end_index])
            std = np.std(current[start_index:end_index])
            std_current[i] = std if std > 0 else filtered_diode_signal.error

    Ea, D0 = fit_arrhenius_equation(averaged_current + 1, beta, 1 / std_current)
    print(Ea, D0)

    plt.plot(beta, averaged_current)
    plt.plot(beta, arrhenius_equation(D0, Ea, beta) - 1)

    plt.xlabel(f'$\\beta$ [$J^{{-1}}$]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.show()

    plt.plot(temperature, averaged_current)
    plt.plot(temperature, arrhenius_equation(D0, Ea, beta) - 1)

    plt.xlabel(f'Temperature [{temperature_converter.unit}]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.show()
