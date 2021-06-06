import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import k
from scipy.special import factorial
from specc.data.signal import Signal

from tools import DiodeConverter, TemperatureConverter, fit_arrhenius_equation

filter_frequency = 1
resistor = 0.511e6

timestamps = [
    '1622627302',
    # '1622629593',
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
    filtered_diode_signal = Signal(diode_signal.sample_rate, np.real(np.fft.ifft(filtered_diode_fft)),
                                   converter=diode_converter)

    table_headers = ['Observable', 'Unit', 'Mean', 'STD']
    table_content = [
        ['Temperature', temperature_signal.converter.unit, np.mean(temperature_signal.csamples),
         temperature_signal.error],
        # ['Diode current', diode_signal.converter.unit, np.mean(diode_signal.csamples), diode_signal.error],
        ['Filtered diode current', filtered_diode_signal.converter.unit, np.mean(filtered_diode_signal.csamples),
         filtered_diode_signal.error]
    ]

    # print(tabulate(table_content, table_headers))

    # plot(temperature_signal).show()
    # plot(diode_signal).show()
    # plot(filtered_diode_signal).show()

    beta = 1 / (k * temperature_signal.csamples)
    current = filtered_diode_signal.csamples

    sort_indices = np.argsort(beta)
    beta = np.take_along_axis(beta, sort_indices, axis=0)
    current = np.take_along_axis(current, sort_indices, axis=0)

    beta, unique_indices = np.unique(beta, return_index=True)

    averaged_current = np.empty((len(unique_indices)))
    std_current = np.empty((len(unique_indices)))

    for i, index_pair in enumerate(zip(unique_indices[0:-1], unique_indices[1:])):
        start_index, end_index = index_pair
        averaged_current[i] = np.mean(current[start_index:end_index])
        std_current[i] = np.std(current[start_index:end_index])

        if i == len(unique_indices) - 2:
            averaged_current[-1] = np.mean(current[end_index:])
            std_current[-1] = np.std(current[end_index:])

    averaged_current += 1

    Ea, D0 = fit_arrhenius_equation(averaged_current, beta)

    averaged_current -= 1

    plt.plot(beta, averaged_current)
    plt.plot(beta, D0 * np.exp(-Ea * beta) - 1)

    plt.xlabel(f'$\\beta$ [J]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.show()
