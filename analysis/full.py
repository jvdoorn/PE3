import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import k
from specc.data.signal import Signal
from tabulate import tabulate

from analysis import FILTER_FREQUENCY, SELECTED_DATASETS
from tools import DiodeConverter, TemperatureConverter, arrhenius_equation, fit_arrhenius_equation

temperature_converter = TemperatureConverter()
diode_converter = DiodeConverter()

table_headers = ['Timestamp', 'Name', 'D0 (nA)', 'E_A (J)']
table_content = []

for timestamp, name in SELECTED_DATASETS.items():
    temperature_file = f'data/temperature-{timestamp}.npz'
    diode_file = f'data/current-{timestamp}.npz'

    temperature_signal = Signal.load(temperature_file, converter=temperature_converter)
    try:
        diode_signal = Signal.load(diode_file, converter=diode_converter)
    except FileNotFoundError:
        continue

    filtered_diode_fft = diode_signal.fft
    filtered_diode_fft[np.abs(diode_signal.frequencies) >= FILTER_FREQUENCY] = 0
    filtered_diode_signal = Signal(diode_signal.sample_rate, np.real(np.fft.ifft(filtered_diode_fft)),
                                   converter=diode_converter)

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

    offset = 1 - np.min(averaged_current)
    Ea, D0 = fit_arrhenius_equation(averaged_current + offset, beta, 1 / std_current)
    table_content.append([timestamp, name, D0, Ea])

    plt.plot(beta, averaged_current)
    plt.plot(beta, arrhenius_equation(D0, Ea, beta) - offset)

    plt.title(f'{name} (filtered)\ndataset {timestamp}')
    plt.xlabel(f'$\\beta$ [$J^{{-1}}$]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.savefig(f'svg/full/{timestamp}-over-beta.svg')
    plt.show()

    plt.plot(temperature, averaged_current)
    plt.plot(temperature, arrhenius_equation(D0, Ea, beta) - offset)

    plt.title(f'{name} (filtered)\ndataset {timestamp}')
    plt.xlabel(f'Temperature [{temperature_converter.unit}]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.savefig(f'svg/full/{timestamp}-over-temperature.svg')
    plt.show()

print(tabulate(table_content, table_headers))
