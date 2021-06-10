import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import electron_volt, k
from specc.data.signal import Signal
from tabulate import tabulate

from analysis import FILTER_FREQUENCY, SELECTED_DATASETS
from tools import DiodeConverter, TemperatureConverter, arrhenius_equation, fit_arrhenius_equation

temperature_converter = TemperatureConverter(0.3)

table_headers = ['Timestamp', 'Name', 'D0 (nA)', 'E_A (eV)', 'D0 STD (nA)', 'E_A STD (eV)']
table_content = []

error_every = 32

for timestamp, name in SELECTED_DATASETS.items():
    temperature_file = f'data/temperature-{timestamp}.npz'
    diode_file = f'data/current-{timestamp}.npz'

    diode_converter = DiodeConverter() if not timestamp == 1622623502 else DiodeConverter(0.1e6)

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
    D0, Ea, D0_std, Ea_std = fit_arrhenius_equation(averaged_current + offset, beta, 1 / std_current)
    table_content.append([timestamp, name, D0, Ea / electron_volt, D0_std, Ea_std / electron_volt])

    plt.clf()
    plt.errorbar(beta, averaged_current, yerr=std_current, fmt='o', markersize=2, capsize=3, errorevery=error_every,
                 alpha=0.5)

    plt.title(f'{name} (sanitized)\ndataset {timestamp}')
    plt.xlabel(f'$\\beta$ [$J^{{-1}}$]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.savefig(f'svg/full/{timestamp}-over-beta.svg')
    plt.savefig(f'/Users/julian/Dropbox/University/PE3/report/figures/full/{timestamp}-over-beta.png')
    plt.show()

    plt.errorbar(beta, averaged_current, yerr=std_current, fmt='o', markersize=2, capsize=3, errorevery=error_every,
                 alpha=0.5)
    plt.plot(beta, arrhenius_equation(D0, Ea, beta) - offset, zorder=10)

    plt.title(f'{name} (fitted)\ndataset {timestamp}')
    plt.xlabel(f'$\\beta$ [$J^{{-1}}$]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.savefig(f'svg/full/{timestamp}-over-beta-fit.svg')
    plt.savefig(f'/Users/julian/Dropbox/University/PE3/report/figures/full/{timestamp}-over-beta-fit.png')
    plt.show()

    plt.errorbar(temperature, averaged_current, yerr=std_current, fmt='o', markersize=2, capsize=3,
                 errorevery=error_every, alpha=0.5)

    plt.title(f'{name} (sanitized)\ndataset {timestamp}')
    plt.xlabel(f'Temperature [{temperature_converter.unit}]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.savefig(f'svg/full/{timestamp}-over-temperature.svg')
    plt.savefig(f'/Users/julian/Dropbox/University/PE3/report/figures/full/{timestamp}-over-temperature.png')
    plt.show()

    plt.errorbar(temperature, averaged_current, yerr=std_current, fmt='o', markersize=2, capsize=3,
                 errorevery=error_every, alpha=0.5)
    plt.plot(temperature, arrhenius_equation(D0, Ea, beta) - offset, zorder=10)

    plt.title(f'{name} (fitted)\ndataset {timestamp}')
    plt.xlabel(f'Temperature [{temperature_converter.unit}]')
    plt.ylabel(f'Filtered diode current [{diode_converter.unit}]')

    plt.tight_layout()
    plt.savefig(f'svg/full/{timestamp}-over-temperature-fit.svg')
    plt.savefig(f'/Users/julian/Dropbox/University/PE3/report/figures/full/{timestamp}-over-temperature-fit.png')
    plt.show()

print(tabulate(table_content, table_headers))
