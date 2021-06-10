import numpy as np
from specc.data.signal import Signal
from specc.plotting import plot
from tabulate import tabulate

from analysis import FILTER_FREQUENCY, SELECTED_DATASETS
from tools import DiodeConverter

diode_converter = DiodeConverter()

table_headers = ['Timestamp', 'Name', f'Mean ({diode_converter.unit})', 'Mean STD',
                 f'Filtered mean ({diode_converter.unit})', 'Mean filtered STD']
table_content = []

for timestamp, name in SELECTED_DATASETS.items():
    diode_file = f'data/current-{timestamp}.npz'
    try:
        diode_signal = Signal.load(diode_file, converter=diode_converter)
    except FileNotFoundError:
        continue

    filtered_diode_fft = diode_signal.fft
    filtered_diode_fft[np.abs(diode_signal.frequencies) >= FILTER_FREQUENCY] = 0
    filtered_diode_signal = Signal(diode_signal.sample_rate, np.real(np.fft.ifft(filtered_diode_fft)),
                                   converter=diode_converter)

    table_content.append([timestamp, name, np.mean(diode_signal.csamples), np.mean(diode_signal.error),
                          np.mean(filtered_diode_signal.csamples), np.mean(filtered_diode_signal.error)])

    plt = plot(diode_signal, title=f'{name}\ndataset {timestamp}')
    plt.savefig(f'svg/diode/{timestamp}.svg')
    plt.savefig(f'/Users/julian/Dropbox/University/PE3/report/figures/diode/{timestamp}.png')
    plt.show()

    plt = plot(filtered_diode_signal, title=f'{name} (filtered)\ndataset {timestamp}')
    plt.savefig(f'svg/diode/{timestamp}-filtered.svg')
    plt.savefig(f'/Users/julian/Dropbox/University/PE3/report/figures/diode/{timestamp}-filtered.png')
    plt.show()

print(tabulate(table_content, table_headers))
