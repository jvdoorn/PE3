import numpy as np
from specc.data.signal import Signal
from specc.plotting import plot
from tabulate import tabulate

from analysis import SELECTED_DATASETS
from tools import TemperatureConverter

temperature_converter = TemperatureConverter()

table_headers = ['Timestamp', 'Name', f'Mean ({temperature_converter.unit})', 'STD']
table_content = []

for timestamp, name in SELECTED_DATASETS.items():
    temperature_file = f'data/temperature-{timestamp}.npz'
    temperature_signal = Signal.load(temperature_file, converter=temperature_converter)

    table_content.append([timestamp, name, np.mean(temperature_signal.csamples), np.std(temperature_signal.csamples)])

    plt = plot(temperature_signal, title=f'{name}\ndataset {timestamp}')
    plt.savefig(f'svg/temperature/{timestamp}.svg')
    plt.show()

print(tabulate(table_content, table_headers))
