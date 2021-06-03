import numpy as np
from specc.data.results import SystemBehaviour
from specc.data.signal import Signal
from specc.plotting import plot

digital_filter_frequency = 10
resistor = 500e3

timestamp = '1622620565'
diode_voltage_file = f'data/current-{timestamp}.npz'

diode_signal = Signal.load(diode_voltage_file) * (1 / resistor)
plot(diode_signal, unit='A').show()

filtered_diode_fft = diode_signal.fft
filtered_diode_fft[np.abs(diode_signal.frequencies) >= digital_filter_frequency] = 0

filtered_diode_signal = Signal(diode_signal.sample_rate, np.fft.ifft(filtered_diode_fft))
plot(filtered_diode_signal, unit='A').show()

diode_behaviour = SystemBehaviour.from_array(diode_signal.masked_frequencies, diode_signal.masked_nfft)
filtered_diode_behaviour = SystemBehaviour.from_array(filtered_diode_signal.masked_frequencies,
                                                      filtered_diode_signal.masked_nfft)
plot([diode_behaviour, filtered_diode_behaviour], labels=['Raw signal', 'Filtered signal']).show()
