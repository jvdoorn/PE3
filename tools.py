from typing import Union

import numpy as np
from specc.analysis.converter import Converter


def to_kelvin(T_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return T_celsius + 273.25


def to_celsius(T_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return T_kelvin - 273.25


class DiodeConverter(Converter):
    def __init__(self, R: float):
        super().__init__(unit='nA')

        self.R = R

    def convert(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (V / self.R) * 1e9

    def error(self, V: Union[float, np.ndarray]) -> float:
        # return np.std(self.convert(V))
        return 0.05


class TemperatureConverter(Converter):
    def __init__(self):
        super().__init__('°C')
        self.V_in = 5

        self.R1 = 5.08e3

        # TTC05682 properties
        self.R0 = 6800
        self.T0 = to_kelvin(25)

        self.B = 4050

    def R(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.R1 * (self.V_in / V - 1)

    def convert(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return to_celsius(1 / (1 / self.T0 + (1 / self.B) * np.log(self.R(V) / self.R0))) - 4.8

    def error(self, V: Union[float, np.ndarray]) -> float:
        # return np.std(self.convert(V))
        return 0.1
