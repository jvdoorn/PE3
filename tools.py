from typing import Union

import numpy as np
from specc.analysis.converter import Converter


def to_kelvin(T_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return T_celsius + 273.25


def to_celsius(T_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return T_kelvin - 273.25


class DiodeConverter(Converter):
    def __init__(self):
        super().__init__(unit='nA')

        self.R = 0.511e6
        self.tolerance_R = 0.05
        self.sigma_R = self.tolerance_R * self.R

    def sigma_V(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 0.005 * V + 4 / (2 ** 16 - 1)

    @property
    def dv_I_V(self) -> float:
        return 1 / self.R

    def dv_I_R(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -V / self.R ** 2

    def convert(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (V / self.R) * 1e9

    def error(self, V: Union[float, np.ndarray]) -> float:
        return np.sqrt(
            (self.dv_I_V * self.sigma_V(V)) ** 2
            + (self.dv_I_R(V) * self.sigma_R) ** 2
        ) * 1e9


class TemperatureConverter(Converter):
    def __init__(self):
        super().__init__('K')
        self.V_in = 5

        self.R1 = 5.08e3
        self.tolerance_R1 = 0.01
        self.sigma_R1 = self.tolerance_R1 * self.R1

        # TTC05682 properties
        self.R0 = 6800
        self.tolerance_R0 = 0.05
        self.sigma_R0 = self.tolerance_R0 * self.R0
        self.T0 = to_kelvin(25)

        self.B = 4050

    def R(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.R1 * (self.V_in / V - 1)

    def dv_R_R1(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.V_in / V - 1

    def dv_R_V(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -self.V_in / V ** 2

    def dv_T_R(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -self.convert(V) ** 2 * 1 / self.B * 1 / self.R(V)

    def dv_T_R0(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -self.convert(V) ** 2 * 1 / self.B * -1 / self.R0

    def sigma_V(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 0.005 * V + 20 / (2 ** 16 - 1)

    def convert(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Converts the voltage V to a temperature in K.
        """
        return 1 / (1 / self.T0 + (1 / self.B) * np.log(self.R(V) / self.R0))

    def error(self, V: Union[float, np.ndarray]) -> float:
        return np.sqrt(
            (self.dv_T_R(V) * self.dv_R_R1(V) * self.sigma_R1) ** 2
            + (self.dv_T_R0(V) * self.sigma_R0) ** 2
            + (self.dv_T_R(V) * self.dv_R_V(V) * self.sigma_V(V)) ** 2
        )


def arrhenius_equation(D0, Ea, beta):
    return D0 * np.exp(-Ea * beta)


def fit_arrhenius_equation(D, beta, weight):
    popt, cov = np.polyfit(beta, np.log(D), 1, w=weight, cov=True)

    D0 = np.exp(popt[1])
    Ea = -popt[0]

    std = np.sqrt(np.diag(cov))
    D0_std = np.exp(std[1])
    Ea_std = std[0]

    return D0, Ea, D0_std, Ea_std
