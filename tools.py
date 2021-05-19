from typing import Union

import numpy as np


def to_kelvin(T_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return T_celsius + 273.25


def to_celsius(T_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return T_kelvin - 273.25


class IVConverter:
    def __init__(self):
        self.sigma_V = 0.3052e-3

        self.R1 = 1e3
        self.R2 = 10
        self.R3 = 10e3

        self.R1_tolerance = self.R2_tolerance = self.R3_tolerance = 0.01

    @property
    def R(self) -> float:
        return self.R1 * self.R3 / self.R2

    @property
    def sigma_R1(self) -> float:
        return self.R1 * self.R1_tolerance

    @property
    def sigma_R2(self) -> float:
        return self.R2 * self.R2_tolerance

    @property
    def sigma_R3(self) -> float:
        return self.R3 * self.R3_tolerance

    @property
    def sigma_R(self) -> float:
        return np.sqrt(
            (self.R3 * self.sigma_R1 / self.R2) ** 2 +
            (self.R1 * self.R3 * self.sigma_R2 / self.R2 ** 2) ** 2 +
            (self.R1 * self.sigma_R3 / self.R2) ** 2
        )

    def I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return V / self.R

    def sigma_I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.sqrt(
            (self.sigma_V / self.R) ** 2 +
            (V * self.sigma_R / self.R ** 2) ** 2
        )


class TemperatureConverter:
    def __init__(self):
        self.V_in = 5
        self.sigma_V_in = 0.2
        self.sigma_V = 0.3052e-3

        self.R1 = 5.08e3
        self.R1_tolerance = 0.01

        # TTC05682 properties
        self.R0 = 6800
        self.R0_tolerance = 0.05
        self.T0 = to_kelvin(25)

        self.B = 4050

    @property
    def sigma_R1(self) -> float:
        return self.R1 * self.R1_tolerance

    @property
    def sigma_R0(self) -> float:
        return self.R0 * self.R0_tolerance

    def sigma_R(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.sqrt((self.V_in * self.sigma_R1 / (V - 1)) ** 2 + (self.R1 * self.sigma_V_in / (V - 1)) ** 2 + (
            (self.R1 * self.V_in * self.sigma_V / (V - 1) ** 2)) ** 2)

    def R(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.R1 * (self.V_in / V - 1)

    def T(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1 / (1 / self.T0 + (1 / self.B) * np.log(self.R(V) / self.R0))

    def sigma_T(self, V: np.ndarray):
        return np.std(self.T(V))
