from typing import Union

import numpy as np
from scipy.stats import linregress


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
        self.R = 6800

        # TTC05682 property
        self.R0 = 6800
        self.T0 = to_kelvin(25)

        calibration_data = np.asarray([
            [120, 0.2],
            [110, 0.3],
            [100, 0.4],
            [90, 0.5],
            [80, 0.7],
            [30, 5],
            [20, 9],
            [0, 20],
            [-20, 60],
            [-30, 100]
        ]).T
        temperature_data = to_kelvin(calibration_data[0])
        resistance_data = calibration_data[1] * 1e3
        print(temperature_data, resistance_data)

        self.B, _, _, _, self.sigma_B = linregress(1 / temperature_data, np.log(resistance_data))
        print(self.B, self.sigma_B)

    def R_thermistor(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.R * (self.V_in / V - 1)

    def T(self, R) -> Union[float, np.ndarray]:
        return 1 / (1 / self.T0 + (1 / self.B) * np.log(R / self.R0))

    def sigma_T(self, T: np.ndarray) -> float:
        return np.sqrt(np.sum((T - T.mean()) ** 2) / len(T))
