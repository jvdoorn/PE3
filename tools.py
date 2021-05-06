from typing import Union

import numpy as np


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
        pass

    def T(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return V

    def sigma_T(self, T: np.ndarray) -> float:
        return np.sqrt(np.sum((T - T.mean()) ** 2) / len(T))
