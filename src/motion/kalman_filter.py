from __future__ import annotations

import numpy as np


class SimpleKalmanFilter:
    """State = [x, y, vx, vy].

    状态向量为 [x, y, vx, vy]。
    """

    def __init__(self, dt: float = 1.0, process_var: float = 1.0, measure_var: float = 10.0) -> None:
        self.dt = dt
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.Q = np.eye(4, dtype=np.float32) * process_var
        self.R = np.eye(2, dtype=np.float32) * measure_var
        self.P0 = np.eye(4, dtype=np.float32) * 100.0

    def initiate(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        mean = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        cov = self.P0.copy()
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + self.Q
        return mean.astype(np.float32), cov.astype(np.float32)

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected_mean = self.H @ mean
        projected_cov = self.H @ cov @ self.H.T + self.R
        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, measurement: tuple[float, float]) -> float:
        proj_mean, proj_cov = self.project(mean, cov)
        delta = np.array(measurement, dtype=np.float32) - proj_mean
        inv_cov = np.linalg.inv(proj_cov)
        distance = float(delta.T @ inv_cov @ delta)
        return distance

    def update(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        measurement: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        z = np.array(measurement, dtype=np.float32)
        projected_mean, projected_cov = self.project(mean, cov)
        innovation = z - projected_mean
        kalman_gain = cov @ self.H.T @ np.linalg.inv(projected_cov)
        mean = mean + kalman_gain @ innovation
        cov = (np.eye(4, dtype=np.float32) - kalman_gain @ self.H) @ cov
        return mean.astype(np.float32), cov.astype(np.float32)
