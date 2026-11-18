"""Kuramoto 方程右端和相位相互作用函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from coupling import PhaseField

    FloatArray = NDArray[np.float64]
    Interaction = Callable[[FloatArray], FloatArray]


def sine_interaction(delta: FloatArray) -> FloatArray:
    """逐元素计算 sin(delta)，保持输入形状。"""
    return np.sin(delta)


def second_harmonic_interaction(
    delta: FloatArray, *, amplitude: float, phase_shift: float = 0.0
) -> FloatArray:
    """计算 sin(delta) + amplitude * sin(2 * delta + phase_shift)。"""
    return np.sin(delta) + amplitude * np.sin(2 * delta + phase_shift)


def rhs(
    t: float,
    theta: FloatArray,
    omega: FloatArray,
    K: float,
    W: FloatArray,
    interaction: Interaction,
) -> FloatArray:
    """用给定权重和相位相互作用计算相位速度，返回形状 `(n,)`。

    `theta` 和 `omega` 的形状为 `(n,)`，`W` 的形状为 `(n, n)`。
    `W[i, j]` 是 j 对 i 的权重，已包含归一化因子。`t` 供积分器传入。
    构造 delta[i, j] = theta[j] - theta[i]，对每一行的加权相互作用求和，
    乘 K 后加到 omega 上。
    """
    delta = theta[None, :] - theta[:, None]
    interaction_sum = np.sum(W * interaction(delta), axis=1)
    return omega + K * interaction_sum


def sine_rhs(
    t: float, theta: FloatArray, omega: FloatArray, K: float, field: PhaseField
) -> FloatArray:
    """通过加权相位场计算纯正弦耦合。"""
    return harmonic_rhs(t, theta, omega, K, field)


def harmonic_rhs(
    t: float, theta: FloatArray, omega: FloatArray, K: float, field: PhaseField,
    *, second_coefficient: complex = 0j,
) -> FloatArray:
    """通过相位场计算一阶和可选二阶谐波，二阶系数为 a * exp(i * phase_shift)。"""
    if K == 0:
        return omega

    phases = np.exp(1j * theta)
    # Im(conj(z_i) * sum_j W_ij * z_j) = sum_j W_ij * sin(theta_j - theta_i).
    interaction_sum = (phases.conj() * field(phases)).imag
    if second_coefficient != 0j:
        squared = phases**2
        interaction_sum += (second_coefficient * squared.conj() * field(squared)).imag
    return omega + K * interaction_sum
