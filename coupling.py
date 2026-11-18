"""构造耦合权重，统一约定 W[i, j] 表示 j 对 i 的作用。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    ComplexArray = NDArray[np.complex128]
    PhaseField = Callable[[ComplexArray], ComplexArray]
    SpatialKernel = Callable[[FloatArray], FloatArray]


def all_to_all_weights(n: int) -> FloatArray:
    """返回形状 `(n, n)` 的矩阵，对角线为零，其余权重为 1/n。"""
    W = np.ones((n, n)) / n
    np.fill_diagonal(W, 0)
    return W


def all_to_all_field(phases: ComplexArray) -> ComplexArray:
    """计算 1/N 加权的全连接相位场，并排除各节点自身。"""
    return phases.mean() - phases / phases.size


def ring_weights(n: int) -> FloatArray:
    """为至少三个节点构造周期环，每个节点的左右邻居各占 1/2。"""
    if n < 3:
        raise ValueError("周期环至少需要三个节点。")

    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i - 1) % n] = 0.5
        W[i, (i + 1) % n] = 0.5
    return W


def ring_field(phases: ComplexArray) -> ComplexArray:
    """返回每个环形节点的左右邻居相位向量平均值。"""
    if phases.size < 3:
        raise ValueError("周期环至少需要三个节点。")
    return 0.5 * (np.roll(phases, 1) + np.roll(phases, -1))


def periodic_grid_weights(
    grid_shape: tuple[int, int], *, kernel: SpatialKernel | None = None
) -> FloatArray:
    """构造小型二维周期网格的稠密权重矩阵，两轴长度至少为 3。

    按行优先排列节点，排除自耦合。`kernel=None` 时仅连接上下左右
    四邻居，各占 1/4。传入核函数时，用最短周期距离计算权重，并按每行
    权重的绝对值之和归一化。行内权重全为零时保留零行。
    """
    rows, cols = grid_shape
    if rows < 3 or cols < 3:
        raise ValueError("周期网格的行数和列数都至少为 3。")

    n = rows * cols
    nodes = np.arange(n)
    r = nodes // cols
    c = nodes % cols
    if kernel is None:
        W = np.zeros((n, n), float)
        up = ((r - 1) % rows) * cols + c
        down = ((r + 1) % rows) * cols + c
        left = r * cols + (c - 1) % cols
        right = r * cols + (c + 1) % cols
        W[nodes, up] = 0.25
        W[nodes, down] = 0.25
        W[nodes, left] = 0.25
        W[nodes, right] = 0.25
        return W
    dr = np.abs(r[None, :] - r[:, None])
    dc = np.abs(c[None, :] - c[:, None])
    dr = np.minimum(dr, rows - dr)
    dc = np.minimum(dc, cols - dc)
    distance = np.hypot(dr, dc)
    W = kernel(distance)
    np.fill_diagonal(W, 0.0)
    row_norm = np.sum(np.abs(W), axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    W /= row_norm
    return W


def gaussian_kernel(distance: FloatArray, *, sigma: float) -> FloatArray:
    """逐元素计算以 sigma 为宽度的高斯空间核。"""
    return np.exp(-distance**2 / (2 * sigma**2))


def periodic_grid_field(
    grid_shape: tuple[int, int], *, kernel: SpatialKernel | None = None
) -> PhaseField:
    """返回周期网格的相位场计算函数，默认对四邻居求平均。"""
    rows, cols = grid_shape
    if rows < 3 or cols < 3:
        raise ValueError("周期网格的行数和列数都至少为 3。")

    if kernel is None:
        def neighbor_field(phases: ComplexArray) -> ComplexArray:
            grid = phases.reshape(grid_shape)
            neighbors = (
                np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0)
                + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)
            )
            return 0.25 * neighbors.ravel()

        return neighbor_field

    r = np.arange(rows)
    c = np.arange(cols)
    dr = np.minimum(r, rows - r)
    dc = np.minimum(c, cols - c)
    stencil = kernel(np.hypot(dr[:, None], dc[None, :]))
    stencil[0, 0] = 0.0
    norm = np.abs(stencil).sum()
    if norm > 0:
        stencil /= norm
    kernel_spectrum = np.fft.fft2(stencil)

    def field(phases: ComplexArray) -> ComplexArray:
        grid = phases.reshape(grid_shape)
        return np.fft.ifft2(kernel_spectrum * np.fft.fft2(grid)).ravel()

    return field


def mexican_hat_kernel(
    distance: FloatArray,
    *,
    sigma_exc: float,
    sigma_inh: float,
    inhibitory_scale: float,
) -> FloatArray:
    """用窄高斯减去缩放后的宽高斯，构造中心兴奋、外围抑制的核。"""
    excitatory = gaussian_kernel(distance, sigma=sigma_exc)
    inhibitory = gaussian_kernel(distance, sigma=sigma_inh)
    return excitatory - inhibitory_scale * inhibitory
