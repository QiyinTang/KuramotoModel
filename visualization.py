"""从相位轨迹计算同步序参量并绘图。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]


PHASE_COLORS = LinearSegmentedColormap.from_list(
    "kuramoto_phase",
    ["#3653a4", "#338dc2", "#6ec9a4", "#f6de68", "#ed784e", "#b64a85", "#3653a4"],
)
PHASE_BIN_EDGES = np.linspace(0, 2 * np.pi, 121)
MAX_CIRCLE_POINTS = 512


def _mean_phase_vector(theta: FloatArray) -> NDArray[np.complex128]:
    """对每一帧的单位相位向量求平均，得到复序参量。"""
    return np.cos(theta).mean(axis=1) + 1j * np.sin(theta).mean(axis=1)


def order_parameter(theta: FloatArray) -> FloatArray:
    """对形状 `(采样时间数, 振子数)` 的轨迹逐帧计算 |mean(exp(i*theta))|。"""
    return np.abs(_mean_phase_vector(theta))


def plot_order_parameter(t: FloatArray, by_coupling: Mapping[float, FloatArray]) -> Figure:
    """按耦合强度 K 绘制多条 R(t)，返回 Figure。"""
    fig, ax = plt.subplots()
    for K, R in by_coupling.items():
        ax.plot(t, R, label=f"K = {K:g}")

    ax.set(xlabel="Time", ylabel="R(t)", ylim=(0, 1.05))
    ax.legend()
    return fig


def _setup_phase_circle(
    ax: Axes, n: int
) -> tuple[PathCollection, slice | NDArray[np.intp], LineCollection | None]:
    circle = np.column_stack((np.cos(PHASE_BIN_EDGES), np.sin(PHASE_BIN_EDGES)))
    ax.plot(circle[:, 0], circle[:, 1], color="0.75")
    sample: slice | NDArray[np.intp] = slice(None)
    density = None
    if n > MAX_CIRCLE_POINTS:
        sample = np.random.default_rng(0).choice(n, size=MAX_CIRCLE_POINTS, replace=False)
        outer_circle = 1.07 * circle
        segments = np.stack((outer_circle[:-1], outer_circle[1:]), axis=1)
        density = LineCollection(segments, cmap="Blues", norm=Normalize(0, 1), linewidths=3)
        ax.add_collection(density)

    points = ax.scatter([], [], s=16 if density is None else 7, color="tab:blue", alpha=0.55)
    ax.set(
        xlabel=r"$\cos\theta$", ylabel=r"$\sin\theta$",
        xlim=(-1.15, 1.15), ylim=(-1.15, 1.15),
    )
    ax.set_aspect("equal")
    return points, sample, density


def _update_phase_circle(
    phases: FloatArray, points: PathCollection, sample: slice | NDArray[np.intp],
    density: LineCollection | None,
) -> None:
    shown = phases[sample]
    points.set_offsets(np.column_stack((np.cos(shown), np.sin(shown))))
    if density is not None:
        counts, _ = np.histogram(phases % (2 * np.pi), bins=PHASE_BIN_EDGES)
        density.set_array(np.sqrt(counts / max(1, counts.max())))


def plot_unit_circle(theta: FloatArray) -> Figure:
    """绘制单位圆散点，振子较多时同时显示全体相位密度。"""
    fig, ax = plt.subplots()
    points, sample, density = _setup_phase_circle(ax, theta.size)
    _update_phase_circle(theta, points, sample, density)
    if density is not None:
        ax.set_title("Phase density and sampled oscillators")
    return fig


def plot_ring_phases(t: FloatArray, theta: FloatArray) -> Figure:
    """绘制环上节点随时间变化的相位图，返回 Figure。"""
    fig, ax = plt.subplots()
    nodes = np.arange(theta.shape[1])
    wrapped = theta.T % (2 * np.pi)

    mesh = ax.pcolormesh(
        t, nodes, wrapped, shading="nearest",
        cmap=PHASE_COLORS, vmin=0, vmax=2 * np.pi,
    )
    ax.set(xlabel="Time", ylabel="Node index")
    fig.colorbar(mesh, ax=ax, label="Phase (rad)")
    return fig


def _draw_grid_phase(ax: Axes, theta: FloatArray, grid_shape: tuple[int, int]) -> AxesImage:
    image = ax.imshow(
        theta.reshape(grid_shape) % (2 * np.pi),
        origin="upper", interpolation="nearest", cmap=PHASE_COLORS,
        vmin=0, vmax=2 * np.pi,
    )
    ax.set(xlabel="Column", ylabel="Row")
    colorbar = ax.figure.colorbar(image, ax=ax, ticks=[0, np.pi, 2 * np.pi])
    colorbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])
    colorbar.set_label("Phase (rad)")
    return image


def plot_grid_phase(theta: FloatArray, grid_shape: tuple[int, int]) -> Figure:
    """将一帧相位按行优先还原为网格，用固定在 [0, 2π] 的循环色标绘图。"""
    fig, ax = plt.subplots()
    _draw_grid_phase(ax, theta, grid_shape)
    return fig


def animate_grid_dynamics(
    t: FloatArray, theta: FloatArray, grid_shape: tuple[int, int], *, fps: int = 20
) -> FuncAnimation:
    """同步播放二维相位、单位圆散点和全局同步序参量。"""
    rows, cols = grid_shape
    if t.ndim != 1 or t.size < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("t 必须是一维数组，并包含至少两个严格递增的时刻。")
    if rows < 1 or cols < 1 or theta.shape != (t.size, rows * cols):
        raise ValueError("theta 的形状必须是 (时间点数, 网格节点数)。")
    if fps <= 0:
        raise ValueError("fps 必须大于零。")

    mean_vector = _mean_phase_vector(theta)
    coherence = np.abs(mean_vector)
    fig, (grid_ax, circle_ax, curve_ax) = plt.subplots(
        1, 3, figsize=(12, 4), width_ratios=(1.1, 1, 1.3), layout="constrained"
    )
    time_label = fig.suptitle("")

    image = _draw_grid_phase(grid_ax, theta[0], grid_shape)
    grid_ax.set_title("Grid phase")

    points, sample, density = _setup_phase_circle(circle_ax, rows * cols)
    circle_ax.set_title("Phases and mean vector" if density is None else "Phase density and mean vector")
    arrow = circle_ax.quiver(
        [0], [0], [0], [0], angles="xy", scale_units="xy", scale=1,
        color="tab:red", zorder=3,
    )

    curve, = curve_ax.plot([], [], color="tab:red")
    marker, = curve_ax.plot([], [], "o", color="tab:red", markersize=5)
    curve_ax.set(
        title="Global synchronization", xlabel="Time", ylabel="R(t)",
        xlim=(t[0], t[-1]), ylim=(0, 1.05),
    )
    curve_ax.grid(alpha=0.2)

    def update(frame: int) -> None:
        phases = theta[frame]
        image.set_data(phases.reshape(grid_shape) % (2 * np.pi))
        _update_phase_circle(phases, points, sample, density)
        arrow.set_UVC([mean_vector[frame].real], [mean_vector[frame].imag])
        curve.set_data(t[:frame + 1], coherence[:frame + 1])
        marker.set_data([t[frame]], [coherence[frame]])
        time_label.set_text(f"t = {t[frame]:.2f}    |    R = {coherence[frame]:.3f}")

    update(0)
    return FuncAnimation(fig, update, frames=t.size, interval=1000 / fps)
