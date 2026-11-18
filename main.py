"""交互式配置 Kuramoto 实验，播放或导出三联动画。"""

from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from functools import partial
from math import isfinite
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FFMpegWriter, PillowWriter

from coupling import (
    all_to_all_field,
    gaussian_kernel,
    mexican_hat_kernel,
    periodic_grid_field,
    ring_field,
)
from model import harmonic_rhs
from simulation import simulate
from visualization import animate_grid_dynamics

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable

    from matplotlib.animation import FuncAnimation
    from numpy.typing import NDArray

    from coupling import PhaseField

    FloatArray = NDArray[np.float64]


TOPOLOGIES = {
    "all-to-all": "经典全连接",
    "ring": "一维环形近邻",
    "grid-neighbors": "二维周期四邻居",
    "gaussian-grid": "二维周期高斯核",
    "mexican-grid": "二维周期墨西哥帽核",
}
INTERACTIONS = {"sine": "经典正弦", "second-harmonic": "正弦加二次谐波"}
DEFAULT_FPS = 20
DEFAULT_MP4_FPS = 10
Number = TypeVar("Number", int, float)


@dataclass(frozen=True)
class ExperimentSettings:
    interaction: str = "sine"
    amplitude: float = 0.25
    phase_shift: float = 0.0
    sigma: float = 1.5
    sigma_exc: float = 0.8
    sigma_inh: float = 1.8
    inhibitory_scale: float = 0.5
    seed: int = 0
    omega_std: float = 0.5
    duration: float = 40.0
    num_samples: int = 401


def make_experiment_inputs(
    n: int, seed: int, omega_std: float, duration: float, num_samples: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """返回 `(theta0, omega, t_eval)`，前两项形状为 `(n,)`。

    用同一个随机数生成器采样均匀初相位和零均值高斯固有频率，
    并生成覆盖 `[0, duration]` 的等间距采样时间。
    """
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0, 2 * np.pi, size=n)
    omega = rng.normal(0, omega_std, size=n)
    t_eval = np.linspace(0, duration, num_samples)
    return theta0, omega, t_eval


def _make_field(grid_shape: tuple[int, int], topology: str, settings: ExperimentSettings) -> PhaseField:
    if topology == "all-to-all":
        return all_to_all_field
    if topology == "ring":
        return ring_field
    if topology == "grid-neighbors":
        return periodic_grid_field(grid_shape)
    if topology == "gaussian-grid":
        kernel = partial(gaussian_kernel, sigma=settings.sigma)
        return periodic_grid_field(grid_shape, kernel=kernel)
    if topology == "mexican-grid":
        kernel = partial(
            mexican_hat_kernel, sigma_exc=settings.sigma_exc,
            sigma_inh=settings.sigma_inh, inhibitory_scale=settings.inhibitory_scale,
        )
        return periodic_grid_field(grid_shape, kernel=kernel)
    raise ValueError(f"未知的连接拓扑：{topology}")


def run_experiment(
    grid_shape: tuple[int, int], coupling_strengths: tuple[float, ...], *,
    topology: str, settings: ExperimentSettings = ExperimentSettings(),
) -> tuple[FloatArray, dict[float, FloatArray]]:
    """用同一组初值和固有频率生成所选实验的轨迹。"""
    rows, cols = grid_shape
    if rows < 1 or cols < 1:
        raise ValueError("网格的行数和列数必须为正。")
    if topology == "ring" and rows * cols < 3:
        raise ValueError("周期环至少需要三个节点。")
    if settings.interaction not in INTERACTIONS:
        raise ValueError(f"未知的相位相互作用：{settings.interaction}")

    field = _make_field(grid_shape, topology, settings)
    second_coefficient = 0j
    if settings.interaction == "second-harmonic":
        second_coefficient = settings.amplitude * np.exp(1j * settings.phase_shift)
    theta0, omega, t_eval = make_experiment_inputs(
        rows * cols, seed=settings.seed, omega_std=settings.omega_std,
        duration=settings.duration, num_samples=settings.num_samples,
    )
    phases_by_coupling = {}
    for strength in coupling_strengths:
        derivative = partial(
            harmonic_rhs, omega=omega, K=strength, field=field,
            second_coefficient=second_coefficient,
        )
        _, theta = simulate(derivative, theta0, t_eval)
        phases_by_coupling[strength] = theta
    return t_eval, phases_by_coupling


def make_grid_animation(
    fps: int, *, side: int = 32, topology: str = "gaussian-grid", k: float = 2.0,
    settings: ExperimentSettings = ExperimentSettings(num_samples=101),
) -> FuncAnimation:
    """用所选实验的轨迹生成三联动画。"""
    grid_shape = (side, side)
    t, phases = run_experiment(grid_shape, (k,), topology=topology, settings=settings)
    return animate_grid_dynamics(t, phases[k], grid_shape, fps=fps)


def _integer(text: str, *, minimum: int) -> int:
    try:
        number = int(text)
    except ValueError:
        raise ArgumentTypeError("请输入整数。") from None
    if number < minimum:
        raise ArgumentTypeError(f"数值至少为 {minimum}。")
    return number


def _real(text: str, *, minimum: float | None = None, strict: bool = False) -> float:
    try:
        number = float(text)
    except ValueError:
        raise ArgumentTypeError("请输入数字。") from None
    if not isfinite(number):
        raise ArgumentTypeError("请输入有限数值。")
    if minimum is not None and (number < minimum or (strict and number == minimum)):
        relation = "大于" if strict else "大于等于"
        raise ArgumentTypeError(f"数值必须{relation} {minimum:g}。")
    return number


def _minimum_side(topology: str) -> int:
    return {"all-to-all": 1, "ring": 2}.get(topology, 3)


def _build_parser() -> ArgumentParser:
    defaults = ExperimentSettings(num_samples=101)
    parser = ArgumentParser(description=__doc__, epilog="不带参数运行时进入交互菜单。")
    parser.add_argument("--animate", action="store_true", help="直接播放三联动画")
    parser.add_argument("--save", type=Path, metavar="FILE", help="导出 GIF 或 MP4")
    parser.add_argument(
        "--topology", choices=TOPOLOGIES, default="gaussian-grid", help="连接拓扑"
    )
    parser.add_argument(
        "--interaction", choices=INTERACTIONS, default=defaults.interaction, help="相位相互作用"
    )
    parser.add_argument("--side", type=int, default=32, metavar="N", help="网格边长，总振子数为 N×N")
    parser.add_argument("--k", type=_real, default=2.0, help="耦合强度 K")
    positive_real = partial(_real, minimum=0, strict=True)
    nonnegative_real = partial(_real, minimum=0)
    parser.add_argument("--duration", type=positive_real, default=defaults.duration, help="模拟时长")
    parser.add_argument("--omega-std", type=nonnegative_real, default=defaults.omega_std, help="固有频率标准差")
    parser.add_argument("--seed", type=partial(_integer, minimum=0), default=defaults.seed, help="随机种子")
    parser.add_argument("--frames", type=partial(_integer, minimum=2), default=defaults.num_samples, help="采样帧数")
    parser.add_argument(
        "--fps", type=partial(_integer, minimum=1),
        help=f"播放或导出帧率（窗口和 GIF 默认 {DEFAULT_FPS}，MP4 默认 {DEFAULT_MP4_FPS}）",
    )
    parser.add_argument("--amplitude", type=_real, default=defaults.amplitude, help="二次谐波幅度")
    parser.add_argument("--phase-shift", type=_real, default=defaults.phase_shift, help="二次谐波相移（弧度）")
    parser.add_argument("--sigma", type=positive_real, default=defaults.sigma, help="高斯核宽度")
    parser.add_argument("--sigma-exc", type=positive_real, default=defaults.sigma_exc, help="墨西哥帽兴奋宽度")
    parser.add_argument("--sigma-inh", type=positive_real, default=defaults.sigma_inh, help="墨西哥帽抑制宽度")
    parser.add_argument("--inhibitory-scale", type=nonnegative_real, default=defaults.inhibitory_scale, help="墨西哥帽抑制系数")
    return parser


def _prompt_choice(label: str, choices: dict[str, str], default: str) -> str:
    names = tuple(choices)
    default_number = names.index(default) + 1
    print(label)
    for number, name in enumerate(names, start=1):
        print(f"  {number}. {choices[name]}")
    while True:
        selected = input(f"选择编号 [{default_number}]：").strip()
        if not selected:
            return default
        try:
            number = int(selected)
        except ValueError:
            number = 0
        if 1 <= number <= len(names):
            return names[number - 1]
        print(f"请输入 1 到 {len(names)} 之间的编号。")


def _prompt_number(label: str, default: Number, convert: Callable[[str], Number]) -> Number:
    while True:
        value = input(f"{label} [{default:g}]：").strip()
        if not value:
            return default
        try:
            return convert(value)
        except ArgumentTypeError as exc:
            print(exc)


def _prompt_experiment(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args([])
    print("Kuramoto 模拟（按回车使用默认值）")
    args.topology = _prompt_choice("连接方式", TOPOLOGIES, args.topology)
    args.interaction = _prompt_choice("相位相互作用", INTERACTIONS, args.interaction)
    args.side = _prompt_number(
        "网格边长 N（总振子数为 N×N）", args.side,
        partial(_integer, minimum=_minimum_side(args.topology)),
    )
    args.k = _prompt_number("耦合强度 K", args.k, _real)
    positive_real = partial(_real, minimum=0, strict=True)
    nonnegative_real = partial(_real, minimum=0)
    args.duration = _prompt_number("模拟时长", args.duration, positive_real)
    args.omega_std = _prompt_number("固有频率标准差", args.omega_std, nonnegative_real)
    if args.topology == "gaussian-grid":
        args.sigma = _prompt_number("高斯核宽度", args.sigma, positive_real)
    elif args.topology == "mexican-grid":
        args.sigma_exc = _prompt_number("兴奋宽度", args.sigma_exc, positive_real)
        args.sigma_inh = _prompt_number("抑制宽度", args.sigma_inh, positive_real)
        args.inhibitory_scale = _prompt_number("抑制系数", args.inhibitory_scale, nonnegative_real)
    if args.interaction == "second-harmonic":
        args.amplitude = _prompt_number("二次谐波幅度", args.amplitude, _real)
        args.phase_shift = _prompt_number("二次谐波相移（弧度）", args.phase_shift, _real)
    output = input("导出路径（.gif 或 .mp4，留空播放）：").strip()
    args.save = Path(output) if output else None
    if args.save is not None and args.save.suffix.lower() == ".mp4":
        args.fps = _prompt_number(
            "MP4 导出帧率", DEFAULT_MP4_FPS, partial(_integer, minimum=1)
        )
    return args


def _ffmpeg_available() -> bool:
    """优先使用系统 FFmpeg，再查找当前 Python 环境提供的版本。"""
    if FFMpegWriter.isAvailable():
        return True

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return False
    return FFMpegWriter.isAvailable()


def cli() -> None:
    parser = _build_parser()
    try:
        args = _prompt_experiment(parser) if len(sys.argv) == 1 else parser.parse_args()
    except EOFError:
        parser.error("交互菜单需要终端输入，也可以用 --help 查看命令行参数。")
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return

    minimum_side = _minimum_side(args.topology)
    if args.side < minimum_side:
        parser.error(f"{args.topology} 的网格边长至少为 {minimum_side}。")

    suffix = args.save.suffix.lower() if args.save is not None else ""
    fps = args.fps if args.fps is not None else (
        DEFAULT_MP4_FPS if suffix == ".mp4" else DEFAULT_FPS
    )
    writer = None
    if args.save is not None:
        if suffix == ".gif":
            writer = PillowWriter(fps=fps)
        elif suffix == ".mp4":
            if not _ffmpeg_available():
                parser.error(
                    "导出 MP4 需要 FFmpeg。请在项目环境运行 "
                    r".\.venv\Scripts\python.exe -m pip install imageio-ffmpeg，"
                    "或使用 .gif 导出。"
                )
            writer = FFMpegWriter(fps=fps)
        else:
            parser.error("输出文件后缀必须为 .gif 或 .mp4。")

    settings = ExperimentSettings(
        interaction=args.interaction, amplitude=args.amplitude, phase_shift=args.phase_shift,
        sigma=args.sigma, sigma_exc=args.sigma_exc, sigma_inh=args.sigma_inh,
        inhibitory_scale=args.inhibitory_scale, seed=args.seed, omega_std=args.omega_std,
        duration=args.duration, num_samples=args.frames,
    )
    print(
        f"模拟 {args.side ** 2:,} 个振子，{TOPOLOGIES[args.topology]}，"
        f"{INTERACTIONS[args.interaction]}，K={args.k:g}。",
        flush=True,
    )
    animation = make_grid_animation(
        fps, side=args.side, topology=args.topology, k=args.k, settings=settings
    )
    if args.save is None:
        plt.show()
    else:
        animation.save(args.save, writer=writer)
        print(f"已导出到 {args.save}")


if __name__ == "__main__":
    cli()
