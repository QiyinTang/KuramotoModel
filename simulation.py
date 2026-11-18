"""从给定初值积分相位轨迹。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    Derivative = Callable[[float, FloatArray], FloatArray]


def simulate(
    rhs: Derivative, theta0: FloatArray, t_eval: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """返回 `(t, theta)`，其中 `theta` 的形状为 `(采样时间数, 振子数)`。

    用 `scipy.integrate.solve_ivp` 从 t_eval[0] 积分到 t_eval[-1]。
    积分失败时抛出 RuntimeError。
    """
    from scipy.integrate import solve_ivp

    solution = solve_ivp(
        rhs,
        t_span=(t_eval[0], t_eval[-1]),
        y0=theta0,
        t_eval=t_eval,
    )
    if not solution.success:
        raise RuntimeError(f"数值积分失败：{solution.message}")

    return solution.t, solution.y.T
