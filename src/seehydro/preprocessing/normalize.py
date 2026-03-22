"""遥感影像归一化工具模块，支持百分位数和Min-Max归一化。"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


def _validate_and_cast(data: np.ndarray) -> np.ndarray:
    """验证输入为2D或3D数组并转换为float32。"""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data 必须是 np.ndarray，得到 {type(data)!r}")
    if data.ndim not in (2, 3):
        raise ValueError(f"data 必须是 2D(H,W) 或 3D(C,H,W)，得到 shape={data.shape}")
    return data.astype(np.float32, copy=False)


def normalize_percentile(
    data: np.ndarray,
    low: float = 2,
    high: float = 98,
) -> np.ndarray:
    """百分位数归一化到[0,1]。

    - HxW：整体计算百分位
    - CxHxW：逐通道独立归一化
    - low==high时该通道填0，结果clip到[0,1]
    """
    if not (0 <= low <= 100 and 0 <= high <= 100):
        raise ValueError(f"low/high 须在[0,100]，得到 low={low}, high={high}")
    if low > high:
        raise ValueError(f"low 须 <= high，得到 low={low}, high={high}")
    arr = _validate_and_cast(data)
    if arr.ndim == 2:
        p_low = float(np.percentile(arr, low))
        p_high = float(np.percentile(arr, high))
        denom = p_high - p_low
        if denom == 0:
            logger.warning("normalize_percentile: 2D输入百分位范围为0，返回全零。")
            return np.zeros_like(arr)
        return np.clip((arr - p_low) / denom, 0.0, 1.0).astype(np.float32)
    c = arr.shape[0]
    flat = arr.reshape(c, -1)
    p_low_arr = np.percentile(flat, low, axis=1)
    p_high_arr = np.percentile(flat, high, axis=1)
    denom_arr = p_high_arr - p_low_arr
    degenerate = denom_arr == 0
    if np.any(degenerate):
        logger.warning("normalize_percentile: 通道 {} 百分位范围为0，该通道填零。", np.where(degenerate)[0].tolist())
    safe_denom = np.where(degenerate, 1.0, denom_arr)
    p_low_bc = p_low_arr[:, None, None]
    safe_denom_bc = safe_denom[:, None, None]
    out = (arr - p_low_bc) / safe_denom_bc
    out = np.where(degenerate[:, None, None], 0.0, out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def normalize_minmax(data: np.ndarray) -> np.ndarray:
    """Min-Max归一化到[0,1]，逐通道处理（CxHxW）或整体处理（HxW）。"""
    arr = _validate_and_cast(data)
    if arr.ndim == 2:
        vmin = float(arr.min())
        vmax = float(arr.max())
        denom = vmax - vmin
        if denom == 0:
            logger.warning("normalize_minmax: 2D输入值域为0，返回全零。")
            return np.zeros_like(arr)
        return np.clip((arr - vmin) / denom, 0.0, 1.0).astype(np.float32)
    c = arr.shape[0]
    flat = arr.reshape(c, -1)
    vmin_arr = flat.min(axis=1)
    vmax_arr = flat.max(axis=1)
    denom_arr = vmax_arr - vmin_arr
    degenerate = denom_arr == 0
    if np.any(degenerate):
        logger.warning("normalize_minmax: 通道 {} 值域为0，该通道填零。", np.where(degenerate)[0].tolist())
    safe_denom = np.where(degenerate, 1.0, denom_arr)
    vmin_bc = vmin_arr[:, None, None]
    safe_denom_bc = safe_denom[:, None, None]
    out = (arr - vmin_bc) / safe_denom_bc
    out = np.where(degenerate[:, None, None], 0.0, out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def normalize_image(
    data: np.ndarray,
    method: str = "percentile",
    **kwargs: Any,
) -> np.ndarray:
    """根据方法名选择归一化方式。

    Args:
        data: 输入数组，2D(H,W) 或 3D(C,H,W)
        method: "percentile" 或 "minmax"
        **kwargs: 传递给对应归一化函数的参数

    Returns:
        归一化后的float32数组，范围[0,1]
    """
    method_norm = method.lower().strip()
    logger.debug("normalize_image: method={}, kwargs={}", method_norm, kwargs)
    if method_norm == "percentile":
        return normalize_percentile(data, **kwargs)
    if method_norm == "minmax":
        return normalize_minmax(data, **kwargs)
    raise ValueError(f"不支持的归一化方法: {method!r}，可选 percentile 或 minmax")

