"""波段增强处理模块，用于遥感影像水体提取预处理。"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算归一化水体指数 NDWI = (Green - NIR) / (Green + NIR)，分母为0处填0，范围[-1,1]。"""
    g = np.asarray(green, dtype=np.float32)
    n = np.asarray(nir, dtype=np.float32)
    denom = g + n
    result = np.zeros_like(denom)
    np.divide(g - n, denom, out=result, where=denom != 0.0)
    return np.clip(result, -1.0, 1.0)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算归一化植被指数 NDVI = (NIR - Red) / (NIR + Red)，分母为0处填0，范围[-1,1]。"""
    r = np.asarray(red, dtype=np.float32)
    n = np.asarray(nir, dtype=np.float32)
    denom = n + r
    result = np.zeros_like(denom)
    np.divide(n - r, denom, out=result, where=denom != 0.0)
    return np.clip(result, -1.0, 1.0)


def _linear_stretch_to_uint8(channel: np.ndarray) -> tuple[np.ndarray, float, float]:
    """将单通道线性拉伸到uint8范围[0,255]，返回(uint8数组, vmin, vmax)。"""
    ch = np.asarray(channel, dtype=np.float32)
    vmin = float(ch.min())
    vmax = float(ch.max())
    vrange = vmax - vmin
    if vrange == 0.0:
        return np.zeros(ch.shape, dtype=np.uint8), vmin, vmax
    stretched = np.clip((ch - vmin) / vrange * 255.0, 0.0, 255.0).astype(np.uint8)
    return stretched, vmin, vmax


def _restore_from_uint8(channel_u8: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """从uint8通过逆线性变换恢复float32。"""
    vrange = vmax - vmin
    if vrange == 0.0:
        return np.full(channel_u8.shape, vmin, dtype=np.float32)
    return (channel_u8.astype(np.float32) / 255.0 * vrange + vmin)


def _apply_clahe_single_channel(channel: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """对单个2D通道应用CLAHE。"""
    if channel.ndim != 2:
        raise ValueError(f"单通道输入必须是2D，得到 ndim={channel.ndim}")
    if channel.dtype == np.uint8:
        return clahe.apply(channel)
    ch_f32 = np.asarray(channel, dtype=np.float32)
    ch_u8, vmin, vmax = _linear_stretch_to_uint8(ch_f32)
    enhanced_u8 = clahe.apply(ch_u8)
    return _restore_from_uint8(enhanced_u8, vmin, vmax)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """对图像应用CLAHE自适应直方图均衡化。

    - 单通道(HxW)：直接处理
    - 多通道(CxHxW)：逐通道处理
    - float32输入先线性拉伸到uint8，处理后还原float32
    """
    if clip_limit <= 0:
        raise ValueError(f"clip_limit 必须大于0，得到 {clip_limit}")
    if grid_size <= 0:
        raise ValueError(f"grid_size 必须大于0，得到 {grid_size}")
    image_arr = np.asarray(image)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size)))
    logger.debug("CLAHE: shape={}, dtype={}", image_arr.shape, image_arr.dtype)
    if image_arr.ndim == 2:
        return _apply_clahe_single_channel(image_arr, clahe)
    if image_arr.ndim == 3:
        channels = [_apply_clahe_single_channel(image_arr[c], clahe) for c in range(image_arr.shape[0])]
        return np.stack(channels, axis=0)
    raise ValueError(f"不支持的图像维度 ndim={image_arr.ndim}，期望 2 或 3")


def enhance_for_water(
    bands: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """水体增强处理流程。

    输入bands字典，键为: green, red, nir（必须），swir（可选）。
    对所有输入波段应用CLAHE增强，输出增加ndwi, ndvi通道。
    缺少必须波段时抛出KeyError。
    """
    required_keys = {"green", "red", "nir"}
    missing = sorted(required_keys - set(bands.keys()))
    if missing:
        raise KeyError(f"缺少必须波段: {missing}")

    logger.info("开始水体增强处理，输入波段: {}", sorted(bands.keys()))

    enhanced: dict[str, np.ndarray] = {}
    for name, band in bands.items():
        enhanced[name] = apply_clahe(band)

    green = enhanced["green"]
    red = enhanced["red"]
    nir = enhanced["nir"]

    if not (green.shape == red.shape == nir.shape):
        raise ValueError(f"必须波段形状不一致: green={green.shape}, red={red.shape}, nir={nir.shape}")

    enhanced["ndwi"] = compute_ndwi(green, nir)
    enhanced["ndvi"] = compute_ndvi(red, nir)

    logger.info("水体增强完成，输出波段: {}", sorted(enhanced.keys()))
    return enhanced

