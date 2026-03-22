"""栅格读写与遥感指数计算工具函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def read_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """读取栅格文件，返回 (data_array, profile)，data_array shape=(bands, H, W)。"""
    with rasterio.open(Path(path)) as src:
        data = src.read()
        profile = src.profile.copy()
    return data, profile


def write_raster(path: str | Path, data: np.ndarray, profile: dict[str, Any]) -> None:
    """将 data 写入栅格文件，profile 来自 rasterio。"""
    raster_path = Path(path)
    raster_path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        count = 1
        height, width = data.shape
    elif data.ndim == 3:
        count, height, width = data.shape
    else:
        raise ValueError(f"data 维度必须为 2 或 3，实际为 {data.ndim}。")

    out_profile = profile.copy()
    out_profile.update({"height": height, "width": width, "count": count, "dtype": str(data.dtype)})

    with rasterio.open(raster_path, "w", **out_profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def get_raster_bounds(path: str | Path) -> tuple[float, float, float, float]:
    """获取栅格边界，返回 (left, bottom, right, top)。"""
    with rasterio.open(Path(path)) as src:
        b = src.bounds
    return b.left, b.bottom, b.right, b.top


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算 NDWI = (Green - NIR) / (Green + NIR)，避免除零，返回 float32。"""
    green_f = green.astype(np.float32, copy=False)
    nir_f = nir.astype(np.float32, copy=False)
    denom = green_f + nir_f
    result = np.where(denom != 0, (green_f - nir_f) / denom, 0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算 NDVI = (NIR - Red) / (NIR + Red)，避免除零，返回 float32。"""
    red_f = red.astype(np.float32, copy=False)
    nir_f = nir.astype(np.float32, copy=False)
    denom = nir_f + red_f
    result = np.where(denom != 0, (nir_f - red_f) / denom, 0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)
