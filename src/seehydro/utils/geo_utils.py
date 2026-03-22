"""GIS 工具函数：像素/地理坐标互转、距离测量、投影变换与 UTM 推算。"""

from __future__ import annotations

import math
from typing import Any

from geopandas import GeoDataFrame
from pyproj import CRS, Geod


def pixel_to_geo(pixel_xy: tuple[int, int], transform: Any) -> tuple[float, float]:
    """像素坐标(col, row)转地理坐标(lon, lat)，transform 为 rasterio Affine。"""
    col, row = pixel_xy
    lon, lat = transform * (col, row)
    return float(lon), float(lat)


def geo_to_pixel(lon: float, lat: float, transform: Any) -> tuple[int, int]:
    """地理坐标(lon, lat)转像素坐标(col, row)，向下取整。"""
    col_f, row_f = (~transform) * (lon, lat)
    return int(math.floor(col_f)), int(math.floor(row_f))


def measure_distance_m(
    point1: tuple[float, float],
    point2: tuple[float, float],
    crs: str = "EPSG:4326",
) -> float:
    """计算两点间大地线距离（米），point=(lon, lat) 或投影坐标。"""
    crs_obj = CRS.from_user_input(crs)
    x1, y1 = point1
    x2, y2 = point2

    if crs_obj.to_epsg() == 4326:
        geod = Geod(ellps="WGS84")
        _, _, distance_m = geod.inv(x1, y1, x2, y2)
        return float(distance_m)

    return float(math.hypot(x2 - x1, y2 - y1))


def reproject_gdf(gdf: GeoDataFrame, target_crs: str) -> GeoDataFrame:
    """将 GeoDataFrame 重投影到目标 CRS，返回新对象。"""
    return gdf.to_crs(target_crs)


def get_utm_crs(lon: float, lat: float) -> str:
    """根据经纬度推算最合适的 UTM CRS 字符串（如 'EPSG:32649'）。"""
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(60, zone))
    epsg_base = 32600 if lat >= 0 else 32700
    return f"EPSG:{epsg_base + zone}"
