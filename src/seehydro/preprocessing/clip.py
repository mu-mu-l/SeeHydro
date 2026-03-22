"""影像裁剪预处理模块，用于按矢量几何体裁剪遥感栅格影像。"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from loguru import logger
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry


def clip_raster_by_geometry(
    raster_path: str | Path,
    geometry: BaseGeometry | gpd.GeoDataFrame,
    output_path: str | Path,
    nodata: float = 0,
) -> Path:
    """用矢量几何体裁剪栅格影像。

    使用 rasterio.mask.mask 实现，保留CRS和变换信息。
    geometry可以是shapely几何体或GeoDataFrame（自动统一到栅格CRS）。
    """
    raster_path = Path(raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry.empty:
                raise ValueError("输入的 GeoDataFrame 为空，无法执行裁剪。")
            if geometry.crs is None:
                raise ValueError("输入的 GeoDataFrame 未设置 CRS，无法统一到栅格坐标系。")
            geom_in_crs = geometry.to_crs(raster_crs) if geometry.crs != raster_crs else geometry
            shapes = [mapping(g) for g in geom_in_crs.geometry if g is not None and not g.is_empty]
            if not shapes:
                raise ValueError("GeoDataFrame 中没有有效几何体，无法执行裁剪。")
        else:
            if geometry.is_empty:
                raise ValueError("输入的几何体为空，无法执行裁剪。")
            shapes = [mapping(geometry)]

        out_image, out_transform = mask(src, shapes, crop=True, filled=True, nodata=nodata)

        profile = src.profile.copy()
        profile.update(
            driver=profile.get("driver", "GTiff"),
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            nodata=nodata,
            count=out_image.shape[0],
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_image)

    logger.info("裁剪完成: {} -> {}", raster_path, output_path)
    return output_path


def clip_along_route(
    raster_path: str | Path,
    route_gdf: gpd.GeoDataFrame,
    buffer_m: float,
    output_path: str | Path,
) -> Path:
    """沿线路缓冲区裁剪影像。

    步骤：1) 将线路投影到UTM（用estimate_utm_crs）
                    2) 做buffer_m米缓冲区
                    3) 转回原栅格CRS
                    4) 调用clip_raster_by_geometry裁剪
    """
    if route_gdf.empty:
        raise ValueError("route_gdf 为空，无法执行沿线裁剪。")
    if route_gdf.crs is None:
        raise ValueError("route_gdf 未设置 CRS，无法执行投影与缓冲。")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    utm_crs = route_gdf.estimate_utm_crs()
    if utm_crs is None:
        raise ValueError("无法估算 UTM 坐标系，请检查 route_gdf 的地理范围和 CRS。")

    route_utm = route_gdf.to_crs(utm_crs)
    buffer_series = route_utm.geometry.buffer(buffer_m)
    buffer_gdf = gpd.GeoDataFrame(geometry=buffer_series, crs=utm_crs)
    buffer_in_raster_crs = buffer_gdf.to_crs(raster_crs)

    logger.info("沿线裁剪: raster={}, buffer_m={}", raster_path, buffer_m)
    return clip_raster_by_geometry(raster_path, buffer_in_raster_crs, output_path)


def batch_clip(
    raster_dir: str | Path,
    route_gdf: gpd.GeoDataFrame,
    buffer_m: float,
    output_dir: str | Path,
) -> list[Path]:
    """批量裁剪目录下所有栅格文件（*.tif, *.tiff）。

    输出目录不存在时自动创建，记录成功/失败数量到日志。
    """
    raster_dir = Path(raster_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raster_files = sorted(
        p for p in raster_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )

    results: list[Path] = []
    success_count = 0
    fail_count = 0

    logger.info("开始批量裁剪，输入目录: {}，文件数: {}", raster_dir, len(raster_files))

    for raster_file in raster_files:
        output_path = output_dir / raster_file.name
        try:
            clipped_path = clip_along_route(
                raster_path=raster_file,
                route_gdf=route_gdf,
                buffer_m=buffer_m,
                output_path=output_path,
            )
            results.append(clipped_path)
            success_count += 1
        except Exception as exc:
            fail_count += 1
            logger.error("裁剪失败: {}，错误: {}", raster_file, exc)

    logger.info("批量裁剪完成: 成功 {}，失败 {}，总计 {}", success_count, fail_count, len(raster_files))
    return results

