"""桥梁参数提取."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import Point, box

from seehydro.extraction.geo_measure import get_utm_crs, measure_distance_m


def extract_bridge_params(
    detections: list[dict],
    tile_transform,
    crs: str = "EPSG:4326",
    canal_width_m: float | None = None,
) -> gpd.GeoDataFrame:
    """从检测结果中提取桥梁参数.

    Args:
        detections: 检测结果列表，每个含 bbox, class_id, class_name, confidence
        tile_transform: 瓦片的仿射变换
        crs: 坐标系
        canal_width_m: 渠道宽度（如已知，用于校正跨度）

    Returns:
        GeoDataFrame，含 geometry(Point), bridge_type, span_m, confidence
    """
    bridge_classes = {"highway_bridge", "railway_bridge"}

    records = []
    for det in detections:
        if det["class_name"] not in bridge_classes:
            continue

        bbox = det["bbox"]  # [x1, y1, x2, y2] 像素坐标
        cx_px = (bbox[0] + bbox[2]) / 2
        cy_px = (bbox[1] + bbox[3]) / 2

        # 像素坐标转地理坐标
        lon, lat = tile_transform * (cx_px + 0.5, cy_px + 0.5)

        # 计算跨度（bbox的对角线长度的短边作为近似跨度）
        w_px = bbox[2] - bbox[0]
        h_px = bbox[3] - bbox[1]
        span_px = min(w_px, h_px)

        # 转为米
        # 近似：通过bbox两端点的地理距离
        lon1, lat1 = tile_transform * (bbox[0] + 0.5, cy_px + 0.5)
        lon2, lat2 = tile_transform * (bbox[2] + 0.5, cy_px + 0.5)
        span_h = measure_distance_m((lon1, lat1), (lon2, lat2))

        lon1, lat1 = tile_transform * (cx_px + 0.5, bbox[1] + 0.5)
        lon2, lat2 = tile_transform * (cx_px + 0.5, bbox[3] + 0.5)
        span_v = measure_distance_m((lon1, lat1), (lon2, lat2))

        # 跨度取较短边（桥梁跨越渠道方向通常是短边）
        span_m = min(span_h, span_v)

        # 如果已知渠道宽度，可做合理性校验
        if canal_width_m and span_m < canal_width_m * 0.5:
            span_m = canal_width_m  # 校正

        records.append({
            "geometry": Point(lon, lat),
            "bridge_type": det["class_name"],
            "bridge_type_cn": "公路桥" if det["class_name"] == "highway_bridge" else "铁路桥",
            "span_m": round(span_m, 1),
            "bbox_width_m": round(span_h, 1),
            "bbox_height_m": round(span_v, 1),
            "confidence": round(det["confidence"], 3),
        })

    gdf = gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame(
        columns=["geometry", "bridge_type", "span_m", "confidence"], crs=crs
    )
    logger.info(f"提取桥梁参数: {len(gdf)} 座桥")
    return gdf
