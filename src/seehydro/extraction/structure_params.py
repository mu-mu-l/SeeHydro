"""其他建筑物参数提取（倒虹吸、渡槽、闸门等）."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import Point

from seehydro.extraction.geo_measure import measure_distance_m


# 建筑物类别分组
STRUCTURE_GROUPS = {
    "siphon": {"siphon_inlet", "siphon_outlet"},
    "aqueduct": {"aqueduct"},
    "gate": {"check_gate", "drain_gate"},
    "diversion": {"diversion"},
}


def extract_siphon_params(
    detections: list[dict],
    tile_transform,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """提取倒虹吸参数.

    匹配相邻的入口和出口，计算倒虹吸长度。
    """
    inlets = []
    outlets = []

    for det in detections:
        bbox = det["bbox"]
        cx_px = (bbox[0] + bbox[2]) / 2
        cy_px = (bbox[1] + bbox[3]) / 2
        lon, lat = tile_transform * (cx_px + 0.5, cy_px + 0.5)

        entry = {
            "geometry": Point(lon, lat),
            "confidence": det["confidence"],
            "lon": lon,
            "lat": lat,
        }

        if det["class_name"] == "siphon_inlet":
            inlets.append(entry)
        elif det["class_name"] == "siphon_outlet":
            outlets.append(entry)

    # 匹配入口和出口（最近邻配对）
    records = []
    used_outlets = set()

    for inlet in inlets:
        best_dist = float("inf")
        best_outlet_idx = -1

        for j, outlet in enumerate(outlets):
            if j in used_outlets:
                continue
            dist = measure_distance_m((inlet["lon"], inlet["lat"]), (outlet["lon"], outlet["lat"]))
            if dist < best_dist:
                best_dist = dist
                best_outlet_idx = j

        if best_outlet_idx >= 0 and best_dist < 5000:  # 最大匹配距离5km
            used_outlets.add(best_outlet_idx)
            outlet = outlets[best_outlet_idx]

            records.append({
                "geometry": Point(
                    (inlet["lon"] + outlet["lon"]) / 2,
                    (inlet["lat"] + outlet["lat"]) / 2,
                ),
                "type": "inverted_siphon",
                "type_cn": "倒虹吸",
                "length_m": round(best_dist, 1),
                "inlet_lon": inlet["lon"],
                "inlet_lat": inlet["lat"],
                "outlet_lon": outlet["lon"],
                "outlet_lat": outlet["lat"],
                "confidence": round(min(inlet["confidence"], outlet["confidence"]), 3),
            })

    # 未匹配的入口/出口单独记录
    for i, inlet in enumerate(inlets):
        if not any(r.get("inlet_lon") == inlet["lon"] for r in records):
            records.append({
                "geometry": inlet["geometry"],
                "type": "siphon_inlet_unmatched",
                "type_cn": "倒虹吸入口(未匹配)",
                "length_m": None,
                "confidence": round(inlet["confidence"], 3),
            })

    gdf = gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame(
        columns=["geometry", "type", "length_m", "confidence"], crs=crs
    )
    logger.info(f"提取倒虹吸参数: {len(gdf)} 个")
    return gdf


def extract_aqueduct_params(
    detections: list[dict],
    tile_transform,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """提取渡槽参数."""
    records = []

    for det in detections:
        if det["class_name"] != "aqueduct":
            continue

        bbox = det["bbox"]
        cx_px = (bbox[0] + bbox[2]) / 2
        cy_px = (bbox[1] + bbox[3]) / 2
        lon, lat = tile_transform * (cx_px + 0.5, cy_px + 0.5)

        # 渡槽长度近似为bbox长边
        lon1, lat1 = tile_transform * (bbox[0] + 0.5, bbox[1] + 0.5)
        lon2, lat2 = tile_transform * (bbox[2] + 0.5, bbox[3] + 0.5)
        w = measure_distance_m((lon1, lat1), (lon2, lat1))
        h = measure_distance_m((lon1, lat1), (lon1, lat2))

        records.append({
            "geometry": Point(lon, lat),
            "type": "aqueduct",
            "type_cn": "渡槽",
            "length_m": round(max(w, h), 1),
            "span_m": round(min(w, h), 1),
            "confidence": round(det["confidence"], 3),
        })

    gdf = gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame(
        columns=["geometry", "type", "length_m", "confidence"], crs=crs
    )
    logger.info(f"提取渡槽参数: {len(gdf)} 个")
    return gdf


def extract_gate_params(
    detections: list[dict],
    tile_transform,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """提取闸门参数."""
    gate_classes = {"check_gate", "drain_gate", "diversion"}
    gate_cn = {
        "check_gate": "节制闸",
        "drain_gate": "退水闸",
        "diversion": "分水口",
    }

    records = []
    for det in detections:
        if det["class_name"] not in gate_classes:
            continue

        bbox = det["bbox"]
        cx_px = (bbox[0] + bbox[2]) / 2
        cy_px = (bbox[1] + bbox[3]) / 2
        lon, lat = tile_transform * (cx_px + 0.5, cy_px + 0.5)

        records.append({
            "geometry": Point(lon, lat),
            "type": det["class_name"],
            "type_cn": gate_cn.get(det["class_name"], det["class_name"]),
            "confidence": round(det["confidence"], 3),
        })

    gdf = gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame(
        columns=["geometry", "type", "confidence"], crs=crs
    )
    logger.info(f"提取闸门/分水口参数: {len(gdf)} 个")
    return gdf


def extract_all_structures(
    detections: list[dict],
    tile_transform,
    crs: str = "EPSG:4326",
) -> dict[str, gpd.GeoDataFrame]:
    """提取所有建筑物参数.

    Returns:
        {"siphons": GeoDataFrame, "aqueducts": GeoDataFrame, "gates": GeoDataFrame}
    """
    return {
        "siphons": extract_siphon_params(detections, tile_transform, crs),
        "aqueducts": extract_aqueduct_params(detections, tile_transform, crs),
        "gates": extract_gate_params(detections, tile_transform, crs),
    }
