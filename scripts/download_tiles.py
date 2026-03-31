#!/usr/bin/env python3
"""最小可用的在线卫星瓦片下载脚本。"""

from __future__ import annotations

import argparse
import io
import math
import os
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def lon_lat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    y = int(math.floor((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n))
    return x, y


def tile_to_lon_lat(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = (x / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def build_url(provider: str, x: int, y: int, z: int, api_key: str | None) -> str:
    if provider == "google_satellite":
        return f"https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    if provider == "tianditu_img":
        if not api_key:
            raise ValueError("tianditu_img 需要 --api-key")
        return (
            "https://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            f"&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILECOL={x}&TILEROW={y}"
            f"&TILEMATRIX={z}&tk={api_key}"
        )
    raise ValueError(f"不支持的 provider: {provider}")


def download_tile(session: requests.Session, provider: str, x: int, y: int, z: int, api_key: str | None) -> np.ndarray:
    url = build_url(provider, x, y, z, api_key)
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载一小块在线卫星图并拼接为 GeoTIFF")
    parser.add_argument("--west", type=float, default=111.5000, help="经度西边界")
    parser.add_argument("--south", type=float, default=32.6700, help="纬度南边界")
    parser.add_argument("--east", type=float, default=111.5050, help="经度东边界")
    parser.add_argument("--north", type=float, default=32.6750, help="纬度北边界")
    parser.add_argument("--zoom", type=int, default=15, help="缩放级别，建议先用 15 或 16")
    parser.add_argument(
        "--provider",
        default="google_satellite",
        choices=["google_satellite", "tianditu_img"],
        help="底图来源",
    )
    parser.add_argument("--api-key", default=None, help="天地图 key；也可用环境变量 TDT_KEY")
    parser.add_argument("--output-dir", default="data/cache/tiles_demo", help="输出目录")
    args = parser.parse_args()

    west, south, east, north = args.west, args.south, args.east, args.north
    api_key = args.api_key or os.environ.get("TDT_KEY")
    x_min, y_min = lon_lat_to_tile(west, north, args.zoom)
    x_max, y_max = lon_lat_to_tile(east, south, args.zoom)
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_cols = x_max - x_min + 1
    tile_rows = y_max - y_min + 1
    total_width = tile_cols * 256
    total_height = tile_rows * 256
    mosaic = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    session = requests.Session()
    session.headers.update({"User-Agent": "SeeHydro-Demo/0.1"})

    total = tile_cols * tile_rows
    done = 0
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            tile = download_tile(session, args.provider, x, y, args.zoom, api_key)
            x_offset = (x - x_min) * 256
            y_offset = (y - y_min) * 256
            mosaic[y_offset:y_offset + 256, x_offset:x_offset + 256, :] = tile
            done += 1
            print(f"[{done}/{total}] 下载瓦片 z={args.zoom}, x={x}, y={y}")

    left, top = tile_to_lon_lat(x_min, y_min, args.zoom)
    right, bottom = tile_to_lon_lat(x_max + 1, y_max + 1, args.zoom)
    transform = from_bounds(left, bottom, right, top, total_width, total_height)

    out_path = output_dir / f"{args.provider}_{args.zoom}_{x_min}_{y_min}_{x_max}_{y_max}.tif"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=total_height,
        width=total_width,
        count=3,
        dtype=np.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(mosaic[:, :, 0], 1)
        dst.write(mosaic[:, :, 1], 2)
        dst.write(mosaic[:, :, 2], 3)

    print(f"下载完成: {out_path}")


if __name__ == "__main__":
    main()
