#!/usr/bin/env python3
"""最小可用的 Sentinel-2 下载脚本包装。"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from seehydro.acquisition.gee import GEEDownloader


def main() -> None:
    parser = argparse.ArgumentParser(description="按线路分段下载 Sentinel-2 影像")
    parser.add_argument("--route", default="data/route/snbd_centerline.geojson", help="线路 GeoJSON/Shapefile 路径")
    parser.add_argument("--output-dir", default="data/sentinel2", help="输出目录")
    parser.add_argument("--start-date", default="2024-01-01", help="开始日期")
    parser.add_argument("--end-date", default="2025-12-31", help="结束日期")
    parser.add_argument("--segment-length", type=float, default=10000, help="分段长度（米）")
    parser.add_argument("--buffer", type=float, default=2000, help="缓冲宽度（米）")
    parser.add_argument("--project-id", default=None, help="Earth Engine 项目 ID，可留空")
    args = parser.parse_args()

    route_path = Path(args.route)
    if not route_path.exists():
        raise FileNotFoundError(f"未找到线路文件: {route_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_gdf = gpd.read_file(route_path)
    downloader = GEEDownloader(project_id=args.project_id)
    files = downloader.download_by_segments(
        route_gdf=route_gdf,
        segment_length_m=args.segment_length,
        buffer_m=args.buffer,
        date_range=(args.start_date, args.end_date),
        output_dir=output_dir,
    )

    print(f"下载完成，共 {len(files)} 个文件")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
