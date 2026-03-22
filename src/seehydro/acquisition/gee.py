"""该模块提供基于 Google Earth Engine 的 Sentinel-2 影像获取与分段下载功能。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import linemerge, substring
from tqdm import tqdm

try:
    import ee
    import geemap
except ImportError as e:
    raise ImportError(f"请安装 earthengine-api 和 geemap: {e}") from e

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


class GEEDownloader:
    """Google Earth Engine Sentinel-2 影像下载器."""

    def __init__(self, project_id: str | None = None) -> None:
        """初始化下载器并执行 GEE 认证。

        Args:
            project_id: GEE 项目 ID，为 None 时使用默认项目。
        """
        self.project_id = project_id
        self.authenticate()

    def authenticate(self) -> None:
        """执行 GEE 认证，优先使用已有凭据，失败后触发交互认证。"""
        try:
            ee.Initialize(project=self.project_id)
            logger.info("GEE 初始化成功，使用已有凭据。")
        except ee.EEException:
            logger.warning("GEE 初始化失败，开始执行 ee.Authenticate()。")
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            logger.info("GEE 认证并初始化成功。")

    def get_sentinel2(
        self,
        geometry: ee.Geometry,
        date_range: tuple[str, str],
        cloud_pct_max: int = 10,
        bands: list[str] | None = None,
    ) -> ee.Image:
        """获取指定区域和时间范围的 Sentinel-2 中值合成影像。

        Args:
            geometry: GEE 地理范围。
            date_range: 时间范围 (start_date, end_date)，格式 "YYYY-MM-DD"。
            cloud_pct_max: 最大云量百分比阈值。
            bands: 波段列表，默认 ["B2", "B3", "B4", "B8", "B11", "B12"]。

        Returns:
            云量最少时段的中位数合成 ee.Image。
        """
        selected_bands = bands or ["B2", "B3", "B4", "B8", "B11", "B12"]

        try:
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geometry)
                .filterDate(date_range[0], date_range[1])
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
            )
            image = collection.median().select(selected_bands)
            return image
        except ee.EEException as exc:
            logger.error("获取 Sentinel-2 影像失败: {}", exc)
            raise

    def download_image(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        output_path: str | Path,
        scale: int = 10,
    ) -> Path:
        """将 ee.Image 导出为本地 GeoTIFF 文件。

        Args:
            image: 待下载的 GEE 影像。
            geometry: 导出范围。
            output_path: 本地输出路径（.tif）。
            scale: 空间分辨率（米），默认 10m。

        Returns:
            下载完成的文件路径。
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            geemap.ee_export_image(
                image,
                filename=str(out_path),
                scale=scale,
                region=geometry,
                file_per_band=False,
            )
            logger.info("影像下载完成: {}", out_path)
            return out_path
        except ee.EEException as exc:
            logger.error("影像下载失败 {}: {}", out_path, exc)
            raise

    def download_by_segments(
        self,
        route_gdf: gpd.GeoDataFrame,
        segment_length_m: float = 10000,
        buffer_m: float = 2000,
        date_range: tuple[str, str] = ("2024-01-01", "2025-12-31"),
        output_dir: str | Path = "data/sentinel2",
    ) -> list[Path]:
        """按路线分段并批量下载每段缓冲区对应的 Sentinel-2 影像。

        Args:
            route_gdf: 线路 GeoDataFrame。
            segment_length_m: 每段长度（米），默认 10000m。
            buffer_m: 缓冲区半径（米），默认 2000m。
            date_range: 时间范围，默认 2024-01-01 至 2025-12-31。
            output_dir: 影像输出目录。

        Returns:
            已成功下载的影像路径列表。
        """
        if route_gdf.empty:
            logger.warning("输入 route_gdf 为空，未执行下载。")
            return []

        if segment_length_m <= 0:
            raise ValueError("segment_length_m 必须大于 0。")
        if buffer_m <= 0:
            raise ValueError("buffer_m 必须大于 0。")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        route_3857 = route_gdf.to_crs(epsg=3857)
        merged = route_3857.geometry.unary_union
        merged_lines = linemerge(merged)
        segments = self._split_to_segments(merged_lines, segment_length_m)

        if not segments:
            logger.warning("未生成任何线段，请检查输入几何。")
            return []

        downloaded_paths: list[Path] = []
        for i, segment in enumerate(tqdm(segments, desc="下载影像段")):
            segment_file = out_dir / f"segment_{i:03d}.tif"
            if segment_file.exists():
                logger.info("缓存命中: {}", segment_file)
                downloaded_paths.append(segment_file)
                continue

            try:
                buffered_3857 = segment.buffer(buffer_m)
                buffered_4326 = gpd.GeoSeries([buffered_3857], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
                ee_geometry = ee.Geometry(mapping(buffered_4326))

                image = self.get_sentinel2(geometry=ee_geometry, date_range=date_range)
                saved = self.download_image(
                    image=image,
                    geometry=ee_geometry,
                    output_path=segment_file,
                )
                downloaded_paths.append(saved)
            except Exception as exc:  # noqa: BLE001
                logger.warning("线段 {} 下载失败，跳过。错误: {}", i, exc)
                continue

        return downloaded_paths

    def _split_to_segments(self, geometry: BaseGeometry, segment_length_m: float) -> list[LineString]:
        """将线几何按固定长度切分为线段列表。

        Args:
            geometry: 输入线几何（LineString 或 MultiLineString）。
            segment_length_m: 每段长度（米）。

        Returns:
            切分后的 LineString 列表。
        """
        lines = self._extract_lines(geometry)
        segments: list[LineString] = []

        for line in lines:
            length = line.length
            if length <= 0:
                continue

            n_segments = max(1, math.ceil(length / segment_length_m))
            starts = np.arange(0.0, n_segments * segment_length_m, segment_length_m)

            for start in starts:
                end = min(start + segment_length_m, length)
                if end <= start:
                    continue

                piece = substring(line, float(start), float(end))
                if isinstance(piece, LineString) and piece.length > 0:
                    segments.append(piece)
                elif isinstance(piece, MultiLineString):
                    for sub in piece.geoms:
                        if isinstance(sub, LineString) and sub.length > 0:
                            segments.append(sub)

        return segments

    def _extract_lines(self, geometry: BaseGeometry) -> list[LineString]:
        """从任意几何中递归提取 LineString。

        Args:
            geometry: 输入几何对象。

        Returns:
            LineString 列表。
        """
        if geometry.is_empty:
            return []

        if isinstance(geometry, LineString):
            return [geometry]

        if isinstance(geometry, MultiLineString):
            return [g for g in geometry.geoms if isinstance(g, LineString)]

        if hasattr(geometry, "geoms"):
            lines: list[LineString] = []
            for geom in geometry.geoms:  # type: ignore[attr-defined]
                lines.extend(self._extract_lines(geom))
            return lines

        return []
