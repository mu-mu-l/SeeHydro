"""南水北调中线线路数据获取模块."""

from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

DEFAULT_BBOX = (32.0, 111.0, 40.5, 117.0)  # (south, west, north, east)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class RouteDataLoader:
    """南水北调中线线路数据加载器."""

    @staticmethod
    def _empty_gdf() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"geometry": [], "name": [], "osm_id": []},
            geometry="geometry",
            crs="EPSG:4326",
        )

    def from_osm(self, bbox: tuple | None = None) -> gpd.GeoDataFrame:
        """从 OpenStreetMap 下载南水北调中线数据.

        使用 Overpass API 查询 waterway=canal + name 含南水北调的数据。
        如果 bbox 为 None，使用默认范围（丹江口到北京）。
        bbox 格式：(south, west, north, east)
        默认 bbox: (32.0, 111.0, 40.5, 117.0)

        使用 requests 直接调用 Overpass API，不依赖 osmnx。
        """
        query_bbox = bbox or DEFAULT_BBOX
        south, west, north, east = query_bbox
        overpass_query = (
            f'[out:json][timeout:60];\n'
            f'(\n'
            f'  way["waterway"="canal"]["name"~"南水北调"]({south},{west},{north},{east});\n'
            f'  relation["waterway"="canal"]["name"~"南水北调"]({south},{west},{north},{east});\n'
            f');\n'
            f'out body geom;'
        )
        logger.info(f"开始请求 Overpass API，bbox={query_bbox}")
        try:
            response = requests.post(OVERPASS_URL, data={"data": overpass_query}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            logger.error(f"Overpass 请求超时: {exc}")
            raise
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            logger.error(f"Overpass HTTP 错误，状态码={status}: {exc}")
            raise
        except requests.exceptions.RequestException as exc:
            logger.error(f"Overpass 请求失败: {exc}")
            raise
        try:
            data = response.json()
        except ValueError as exc:
            logger.error(f"Overpass 返回非 JSON 响应: {exc}")
            raise
        elements = data.get("elements", [])
        logger.debug(f"Overpass 返回 elements 数量: {len(elements)}")
        if not elements:
            logger.warning("未查询到任何线路元素，返回空 GeoDataFrame")
            return RouteDataLoader._empty_gdf()
        records: list[dict[str, object]] = []
        for elem in elements:
            elem_type = elem.get("type")
            elem_id = elem.get("id")
            name = (elem.get("tags") or {}).get("name", "")
            if elem_type == "way":
                geometry = elem.get("geometry", [])
                if len(geometry) < 2:
                    continue
                coords = [(pt["lon"], pt["lat"]) for pt in geometry if "lon" in pt and "lat" in pt]
                if len(coords) >= 2:
                    records.append({"geometry": LineString(coords), "name": name, "osm_id": f"way/{elem_id}"})
            elif elem_type == "relation":
                members = elem.get("members", [])
                for member in members:
                    if member.get("type") != "way":
                        continue
                    member_geometry = member.get("geometry", [])
                    if len(member_geometry) < 2:
                        continue
                    member_coords = [(pt["lon"], pt["lat"]) for pt in member_geometry if "lon" in pt and "lat" in pt]
                    if len(member_coords) >= 2:
                        member_id = member.get("ref", "unknown")
                        records.append({
                            "geometry": LineString(member_coords),
                            "name": name,
                            "osm_id": f"relation/{elem_id}/way/{member_id}",
                        })
        if not records:
            logger.warning("查询结果中无可构建的线路几何，返回空 GeoDataFrame")
            return RouteDataLoader._empty_gdf()
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        logger.info(f"OSM 线路加载完成，共 {len(gdf)} 条要素")
        return gdf

    def from_geojson(self, path: str | Path) -> gpd.GeoDataFrame:
        """从本地 GeoJSON 文件加载线路数据."""
        file_path = Path(path)
        logger.info(f"读取 GeoJSON: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"GeoJSON 文件不存在: {file_path}")
        try:
            gdf = gpd.read_file(file_path)
        except OSError as exc:
            logger.error(f"读取 GeoJSON 失败（文件错误）: {file_path}, {exc}")
            raise
        except ValueError as exc:
            logger.error(f"读取 GeoJSON 失败（数据格式错误）: {file_path}, {exc}")
            raise
        logger.info(f"GeoJSON 加载完成，共 {len(gdf)} 条要素")
        return gdf

    def from_shapefile(self, path: str | Path) -> gpd.GeoDataFrame:
        """从本地 Shapefile 加载线路数据."""
        file_path = Path(path)
        logger.info(f"读取 Shapefile: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Shapefile 文件不存在: {file_path}")
        try:
            gdf = gpd.read_file(file_path)
        except OSError as exc:
            logger.error(f"读取 Shapefile 失败（文件错误）: {file_path}, {exc}")
            raise
        except ValueError as exc:
            logger.error(f"读取 Shapefile 失败（数据格式错误）: {file_path}, {exc}")
            raise
        logger.info(f"Shapefile 加载完成，共 {len(gdf)} 条要素")
        return gdf

    def buffer(self, gdf: gpd.GeoDataFrame, width_m: float, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """沿线路生成缓冲区.

        先投影到 UTM 做 buffer，再转回原 CRS。
        """
        if width_m <= 0:
            raise ValueError(f"width_m 必须大于 0，当前値: {width_m}")
        if gdf.empty:
            logger.warning("输入 GeoDataFrame 为空，直接返回拷贝")
            return gdf.copy()
        utm_crs = gdf.estimate_utm_crs()
        if utm_crs is None:
            raise ValueError("无法估算 UTM CRS，无法进行缓冲区计算")
        logger.info(f"开始生成缓冲区，宽度={width_m}m，UTM={utm_crs}")
        projected = gdf.to_crs(utm_crs)
        result = projected.copy()
        result["geometry"] = projected.geometry.buffer(width_m)
        result = result.to_crs(crs)
        logger.info(f"缓冲区生成完成，共 {len(result)} 条要素")
        return result

    def split_segments(self, gdf: gpd.GeoDataFrame, length_m: float) -> list[gpd.GeoDataFrame]:
        """将线路按指定长度(米)分段.

        先投影到 UTM，按 length_m 切分 LineString，返回分段列表。
        """
        if length_m <= 0:
            raise ValueError(f"length_m 必须大于 0，当前値: {length_m}")
        if gdf.empty:
            logger.warning("输入 GeoDataFrame 为空，返回空列表")
            return []
        if gdf.crs is None:
            raise ValueError("输入 GeoDataFrame 缺少 CRS，无法执行分段")
        utm_crs = gdf.estimate_utm_crs()
        if utm_crs is None:
            raise ValueError("无法估算 UTM CRS，无法执行分段")
        logger.info(f"开始分段，目标长度={length_m}m，UTM={utm_crs}")
        projected = gdf.to_crs(utm_crs)
        output_segments: list[gpd.GeoDataFrame] = []
        for idx, row in projected.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                logger.debug(f"跳过空几何，索引={idx}")
                continue
            lines: list[LineString] = []
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = [line for line in geom.geoms if isinstance(line, LineString)]
            else:
                logger.debug(f"跳过非线几何，索引={idx}，类型={geom.geom_type}")
                continue
            for line in lines:
                start = 0.0
                total_len = float(line.length)
                while start < total_len:
                    end = min(start + length_m, total_len)
                    seg_geom = substring(line, start, end)
                    if seg_geom.is_empty:
                        start = end
                        continue
                    seg_row = row.copy()
                    seg_row.geometry = seg_geom
                    seg_gdf = gpd.GeoDataFrame([seg_row], geometry="geometry", crs=utm_crs).to_crs(gdf.crs)
                    output_segments.append(seg_gdf)
                    start = end
        logger.info(f"分段完成，共生成 {len(output_segments)} 段")
        return output_segments

    def save(self, gdf: gpd.GeoDataFrame, path: str | Path, driver: str = "GeoJSON") -> None:
        """保存线路数据到文件."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"保存线路数据: {file_path} (driver={driver})")
        try:
            gdf.to_file(file_path, driver=driver)
        except OSError as exc:
            logger.error(f"保存失败（文件系统错误）: {file_path}, {exc}")
            raise
        except ValueError as exc:
            logger.error(f"保存失败（参数或数据错误）: {file_path}, {exc}")
            raise
        logger.info(f"保存完成: {file_path}")

    def get_route_info(self, gdf: gpd.GeoDataFrame) -> dict[str, object]:
        """获取线路基本信息：总长度(km)、起点、终点、段数等."""
        if gdf.empty:
            logger.warning("输入 GeoDataFrame 为空，返回空信息")
            return {
                "total_length_km": 0.0,
                "num_segments": 0,
                "start_point": None,
                "end_point": None,
                "crs": str(gdf.crs) if gdf.crs else None,
            }
        if gdf.crs is None:
            raise ValueError("输入 GeoDataFrame 缺少 CRS，无法计算线路信息")
        utm_crs = gdf.estimate_utm_crs()
        if utm_crs is None:
            raise ValueError("无法估算 UTM CRS，无法计算线路长度")
        projected = gdf.to_crs(utm_crs)
        total_length_m = float(projected.geometry.length.sum())
        total_length_km = total_length_m / 1000.0
        first_geom = gdf.geometry.iloc[0]
        last_geom = gdf.geometry.iloc[-1]

        def _start_point(geom: object) -> tuple[float, float] | None:
            if isinstance(geom, LineString) and len(geom.coords) > 0:
                x, y = geom.coords[0]
                return float(x), float(y)
            if isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
                first_line = geom.geoms[0]
                if len(first_line.coords) > 0:
                    x, y = first_line.coords[0]
                    return float(x), float(y)
            return None

        def _end_point(geom: object) -> tuple[float, float] | None:
            if isinstance(geom, LineString) and len(geom.coords) > 0:
                x, y = geom.coords[-1]
                return float(x), float(y)
            if isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
                last_line = geom.geoms[-1]
                if len(last_line.coords) > 0:
                    x, y = last_line.coords[-1]
                    return float(x), float(y)
            return None

        num_segments = 0
        for geom in gdf.geometry:
            if isinstance(geom, LineString):
                num_segments += 1
            elif isinstance(geom, MultiLineString):
                num_segments += len(geom.geoms)
        info: dict[str, object] = {
            "total_length_km": total_length_km,
            "num_segments": num_segments,
            "start_point": _start_point(first_geom),
            "end_point": _end_point(last_geom),
            "crs": str(gdf.crs),
        }
        k_len, k_seg, k_s, k_e = "total_length_km", "num_segments", "start_point", "end_point"
        logger.info(f"线路信息计算完成: 总长度={info[k_len]:.3f}km, 段数={info[k_seg]}, 起点={info[k_s]}, 终点={info[k_e]}")
        return info


def load_route(source: str = "osm", path: str | Path | None = None, **kwargs: object) -> gpd.GeoDataFrame:
    """便捷函数，根据 source 参数自动选择加载方式.

    Args:
        source: 数据来源，"osm"、"geojson" 或 "shapefile"
        path: 本地文件路径（source 为 geojson/shapefile 时必须提供）
        **kwargs: 传递给对应加载方法的额外参数

    Returns:
        线路 GeoDataFrame
    """
    loader = RouteDataLoader()
    source_lower = source.lower().strip()
    if source_lower == "osm":
        return loader.from_osm(**kwargs)
    if source_lower == "geojson":
        if path is None:
            raise ValueError("数据来源为 geojson 时必须提供 path")
        return loader.from_geojson(path)
    if source_lower == "shapefile":
        if path is None:
            raise ValueError("数据来源为 shapefile 时必须提供 path")
        return loader.from_shapefile(path)
    raise ValueError(f"不支持的数据来源: {source!r}，可选: osm, geojson, shapefile")

