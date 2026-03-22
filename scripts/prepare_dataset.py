#!/usr/bin/env python3
"""数据集准备脚本：线路裁剪 + 切片生成."""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from seehydro.acquisition.route import load_route
from seehydro.preprocessing.clip import batch_clip
from seehydro.preprocessing.tiling import TileGenerator
from seehydro.utils.config import load_config

app = typer.Typer(help="SeeHydro 数据集准备工具")


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", help="配置文件路径")] = Path("configs/default.yaml"),
    input_dir: Annotated[Path, typer.Option("--input-dir", help="输入影像目录")] = Path("data/sentinel2"),
    output_dir: Annotated[Path, typer.Option("--output-dir", help="输出目录")] = Path("data/tiles/sentinel2"),
    route_path: Annotated[Path | None, typer.Option("--route", help="线路数据路径")] = None,
    tile_size: Annotated[int, typer.Option("--tile-size", help="切片大小")] = 512,
    overlap: Annotated[float, typer.Option("--overlap", help="重叠率")] = 0.25,
    buffer_m: Annotated[float, typer.Option("--buffer", help="缓冲宽度(米)")] = 2000,
) -> None:
    """数据集准备流程：加载配置 → 加载线路 → 沿线裁剪 → 生成切片 → 保存索引."""
    # Step 1: 加载配置
    cfg = load_config(config)
    logger.info(f"[1/5] 配置已加载: {config}")

    # Step 2: 加载线路数据
    if route_path and route_path.exists():
        route_gdf = load_route(source="local", path=route_path)
    else:
        default_route = Path(cfg.get("route", {}).get("local_path", "data/route/snbd_centerline.geojson"))
        if default_route.exists():
            route_gdf = load_route(source="local", path=default_route)
        else:
            logger.info("本地线路不存在，尝试从OSM下载...")
            route_gdf = load_route(source="osm")
    logger.info(f"[2/5] 线路数据已加载: {len(route_gdf)} 条记录")

    # Step 3: 沿线裁剪
    clipped_dir = output_dir / "clipped"
    clipped_files = batch_clip(
        raster_dir=input_dir,
        route_gdf=route_gdf,
        buffer_m=buffer_m,
        output_dir=clipped_dir,
    )
    logger.info(f"[3/5] 裁剪完成: {len(clipped_files)} 个文件 → {clipped_dir}")

    # Step 4: 生成切片
    tile_dir = output_dir / "tiles"
    generator = TileGenerator(tile_size=tile_size, overlap=overlap)

    all_tile_infos = []
    for clipped_path in clipped_files:
        tile_infos = generator.generate_tiles(
            image_path=clipped_path,
            output_dir=tile_dir,
            prefix=clipped_path.stem,
        )
        all_tile_infos.extend(tile_infos)
    logger.info(f"[4/5] 切片完成: {len(all_tile_infos)} 个切片 → {tile_dir}")

    # Step 5: 保存索引
    index_path = output_dir / "tile_index.csv"
    generator.save_tile_index(all_tile_infos, index_path)
    logger.info(f"[5/5] 索引已保存: {index_path}")

    typer.echo(f"数据集准备完成！共 {len(all_tile_infos)} 个切片")


if __name__ == "__main__":
    app()
