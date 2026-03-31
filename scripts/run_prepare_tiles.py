#!/usr/bin/env python3
"""将已有 tif 影像切成训练小图块。"""

from __future__ import annotations

import argparse
from pathlib import Path

from seehydro.preprocessing.tiling import TileGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="批量切片已有 GeoTIFF")
    parser.add_argument("--input-dir", default="data/sentinel2", help="输入 tif 目录")
    parser.add_argument("--output-dir", default="data/tiles/sentinel2", help="切片输出目录")
    parser.add_argument("--tile-size", type=int, default=512, help="切片大小")
    parser.add_argument("--overlap", type=float, default=0.25, help="重叠率")
    parser.add_argument("--min-valid-ratio", type=float, default=0.5, help="最小有效像素比例")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(input_dir.glob("*.tif"))
    if not tif_files:
        print(f"没找到 tif 文件: {input_dir}")
        return

    generator = TileGenerator(tile_size=args.tile_size, overlap=args.overlap)
    total_tiles = 0

    for tif in tif_files:
        print(f"正在切片: {tif}")
        tile_infos = generator.generate_tiles(
            image_path=tif,
            output_dir=output_dir,
            prefix=tif.stem,
            min_valid_ratio=args.min_valid_ratio,
        )
        print(f"完成: {len(tile_infos)} 张")
        total_tiles += len(tile_infos)

    print(f"全部完成，总切片数: {total_tiles}")


if __name__ == "__main__":
    main()
