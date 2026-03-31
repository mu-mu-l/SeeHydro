# SeeHydro 傻瓜式快速开始

这份文档只解决一件事：先把下载和切片跑通。

## 1. 先装最小依赖

如果你只是先测试在线卫星图下载，最少先装这些：

```bash
cd /tmp/SeeHydro
python3 -m pip install requests pillow numpy rasterio
```

如果你后面要跑项目完整流程，再装：

```bash
python3 -m pip install -e .
```

如果 `pip install -e .` 太慢或有依赖冲突，先只跑最小下载脚本即可。

## 2. 先测试在线卫星图下载

最简单命令：

```bash
cd /tmp/SeeHydro
python3 scripts/download_tiles.py
```

默认会下载一个很小的测试范围，并输出到：

```bash
data/cache/tiles_demo
```

如果你要换范围：

```bash
python3 scripts/download_tiles.py \
  --west 111.50 \
  --south 32.67 \
  --east 111.55 \
  --north 32.72 \
  --zoom 15
```

建议一开始只用 `zoom=15` 或 `16`，不要先上 `18` 或 `19`。

## 3. 如果你要天地图

先申请天地图 key，然后运行：

```bash
export TDT_KEY='你的天地图key'
python3 scripts/download_tiles.py --provider tianditu_img
```

也可以不用环境变量，直接传参：

```bash
python3 scripts/download_tiles.py \
  --provider tianditu_img \
  --api-key 你的天地图key
```

注意：不要把你真实的 key 直接写进仓库文件并提交到 GitHub。

## 4. 如果你要 Sentinel-2

这条更正式，但依赖更多，而且第一次通常需要 Earth Engine 授权。

先准备线路文件，例如：

```bash
data/route/snbd_centerline.geojson
```

再运行：

```bash
python3 scripts/download_sentinel2.py --route data/route/snbd_centerline.geojson
```

## 5. 下载完以后切片

如果你目录里已经有 `tif`，运行：

```bash
python3 scripts/run_prepare_tiles.py --input-dir data/sentinel2 --output-dir data/tiles/sentinel2
```

## 6. 你现在最应该怎么做

按这个顺序最稳：

1. 先运行 `python3 scripts/download_tiles.py`
2. 确认能生成 `.tif`
3. 再考虑换更大范围
4. 再考虑 Sentinel-2
5. 再考虑标注和训练

## 7. 你最容易踩的坑

- 一开始就下太大范围，导致又慢又容易失败
- 一开始就用很高 zoom，瓦片数量暴涨
- 还没验证下载成功，就直接研究训练
- 把在线卫星底图和 Sentinel-2 科学数据混为一谈

## 8. 一句话结论

先把最小下载跑通，再往后走。当前最省事的入口就是：

```bash
python3 scripts/download_tiles.py
```
