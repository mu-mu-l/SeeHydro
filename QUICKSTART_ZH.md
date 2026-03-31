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

## 8. 下载后如果你自己继续做

下载成功后，不等于已经能直接训练。通常还要继续做下面几步。

### 第一步：确认下载结果

先确认目录里确实有 `.tif` 文件，例如：

```bash
data/cache/tiles_demo
data/sentinel2
```

如果连 `.tif` 都没有，先不要往下走，先把下载问题解决。

### 第二步：切片

训练一般不会直接拿整幅大图去跑，而是要切成很多小图块。

如果你的输入目录里已经有 `.tif`，可以直接运行：

```bash
python3 scripts/run_prepare_tiles.py \
  --input-dir data/cache/tiles_demo \
  --output-dir data/tiles/demo
```

或者：

```bash
python3 scripts/run_prepare_tiles.py \
  --input-dir data/sentinel2 \
  --output-dir data/tiles/sentinel2
```

输出目录里会得到很多切片图，后面标注和训练主要用这些切片。

### 第三步：自己做标注

当前仓库不会在下载后自动给你训练标签。

也就是说：

- 下载影像后，标签通常还是要你自己做
- 如果你做分割任务，要自己做掩膜标注
- 如果你做检测任务，要自己做框标注

建议新手先做分割，不要先做检测。

最简单建议：

1. 从切片目录中挑 30 到 100 张图
2. 先只标最关键类别，例如水面和背景
3. 把图片和标签按同名方式整理好

可以先按这种目录整理：

```bash
data/train_seg/images
data/train_seg/masks
```

例如：

```bash
data/train_seg/images/tile_001.tif
data/train_seg/masks/tile_001.png
```

要求是图片和标签文件名一一对应。

### 第四步：准备训练

你至少需要准备：

- 切片图像
- 对应标签
- 明确的类别定义

对这个项目来说，当前更现实的理解是：

- 下载：脚本已经帮你处理
- 切片：脚本已经提供
- 标注：主要还是你自己做
- 训练：你自己跑

所以这个仓库更像“研究框架”，不是“一键自动出结果”的成熟软件。

### 第五步：先做最小闭环，不要一下子做太大

推荐顺序：

1. 先下载一小块图
2. 先切片
3. 先标几十张
4. 先训一个最小模型
5. 看推理结果能不能出东西

不要一开始就：

- 下特别大范围
- 用特别高的 zoom
- 期望下载完就自动训练
- 期望没有标签也能直接高质量识别

### 第六步：一句话理解你后面要做什么

如果你自己继续做，最现实的顺序就是：

```text
下载 -> 切片 -> 标注 -> 训练 -> 推理 -> 提参数
```

## 9. 一句话结论

先把最小下载跑通，再往后走。当前最省事的入口就是：

```bash
python3 scripts/download_tiles.py
```
