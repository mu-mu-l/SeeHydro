"""PyTorch Dataset 用于分割训练."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from loguru import logger


class SegmentationDataset(Dataset):
    """遥感影像分割数据集.

    目录结构:
        images/     -- 影像切片 (GeoTIFF)
        masks/      -- 分割标注 (GeoTIFF, 单通道, 像素值=类别ID)
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform=None,
        in_channels: int = 3,
        normalize: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.in_channels = in_channels
        self.normalize = normalize

        self.image_files = sorted(self.image_dir.glob("*.tif"))
        self.mask_files = sorted(self.mask_dir.glob("*.tif"))

        # 确保影像和掩膜一一对应
        image_stems = {f.stem for f in self.image_files}
        mask_stems = {f.stem for f in self.mask_files}
        common = image_stems & mask_stems

        self.image_files = sorted([f for f in self.image_files if f.stem in common])
        self.mask_files = sorted([f for f in self.mask_files if f.stem in common])

        logger.info(f"数据集加载: {len(self.image_files)} 对影像-掩膜")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 读取影像
        with rasterio.open(self.image_files[idx]) as src:
            image = src.read().astype(np.float32)  # (C, H, W)

        # 读取掩膜
        with rasterio.open(self.mask_files[idx]) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # 通道选择
        if image.shape[0] > self.in_channels:
            image = image[: self.in_channels]

        # 归一化到 [0, 1]
        if self.normalize:
            for c in range(image.shape[0]):
                p_low, p_high = np.percentile(image[c], [2, 98])
                if p_high > p_low:
                    image[c] = np.clip((image[c] - p_low) / (p_high - p_low), 0, 1)
                else:
                    image[c] = 0

        # albumentations 数据增强
        if self.transform:
            # albumentations 期望 HxWxC
            image_hwc = image.transpose(1, 2, 0)
            augmented = self.transform(image=image_hwc, mask=mask)
            image = augmented["image"].transpose(2, 0, 1)  # 转回 CxHxW
            mask = augmented["mask"]

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
        }


class DetectionDataset:
    """检测数据集管理（YOLO格式）.

    YOLO格式目录:
        images/train/    -- 训练影像
        images/val/      -- 验证影像
        labels/train/    -- 训练标注 (txt, class x_center y_center w h)
        labels/val/      -- 验证标注
        data.yaml        -- 数据集配置
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def create_data_yaml(
        self,
        class_names: list[str],
        train_dir: str = "images/train",
        val_dir: str = "images/val",
    ) -> Path:
        """生成YOLO数据集配置文件."""
        yaml_content = f"""path: {self.data_dir.resolve()}
train: {train_dir}
val: {val_dir}

nc: {len(class_names)}
names: {class_names}
"""
        yaml_path = self.data_dir / "data.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        logger.info(f"创建数据集配置: {yaml_path}")
        return yaml_path

    def get_stats(self) -> dict:
        """统计数据集信息."""
        stats = {}
        for split in ["train", "val"]:
            img_dir = self.data_dir / "images" / split
            lbl_dir = self.data_dir / "labels" / split
            if img_dir.exists():
                stats[split] = {
                    "images": len(list(img_dir.glob("*.*"))),
                    "labels": len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0,
                }
        return stats
