"""目标检测模型封装（YOLOv8）."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger


# 检测类别定义
DET_CLASSES = {
    0: "highway_bridge",   # 公路桥
    1: "railway_bridge",   # 铁路桥
    2: "siphon_inlet",     # 倒虹吸入口
    3: "siphon_outlet",    # 倒虹吸出口
    4: "aqueduct",         # 渡槽
    5: "check_gate",       # 节制闸
    6: "drain_gate",       # 退水闸
    7: "diversion",        # 分水口
}

DET_CLASSES_CN = {
    0: "公路桥",
    1: "铁路桥",
    2: "倒虹吸入口",
    3: "倒虹吸出口",
    4: "渡槽",
    5: "节制闸",
    6: "退水闸",
    7: "分水口",
}


class DetectionModel:
    """YOLOv8目标检测模型封装."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_name: str = "yolov8m.pt",
        device: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """初始化检测模型.

        Args:
            model_path: 训练好的模型权重路径，None则用预训练
            model_name: 预训练模型名称
            device: 推理设备
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        from ultralytics import YOLO

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if model_path and Path(model_path).exists():
            self.model = YOLO(str(model_path))
            logger.info(f"加载检测模型: {model_path}")
        else:
            self.model = YOLO(model_name)
            logger.info(f"加载预训练模型: {model_name}")

    def predict(
        self,
        image: np.ndarray | str | Path,
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[dict]:
        """对单张图像进行目标检测.

        Args:
            image: numpy数组(HxWxC, BGR)或图像路径
            conf: 置信度阈值，None用默认
            iou: IoU阈值，None用默认

        Returns:
            检测结果列表，每个dict包含:
            - bbox: [x1, y1, x2, y2] 像素坐标
            - confidence: 置信度
            - class_id: 类别ID
            - class_name: 类别名称
        """
        results = self.model.predict(
            source=image if isinstance(image, (str, Path)) else image,
            conf=conf or self.conf_threshold,
            iou=iou or self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "confidence": float(boxes.conf[i].cpu()),
                    "class_id": int(boxes.cls[i].cpu()),
                    "class_name": DET_CLASSES.get(int(boxes.cls[i].cpu()), "unknown"),
                }
                detections.append(det)

        return detections

    def predict_batch(
        self,
        images: list[np.ndarray | str | Path],
        conf: float | None = None,
    ) -> list[list[dict]]:
        """批量检测."""
        results = self.model.predict(
            source=images,
            conf=conf or self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        all_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    det = {
                        "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                        "confidence": float(boxes.conf[i].cpu()),
                        "class_id": int(boxes.cls[i].cpu()),
                        "class_name": DET_CLASSES.get(int(boxes.cls[i].cpu()), "unknown"),
                    }
                    detections.append(det)
            all_detections.append(detections)

        return all_detections

    def train(
        self,
        data_yaml: str | Path,
        epochs: int = 200,
        imgsz: int = 1024,
        batch: int = 4,
        project: str = "models/trained/detection",
        name: str = "run",
        **kwargs,
    ) -> Path:
        """训练检测模型.

        Args:
            data_yaml: YOLO格式数据集配置文件
            epochs: 训练轮数
            imgsz: 输入尺寸
            batch: 批大小
            project: 输出目录
            name: 实验名称

        Returns:
            最佳模型权重路径
        """
        logger.info(f"开始训练检测模型: epochs={epochs}, imgsz={imgsz}, batch={batch}")
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=self.device,
            **kwargs,
        )
        best_path = Path(project) / name / "weights" / "best.pt"
        logger.info(f"训练完成，最佳模型: {best_path}")
        return best_path
