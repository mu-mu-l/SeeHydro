"""检测模型训练脚本."""

from pathlib import Path

from loguru import logger

from seehydro.models.det_model import DET_CLASSES, DetectionModel


def train_detection(
    data_yaml: str | Path,
    config: dict,
    output_dir: str | Path = "models/trained/detection",
) -> Path:
    """训练YOLOv8检测模型.

    Args:
        data_yaml: YOLO格式数据集配置文件
        config: 训练配置字典
        output_dir: 输出目录

    Returns:
        最佳模型权重路径
    """
    model_name = config.get("model_name", "yolov8m")
    epochs = config.get("epochs", 200)
    imgsz = config.get("input_size", 1024)
    batch = config.get("batch_size", 4)
    lr = config.get("lr", 1e-3)
    name = config.get("experiment_name", "snbd_det")

    logger.info(f"开始训练检测模型: model={model_name}, epochs={epochs}, imgsz={imgsz}")

    det_model = DetectionModel(model_name=f"{model_name}.pt")
    best_path = det_model.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name=name,
        lr0=lr,
        mosaic=1.0,
        mixup=0.1,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
    )

    return best_path
