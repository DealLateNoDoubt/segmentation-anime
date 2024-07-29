import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('..')  # 添加上级父目录到搜索路径

from ultralytics import YOLO
from wandb_settings import *

if __name__ == "__main__":
    WandbInit()

    # Load a model
    model = YOLO("yolov8l-seg-improve.yaml")
    # model = YOLO("yolov8m-seg-improve_2.yaml")
    # model = YOLO("yolov8m-seg.yaml")
    results = model.train(
        data="data-anine.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,  # 默认值8，用于数据加载时的线程数
        optimizer="SGD",
        save=True,
        patience=100,   # 早停等待轮数
        cos_lr=True,    # 余弦学习率调度器,默认为False
        plots=True,
        overlap_mask=True,
        lr0=1e-4,
        lrf=1e-4,
        mixup=0.4,
        copy_paste=0.3,
        resume=False, # 中断训练后，是否恢复从上一次训练(魔改: 是否使用resume_model来进行继续训练)
        # resume_model="../segmentation_scripts/runs/segment/train/weights/last.pt",
    )

    WandbFinish()