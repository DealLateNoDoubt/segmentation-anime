"""
Wandb有问题，会导致TypeError：model='yolov8l.yaml' 不是受支持的模型格式。
"""
import os
import wandb
from datetime import datetime
from wandb.integration.ultralytics import add_wandb_callback

os.environ["WANDB_API_KEY"] = ''    # Your Wandb-Api

def WandbInit():
    now = datetime.now()
    formatted_time = now.strftime("%d/%m/%Y %H:%M")
    wandb.init(
        project="yolov8-anime-seg",
        name=formatted_time,
        id=wandb.util.generate_id(),
    )

def WandbFinish():
    wandb.finish()


def AddWandbCallback(model):
    add_wandb_callback(model, enable_model_checkpointing=True)