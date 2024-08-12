import torch
from ultralytics import YOLO

if __name__ == "__main__":
    device = torch.device("cuda")

    model = YOLO("./runs/m-starnet-glsa-bifpn-ep100-1/weights/best.pt")
    model.export(format="onnx")


