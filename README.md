# segmentation-anime

![5_add_mask](https://github.com/user-attachments/assets/4b363d9b-0cc9-4ecb-9fd7-e46b133efc24)
基于轻量化改进Yolov8-seg + Sam 实现动漫人物分割

## Yolov8-Seg
### Yolov8轻量化改进-ultralytics/cfg/v8/yolov8-seg-improve.yaml
1. bockbone-> StarNet: 主干调整为CVPR2024-微软StarNet轻量级主干;
2. 基于StarNet中StarBlock结构，改进C2f：将C2f中Bottleneck卷积替换为StarBlock；
3. 基于 [DuAT](https://github.com/Barrett-python/DuAT)引入GLSA聚合模块（能够聚合和表示全局和局部空间特征，这分别有利于定位大物体和小物体）；
4. 引入BiFPN-Neck结构，使用双通道金字塔加强特征；
5. 引入EMA注意力机制，加强感受野注意力；
   
改进后，参数从43w有效减少至20w,OPs也有所提升；
YOLOv8l-seg-improve summary: 830 layers, 20007752 parameters, 20007736 gradients, 125.9 GFLOPs

## Sam
### 使用官方[SAM](https://github.com/facebookresearch/segment-anything)代码，以及vit-h权重对图片进行处理；

## 推理过程
### 1-根据已训练好的yolo权重模型，对图片进行推理，获取到分割mask图片：
![_mask](https://github.com/user-attachments/assets/f53c6c31-062d-4835-bfa5-e188a2158329)

### 2-根据Sam-vlt-h，获取到各个分割碎片信息：
![_sam](https://github.com/user-attachments/assets/0b4d75e8-9a72-491e-bf90-ae79dcf257a9)

### 3-基于分割碎片信息与mask图片进行IOU计算，获取在阈值范围内的分割碎片，并通过分割碎片来进行直接分割：
![_seg](https://github.com/user-attachments/assets/00e59f09-1d9a-4e76-bc30-d3608b47bf41)

### 4-根据分割信息，在原图上绘制出来：
![_add_mask](https://github.com/user-attachments/assets/858dc783-9230-4c10-bf22-d6eaee6b9c7a)
