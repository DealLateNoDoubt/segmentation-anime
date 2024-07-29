# segmentation-anime

![5_add_mask](https://github.com/user-attachments/assets/4b363d9b-0cc9-4ecb-9fd7-e46b133efc24)
基于轻量化改进Yolov8-seg + Sam 实现动漫人物分割

## Yolov8-Improve
1、 bockbone-> StarNet: 主干调整为CVPR2024-微软StarNet轻量级主干;
2、基于StarNet中StarBlock结构，改进C2f：将C2f中Bottleneck卷积替换为StarBlock；
3、基于 [DuAT](https://github.com/Barrett-python/DuAT)引入GLSA聚合模块（能够聚合和表示全局和局部空间特征，这分别有利于定位大物体和小物体）；
4、引入BiFPN-Neck结构，使用双通道金字塔加强特征；
5、引入EMA注意力机制，加强感受野注意力；
