import sys
sys.path.append('..')  # 添加上级父目录到搜索路径

import cv2
import argparse
import numpy as np
from onnx_detects import YoloSegPredict
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_image(image):
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict(parse):
    source = parse.source

    # 加载onnx
    onnx = YoloSegPredict("runs/m-starnet-glsa-bifpn-ep100-1/weights/best.onnx")

    # 加载sam2.0
    model_cfg = "sam2_hiera_l.yaml"
    sam2_checkpoint = "../sam2/checkpoints/sam2_hiera_large.pt"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
    sam2 = sam2.cuda()
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
    )

    detects(source, onnx, mask_generator)

def detects(source, onnx, sam2, threshold=0.7):
    image = cv2.imread(source)

    kernel = np.zeros((3, 3), np.uint8)
    iterations = 3

    _, _, onnx_masks = onnx(image)
    onnx_masks = onnx_masks[0]
    onnx_masks = np.where(onnx_masks > 1, 255, 0).astype(np.uint8)
    onnx_masks = cv2.erode(onnx_masks, kernel, iterations=iterations)   # 腐蚀
    onnx_masks = cv2.dilate(onnx_masks, kernel, iterations=iterations-1)    # 膨胀
    onnx_masks = onnx_masks.astype(np.uint8)
    if len(onnx_masks) <= 0:
        print("no-onnx")
        return
    print("onnx-done")

    sam2_masks = []
    anns = sam2.generate(image)
    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for i, ann in enumerate(anns):
        m = ann['segmentation'].astype(np.uint8)
        sam2_masks.append(m)
        # temp_img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
        # temp_img[m] = [255, 255, 255, 255]
        # img[m] = [255, 255, 255, 255]
        # cv2.imwrite(f"./test/{i}.jpg", temp_img)
    # cv2.imwrite(f"./test/test.jpg", img)

    sam2_masks = np.array(sam2_masks)
    if len(sam2_masks) <= 0:
        print("no-sam2")
        return
    print("sam2-done")

    zeros = np.zeros_like(sam2_masks[0])
    full_mask = np.zeros_like(sam2_masks[0])
    for i, mask in enumerate(sam2_masks):
        full_mask = cv2.bitwise_or(full_mask, mask)

    not_labeled = cv2.bitwise_not(cv2.bitwise_or(zeros, full_mask))

    final_mask = np.zeros_like(sam2_masks[0])
    for i, mask in enumerate(sam2_masks):
        score = np.sum(onnx_masks * mask / mask.sum())
        if (score / 255.0) > threshold:
            final_mask = cv2.bitwise_or(final_mask, mask)  # 或运算

    not_labeled_onnx = cv2.bitwise_and(not_labeled, onnx_masks)
    final_mask = cv2.bitwise_or(final_mask, not_labeled_onnx)
    final_mask = cv2.erode(final_mask, kernel, iterations=iterations)
    final_mask = cv2.dilate(final_mask, kernel, iterations=iterations)
    final_mask = np.array(final_mask)

    image_add_mask = source_add_mask(source, final_mask)
    cv2.imwrite("./test/image_add_mask.png", image_add_mask)

def source_add_mask(source, mask):
    original_image = cv2.imread(source)

    # 确保原图是三通道的
    if original_image.shape[2] == 4:  # 如果原图是四通道（例如带有alpha通道）
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)

    # 创建一个与原图大小相同的浅色图像
    color_mask = np.zeros_like(original_image)
    color_mask[mask > 0] = [128, 0, 128]  # 浅色

    return cv2.addWeighted(original_image, 0.7, color_mask, 1, 0, dtype=cv2.CV_32F)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./test/images/5.jpg")
    parser.add_argument("--save_path", default="./test/save")
    parser.add_argument("--show_mask", default=True)
    parser.add_argument("--show_sam", default=True)
    parser.add_argument("--show_yolo", default=True)
    parser.add_argument("--show_seg", default=True)

    predict(parser.parse_args())