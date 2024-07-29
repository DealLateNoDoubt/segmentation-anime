import sys
sys.path.append('..')  # 添加上级父目录到搜索路径

import os
import cv2
import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sam.segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def pipeline(seg_model, yolo_model, image_path, threshold=0.9):
    image = Image.open(image_path).convert("RGB")
    yolo_masks = yolo_mask_generate(yolo_model, image, half=False)
    if yolo_masks is None:
        return None, None, None, None
    sam_masks = sam_mask_generate(seg_model, image)
    voted_mask = vote_mask_generate(sam_masks, yolo_masks, threshold=threshold)
    image = np.array(image.convert("RGBA"))
    voted_mask = np.array(voted_mask)
    # 把mask给resize到和image一样的大小
    voted_mask = cv2.resize(voted_mask, (image.shape[1], image.shape[0])).astype(np.uint8)
    # mask大于0的地方，image的alpha通道为255，否则为0
    image[:, :, 3] = voted_mask * 255
    out_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return voted_mask, out_image, sam_masks, yolo_masks


def yolo_mask_generate(yolo_model, image, half=False):
    # 生成mask
    masks = yolo_model.predict(
        source=image,
        save=False,
        conf=0.8,   # 检测的最低置信度阈值
        iou=0.8,
        half=half,
        retina_masks=True,
    )
    if masks[0].masks is None:
        return None
    masks = masks[0].masks.data.data.cpu().numpy().sum(axis=0)
    return masks

def sam_mask_generate(seg_model, image):
    if isinstance(image, str):
        image = cv2.imread(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    original_image_size = image.shape[:2]
    # 如果图片太大，就resize到1920以下，最长边为1920
    if max(original_image_size) > 1920:
        scale = 1920 / max(original_image_size)
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    masks = seg_model.generate(image)       # 生成mask
    return_masks = []
    for i, mask in enumerate(masks):
        return_masks.append(mask['segmentation'].astype(np.uint8))  # 把masks给resize到原图大小
    return np.array(return_masks)

def vote_mask_generate(SAM_masks, YOLO_masks, threshold=0.9, erode_iter=3, erode_kernel_size=3):
    scores = []
    #把yolo_masks给resize到和SAM_masks一样的大小
    zeros = np.zeros_like(SAM_masks[0])
    #zeros与所有的SAM_masks进行或运算，得到一个和SAM_masks一样大小的mask 然后取反，得到所有的未标注区域，然后与YOLO_masks相与，得到所有的未标注区域的YOLO_masks
    full_mask = np.zeros_like(SAM_masks[0])
    for i, mask in enumerate(SAM_masks):
        full_mask = cv2.bitwise_or(full_mask, mask)

    # #对YOLO_masks先进行一些腐蚀再膨胀
    kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    YOLO_masks = cv2.erode(YOLO_masks, kernel, iterations=erode_iter)   # 腐蚀
    if erode_iter > 1:
        YOLO_masks = cv2.dilate(YOLO_masks, kernel, iterations=erode_iter-1)    # 膨胀
    YOLO_masks = cv2.resize(YOLO_masks, (SAM_masks[0].shape[1], SAM_masks[0].shape[0])).astype(np.uint8)

    not_labeled = cv2.bitwise_not(cv2.bitwise_or(zeros, full_mask))

    final_mask = np.zeros_like(SAM_masks[0])
    for i, mask in enumerate(SAM_masks):
        score = np.sum(YOLO_masks * mask / mask.sum())
        scores.append(score)
        if score > threshold:
            final_mask = cv2.bitwise_or(final_mask, mask)    # 或运算
    not_labeled_yolo = cv2.bitwise_and(not_labeled, YOLO_masks)

    final_mask = cv2.bitwise_or(final_mask, not_labeled_yolo)
    final_mask = cv2.erode(final_mask, kernel, iterations=erode_iter)
    final_mask = cv2.dilate(final_mask, kernel, iterations=erode_iter)
    return final_mask


def predict(parse):
    source = parse.source

    # 加载sam
    sam = sam_model_registry["vit_h"](checkpoint="../sam/weight/sam_vit_h_4b8939.pth")
    sam = sam.cuda()
    sam_model_generator = SamAutomaticMaskGenerator(
        sam,
        crop_n_layers=1,
    )
    #  加载Yolo
    yolo_model = YOLO("./runs/m-starnet-glsa-bifpn-ep100-1/weights/best.pt")

    if os.path.isdir(source):
        for file in os.listdir(source):
            detects(os.path.join(source, file), yolo_model, sam_model_generator, parse)
    else:
        detects(source, yolo_model, sam_model_generator, parse)

def detects(image_path, yolo_model, sam_model_generator, parse):
    save_path = parse.save_path
    show_mask, show_sam, show_yolo, show_seg = parse.show_mask, parse.show_sam, parse.show_yolo, parse.show_seg

    output_masks, output_images, sam_masks, yolo_masks = pipeline(
        sam_model_generator,
        yolo_model,
        image_path,
        threshold=0.8,
    )
    if output_images is None or output_masks is None or sam_masks is None or yolo_masks is None:
        print("No object detected!")
        return

    image_add_mask = source_add_mask(image_path, output_masks)
    cv2.imwrite(os.path.join(save_path, image_path.split(os.sep)[-1].split(".")[0] + '_add_mask.png'), image_add_mask)
    if show_seg:
        cv2.imwrite(os.path.join(save_path, image_path.split(os.sep)[-1].split(".")[0] + '_seg.png'), output_images)
    if show_mask:
        cv2.imwrite(os.path.join(save_path, image_path.split(os.sep)[-1].split(".")[0] + '_mask.png'), output_masks * 255)
    if show_sam:
        cv2.imwrite(os.path.join(save_path, image_path.split(os.sep)[-1].split(".")[0] + '_sam.png'), plot_sam_mask(sam_masks))
    if show_yolo:
        cv2.imwrite(os.path.join(save_path, image_path.split(os.sep)[-1].split(".")[0] + '_yolo.png'), cv2.resize(yolo_masks * 255, (output_images.shape[1], output_images.shape[0])))


def source_add_mask(source, mask):
    original_image = cv2.imread(source)

    # 确保原图是三通道的
    if original_image.shape[2] == 4:  # 如果原图是四通道（例如带有alpha通道）
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)

    # 创建一个与原图大小相同的浅色图像
    color_mask = np.zeros_like(original_image)
    color_mask[mask > 0] = [128, 0, 128]  # 浅色

    return cv2.addWeighted(original_image, 0.7, color_mask, 1, 0, dtype=cv2.CV_32F)


def plot_sam_mask(masks):
    color = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)
    background = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        # 产生一个三通道的随机颜色
        temp = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
        temp = temp * color[i]
        background = cv2.bitwise_or(background, temp)
    return background


def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./test/images/5.jpg")
    parser.add_argument("--save_path", default="./test/save")
    parser.add_argument("--show_mask", default=True)
    parser.add_argument("--show_sam", default=True)
    parser.add_argument("--show_yolo", default=True)
    parser.add_argument("--show_seg", default=True)

    predict(parser.parse_args())