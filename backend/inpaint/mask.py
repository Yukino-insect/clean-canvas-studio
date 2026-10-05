"""将手工选区和 OCR 检测框统一成 AI/OpenCV 都可使用的二值掩膜。"""

from __future__ import annotations

import cv2
import numpy as np

Rectangle = tuple[int, int, int, int]  # xmin, xmax, ymin, ymax


def normalize_regions(regions: list[Rectangle], width: int, height: int) -> list[Rectangle]:
    """裁剪、排序并丢弃空区域，避免任何 OCR 坐标越出图像。"""
    normalized: list[Rectangle] = []
    for xmin, xmax, ymin, ymax in regions:
        raw_left, raw_right = sorted((int(xmin), int(xmax)))
        raw_top, raw_bottom = sorted((int(ymin), int(ymax)))
        left, right = max(0, raw_left), min(width, raw_right)
        top, bottom = max(0, raw_top), min(height, raw_bottom)
        if right > left and bottom > top:
            normalized.append((left, right, top, bottom))
    return sorted(set(normalized), key=lambda box: (box[2], box[0], box[3], box[1]))


def regions_to_mask(
    image_shape: tuple[int, int] | tuple[int, int, int],
    regions: list[Rectangle],
    *,
    expand_pixels: int = 0,
    feather_pixels: int = 0,
) -> np.ndarray:
    """将矩形区域合并为 0/255 掩膜。

    ``expand_pixels`` 用于覆盖文字描边和抗锯齿残留；AI 和 OpenCV 都接收
    二值白色掩膜。``feather_pixels`` 仅用于未来扩展，当前保持二值输出，防止
    模型把灰度区域误判为未遮罩。
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for left, right, top, bottom in normalize_regions(regions, width, height):
        cv2.rectangle(mask, (left, top), (right - 1, bottom - 1), 255, thickness=-1)
    if expand_pixels > 0 and np.any(mask):
        kernel_size = expand_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    return mask
