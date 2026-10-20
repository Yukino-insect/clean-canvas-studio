"""OpenCV 修复流程需要的遮罩工具。"""

import cv2
import numpy as np

from backend.config import config


def create_mask(size: tuple[int, int], coords_list: list[tuple[int, int, int, int]]) -> np.ndarray:
    """按检测框生成二值遮罩，并向四周扩展少量像素以覆盖文字边缘。

    ``size`` 为 ``(高度, 宽度)``，检测框格式为
    ``(xmin, xmax, ymin, ymax)``。
    """
    height, width = size
    mask = np.zeros(size, dtype=np.uint8)
    padding = config.ocrMaskPadding.value
    for xmin, xmax, ymin, ymax in coords_list:
        x1, y1 = max(0, xmin - padding), max(0, ymin - padding)
        x2, y2 = min(width - 1, xmax + padding), min(height - 1, ymax + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def is_frame_number_in_ab_sections(frame_no: int, ab_sections) -> bool:
    """判断 0-based 帧号是否位于用户设置的任一 A/B 处理区间。"""
    return not ab_sections or any(frame_no in section for section in ab_sections)
