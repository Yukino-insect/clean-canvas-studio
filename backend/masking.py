"""OCR 多边形、用户 ROI 与最终修复掩膜之间的明确数据模型。

PaddleOCR 的 ``dt_polys`` 是文字区域外轮廓。它比外接矩形精确，但并非字符
笔画级语义分割；本模块因此把“候选 ROI”和“最终掩膜”严格拆开。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import cv2
import numpy as np


Rectangle = tuple[int, int, int, int]  # xmin, xmax, ymin, ymax
Point = tuple[int, int]


class MaskMode(str, Enum):
    RECTANGLE = "rectangle"
    OCR_POLYGON = "ocr_polygon"
    OCR_REFINED = "ocr_refined"
    MIXED = "mixed"


class VideoMaskStrategy(str, Enum):
    STATIC = "static"
    SAMPLED_OCR = "sampled_ocr"
    EVERY_FRAME_OCR = "every_frame_ocr"


@dataclass(frozen=True)
class OcrPolygon:
    points: tuple[Point, ...]
    score: float = 1.0

    @property
    def bbox(self) -> Rectangle:
        xs, ys = zip(*self.points)
        return min(xs), max(xs), min(ys), max(ys)

    def to_dict(self) -> dict[str, Any]:
        return {"points": [[x, y] for x, y in self.points], "score": self.score}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OcrPolygon":
        points = tuple((int(point[0]), int(point[1])) for point in value["points"])
        if len(points) < 3:
            raise ValueError("OCR 多边形至少需要三个顶点。")
        return cls(points=points, score=float(value.get("score", 1.0)))


@dataclass
class MaskPlan:
    """确认后冻结并传入子进程的最终掩膜计划。"""

    mode: MaskMode = MaskMode.OCR_POLYGON
    ocr_polygons: list[OcrPolygon] = field(default_factory=list)
    candidate_regions: list[Rectangle] = field(default_factory=list)
    manual_regions: list[Rectangle] = field(default_factory=list)
    expand_pixels: int = 2
    refine_color_threshold: int = 22
    refine_min_component_area: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "ocr_polygons": [polygon.to_dict() for polygon in self.ocr_polygons],
            "candidate_regions": [list(region) for region in self.candidate_regions],
            "manual_regions": [list(region) for region in self.manual_regions],
            "expand_pixels": self.expand_pixels,
            "refine_color_threshold": self.refine_color_threshold,
            "refine_min_component_area": self.refine_min_component_area,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MaskPlan":
        return cls(
            mode=MaskMode(value.get("mode", MaskMode.OCR_POLYGON.value)),
            ocr_polygons=[OcrPolygon.from_dict(item) for item in value.get("ocr_polygons", [])],
            candidate_regions=[tuple(map(int, item)) for item in value.get("candidate_regions", [])],
            manual_regions=[tuple(map(int, item)) for item in value.get("manual_regions", [])],
            expand_pixels=max(0, int(value.get("expand_pixels", 2))),
            refine_color_threshold=max(1, int(value.get("refine_color_threshold", 22))),
            refine_min_component_area=max(1, int(value.get("refine_min_component_area", 8))),
        )

    def with_polygons(self, polygons: Iterable[OcrPolygon]) -> "MaskPlan":
        return MaskPlan(
            mode=self.mode,
            ocr_polygons=list(polygons),
            candidate_regions=list(self.candidate_regions),
            manual_regions=list(self.manual_regions),
            expand_pixels=self.expand_pixels,
            refine_color_threshold=self.refine_color_threshold,
            refine_min_component_area=self.refine_min_component_area,
        )


def _normalized_polygon(polygon: OcrPolygon, width: int, height: int) -> np.ndarray | None:
    points = np.asarray(polygon.points, dtype=np.int32)
    if len(points) < 3:
        return None
    points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
    if cv2.contourArea(points) < 1:
        return None
    return points


def polygon_mask(image_shape: tuple[int, ...], polygons: Iterable[OcrPolygon], expand_pixels: int = 0) -> np.ndarray:
    """只填充 OCR 多边形本身，而非其外接矩形。"""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    normalized = [_normalized_polygon(polygon, width, height) for polygon in polygons]
    normalized = [points for points in normalized if points is not None]
    if normalized:
        cv2.fillPoly(mask, normalized, 255)
    if expand_pixels > 0 and np.any(mask):
        kernel_size = expand_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    return mask


def rectangle_mask(image_shape: tuple[int, ...], regions: Iterable[Rectangle], expand_pixels: int = 0) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for xmin, xmax, ymin, ymax in regions:
        raw_left, raw_right = sorted((int(xmin), int(xmax)))
        raw_top, raw_bottom = sorted((int(ymin), int(ymax)))
        left, right = max(0, raw_left), min(width, raw_right)
        top, bottom = max(0, raw_top), min(height, raw_bottom)
        if right > left and bottom > top:
            cv2.rectangle(mask, (left, top), (right - 1, bottom - 1), 255, thickness=-1)
    if expand_pixels > 0 and np.any(mask):
        kernel_size = expand_pixels * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    return mask


def filter_polygons_by_regions(polygons: Iterable[OcrPolygon], regions: Iterable[Rectangle]) -> list[OcrPolygon]:
    """ROI 只作为候选筛选器：中心点或任一顶点落入 ROI 才保留多边形。"""
    candidates = list(regions)
    if not candidates:
        return list(polygons)
    kept: list[OcrPolygon] = []
    for polygon in polygons:
        xs, ys = zip(*polygon.points)
        center = (sum(xs) / len(xs), sum(ys) / len(ys))
        for xmin, xmax, ymin, ymax in candidates:
            def inside(point: tuple[float, float]) -> bool:
                return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax
            if inside(center) or any(inside(point) for point in polygon.points):
                kept.append(polygon)
                break
    return kept


def refine_polygon_mask(
    image_bgr: np.ndarray,
    base_mask: np.ndarray,
    *,
    color_threshold: int = 22,
    min_component_area: int = 8,
) -> np.ndarray:
    """在 OCR 多边形内用局部背景色差细化掩膜。

    这是一种无额外模型的辅助方法：从多边形周围的窄环估计背景 Lab 中位数，
    只保留多边形内与背景差异足够大的连通区域。无法取得足够背景样本时安全
    回退到原始多边形掩膜。
    """
    if not np.any(base_mask):
        return base_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    outer = cv2.dilate(base_mask, kernel)
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(base_mask))
    samples = image_bgr[ring > 0]
    if len(samples) < 24:
        return base_mask.copy()
    lab = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    background = np.median(cv2.cvtColor(samples.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    refined = np.where((base_mask > 0) & (distance >= color_threshold), 255, 0).astype(np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    result = np.zeros_like(refined)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            result[labels == label] = 255
    # 过于严格的阈值不能产生一个空掩膜；这是“细化”的失败，不应变成漏擦。
    return result if np.any(result) else base_mask.copy()


def build_mask(image_bgr: np.ndarray, plan: MaskPlan) -> np.ndarray:
    """根据冻结计划构造最终掩膜，供所有修复引擎使用。"""
    polygon = polygon_mask(image_bgr.shape, plan.ocr_polygons, plan.expand_pixels)
    if plan.mode is MaskMode.RECTANGLE:
        return rectangle_mask(image_bgr.shape, plan.manual_regions, plan.expand_pixels)
    if plan.mode is MaskMode.OCR_POLYGON:
        return polygon
    if plan.mode is MaskMode.OCR_REFINED:
        return refine_polygon_mask(
            image_bgr,
            polygon,
            color_threshold=plan.refine_color_threshold,
            min_component_area=plan.refine_min_component_area,
        )
    manual = rectangle_mask(image_bgr.shape, plan.manual_regions, plan.expand_pixels)
    return cv2.bitwise_or(polygon, manual)
