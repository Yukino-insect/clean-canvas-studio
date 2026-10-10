"""用户选择的 ROI、最终修复掩膜与动态 OCR 多边形的明确数据模型。

PaddleOCR 的 ``dt_polys`` 是文字区域外轮廓。它比外接矩形精确，但并非字符
笔画级语义分割；因此 OCR 候选与静态最终掩膜严格分属不同的处理路径。
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
    BRUSH = "brush"
    COLOR = "color"


class VideoMaskStrategy(str, Enum):
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
    """确认后冻结并传入子进程的修复区域组合计划。

    ``brush_strokes`` 使用相对源图坐标 ``(x, y)`` 保存，并在每一笔中冻结
    当时的 ``radius``，因此同一计划可以在原始分辨率与预览分辨率不同、或
    用户后来改了笔刷粗细的情况下稳定重建。颜色条件只在
    ``candidate_regions`` 内工作，避免用户误把整张复杂画面交给阈值算法。这里
    的 ``candidate_regions`` 是颜色筛选的手工 ROI，绝不是 OCR 候选结果；OCR
    多边形只在动态 OCR 视频路径中由 ``polygon_mask`` 或
    ``refine_ocr_polygon_mask`` 直接使用。

    ``mode`` 保留为兼容旧任务快照和记录最后一次确认方式的字段；真正的最终
    掩膜由三个来源的并集构成：``manual_regions``、``brush_strokes`` 与
    ``candidate_regions``。因此用户确认矩形后再确认涂鸦，不会把已确认矩形
    悄悄替换掉。所有字段都可序列化，子进程不会读取之后才被用户更改的 GUI
    配置。
    """

    mode: MaskMode = MaskMode.RECTANGLE
    candidate_regions: list[Rectangle] = field(default_factory=list)
    manual_regions: list[Rectangle] = field(default_factory=list)
    brush_strokes: list[dict[str, Any] | list[tuple[float, float]]] = field(default_factory=list)
    brush_radius: int = 12
    color_foreground: str = "#FFFFFF"
    color_background: str = "#000000"
    color_tolerance: int = 30
    # 颜色字形筛出后的小幅膨胀；旧任务不含此字段时保持 0，避免回放结果变化。
    color_outline_expand_pixels: int = 0
    expand_pixels: int = 2

    @property
    def source_modes(self) -> list[MaskMode]:
        """返回计划中实际已确认的来源，顺序用于日志和界面说明。"""
        result: list[MaskMode] = []
        if self.manual_regions:
            result.append(MaskMode.RECTANGLE)
        if self.brush_strokes:
            result.append(MaskMode.BRUSH)
        if self.candidate_regions:
            result.append(MaskMode.COLOR)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "candidate_regions": [list(region) for region in self.candidate_regions],
            "manual_regions": [list(region) for region in self.manual_regions],
            "brush_strokes": _serialize_brush_strokes(self.brush_strokes, self.brush_radius),
            "brush_radius": self.brush_radius,
            "color_foreground": self.color_foreground,
            "color_background": self.color_background,
            "color_tolerance": self.color_tolerance,
            "color_outline_expand_pixels": self.color_outline_expand_pixels,
            "expand_pixels": self.expand_pixels,
            "source_modes": [mode.value for mode in self.source_modes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MaskPlan":
        return cls(
            mode=MaskMode(value.get("mode", MaskMode.RECTANGLE.value)),
            candidate_regions=[tuple(map(int, item)) for item in value.get("candidate_regions", [])],
            manual_regions=[tuple(map(int, item)) for item in value.get("manual_regions", [])],
            brush_radius=max(1, int(value.get("brush_radius", 12))),
            color_foreground=str(value.get("color_foreground", "#FFFFFF")),
            color_background=str(value.get("color_background", "#000000")),
            color_tolerance=max(0, int(value.get("color_tolerance", 30))),
            # 缺少字段意味着这是旧快照，必须保持当时没有颜色描边扩张的语义。
            color_outline_expand_pixels=max(0, int(value.get("color_outline_expand_pixels", 0))),
            expand_pixels=max(0, int(value.get("expand_pixels", 2))),
        )._with_normalized_brush_strokes(value.get("brush_strokes", []))

    def _with_normalized_brush_strokes(self, strokes: Iterable[Any]) -> "MaskPlan":
        """兼容旧版点列，同时把新计划规范为含半径的逐笔记录。"""
        self.brush_strokes = _normalize_brush_strokes(strokes, self.brush_radius)
        return self

def _normalized_polygon(polygon: OcrPolygon, width: int, height: int) -> np.ndarray | None:
    points = np.asarray(polygon.points, dtype=np.int32)
    if len(points) < 3:
        return None
    points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
    if cv2.contourArea(points) < 1:
        return None
    return points


def _dilate_mask(mask: np.ndarray, expand_pixels: int) -> np.ndarray:
    """仅在有前景时扩张二值 mask，统一各掩膜来源的描边补偿语义。"""
    expand = max(0, int(expand_pixels))
    if not expand or not np.any(mask):
        return mask
    kernel_size = expand * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel)


def polygon_mask(image_shape: tuple[int, ...], polygons: Iterable[OcrPolygon], expand_pixels: int = 0) -> np.ndarray:
    """只填充 OCR 多边形本身，而非其外接矩形。"""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    normalized = [_normalized_polygon(polygon, width, height) for polygon in polygons]
    normalized = [points for points in normalized if points is not None]
    if normalized:
        cv2.fillPoly(mask, normalized, 255)
    return _dilate_mask(mask, expand_pixels)


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
    return _dilate_mask(mask, expand_pixels)


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


def _normalize_brush_strokes(strokes: Iterable[Any], fallback_radius: int) -> list[dict[str, Any]]:
    """将旧点列与新逐笔记录统一为 ``{points, radius}`` 的可序列化结构。"""
    normalized: list[dict[str, Any]] = []
    for stroke in strokes or []:
        raw_points = stroke.get("points", []) if isinstance(stroke, dict) else stroke
        radius = stroke.get("radius", fallback_radius) if isinstance(stroke, dict) else fallback_radius
        points: list[list[float]] = []
        for point in raw_points or []:
            if len(point) >= 2:
                points.append([
                    max(0.0, min(1.0, float(point[0]))),
                    max(0.0, min(1.0, float(point[1]))),
                ])
        if points:
            normalized.append({"points": points, "radius": max(1, int(radius))})
    return normalized


def _serialize_brush_strokes(strokes: Iterable[Any], fallback_radius: int) -> list[dict[str, Any]]:
    """输出稳定 JSON 结构；旧任务读取后会在下次确认时自然升级。"""
    return _normalize_brush_strokes(strokes, fallback_radius)


def brush_mask(image_shape: tuple[int, ...], strokes: Iterable[Any], radius: int) -> np.ndarray:
    """把用户涂鸦路径转换为二值掩膜；每笔使用自己冻结的原图像素半径。"""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in _normalize_brush_strokes(strokes, radius):
        stroke_radius = max(1, int(stroke["radius"]))
        thickness = stroke_radius * 2
        points = [
            (round(max(0.0, min(1.0, x)) * max(0, width - 1)), round(max(0.0, min(1.0, y)) * max(0, height - 1)))
            for x, y in stroke["points"]
        ]
        if not points:
            continue
        if len(points) == 1:
            cv2.circle(mask, points[0], stroke_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(mask, [np.asarray(points, dtype=np.int32)], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.circle(mask, points[0], stroke_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(mask, points[-1], stroke_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask


def _hex_to_bgr(value: str) -> np.ndarray:
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError("颜色必须是 #RRGGBB 格式。")
    rgb = tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))
    return np.asarray(rgb[::-1], dtype=np.uint8)


def foreground_mask_in_roi(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    foreground: str,
    background: str,
    tolerance: int,
) -> np.ndarray:
    """在任意二值 ROI 内保留接近前景、且更接近前景而非背景的像素。

    颜色筛选与动态 OCR 精细定位共用这一逻辑，避免两条路径对相同颜色设置
    得出不同的结果。LAB 距离对亮度变化比直接 RGB 差值更稳定；允许容差为
    ``0``，以便追求极小 mask 的用户使用严格匹配。
    """
    if roi_mask is None or not np.any(roi_mask):
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    try:
        foreground_bgr, background_bgr = _hex_to_bgr(foreground), _hex_to_bgr(background)
    except (TypeError, ValueError):
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    lab = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    foreground_lab = cv2.cvtColor(
        foreground_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB,
    ).reshape(3).astype(np.float32)
    background_lab = cv2.cvtColor(
        background_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB,
    ).reshape(3).astype(np.float32)
    foreground_distance = np.linalg.norm(lab - foreground_lab, axis=2)
    background_distance = np.linalg.norm(lab - background_lab, axis=2)
    selected = (
        (roi_mask > 0)
        & (foreground_distance <= max(0, int(tolerance)))
        & (foreground_distance < background_distance)
    )
    return np.where(selected, 255, 0).astype(np.uint8)


def color_mask(
    image_bgr: np.ndarray,
    regions: Iterable[Rectangle],
    foreground: str,
    background: str,
    tolerance: int,
    outline_expand_pixels: int = 0,
) -> np.ndarray:
    """在用户 ROI 内筛出前景字形，再小幅补偿其描边与抗锯齿边缘。

    膨胀发生在颜色筛选之后，故不会把整个 ROI 误当作待修复区域；它也有意
    不再裁回 ROI，以便覆盖恰好压在用户框边缘的文字描边。
    """
    roi = rectangle_mask(image_bgr.shape, regions)
    selected = foreground_mask_in_roi(image_bgr, roi, foreground, background, tolerance)
    return _dilate_mask(selected, outline_expand_pixels)


def refine_ocr_polygon_mask(
    image_bgr: np.ndarray,
    polygons: Iterable[OcrPolygon],
    foreground: str,
    background: str,
    tolerance: int,
    outline_expand_pixels: int = 1,
) -> np.ndarray:
    """把 OCR 文本行多边形缩小为字幕前景像素，并补偿常见黑色描边。

    PaddleOCR 给出的多边形是“文字可能存在的区域”，而不是字形分割。这里先
    把多边形作为严格 ROI，再依据前景/背景 LAB 距离筛选像素。最后只对筛选出
    的字形进行小幅膨胀，让白字周围的黑描边、抗锯齿边缘进入修复 mask；这比
    对整块 OCR 区域外扩更不容易吞掉背景。
    """
    roi = polygon_mask(image_bgr.shape, polygons)
    refined = foreground_mask_in_roi(image_bgr, roi, foreground, background, tolerance)
    return _dilate_mask(refined, outline_expand_pixels)


def build_mask(image_bgr: np.ndarray, plan: MaskPlan) -> np.ndarray:
    """根据冻结计划合并全部已确认来源，供所有修复引擎使用。"""
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    if plan.manual_regions:
        mask = cv2.bitwise_or(mask, rectangle_mask(image_bgr.shape, plan.manual_regions, plan.expand_pixels))
    if plan.brush_strokes:
        mask = cv2.bitwise_or(mask, brush_mask(image_bgr.shape, plan.brush_strokes, plan.brush_radius))
    if plan.candidate_regions:
        mask = cv2.bitwise_or(
            mask,
            color_mask(
                image_bgr, plan.candidate_regions, plan.color_foreground,
                plan.color_background, plan.color_tolerance, plan.color_outline_expand_pixels,
            ),
        )
    return mask
