"""PaddleOCR 文字候选检测与视频动态帧映射。"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Any

import cv2

from backend.config import config
from backend.masking import OcrPolygon, VideoMaskStrategy, filter_polygons_by_regions
from backend.tools.common_tools import get_readable_path, is_image_file
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections
from .model_config import ModelConfig


def _error_summary(error: BaseException, limit: int = 240) -> str:
    """把后端异常压缩为适合任务日志的一行，避免 CUDA 堆栈淹没处理反馈。"""
    message = " ".join(str(error).split()) or error.__class__.__name__
    return message[:limit - 1] + "…" if len(message) > limit else message


class TextRegionDetector:
    """检测当前帧文字候选，并为视频生成动态文字区域映射。

    OCR 不跟随 PyTorch 的 ``AI 设备`` 选项：它由 PaddlePaddle 独立执行。
    初始化时优先使用已安装且可工作的 NVIDIA GPU 后端；驱动、CUDA runtime、
    显存或 GPU Paddle wheel 不可用时，重新以 CPU 后端创建检测器。两条路径均
    失败才将错误交给调用方，避免一台没有 GPU 的机器因此完全失去 OCR。
    """

    SAMPLE_STEP = 3

    def __init__(self, media_path: str, ocr_regions=None) -> None:
        self.media_path = media_path
        # ``ocr_regions`` 是可选的检测 ROI；为空即在整帧搜索。
        self.ocr_regions = ocr_regions or []
        self._init_sample_step()

    def _init_sample_step(self) -> None:
        if is_image_file(self.media_path):
            self.SAMPLE_STEP = 1
            return
        capture = cv2.VideoCapture(get_readable_path(self.media_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if fps >= 60:
            self.SAMPLE_STEP = 4
        elif fps >= 30:
            self.SAMPLE_STEP = 3
        else:
            self.SAMPLE_STEP = 2

    @cached_property
    def text_detector(self) -> Any:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        # 不预先只检查显卡是否存在：真实可用性还取决于 GPU Paddle wheel、CUDA
        # runtime、驱动和首次模型初始化。直接完整创建，才能保证“可用”的含义。
        try:
            detector = self._create_detector("gpu:0")
        except Exception as gpu_error:
            self.ocr_device = "cpu"
            try:
                detector = self._create_detector("cpu")
            except Exception as cpu_error:
                raise RuntimeError(
                    "PaddleOCR 初始化失败：GPU 与 CPU 后端均不可用。"
                    f" GPU 原因：{_error_summary(gpu_error)}；"
                    f" CPU 原因：{_error_summary(cpu_error)}。"
                ) from cpu_error
            print(
                "PaddleOCR GPU 初始化不可用，已自动回退到 CPU："
                f"{_error_summary(gpu_error)}"
            )
            return detector
        self.ocr_device = "gpu:0"
        return detector

    def detect_polygons(self, image: Any) -> list[OcrPolygon]:
        """保留 PaddleOCR 多边形，用于候选回显或视频当帧动态 mask。"""
        polygons: list[OcrPolygon] = []
        try:
            results = self.text_detector.predict(image)
        except Exception as gpu_error:
            # 某些 Paddle/CUDA 问题只会在第一次推理而非构造模型时暴露。若当前
            # 确实是 GPU 实例，则同样重建 CPU 检测器并重试这一帧；CPU 自身失败
            # 必须原样抛出，不能把真实 OCR 错误伪装成“未检测到文字”。
            if getattr(self, "ocr_device", None) != "gpu:0":
                raise
            self.text_detector = self._create_cpu_detector()
            self.ocr_device = "cpu"
            print(
                "PaddleOCR GPU 推理失败，已自动回退到 CPU："
                f"{_error_summary(gpu_error)}"
            )
            results = self.text_detector.predict(image)
        for result in results:
            points_list = result.get("dt_polys")
            if points_list is None or len(points_list) == 0:
                continue
            scores = result.get("dt_scores")
            if scores is None:
                scores = []
            for index, points in enumerate(points_list):
                parsed = tuple((int(point[0]), int(point[1])) for point in points)
                if len(parsed) >= 3:
                    score = float(scores[index]) if index < len(scores) else 1.0
                    polygons.append(OcrPolygon(parsed, score))
        roi_regions = [(xmin, xmax, ymin, ymax) for ymin, ymax, xmin, xmax in self.ocr_regions]
        return sorted(
            filter_polygons_by_regions(polygons, roi_regions),
            key=lambda polygon: (polygon.bbox[2], polygon.bbox[0], -polygon.score),
        )

    def _create_cpu_detector(self) -> Any:
        """在 GPU 的首次推理失败后重建等价的 CPU 检测器。"""
        return self._create_detector("cpu")

    @staticmethod
    def _create_detector(device: str) -> Any:
        """以指定 Paddle 设备创建检测器，确保 GPU/CPU 配置完全一致。"""
        from paddleocr import TextDetection

        model_config = ModelConfig()
        return TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device=device,
            enable_hpi=False,
            thresh=config.ocrPixelThreshold.value / 100,
            box_thresh=config.ocrBoxThreshold.value / 100,
            unclip_ratio=config.ocrUnclipRatio.value / 100,
        )

    @staticmethod
    def _merge_polygon_groups(current: list[OcrPolygon], following: list[OcrPolygon]) -> list[OcrPolygon]:
        """采样检测之间以两侧候选的并集保守填补中间帧。"""
        result: list[OcrPolygon] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for polygon in [*current, *following]:
            if polygon.points not in seen:
                seen.add(polygon.points)
                result.append(polygon)
        return result

    def find_polygon_frame_map(
        self,
        strategy: VideoMaskStrategy | str,
        repair_processor: Any | None = None,
        sample_step: int | None = None,
    ) -> dict[int, list[OcrPolygon]]:
        """按采样或逐帧频率检测视频文字，返回 1-based 帧号到多边形的映射。"""
        selected = strategy if isinstance(strategy, VideoMaskStrategy) else VideoMaskStrategy(strategy)
        step = 1 if selected is VideoMaskStrategy.EVERY_FRAME_OCR else max(1, sample_step or self.SAMPLE_STEP)
        capture = cv2.VideoCapture(get_readable_path(self.media_path))
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        sampled: dict[int, list[OcrPolygon]] = {}
        frame_no = 0
        last_logged_percent = 0
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            frame_no += 1
            ab_sections = getattr(repair_processor, "ab_sections", None)
            if is_frame_number_in_ab_sections(frame_no - 1, ab_sections) and (frame_no - 1) % step == 0:
                polygons = self.detect_polygons(frame)
                if polygons:
                    sampled[frame_no] = polygons
            if repair_processor:
                # OCR 预扫描尚未写出任何修复帧，不能驱动任务进度或预览时间轴。
                # 每推进约 10% 仅写一条日志，让用户知道后台仍在定位文字。
                percent = min(100, int(100 * frame_no / frame_count))
                if percent >= last_logged_percent + 10 or percent == 100:
                    append_output = getattr(repair_processor, "append_output", None)
                    if callable(append_output):
                        append_output(f"动态 OCR 正在扫描文字位置：{percent}%（定位阶段）。")
                    last_logged_percent = percent
        capture.release()
        if selected is VideoMaskStrategy.EVERY_FRAME_OCR:
            return sampled

        completed: dict[int, list[OcrPolygon]] = {}
        frame_numbers = sorted(sampled)
        for current_no, next_no in zip(frame_numbers, frame_numbers[1:]):
            completed[current_no] = sampled[current_no]
            if next_no - current_no <= step * 2:
                fill = self._merge_polygon_groups(sampled[current_no], sampled[next_no])
                for fill_no in range(current_no + 1, next_no):
                    completed[fill_no] = fill
        if frame_numbers:
            completed[frame_numbers[-1]] = sampled[frame_numbers[-1]]
        return completed
