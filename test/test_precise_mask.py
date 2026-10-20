"""精确 OCR 多边形掩膜的无网络回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import cv2
import numpy as np

from backend.masking import (
    MaskMode,
    MaskPlan,
    OcrPolygon,
    VideoMaskStrategy,
    build_mask,
    brush_mask,
    color_mask,
    filter_polygons_by_regions,
    polygon_mask,
    refine_ocr_polygon_mask,
)
from backend.main import ContentRepairProcessor
from backend.tools.text_region_detector import TextRegionDetector


class PreciseMaskTests(unittest.TestCase):
    def setUp(self):
        self.shape = (30, 40, 3)
        self.polygon = OcrPolygon(((4, 4), (32, 8), (25, 19), (6, 15)), 0.88)

    def test_polygon_mask_is_not_its_bounding_rectangle(self):
        actual = polygon_mask(self.shape, [self.polygon])
        bounding = np.zeros(self.shape[:2], dtype=np.uint8)
        cv2.rectangle(bounding, (4, 4), (32, 19), 255, -1)
        self.assertGreater(int(actual.sum()), 0)
        self.assertLess(int(actual.sum()), int(bounding.sum()))
        self.assertEqual(int(actual[5, 5]), 255)
        self.assertEqual(int(actual[5, 30]), 0)

    def test_roi_filters_polygons_but_does_not_change_vertices(self):
        far = OcrPolygon(((34, 24), (38, 24), (38, 28), (34, 28)), 0.9)
        kept = filter_polygons_by_regions([self.polygon, far], [(0, 33, 0, 22)])
        self.assertEqual(kept, [self.polygon])
        self.assertEqual(kept[0].points, self.polygon.points)

    def test_brush_mask_uses_relative_source_coordinates(self):
        mask = brush_mask(self.shape, [[(0.2, 0.5), (0.8, 0.5)]], radius=1)
        self.assertEqual(int(mask[15, 8]), 255)
        self.assertEqual(int(mask[15, 31]), 255)
        self.assertEqual(int(mask[2, 2]), 0)

    def test_each_brush_stroke_keeps_its_own_frozen_radius(self):
        """改笔刷粗细只能影响下一笔，历史路径不能被重新绘制成新粗细。"""
        shape = (40, 40, 3)
        strokes = [
            {"points": [[0.25, 0.5]], "radius": 1},
            {"points": [[0.75, 0.5]], "radius": 4},
        ]
        mask = brush_mask(shape, strokes, radius=12)
        self.assertEqual(int(mask[19, 10]), 255)
        self.assertEqual(int(mask[16, 10]), 0)   # 半径 1 的历史笔迹没有变粗
        self.assertEqual(int(mask[16, 29]), 255)  # 半径 4 的新笔迹保持更粗
        restored = MaskPlan(brush_strokes=strokes, brush_radius=12).to_dict()
        self.assertEqual(restored["brush_strokes"][0]["radius"], 1)
        self.assertEqual(restored["brush_strokes"][1]["radius"], 4)

    def test_color_mask_keeps_foreground_only_inside_user_roi(self):
        image = np.full(self.shape, (20, 40, 60), dtype=np.uint8)
        # BGR=(240, 240, 240) corresponds to the configured #F0F0F0 foreground.
        image[10:16, 12:28] = (240, 240, 240)
        image[10:16, 30:36] = (240, 240, 240)
        mask = color_mask(image, [(8, 29, 7, 20)], "#F0F0F0", "#3C2814", 8)
        self.assertEqual(int(mask[12, 18]), 255)
        self.assertEqual(int(mask[12, 33]), 0)

    def test_color_mask_outline_expands_only_selected_glyphs(self):
        image = np.full(self.shape, (20, 40, 60), dtype=np.uint8)
        image[12, 18] = (240, 240, 240)
        # 扩张后的范围可覆盖字形周围，但 ROI 内其它背景不能因此整块入选。
        mask = color_mask(
            image, [(8, 29, 7, 20)], "#F0F0F0", "#3C2814", 8, outline_expand_pixels=1,
        )
        self.assertEqual(int(mask[12, 18]), 255)
        self.assertEqual(int(mask[12, 17]), 255)
        self.assertEqual(int(mask[12, 24]), 0)

    def test_legacy_color_plan_does_not_gain_outline_expansion(self):
        restored = MaskPlan.from_dict({
            "mode": "color", "candidate_regions": [[1, 2, 1, 2]],
            "color_foreground": "#FFFFFF", "color_background": "#000000", "color_tolerance": 4,
        })
        self.assertEqual(restored.color_outline_expand_pixels, 0)

    def test_plan_serializes_and_rectangle_mode_keeps_manual_region(self):
        plan = MaskPlan(
            mode=MaskMode.RECTANGLE,
            manual_regions=[(1, 3, 20, 24)],
            expand_pixels=0,
        )
        restored = MaskPlan.from_dict(plan.to_dict())
        mask = build_mask(np.full(self.shape, 100, dtype=np.uint8), restored)
        self.assertEqual(restored.mode, MaskMode.RECTANGLE)
        self.assertEqual(restored.manual_regions, [(1, 3, 20, 24)])
        self.assertEqual(int(mask[22, 2]), 255)
        self.assertEqual(int(mask[5, 30]), 0)

    def test_confirmed_sources_are_combined_in_one_frozen_plan(self):
        """矩形、涂鸦和颜色 ROI 必须取并集，不由最后一次确认覆盖。"""
        image = np.full((30, 40, 3), (20, 20, 20), dtype=np.uint8)
        image[20:24, 30:34] = (255, 255, 255)
        plan = MaskPlan(
            mode=MaskMode.COLOR,
            manual_regions=[(1, 4, 1, 4)],
            brush_strokes=[[(0.5, 0.5)]],
            brush_radius=1,
            candidate_regions=[(28, 36, 18, 26)],
            color_foreground="#FFFFFF",
            color_background="#141414",
            color_tolerance=4,
            expand_pixels=0,
        )
        restored = MaskPlan.from_dict(plan.to_dict())
        mask = build_mask(image, restored)
        self.assertEqual(restored.source_modes, [MaskMode.RECTANGLE, MaskMode.BRUSH, MaskMode.COLOR])
        self.assertEqual(int(mask[2, 2]), 255)   # 手工矩形
        self.assertEqual(int(mask[15, 20]), 255)  # 涂鸦
        self.assertEqual(int(mask[21, 31]), 255)  # 颜色筛选

    def test_dynamic_ocr_refinement_keeps_white_glyph_and_small_outline(self):
        image = np.full((20, 30, 3), (24, 32, 48), dtype=np.uint8)
        # 白色字形包在黑色描边中，外侧是与默认背景参考色接近的画面。
        image[7:13, 10:20] = (0, 0, 0)
        image[8:12, 11:19] = (255, 255, 255)
        polygon = OcrPolygon(((8, 5), (22, 5), (22, 15), (8, 15)))
        mask = refine_ocr_polygon_mask(
            image, [polygon], "#FFFFFF", "#182030", tolerance=4, outline_expand_pixels=1,
        )
        self.assertEqual(int(mask[9, 12]), 255)  # 白色字幕字形
        self.assertEqual(int(mask[7, 11]), 255)  # 扩张后覆盖黑色描边
        self.assertEqual(int(mask[2, 2]), 0)     # OCR ROI 外背景不被吞入

    def test_dynamic_ocr_refinement_path_ignores_static_plan_and_non_glyph_roi(self):
        """业务入口必须真正使用精细 mask，且不能混入冻结的静态组合计划。"""
        image = np.full((40, 50, 3), (30, 30, 30), dtype=np.uint8)
        image[12:16, 15:23] = (255, 255, 255)
        remover = object.__new__(ContentRepairProcessor)
        remover.mask_plan = MaskPlan(manual_regions=[(40, 48, 30, 38)]).to_dict()
        remover.dynamic_ocr_refine_mask = True
        remover.dynamic_ocr_foreground = "#FFFFFF"
        remover.dynamic_ocr_background = "#1E1E1E"
        remover.dynamic_ocr_refine_tolerance = 4
        remover.dynamic_ocr_outline_expand = 0
        polygon = OcrPolygon(((10, 8), (28, 8), (28, 20), (10, 20)))
        mask = remover._mask(image, [polygon])
        self.assertEqual(int(mask[13, 16]), 255)
        self.assertEqual(int(mask[9, 11]), 0)    # 多边形内非白色背景不应整块入选
        self.assertEqual(int(mask[34, 44]), 0)   # 冻结手工区域不参与动态 OCR

    def test_sampled_dynamic_ocr_never_uses_stale_pixel_refinement(self):
        """采样任务快照即使带着旧 refine 标志，也必须在运行端归零。"""
        remover = object.__new__(ContentRepairProcessor)
        remover.video_mask_strategy = VideoMaskStrategy.SAMPLED_OCR.value
        remover.dynamic_ocr_refine_mask = True
        logs = []
        remover.append_output = lambda message: logs.append(message)
        self.assertIs(remover._effective_dynamic_ocr_strategy(), VideoMaskStrategy.SAMPLED_OCR)
        self.assertFalse(remover.dynamic_ocr_refine_mask)
        self.assertIn("已忽略任务快照", logs[0])

    def test_dynamic_ocr_polygon_does_not_reuse_frozen_manual_mask(self):
        remover = object.__new__(ContentRepairProcessor)
        remover.mask_plan = MaskPlan(
            mode=MaskMode.RECTANGLE,
            manual_regions=[(60, 70, 60, 70)],
            expand_pixels=0,
        ).to_dict()
        polygon = OcrPolygon(((10, 10), (30, 12), (25, 24), (12, 20)), 0.9)
        mask = remover._mask(np.zeros((80, 80, 3), dtype=np.uint8), [polygon])
        self.assertEqual(int(mask[65, 65]), 0)  # 冻结手工框没有参与动态 OCR。
        self.assertEqual(int(mask[15, 15]), 255)

    def test_final_mask_modes_do_not_include_ocr_candidate(self):
        self.assertEqual(
            {mode.value for mode in MaskMode},
            {"rectangle", "brush", "color"},
        )

    def test_video_rejects_dynamic_ocr_and_static_mask_together(self):
        remover = object.__new__(ContentRepairProcessor)
        remover.video_dynamic_ocr = True
        remover.is_image = False
        remover.mask_plan = MaskPlan(mode=MaskMode.RECTANGLE).to_dict()
        with self.assertRaisesRegex(RuntimeError, "不能在同一视频任务中同时执行"):
            remover.run()

    def test_dynamic_ocr_progress_starts_when_repair_frames_are_written(self):
        """OCR 预扫描写日志即可，任务进度与视频时间轴只表示实际修复帧。"""
        class FakeProgressBar:
            total = 4
            n = 0

            def update(self, amount):
                self.n += amount

        remover = object.__new__(ContentRepairProcessor)
        remover.video_dynamic_ocr = True
        remover.is_image = False
        remover.progress_total = 0
        remover.progress_remover = 0
        remover.isFinished = False
        remover.progress_listeners = []
        remover.append_output = lambda *_args: None
        received = []
        remover.add_progress_listener(lambda progress, _finished: received.append(progress))

        remover.begin_dynamic_ocr_scan(VideoMaskStrategy.SAMPLED_OCR)
        self.assertEqual(received, [])
        tbar = FakeProgressBar()
        for _ in range(4):
            remover.update_progress(tbar, 1)

        self.assertEqual(received[0], 25)
        self.assertEqual(received[-1], 100)
        self.assertEqual(received, sorted(received))


class PolygonVideoStrategyTests(unittest.TestCase):
    def test_detector_tries_gpu_then_recreates_on_cpu(self):
        """CPU Paddle 或 CUDA 初始化失败时，OCR 必须仍然可用。"""
        calls = []

        class FakeTextDetection:
            def __init__(self, *, device, **_kwargs):
                calls.append(device)
                if device == "gpu:0":
                    raise RuntimeError("CUDA runtime unavailable")

        detector = object.__new__(TextRegionDetector)
        with patch.dict(sys.modules, {"paddleocr": SimpleNamespace(TextDetection=FakeTextDetection)}), \
                patch("builtins.print") as log:
            instance = detector.text_detector

        self.assertIsInstance(instance, FakeTextDetection)
        self.assertEqual(calls, ["gpu:0", "cpu"])
        self.assertEqual(detector.ocr_device, "cpu")
        self.assertIn("自动回退到 CPU", log.call_args.args[0])

    def test_detector_retries_current_frame_on_cpu_when_gpu_predict_fails(self):
        """延迟到首帧推理才暴露的 CUDA 错误也不能中断 OCR 任务。"""
        class BrokenGpuDetector:
            def predict(self, _image):
                raise RuntimeError("GPU inference failed")

        class CpuDetector:
            def predict(self, _image):
                return [{
                    "dt_polys": np.array([[[1, 2], [9, 3], [8, 7], [2, 6]]], dtype=np.int32),
                    "dt_scores": [0.73],
                }]

        detector = object.__new__(TextRegionDetector)
        detector.media_path = "unused.png"
        detector.ocr_regions = []
        detector.ocr_device = "gpu:0"
        detector.__dict__["text_detector"] = BrokenGpuDetector()
        with patch.object(detector, "_create_cpu_detector", return_value=CpuDetector()) as create_cpu, \
                patch("builtins.print") as log:
            polygons = detector.detect_polygons(np.zeros((10, 10, 3), dtype=np.uint8))

        create_cpu.assert_called_once()
        self.assertEqual(detector.ocr_device, "cpu")
        self.assertEqual(polygons, [OcrPolygon(((1, 2), (9, 3), (8, 7), (2, 6)), 0.73)])
        self.assertIn("GPU 推理失败", log.call_args.args[0])

    def test_detector_preserves_paddle_polygons_and_scores(self):
        class FakeDetector:
            def predict(self, image):
                return [{
                    "dt_polys": np.array([[[1, 2], [9, 3], [8, 7], [2, 6]]], dtype=np.int32),
                    "dt_scores": [0.73],
                }]

        detector = object.__new__(TextRegionDetector)
        detector.media_path = "unused.png"
        detector.ocr_regions = []
        detector.__dict__["text_detector"] = FakeDetector()
        polygons = detector.detect_polygons(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(polygons, [OcrPolygon(((1, 2), (9, 3), (8, 7), (2, 6)), 0.73)])

    def _video(self, directory: Path) -> str:
        path = directory / "frames.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        for index in range(5):
            writer.write(np.full((24, 32, 3), index * 30, dtype=np.uint8))
        writer.release()
        return str(path)

    def test_sampled_and_every_frame_strategies_call_ocr_expected_times(self):
        class CountingDetector(TextRegionDetector):
            def __init__(self, path):
                self.media_path = path
                self.ocr_regions = []
                self.calls = 0
                self.SAMPLE_STEP = 2

            def detect_polygons(self, frame):
                self.calls += 1
                return [OcrPolygon(((2, 2), (10, 2), (10, 8), (2, 8)), 0.9)]

        with tempfile.TemporaryDirectory() as directory:
            detector = CountingDetector(self._video(Path(directory)))
            class ScanLogger:
                def __init__(self):
                    self.ab_sections = None
                    self.logs = []

                def append_output(self, message):
                    self.logs.append(message)

            scan_logger = ScanLogger()
            sampled = detector.find_polygon_frame_map(
                VideoMaskStrategy.SAMPLED_OCR, repair_processor=scan_logger, sample_step=2,
            )
            self.assertEqual(detector.calls, 3)
            self.assertIn(2, sampled)  # 采样帧间会填补中间帧。
            self.assertTrue(any("动态 OCR 正在扫描文字位置" in message for message in scan_logger.logs))

            detector = CountingDetector(detector.media_path)
            every = detector.find_polygon_frame_map(VideoMaskStrategy.EVERY_FRAME_OCR)
            self.assertEqual(detector.calls, 5)
            self.assertEqual(set(every), {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
