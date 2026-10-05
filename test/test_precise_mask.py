"""精确 OCR 多边形掩膜的无网络回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.masking import (
    MaskMode,
    MaskPlan,
    OcrPolygon,
    VideoMaskStrategy,
    build_mask,
    filter_polygons_by_regions,
    polygon_mask,
    refine_polygon_mask,
)
from backend.tools.subtitle_detect import SubtitleDetect


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

    def test_refinement_keeps_high_contrast_strokes_and_is_smaller(self):
        image = np.full(self.shape, 60, dtype=np.uint8)
        base = np.zeros(self.shape[:2], dtype=np.uint8)
        cv2.rectangle(base, (5, 5), (34, 22), 255, -1)
        cv2.rectangle(image, (12, 9), (26, 15), (240, 240, 240), -1)
        refined = refine_polygon_mask(image, base, color_threshold=20, min_component_area=4)
        self.assertGreater(int(refined.sum()), 0)
        self.assertLess(int(refined.sum()), int(base.sum()))
        self.assertEqual(int(refined[12, 18]), 255)
        self.assertEqual(int(refined[6, 6]), 0)

    def test_plan_serializes_and_mixed_mode_does_not_drop_manual_region(self):
        plan = MaskPlan(
            mode=MaskMode.MIXED,
            ocr_polygons=[self.polygon],
            candidate_regions=[(4, 32, 4, 19)],
            manual_regions=[(1, 3, 20, 24)],
            expand_pixels=0,
        )
        restored = MaskPlan.from_dict(plan.to_dict())
        mask = build_mask(np.full(self.shape, 100, dtype=np.uint8), restored)
        self.assertEqual(restored.ocr_polygons[0], self.polygon)
        self.assertEqual(int(mask[22, 2]), 255)
        self.assertEqual(int(mask[5, 30]), 0)


class PolygonVideoStrategyTests(unittest.TestCase):
    def test_detector_preserves_paddle_polygons_and_scores(self):
        class FakeDetector:
            def predict(self, image):
                return [{
                    "dt_polys": np.array([[[1, 2], [9, 3], [8, 7], [2, 6]]], dtype=np.int32),
                    "dt_scores": [0.73],
                }]

        detector = object.__new__(SubtitleDetect)
        detector.video_path = "unused.png"
        detector.sub_areas = []
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
        class CountingDetector(SubtitleDetect):
            def __init__(self, path):
                self.video_path = path
                self.sub_areas = []
                self.calls = 0
                self.SAMPLE_STEP = 2

            def detect_polygons(self, frame):
                self.calls += 1
                return [OcrPolygon(((2, 2), (10, 2), (10, 8), (2, 8)), 0.9)]

        with tempfile.TemporaryDirectory() as directory:
            detector = CountingDetector(self._video(Path(directory)))
            sampled = detector.find_polygon_frame_map(VideoMaskStrategy.SAMPLED_OCR, sample_step=2)
            self.assertEqual(detector.calls, 3)
            self.assertIn(2, sampled)  # 采样帧间会填补中间帧。

            detector = CountingDetector(detector.video_path)
            every = detector.find_polygon_frame_map(VideoMaskStrategy.EVERY_FRAME_OCR)
            self.assertEqual(detector.calls, 5)
            self.assertEqual(set(every), {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
