"""不下载模型的 AI 升级回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.enhance.media_enhancer import MediaEnhancer
from backend.enhance.realesrgan import ANIME_ASSET, RealESRGANEnhancer, _build_rrdb_net, _build_srvgg_net
from backend.inpaint.engine import (
    InpaintError,
    InpaintMode,
    _finish_ink_on_flat_background,
    create_inpaint_engine,
    device_candidates,
)
from backend.inpaint.mask import normalize_regions, regions_to_mask
from backend.main import SubtitleRemover
from backend.masking import MaskMode, MaskPlan
from ui.component.enhance_task_list_component import EnhanceTaskListComponent, EnhanceTaskStatus
from ui.component.task_list_component import TaskListComponent, TaskStatus


class MaskAndEngineTests(unittest.TestCase):
    def test_auto_device_candidates_try_cuda_before_cpu(self):
        self.assertEqual(device_candidates("auto"), ("cuda", "cpu"))
        self.assertEqual(device_candidates("cpu"), ("cpu",))
        self.assertEqual(device_candidates("cuda"), ("cuda",))

    def test_auto_inpaint_tries_cpu_after_cuda_initialization_fails(self):
        expected = object()
        with patch("backend.inpaint.engine.TorchScriptLaMaEngine", side_effect=[InpaintError("CUDA unavailable"), expected]) as builder:
            result = create_inpaint_engine(InpaintMode.LAMA, device="auto", model_dir="models")

        self.assertIs(result, expected)
        self.assertEqual(
            [call.args[1] for call in builder.call_args_list],
            ["cuda", "cpu"],
        )

    def test_regions_are_clamped_sorted_and_expanded(self):
        regions = normalize_regions([(20, -4, 12, -1), (2, 6, 3, 8), (2, 6, 3, 8)], 10, 10)
        self.assertEqual(regions, [(0, 10, 0, 10), (2, 6, 3, 8)])
        mask = regions_to_mask((10, 10, 3), [(4, 6, 4, 6)], expand_pixels=1)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(int(mask[3, 4]), 255)
        self.assertEqual(int(mask[0, 0]), 0)

    def test_opencv_mode_remains_available(self):
        source = np.full((32, 32, 3), 128, dtype=np.uint8)
        source[10:20, 10:20] = 255
        mask = regions_to_mask(source.shape, [(10, 20, 10, 20)])
        result = create_inpaint_engine(InpaintMode.OPENCV).inpaint(source, mask)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(result[mask == 0], source[mask == 0]))

    def test_invalid_mode_is_actionable(self):
        with self.assertRaises(InpaintError):
            create_inpaint_engine("not-a-mode")

    def test_flat_background_postprocess_removes_ink_residue_only_inside_mask(self):
        """白色气泡中的字形应回填真实底色，不能触碰 mask 外区域。"""
        source = np.full((40, 50, 3), (238, 241, 244), dtype=np.uint8)
        source[12:28, 18:21] = (42, 39, 36)
        source[20:23, 12:28] = (42, 39, 36)
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        mask[12:28, 18:21] = 1
        mask[20:23, 12:28] = 1
        model_result = source.copy()
        model_result[mask.astype(bool)] = (155, 157, 159)  # 模拟 AI 残留的灰色笔画

        result = _finish_ink_on_flat_background(model_result, source, mask)

        self.assertTrue(np.all(result[mask.astype(bool)] == np.array([238, 241, 244])))
        self.assertTrue(np.array_equal(result[mask == 0], model_result[mask == 0]))

    def test_flat_background_postprocess_keeps_broad_or_textured_masks_for_ai(self):
        """宽泛涂鸦/纹理背景不应被错误地填成单一颜色。"""
        source = np.full((48, 48, 3), 230, dtype=np.uint8)
        source[:, ::4] = (120, 160, 190)  # 采样环是明显纹理，不满足平坦底色条件
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        mask[14:34, 14:34] = 1
        model_result = np.full_like(source, 99)

        result = _finish_ink_on_flat_background(model_result, source, mask)

        self.assertTrue(np.array_equal(result, model_result))


class OutputAndEnhanceRequestTests(unittest.TestCase):
    def test_auto_enhance_tries_cpu_after_cuda_model_initialization_fails(self):
        first_model, second_model, loaded_model = object(), object(), object()
        with (
            patch("backend.enhance.realesrgan.ModelStore.ensure", return_value=Path("model.pth")),
            patch.object(
                RealESRGANEnhancer,
                "_asset_and_model",
                side_effect=[(ANIME_ASSET, first_model), (ANIME_ASSET, second_model)],
            ),
            patch.object(
                RealESRGANEnhancer,
                "_load_model",
                side_effect=[RuntimeError("CUDA unavailable"), (loaded_model, True)],
            ) as load_model,
            patch("backend.enhance.realesrgan.resolve_device", side_effect=lambda candidate: candidate) as resolve,
        ):
            enhancer = RealESRGANEnhancer(device="auto")

        self.assertEqual([call.args[0] for call in resolve.call_args_list], ["cuda", "cpu"])
        self.assertEqual([call.args[2] for call in load_model.call_args_list], ["cuda", "cpu"])
        self.assertIs(enhancer.model, loaded_model)
        self.assertEqual(enhancer.device, "cpu")
        self.assertEqual(enhancer.device_fallback_reason, "CUDA unavailable")

    def test_task_reset_clears_progress_status_and_output_path(self):
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        repair_tasks = TaskListComponent()
        repair_tasks.add_task("example.png")
        repair_task = repair_tasks.get_task(0)
        repair_task.output_path = "old-output.png"
        repair_tasks.update_task_progress(0, 100)
        repair_tasks.update_task_status(0, TaskStatus.COMPLETED)
        repair_tasks.reset_task(0)
        self.assertIs(repair_task.status, TaskStatus.PENDING)
        self.assertEqual(repair_task.progress, 0)
        self.assertIsNone(repair_task._output_path)

        enhance_tasks = EnhanceTaskListComponent()
        enhance_tasks.add_task("example.png", "realesrgan_anime", 2.0, "auto")
        enhance_task = enhance_tasks.get_task(0)
        enhance_task.output_path = "old-enhanced.png"
        enhance_tasks.update_task_progress(0, 100)
        enhance_tasks.update_task_status(0, EnhanceTaskStatus.COMPLETED)
        enhance_tasks.reset_task(0)
        self.assertIs(enhance_task.status, EnhanceTaskStatus.PENDING)
        self.assertEqual(enhance_task.progress, 0)
        self.assertIsNone(enhance_task.output_path)
        repair_tasks.deleteLater()
        enhance_tasks.deleteLater()
    def test_realesrgan_postprocess_accepts_inference_tensor(self):
        """防止视频帧后处理在 inference_mode 外调用 clamp_ 再次崩溃。"""
        import torch

        enhancer = object.__new__(RealESRGANEnhancer)
        enhancer.torch = torch
        enhancer.device = "cpu"
        enhancer.use_half = False
        enhancer.tile_size = 0
        enhancer.native_scale = 4

        def inference_output(_image):
            with torch.inference_mode():
                output = torch.full((1, 3, 4, 4), 2.0)
                output[:, :, 0, 0] = -0.5
                output[:, :, 0, 1] = 0.5
                return output

        enhancer._infer_tiled = inference_output
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        result = enhancer.enhance(image, 4)
        self.assertEqual(result.shape, (4, 4, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result.min()), 0)
        self.assertEqual(int(result.max()), 255)

    def test_realesrgan_networks_have_native_x4_output_without_weights(self):
        import torch
        source = torch.zeros(1, 3, 8, 8)
        with torch.inference_mode():
            self.assertEqual(tuple(_build_rrdb_net(6).eval()(source).shape), (1, 3, 32, 32))
            self.assertEqual(tuple(_build_srvgg_net().eval()(source).shape), (1, 3, 32, 32))

    def test_image_output_uses_output_extension_not_source_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jpg"
            output = root / "result.png"
            image = np.full((32, 48, 3), 180, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), image))
            remover = SubtitleRemover(str(source))
            remover.video_out_path = str(output)
            remover.mask_plan = MaskPlan(
                mode=MaskMode.RECTANGLE,
                manual_regions=[(2, 20, 2, 20)],
            ).to_dict()
            remover.inpaint_mode = InpaintMode.OPENCV.value
            remover.run()
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_enhance_output_name_is_separate_from_source(self):
        task = MediaEnhancer("example.png", scale=2.0)
        self.assertTrue(task.output_path.endswith("example_enhanced_x2.png"))
        self.assertNotEqual(Path(task.output_path).name, "example.png")


if __name__ == "__main__":
    unittest.main()
