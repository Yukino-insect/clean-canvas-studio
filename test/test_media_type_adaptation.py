"""图片/视频状态应驱动设置项和预览控件可见性。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui import CleanCanvasWindow
from backend.config import config
from backend.masking import MaskPlan, build_mask
from ui.home_interface import HomeInterface
from ui.component.task_list_component import TaskOptions


class MediaTypeAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _create_video(path: Path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 32))
        for value in (50, 100, 150):
            writer.write(np.full((32, 48, 3), value, dtype=np.uint8))
        writer.release()

    def test_language_is_advanced_setting_and_content_controls_follow_media_type(self):
        # 用户配置可能保存了“动态 OCR 已开启”；本用例验证切换过程，不能依赖
        # 本机上一次退出时的偏好。
        previous_dynamic_ocr = config.videoDynamicOcr.value
        previous_refine_mask = config.dynamicOcrRefineMask.value
        previous_strategy = config.videoMaskStrategy.value
        config.set(config.videoDynamicOcr, False)
        config.set(config.dynamicOcrRefineMask, False)
        config.set(config.videoMaskStrategy, "every_frame_ocr")
        window = CleanCanvasWindow()
        try:
            window.resize(1360, 720)
            window.show()
            QTest.qWait(80)
            self.app.processEvents()
            home = window.homeInterface
            self.assertFalse(hasattr(home.setting_interface, "interface_combo"))
            self.assertIsNotNone(window.advancedSettingInterface.interface_language)
            # 高级设置只注册一次；避免侧栏出现文字项与孤立齿轮两个入口。
            self.assertEqual(
                list(window.navigationInterface.panel.items).count("AdvancedSettingInterface"), 1
            )
            self.assertEqual(window.navigationInterface.panel.bottomLayout.count(), 1)
            self.assertEqual(home.media_type_selector.currentRouteKey(), "image")
            self.assertFalse(home.setting_interface.video_mask_strategy_combo.isVisible())
            self.assertFalse(home.video_display_component.control_container.isVisible())

            # 内容修复页也必须像增强页一样有显式的图片/视频任务切换；它决定
            # 文件筛选与仅视频可用的时间轴配置，而不依赖用户猜测扩展名。
            home.set_target_media_type("video")
            self.app.processEvents()
            self.assertEqual(home.media_type_selector.currentRouteKey(), "video")
            self.assertTrue(home.setting_interface.video_dynamic_ocr_switch.isVisible())
            self.assertFalse(home.setting_interface.video_mask_strategy_combo.isVisible())
            self.assertFalse(home.setting_interface.dynamic_ocr_refine_switch.isVisible())
            # 动态 OCR 是独立开关：开启后才展示采样/逐帧的检测频率；它与
            # 手工矩形、涂鸦、颜色筛选等最终掩膜来源无关。
            config.set(config.videoDynamicOcr, True)
            self.app.processEvents()
            self.assertTrue(home.setting_interface.video_mask_strategy_combo.isVisible())
            self.assertFalse(home.setting_interface.mask_mode_combo.isVisible())
            self.assertFalse(home.static_mask_actions_widget.isVisible())
            self.assertFalse(home.dynamic_ocr_actions_widget.isVisible())
            self.assertTrue(home.setting_interface.dynamic_ocr_refine_switch.isVisible())
            self.assertFalse(home.dynamic_refine_options_widget.isVisible())
            config.set(config.dynamicOcrRefineMask, True)
            self.app.processEvents()
            self.assertEqual(config.videoMaskStrategy.value, "every_frame_ocr")
            self.assertTrue(home.setting_interface.video_mask_strategy_combo.comboBox.isEnabled())
            self.assertTrue(home.dynamic_ocr_actions_widget.isVisible())
            self.assertTrue(home.dynamic_refine_options_widget.isVisible())
            config.set(config.videoMaskStrategy, "sampled_ocr")
            self.app.processEvents()
            self.assertFalse(home.setting_interface.dynamic_ocr_refine_switch.isVisible())
            self.assertFalse(config.dynamicOcrRefineMask.value)
            self.assertFalse(home.dynamic_ocr_actions_widget.isVisible())
            config.set(config.videoDynamicOcr, False)
            self.app.processEvents()
            self.assertFalse(home.setting_interface.video_mask_strategy_combo.isVisible())
            self.assertTrue(home.setting_interface.mask_mode_combo.isVisible())
            self.assertTrue(home.static_mask_actions_widget.isVisible())
            self.assertFalse(home.dynamic_ocr_actions_widget.isVisible())
            home.set_target_media_type("image")
            self.app.processEvents()
            self.assertFalse(home.setting_interface.video_dynamic_ocr_switch.isVisible())
            self.assertFalse(home.setting_interface.video_mask_strategy_combo.isVisible())

            with patch("ui.home_interface.QtWidgets.QFileDialog.getOpenFileNames", return_value=([], "")) as dialog:
                home.open_file()
                self.assertIs(dialog.call_args.args[0], home)
                self.assertIn("图片文件", dialog.call_args.args[3])

                home.set_target_media_type("video")
                home.open_file()
                self.assertIn("视频文件", dialog.call_args.args[3])

            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "portrait.png"
                cv2.imencode(".png", np.full((80, 40, 3), 160, dtype=np.uint8))[1].tofile(image)
                self.assertTrue(home.load_as_picture(str(image)))
                self.app.processEvents()
                self.assertFalse(home.setting_interface.video_mask_strategy_combo.isVisible())
                self.assertFalse(home.video_display_component.control_container.isVisible())

                video = Path(directory) / "sample.avi"
                self._create_video(video)
                self.assertTrue(home.load_video(str(video)))
                self.app.processEvents()
                self.assertTrue(home.setting_interface.video_dynamic_ocr_switch.isVisible())
                self.assertFalse(home.setting_interface.video_mask_strategy_combo.isVisible())
                self.assertTrue(home.video_display_component.control_container.isVisible())
                # Windows 下 VideoCapture 会独占文件；重新加载图片应释放旧视频句柄。
                self.assertTrue(home.load_as_picture(str(image)))
        finally:
            window.close()
            config.set(config.videoDynamicOcr, previous_dynamic_ocr)
            config.set(config.dynamicOcrRefineMask, previous_refine_mask)
            config.set(config.videoMaskStrategy, previous_strategy)

    def test_switching_to_video_initializes_dynamic_ocr_controls_from_saved_configuration(self):
        """已保存动态 OCR 偏好时，首次切视频不能遗留静态颜色筛选面板。"""
        previous_dynamic_ocr = config.videoDynamicOcr.value
        previous_refine_mask = config.dynamicOcrRefineMask.value
        previous_strategy = config.videoMaskStrategy.value
        window = None
        try:
            config.set(config.videoDynamicOcr, True)
            config.set(config.videoMaskStrategy, "every_frame_ocr")
            config.set(config.dynamicOcrRefineMask, False)
            window = CleanCanvasWindow()
            window.show()
            self.app.processEvents()
            home = window.homeInterface

            # 窗口初始为图片类型；保存的动态 OCR 偏好不应提前影响图片编辑区。
            self.assertTrue(home.static_mask_actions_widget.isVisible())
            home.set_target_media_type("video")
            self.app.processEvents()

            self.assertTrue(home.setting_interface.video_mask_strategy_combo.isVisible())
            self.assertTrue(home.setting_interface.dynamic_ocr_refine_switch.isVisible())
            self.assertFalse(home.static_mask_actions_widget.isVisible())
            self.assertFalse(home.color_tool_widget.isVisible())
            self.assertFalse(home.dynamic_ocr_actions_widget.isVisible())
        finally:
            if window is not None:
                window.close()
            config.set(config.videoDynamicOcr, previous_dynamic_ocr)
            config.set(config.dynamicOcrRefineMask, previous_refine_mask)
            config.set(config.videoMaskStrategy, previous_strategy)

    def test_confirm_empty_regions_uses_existing_component_api_without_crashing(self):
        """OCR 没有候选框时，确认动作应提示用户，而不是访问不存在的接口。"""
        window = CleanCanvasWindow()
        try:
            home = window.homeInterface
            home.task_list_component.add_task("placeholder.png")
            home.task_list_component.current_task_index = 0
            with patch.object(home.video_display_component, "get_selection_rects", return_value=[]) as get_regions:
                home.confirm_current_regions()
            get_regions.assert_called_once_with()
        finally:
            window.close()

    def test_editing_mask_is_not_presented_as_confirmed_and_image_hides_ab_actions(self):
        """红色只表示已确认最终 mask；图片模式没有 A/B 起点菜单。"""
        previous_mode = config.maskMode.value
        window = CleanCanvasWindow()
        try:
            home = window.homeInterface
            home.task_list_component.add_task("placeholder.png")
            home.task_list_component.current_task_index = 0
            home.current_frame = np.full((24, 32, 3), 128, dtype=np.uint8)

            config.set(config.maskMode, "rectangle")
            with patch.object(home, "_show_mask_preview") as show_preview, patch.object(home, "update_preview") as update_preview:
                home._refresh_editing_mask_preview()
            show_preview.assert_not_called()
            update_preview.assert_called_once_with(home.current_frame, store_current=False)

            config.set(config.maskMode, "color")
            with patch.object(home, "_show_mask_preview") as show_preview:
                home._refresh_editing_mask_preview()
            self.assertFalse(show_preview.call_args.kwargs["confirmed"])

            home.set_target_media_type("image")
            component = home.video_display_component
            component._VideoDisplayComponent__update_context_menu_actions()
            self.assertFalse(component.action_mark_ab_start.isVisible())
            self.assertFalse(component.action_mark_ab_end.isVisible())
            self.assertFalse(component.action_mark_ab_delete.isVisible())
        finally:
            window.close()
            config.set(config.maskMode, previous_mode)

    def test_content_repair_switches_to_matching_task_and_keeps_drafts_isolated(self):
        """图片草稿只能随图片任务恢复，切换到视频绝不能残留在预览上。"""
        previous_mode = config.maskMode.value
        config.set(config.maskMode, "rectangle")
        window = CleanCanvasWindow()
        try:
            home = window.homeInterface
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "source.png"
                cv2.imencode(".png", np.full((80, 40, 3), 160, dtype=np.uint8))[1].tofile(image)
                video = Path(directory) / "source.avi"
                self._create_video(video)
                image_index = len(home.task_list_component.tasks)
                home.task_list_component.add_task(str(image))
                video_index = len(home.task_list_component.tasks)
                home.task_list_component.add_task(str(video))

                home.task_list_component.select_task(image_index)
                self.app.processEvents()
                image_draft = [(0.1, 0.4, 0.2, 0.7)]
                home.video_display_component.set_selection_rects(image_draft)
                home.selections_changed(image_draft)
                home.set_target_media_type("video")
                self.app.processEvents()
                self.assertEqual(home.task_list_component.get_current_task_index(), video_index)
                self.assertEqual(home._target_media_type, "video")
                self.assertEqual(home.video_display_component.get_selection_rects(), [])
                self.assertEqual(home.current_frame.shape[:2], (32, 48))

                home.set_target_media_type("image")
                self.app.processEvents()
                self.assertEqual(home.task_list_component.get_current_task_index(), image_index)
                self.assertEqual(home.video_display_component.get_selection_rects(), image_draft)
        finally:
            window.close()
            config.set(config.maskMode, previous_mode)

    def test_video_range_controls_create_closed_interval_and_can_clear(self):
        """处理区间需有显式起点、终点；反向选择也应得到合法的闭区间。"""
        window = CleanCanvasWindow()
        try:
            home = window.homeInterface
            component = home.video_display_component
            home.set_target_media_type("video")
            component.set_video_controls_visible(True)
            component.video_slider.setRange(1, 10)
            component.video_slider.setValue(8)
            self.assertTrue(component.mark_ab_start())
            component.video_slider.setValue(3)
            self.assertTrue(component.mark_ab_end())
            sections = component.get_ab_sections()
            self.assertEqual([(section.start, section.stop) for section in sections], [(2, 8)])
            component.clear_ab_sections()
            self.assertEqual(component.get_ab_sections(), [])
            component._VideoDisplayComponent__update_context_menu_actions()
            self.assertTrue(component.action_mark_ab_start.isVisible())
            self.assertFalse(component.action_mark_ab_end.isVisible())
            self.assertFalse(component.action_clear_ab_sections.isVisible())
        finally:
            window.close()

    def test_color_filter_confirmation_freezes_visible_nonempty_mask(self):
        """颜色筛选确认后，任务计划必须有可重建的非空像素 mask。"""
        previous = (
            config.maskMode.value, config.colorMaskForeground.value,
            config.colorMaskBackground.value, config.colorMaskTolerance.value,
        )
        config.set(config.maskMode, "color")
        config.set(config.colorMaskForeground, "#000000")
        config.set(config.colorMaskBackground, "#FFFFFF")
        config.set(config.colorMaskTolerance, 8)
        window = CleanCanvasWindow()
        try:
            home = window.homeInterface
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "ink.png"
                pixels = np.full((80, 80, 3), 255, dtype=np.uint8)
                pixels[20:40, 20:40] = 0
                cv2.imencode(".png", pixels)[1].tofile(image)
                index = len(home.task_list_component.tasks)
                home.task_list_component.add_task(str(image))
                home.task_list_component.select_task(index)
                self.app.processEvents()
                roi = [(0.1, 0.7, 0.1, 0.7)]
                home.video_display_component.set_selection_rects(roi)
                home.selections_changed(roi)
                home.confirm_current_regions()
                raw_plan = home.task_list_component.get_task_option(index, TaskOptions.MASK_PLAN)
                self.assertIsNotNone(raw_plan)
                mask = build_mask(home.current_frame, MaskPlan.from_dict(raw_plan))
                self.assertGreater(int(np.count_nonzero(mask)), 0)
                self.assertEqual(int(mask[30, 30]), 255)
                # 确认后临时 ROI 会收起，红色冻结预览不会被橙色草稿覆盖。
                self.assertEqual(home.video_display_component.get_selection_rects(), [])
        finally:
            window.close()
            config.set(config.maskMode, previous[0])
            config.set(config.colorMaskForeground, previous[1])
            config.set(config.colorMaskBackground, previous[2])
            config.set(config.colorMaskTolerance, previous[3])

    def test_content_repair_no_longer_exposes_ocr_candidate_button(self):
        previous_mode = config.maskMode.value
        window = CleanCanvasWindow()
        try:
            window.show()
            self.app.processEvents()
            home = window.homeInterface
            for mode in ("rectangle", "color", "brush"):
                config.set(config.maskMode, mode)
                self.app.processEvents()
                self.assertFalse(home.detect_ocr_button.isVisible())
        finally:
            window.close()
            config.set(config.maskMode, previous_mode)

    def test_sampled_task_snapshot_strips_dynamic_ocr_refinement_before_process(self):
        options = {
            TaskOptions.VIDEO_DYNAMIC_OCR.value: True,
            TaskOptions.VIDEO_MASK_STRATEGY.value: "sampled_ocr",
            TaskOptions.DYNAMIC_OCR_REFINE_MASK.value: True,
        }
        self.assertTrue(
            HomeInterface._strip_refine_from_sampled_dynamic_ocr(
                options, is_video=True,
            )
        )
        self.assertFalse(options[TaskOptions.DYNAMIC_OCR_REFINE_MASK.value])

    def test_advanced_settings_group_parameters_by_scope_and_expose_help(self):
        """高级页不应再把历史字幕参数伪装成当前流程仍会生效的配置。"""
        window = CleanCanvasWindow()
        try:
            advanced = window.advancedSettingInterface
            home = window.homeInterface
            self.assertEqual(advanced.help_group.titleLabel.text(), "帮助")
            self.assertEqual(advanced.ocr_group.titleLabel.text(), "OCR 候选检测")
            self.assertEqual(advanced.mask_group.titleLabel.text(), "精确掩膜")
            self.assertEqual(advanced.video_group.titleLabel.text(), "视频采样与时序")
            self.assertIn("最终掩膜", advanced.mask_expand_pixels.titleLabel.text())
            self.assertEqual(advanced.project_help.titleLabel.text(), "项目使用与参数指南")
            # 动态 OCR 精细定位属于内容修复主页面；高级页不再堆放图片任务
            # 永远不会生效的参数。颜色则通过可见色块表达。
            self.assertFalse(hasattr(advanced, "dynamic_ocr_foreground"))
            self.assertIn("background-color:", home.dynamic_foreground_swatch.styleSheet())
            guide = advanced._help_document()
            self.assertIn("内容修复", guide)
            self.assertIn("画质增强", guide)
            self.assertIn("最终掩膜扩边", guide)
        finally:
            window.close()

    def test_enhance_media_type_switch_and_video_slider(self):
        window = CleanCanvasWindow()
        try:
            window.resize(1360, 720)
            window.show()
            window.switchTo(window.enhanceInterface)
            QTest.qWait(80)
            self.app.processEvents()
            page = window.enhanceInterface
            self.assertEqual(page.media_type_selector.currentRouteKey(), "image")
            self.assertFalse(hasattr(page.setting_interface, "video_note"))
            page.set_target_media_type("video")
            self.app.processEvents()
            self.assertEqual(page.media_type_selector.currentRouteKey(), "video")
            self.assertIn("按帧增强并尝试回封装原音频", page.output_text.toPlainText())
            self.assertTrue(page.preview_component.video_slider.isVisible())
            self.assertEqual(page.preview_component.video_slider.value(), 1)
            self.assertEqual(page.preview_component.video_slider.maximum(), 100)

            with tempfile.TemporaryDirectory() as directory:
                video = Path(directory) / "sample.avi"
                self._create_video(video)
                index = page.task_list.add_task(str(video), "realesrgan_anime", 2.0, "cpu")
                page.task_list.select_task(index)
                self.app.processEvents()
                self.assertTrue(page.preview_component.video_slider.isVisible())
                self.assertEqual(page.preview_component.video_slider.minimum(), 1)
                self.assertEqual(page.preview_component.video_slider.maximum(), 3)
                self.assertEqual(page.preview_component.video_slider.value(), 1)
                page.preview_component.video_slider.setValue(3)
                QTest.qWait(20)
                self.app.processEvents()
                self.assertFalse(page.preview_component.title_label.isVisible())
                self.assertIsNotNone(page.preview_component._source_pixmap)
                page._release_preview_capture()
        finally:
            window.close()

    def test_enhance_switches_to_matching_task_or_clears_stale_preview(self):
        """增强页切换类型时不能把图片预览遗留给视频任务。"""
        window = CleanCanvasWindow()
        try:
            window.show()
            window.switchTo(window.enhanceInterface)
            QTest.qWait(80)
            page = window.enhanceInterface
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "source.png"
                cv2.imencode(".png", np.full((80, 40, 3), 160, dtype=np.uint8))[1].tofile(image)
                video = Path(directory) / "source.avi"
                self._create_video(video)
                image_index = page.task_list.add_task(str(image), "realesrgan_anime", 2.0, "cpu")
                video_index = page.task_list.add_task(str(video), "realesrgan_anime", 2.0, "cpu")

                page.task_list.select_task(image_index)
                self.app.processEvents()
                self.assertIsNotNone(page.preview_component._source_pixmap)

                page.set_target_media_type("video")
                self.app.processEvents()
                self.assertEqual(page.task_list.get_current_task_index(), video_index)
                self.assertEqual(page._target_media_type, "video")
                self.assertTrue(page.preview_component.video_slider.isVisible())
                self.assertEqual(page.preview_component.video_slider.maximum(), 3)

                # 移除唯一视频任务后再次切换类型，图片预览必须被明确清空。
                page.task_list.delete_task(video_index)
                page.set_target_media_type("image")
                page.set_target_media_type("video")
                self.app.processEvents()
                self.assertEqual(page.task_list.get_current_task_index(), -1)
                self.assertIsNone(page.preview_component._source_pixmap)
                self.assertEqual(page.preview_component.display.text(), "")
                page._release_preview_capture()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
