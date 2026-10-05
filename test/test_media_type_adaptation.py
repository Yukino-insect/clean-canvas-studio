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

from gui import SubtitleExtractorGUI


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
        window = SubtitleExtractorGUI()
        try:
            window.resize(1360, 720)
            window.show()
            QTest.qWait(80)
            self.app.processEvents()
            home = window.homeInterface
            self.assertFalse(hasattr(home.setting_interface, "interface_combo"))
            self.assertIsNotNone(window.advancedSettingInterface.interface_language)
            self.assertEqual(home.media_type_selector.currentRouteKey(), "image")
            self.assertFalse(home.setting_interface.video_mask_strategy_combo.isVisible())
            self.assertFalse(home.video_display_component.control_container.isVisible())

            # 内容修复页也必须像增强页一样有显式的图片/视频任务切换；它决定
            # 文件筛选与仅视频可用的时间轴配置，而不依赖用户猜测扩展名。
            home.set_target_media_type("video")
            self.app.processEvents()
            self.assertEqual(home.media_type_selector.currentRouteKey(), "video")
            self.assertTrue(home.setting_interface.video_mask_strategy_combo.isVisible())
            home.set_target_media_type("image")
            self.app.processEvents()
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
                self.assertTrue(home.setting_interface.video_mask_strategy_combo.isVisible())
                self.assertTrue(home.video_display_component.control_container.isVisible())
                # Windows 下 VideoCapture 会独占文件；重新加载图片应释放旧视频句柄。
                self.assertTrue(home.load_as_picture(str(image)))
        finally:
            window.close()

    def test_confirm_empty_regions_uses_existing_component_api_without_crashing(self):
        """OCR 没有候选框时，确认动作应提示用户，而不是访问不存在的接口。"""
        window = SubtitleExtractorGUI()
        try:
            home = window.homeInterface
            home.task_list_component.add_task("placeholder.png")
            home.task_list_component.current_task_index = 0
            with patch.object(home.video_display_component, "get_selection_rects", return_value=[]) as get_regions:
                home.confirm_current_regions()
            get_regions.assert_called_once_with()
        finally:
            window.close()

    def test_advanced_settings_group_parameters_by_scope_and_expose_help(self):
        """高级页不应再把历史字幕参数伪装成当前流程仍会生效的配置。"""
        window = SubtitleExtractorGUI()
        try:
            advanced = window.advancedSettingInterface
            self.assertEqual(advanced.help_group.titleLabel.text(), "帮助")
            self.assertEqual(advanced.ocr_group.titleLabel.text(), "OCR 候选检测")
            self.assertEqual(advanced.mask_group.titleLabel.text(), "精确掩膜")
            self.assertEqual(advanced.video_group.titleLabel.text(), "视频采样与时序")
            self.assertIn("最终掩膜", advanced.mask_expand_pixels.titleLabel.text())
            self.assertEqual(advanced.project_help.titleLabel.text(), "项目使用与参数指南")
            guide = advanced._help_document()
            self.assertIn("内容修复", guide)
            self.assertIn("画质增强", guide)
            self.assertIn("最终掩膜扩边", guide)
        finally:
            window.close()

    def test_enhance_media_type_switch_and_video_slider(self):
        window = SubtitleExtractorGUI()
        try:
            window.resize(1360, 720)
            window.show()
            window.switchTo(window.enhanceInterface)
            QTest.qWait(80)
            self.app.processEvents()
            page = window.enhanceInterface
            self.assertEqual(page.media_type_selector.currentRouteKey(), "image")
            self.assertFalse(page.setting_interface.video_note.isVisible())
            page.set_target_media_type("video")
            self.app.processEvents()
            self.assertEqual(page.media_type_selector.currentRouteKey(), "video")
            self.assertTrue(page.setting_interface.video_note.isVisible())

            with tempfile.TemporaryDirectory() as directory:
                video = Path(directory) / "sample.avi"
                self._create_video(video)
                index = page.task_list.add_task(str(video), "realesrgan_anime", 2.0, "cpu")
                page.task_list.select_task(index)
                self.app.processEvents()
                self.assertTrue(page.preview_component.video_slider.isVisible())
                self.assertEqual(page.preview_component.video_slider.maximum(), 2)
                page.preview_component.video_slider.setValue(2)
                QTest.qWait(20)
                self.app.processEvents()
                self.assertFalse(page.preview_component.title_label.isVisible())
                self.assertIsNotNone(page.preview_component._source_pixmap)
                page._release_preview_capture()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
