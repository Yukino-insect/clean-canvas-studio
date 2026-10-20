"""画质增强页的任务、日志和输入/输出预览回归测试。"""

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
from ui.component.enhance_task_list_component import EnhanceTaskStatus


class EnhanceQueueUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_task_selection_completion_preview_and_log(self):
        window = CleanCanvasWindow()
        try:
            window.resize(1108, 758)
            window.show()
            window.switchTo(window.enhanceInterface)
            QTest.qWait(80)
            self.app.processEvents()
            page = window.enhanceInterface
            self.assertGreater(page.preview_component.width(), page.setting_interface.parentWidget().width())
            self.assertEqual(page.task_list.table.columnCount(), 3)
            self.assertEqual(page.setting_interface.model_combo.__class__.__name__, "ComboBoxSettingCard")
            self.assertFalse(page.preview_component.title_label.isVisible())
            self.assertEqual(page.preview_component.display.text(), "")
            self.assertEqual(page.add_button.text(), "打开")
            self.assertEqual(page.run_button.text(), "运行")
            self.assertEqual(page.setting_interface.device_combo.comboBox.currentText(), "自动")
            for card in (page.setting_interface.model_combo, page.setting_interface.scale_combo, page.setting_interface.device_combo):
                self.assertGreaterEqual(
                    card.contentLabel.width(),
                    card.contentLabel.fontMetrics().horizontalAdvance(card.contentLabel.text()),
                    card.titleLabel.text(),
                )

            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.png"
                output = Path(directory) / "source_enhanced_x2.png"
                cv2.imencode(".png", np.full((30, 40, 3), 160, dtype=np.uint8))[1].tofile(source)
                index = page.task_list.add_task(str(source), "realesrgan_anime", 2.0, "cpu")
                page.task_list.select_task(index)
                self.app.processEvents()
                self.assertIsNotNone(page.preview_component._source_pixmap)

                cv2.imencode(".png", np.full((60, 80, 3), 180, dtype=np.uint8))[1].tofile(output)
                page._current_processing_index = index
                page._on_complete(index, str(output))
                self.app.processEvents()

                self.assertIs(page.task_list.get_task(index).status, EnhanceTaskStatus.COMPLETED)
                self.assertFalse(page.preview_component.title_label.isVisible())
                self.assertEqual(page.preview_component.title_label.text(), "")
                # 原图 40×30 被缩放为输出的 80×60；两张图并排并以 4px
                # 分隔线分开，确保完成后的增强预览不再只显示输出图。
                self.assertEqual(page.preview_component._source_pixmap.size().width(), 164)
                self.assertEqual(page.preview_component._source_pixmap.size().height(), 60)
                divider_color = page.preview_component._source_pixmap.toImage().pixelColor(80, 20)
                self.assertEqual(divider_color.name().lower(), "#94a3b8")
                self.assertIn("任务完成", page.output_text.toPlainText())

                portrait = np.zeros((960, 540, 3), dtype=np.uint8)
                page.preview_component.set_frame(portrait, "竖屏预览")
                QTest.qWait(40)
                self.app.processEvents()
                self.assertGreater(page.preview_component.display.height(), page.preview_component.display.width())
        finally:
            window.close()

    def test_add_tasks_opens_dialog_with_enhance_page_as_parent(self):
        """PySide6 文件对话框的第一个位置参数必须是 QWidget 父对象。"""
        window = CleanCanvasWindow()
        try:
            page = window.enhanceInterface
            with patch("ui.enhance_interface.QtWidgets.QFileDialog.getOpenFileNames", return_value=([], "")) as dialog:
                page.set_target_media_type("image")
                page.add_media_tasks()
                image_args = dialog.call_args.args
                self.assertIs(image_args[0], page)
                self.assertIn("图片文件", image_args[3])

                page.set_target_media_type("video")
                self.assertFalse(hasattr(page.setting_interface, "video_note"))
                self.assertIn("按帧增强并尝试回封装原音频", page.output_text.toPlainText())
                self.assertIn("可能出现轻微闪烁", page.output_text.toPlainText())
                page.add_media_tasks()
                video_args = dialog.call_args.args
                self.assertIs(video_args[0], page)
                self.assertIn("视频文件", video_args[3])
        finally:
            window.close()

    def test_video_progress_keeps_shared_preview_timeline_visible(self):
        """增强视频的运行进度应复用内容修复同款时间轴，而不是收到预览即隐藏。"""
        window = CleanCanvasWindow()
        try:
            page = window.enhanceInterface
            page.task_list.add_task("sample.mp4", "realesrgan_anime", 2.0, "cpu")
            page._active_preview_task_index = 0
            page.preview_component.set_video_controls_visible(True, frame_count=101, frame_index=0)
            page._on_progress(0, 50, False)
            # 本用例无需展示顶层窗口；检查控制栏没有被业务逻辑显式隐藏。
            self.assertFalse(page.preview_component.control_container.isHidden())
            self.assertEqual(page.preview_component.video_slider.value(), 51)
        finally:
            window.close()

    def test_queue_runs_multiple_tasks_without_loading_real_model(self):
        class FakeEnhancer:
            run_counts = {}

            def __init__(self, media_path, **_kwargs):
                self.media_path = media_path
                self.progress = []
                self.logs = []
                self.previews = []

            def add_progress_listener(self, listener):
                self.progress.append(listener)

            def add_log_listener(self, listener):
                self.logs.append(listener)

            def add_preview_listener(self, listener):
                self.previews.append(listener)

            def cancel(self):
                pass

            def run(self):
                self.run_counts[self.media_path] = self.run_counts.get(self.media_path, 0) + 1
                image = cv2.imread(self.media_path)
                for listener in self.logs:
                    listener(f"假任务处理：{Path(self.media_path).name}")
                for listener in self.progress:
                    listener(50, False)
                for listener in self.previews:
                    listener(image, image)
                for listener in self.progress:
                    listener(100, True)
                return self.media_path

        window = CleanCanvasWindow()
        try:
            window.resize(1108, 758)
            window.show()
            window.switchTo(window.enhanceInterface)
            page = window.enhanceInterface
            with tempfile.TemporaryDirectory() as directory:
                first = Path(directory) / "first.png"
                second = Path(directory) / "second.png"
                cv2.imencode(".png", np.full((20, 20, 3), 100, dtype=np.uint8))[1].tofile(first)
                cv2.imencode(".png", np.full((20, 20, 3), 200, dtype=np.uint8))[1].tofile(second)
                page.task_list.add_task(str(first), "realesrgan_anime", 2.0, "cpu")
                page.task_list.add_task(str(second), "realesrgan_anime", 2.0, "cpu")
                with patch("ui.enhance_interface.MediaEnhancer", FakeEnhancer):
                    page.start_queue()
                    for _ in range(30):
                        QTest.qWait(50)
                        self.app.processEvents()
                        if not page._queue_thread or not page._queue_thread.is_alive():
                            break
                    # worker 结束只表示它不再发消息；complete/status 信号仍可能
                    # 正排在 GUI 事件队列中，需处理最后一批 queued callbacks。
                    QTest.qWait(100)
                    self.app.processEvents()

                self.assertEqual([task.status for task in page.task_list.tasks], [EnhanceTaskStatus.COMPLETED] * 2)
                self.assertEqual(FakeEnhancer.run_counts, {str(first): 1, str(second): 1})
                self.assertIn("画质增强队列已完成", page.output_text.toPlainText())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
