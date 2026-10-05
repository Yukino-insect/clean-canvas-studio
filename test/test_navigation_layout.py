"""原版内嵌导航展开不应改变顶层窗口几何或扭曲视频预览比例。"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui import SubtitleExtractorGUI


class NavigationLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_inline_navigation_toggle_keeps_window_size_and_preview_aspect_ratio(self):
        window = SubtitleExtractorGUI()
        try:
            window.resize(1280, 800)
            window.show()
            QTest.qWait(80)
            self.app.processEvents()

            component = window.homeInterface.video_display_component
            initial_width = window.width()
            self.assertEqual(window.navigationInterface.width(), 48)

            window.navigationInterface.toggle()
            QTest.qWait(400)
            self.app.processEvents()
            self.assertEqual(window.width(), initial_width)
            self.assertEqual(window.navigationInterface.width(), 322)

            window.navigationInterface.toggle()
            QTest.qWait(400)
            self.app.processEvents()
            self.assertEqual(window.width(), initial_width)
            self.assertEqual(window.navigationInterface.width(), 48)

            displayed_ratio = component.video_display.width() / component.video_display.height()
            self.assertAlmostEqual(displayed_ratio, 16 / 9, delta=0.03)

            # 说明文本和操作按钮不能反向挤压视频预览；在常见的非全屏宽度下，
            # 左侧内容区应保持预览优先的比例。
            window.resize(1108, 758)
            QTest.qWait(80)
            self.app.processEvents()
            settings_card = window.homeInterface.setting_interface.parentWidget()
            self.assertGreater(component.width(), settings_card.width())
            self.assertGreaterEqual(component.width(), 550)
            self.assertLessEqual(settings_card.width(), 450)
            action_buttons = (
                window.homeInterface.file_button,
                window.homeInterface.detect_ocr_button,
                window.homeInterface.confirm_regions_button,
                window.homeInterface.clear_regions_button,
                window.homeInterface.run_button,
            )
            for button in action_buttons:
                # 图标、左右边距和焦点边框预留 35px；自然宽度按钮文字不能被裁切。
                minimum_readable_width = button.fontMetrics().horizontalAdvance(button.text()) + 35
                self.assertGreaterEqual(button.width(), minimum_readable_width, button.text())
            setting_cards = (
                window.homeInterface.setting_interface.inpaint_mode_combo,
                window.homeInterface.setting_interface.device_combo,
                window.homeInterface.setting_interface.mask_mode_combo,
                window.homeInterface.setting_interface.video_mask_strategy_combo,
            )
            self.assertTrue(all(not card.contentLabel.wordWrap() for card in setting_cards))
            for card in setting_cards:
                if not card.isVisible():
                    continue
                # 截图中摘要被裁掉会让用户无法判断当前设置；布局后的标签必须
                # 至少容纳其全部可见文字，而不能依赖 tooltip 才看得到末尾。
                self.assertGreaterEqual(
                    card.contentLabel.width(),
                    card.contentLabel.fontMetrics().horizontalAdvance(card.contentLabel.text()),
                    card.titleLabel.text(),
                )
            self.assertEqual(window.homeInterface.setting_interface.device_combo.comboBox.currentText(), "自动")

            # 竖屏媒体应切换为竖屏逻辑画布和实际显示比例，而不是被先填黑边
            # 压缩到固定 16:9 图像中。
            portrait = np.zeros((960, 540, 3), dtype=np.uint8)
            stable_size = window.size()
            window.homeInterface.update_preview(portrait)
            QTest.qWait(40)
            self.app.processEvents()
            component = window.homeInterface.video_display_component
            self.assertGreater(component.video_preview_height, component.video_preview_width)
            self.assertGreater(component.video_display.height(), component.video_display.width())
            landscape = np.zeros((540, 960, 3), dtype=np.uint8)
            window.homeInterface.update_preview(landscape)
            QTest.qWait(40)
            self.app.processEvents()
            self.assertEqual(window.size(), stable_size)

            # 下拉框使用 Qt 为当前文本给出的自然宽度：长模型名不可截断，
            # “自动”等短选项也不应拥有固定的大块无意义留白。
            for card in setting_cards:
                natural = max(card.comboBox.minimumSizeHint().width(), card.comboBox.sizeHint().width())
                expected = min(card.comboBox.maximumWidth(), natural)
                self.assertEqual(card.comboBox.width(), expected)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
