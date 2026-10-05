"""图片转换与 ICO 多尺寸输出的离线回归测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui import SubtitleExtractorGUI
from ui.image_format_interface import ImageFormatInterface


class ImageFormatInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_converter_writes_common_formats_and_multisize_ico(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            rgba = Image.new("RGBA", (40, 30), (220, 30, 50, 0))
            rgba.paste((10, 150, 210, 255), (0, 0, 20, 30))
            rgba.save(source)

            for output_format, suffix in (("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp"), ("BMP", ".bmp"), ("TIFF", ".tiff")):
                target = root / f"converted{suffix}"
                ImageFormatInterface._convert_file(source, target, output_format, 90, [])
                self.assertTrue(target.is_file(), output_format)
                with Image.open(target) as converted:
                    self.assertEqual(converted.size, (40, 30))

            icon = root / "converted.ico"
            ImageFormatInterface._convert_file(source, icon, "ICO", 95, [16, 32, 64])
            self.assertTrue(icon.is_file())
            with Image.open(icon) as converted_icon:
                self.assertTrue({(16, 16), (32, 32), (64, 64)}.issubset(converted_icon.info.get("sizes", set())))

    def test_new_page_toggles_quality_and_ico_options(self):
        window = SubtitleExtractorGUI()
        try:
            page = window.imageFormatInterface
            window.show()
            window.switchTo(page)
            QTest.qWait(30)
            self.app.processEvents()
            self.assertEqual(page.format_combo.currentText(), "PNG")
            self.assertEqual(page.preview.display.text(), "")
            self.assertFalse(page.quality_card.isVisible())
            self.assertFalse(page.ico_size_card.isVisible())

            long_path = Path(r"C:\Pictures\a_very_long_source_image_name_for_clean_canvas_studio_conversion_preview.png")
            page._source_path = long_path
            page._source_label_text = long_path.name
            page._update_source_label()
            self.assertEqual(page.source_label.toolTip(), str(long_path))
            self.assertNotIn("\n", page.source_label.text())
            self.assertLessEqual(
                page.source_label.fontMetrics().horizontalAdvance(page.source_label.text()),
                page.source_label.contentsRect().width(),
            )

            page.format_combo.setCurrentText("JPEG")
            self.app.processEvents()
            self.assertTrue(page.quality_card.isVisible())
            self.assertFalse(page.ico_size_card.isVisible())

            page.format_combo.setCurrentText("ICO")
            self.app.processEvents()
            self.assertFalse(page.quality_card.isVisible())
            self.assertTrue(page.ico_size_card.isVisible())
            self.assertTrue(all(check.isChecked() for check in page._size_checks.values()))
        finally:
            window.close()

    def test_ctrl_c_no_longer_closes_main_window(self):
        window = SubtitleExtractorGUI()
        try:
            window.show()
            QTest.keyClick(window, Qt.Key_C, Qt.ControlModifier)
            self.app.processEvents()
            self.assertTrue(window.isVisible())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
