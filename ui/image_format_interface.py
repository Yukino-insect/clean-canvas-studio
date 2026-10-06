"""本地图片格式转换与多尺寸 ICO 制作页面。"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtWidgets
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CheckBox,
    ComboBox,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    Slider,
    TextEdit,
)

from ui.component.media_preview_component import MediaPreviewComponent


class ImageFormatInterface(QtWidgets.QWidget):
    """常见位图格式转换与多尺寸 ICO 制作页面。

    Pillow 的文件读写放在后台线程中，避免大图编码阻塞 Qt 事件循环；完成
    信号只携带元数据，由主线程负责更新预览、日志和 InfoBar。
    """

    # result 为输出路径与图像属性字典；error 为转换异常。二者互斥。
    conversion_finished = QtCore.Signal(object, object)

    _FORMAT_SUFFIXES = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tiff",
        "ICO": ".ico",
    }
    _SAVE_FILTERS = {
        "PNG": "PNG 图片 (*.png)",
        "JPEG": "JPEG 图片 (*.jpg *.jpeg)",
        "WEBP": "WebP 图片 (*.webp)",
        "BMP": "BMP 图片 (*.bmp)",
        "TIFF": "TIFF 图片 (*.tiff *.tif)",
        "ICO": "Windows 图标 (*.ico)",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageFormatInterface")
        self._source_path: Path | None = None
        self._source_label_text = "尚未选择文件"
        self._working = False
        self._size_checks: dict[int, CheckBox] = {}
        self._create_widgets()
        self.conversion_finished.connect(self._on_conversion_finished)

    def _create_widgets(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(8)
        self.preview = MediaPreviewComponent("图片转换预览", self, placeholder="")
        left_layout.addWidget(self.preview, 2)
        self.log = TextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(165)
        self.log.document().setDocumentMargin(10)
        log_card = CardWidget(self)
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self.log)
        left_layout.addWidget(log_card, 1)
        layout.addLayout(left_layout, 2)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(10)
        source_card = CardWidget(self)
        source_layout = QtWidgets.QVBoxLayout(source_card)
        source_layout.setContentsMargins(16, 14, 16, 14)
        source_layout.setSpacing(5)
        source_layout.addWidget(BodyLabel("源图片", source_card))
        self.source_label = BodyLabel("尚未选择文件", source_card)
        self.source_label.setWordWrap(False)
        self.source_label.setStyleSheet("color: #667085;")
        source_layout.addWidget(self.source_label)
        right_layout.addWidget(source_card)

        self.format_card = self._make_setting_card(
            "输出格式",
            "选择常见图片格式，转换结果将另存为新文件。",
        )
        self.format_combo = ComboBox(self.format_card)
        self.format_combo.addItems(list(self._FORMAT_SUFFIXES))
        self.format_combo.setCurrentText("PNG")
        self._setting_control_layout(self.format_card).addWidget(self.format_combo)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        right_layout.addWidget(self.format_card)

        self.quality_card = self._make_setting_card(
            "输出质量",
            "仅 JPEG 和 WebP 使用；数值越高，文件通常越大。",
        )
        quality_layout = self._setting_control_layout(self.quality_card)
        self.quality_value = BodyLabel("95", self.quality_card)
        self.quality_slider = Slider(QtCore.Qt.Horizontal, self.quality_card)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.valueChanged.connect(lambda value: self.quality_value.setText(str(value)))
        quality_layout.addWidget(self.quality_value)
        quality_layout.addWidget(self.quality_slider, 1)
        right_layout.addWidget(self.quality_card)

        self.ico_size_card = CardWidget(self)
        ico_layout = QtWidgets.QVBoxLayout(self.ico_size_card)
        ico_layout.setContentsMargins(16, 14, 16, 14)
        ico_layout.setSpacing(8)
        ico_layout.addWidget(BodyLabel("ICO 图标尺寸", self.ico_size_card))
        ico_note = BodyLabel("一个 ICO 文件可同时包含多个尺寸；Windows 会自动选择合适的版本。", self.ico_size_card)
        ico_note.setWordWrap(True)
        ico_note.setStyleSheet("color: #667085;")
        ico_layout.addWidget(ico_note)
        size_row = QtWidgets.QHBoxLayout()
        size_row.setSpacing(6)
        for size in (16, 32, 48, 64, 128, 256):
            check = CheckBox(f"{size}px", self.ico_size_card)
            check.setChecked(True)
            self._size_checks[size] = check
            size_row.addWidget(check)
        ico_layout.addLayout(size_row)
        right_layout.addWidget(self.ico_size_card)

        info_card = CardWidget(self)
        info_layout = QtWidgets.QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 14, 16, 14)
        info_layout.setSpacing(5)
        info_layout.addWidget(BodyLabel("转换说明", info_card))
        self.info_label = BodyLabel("PNG、JPEG、WebP、BMP、TIFF 可互相转换。JPEG 和 BMP 不保留透明通道，透明区域会填充为白色。", info_card)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #667085;")
        info_layout.addWidget(self.info_label)
        right_layout.addWidget(info_card)
        right_layout.addStretch(1)

        action_card = CardWidget(self)
        action_layout = QtWidgets.QHBoxLayout(action_card)
        action_layout.setContentsMargins(16, 16, 16, 16)
        action_layout.setSpacing(8)
        self.open_button = PushButton("打开图片", action_card)
        self.open_button.setIcon(FluentIcon.FOLDER)
        self.open_button.clicked.connect(self.open_source)
        action_layout.addWidget(self.open_button)
        self.convert_button = PrimaryPushButton("转换并保存", action_card)
        self.convert_button.setIcon(FluentIcon.SAVE_AS)
        self.convert_button.clicked.connect(self.convert_and_save)
        action_layout.addWidget(self.convert_button)
        action_layout.addStretch(1)
        right_layout.addWidget(action_card)
        layout.addLayout(right_layout, 1)
        self._on_format_changed(self.format_combo.currentText())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 右侧面板随窗口尺寸变化；延迟到布局完成后重新计算中间省略文本。
        QtCore.QTimer.singleShot(0, self._update_source_label)

    @staticmethod
    def _make_setting_card(title: str, description: str) -> CardWidget:
        card = CardWidget()
        layout = QtWidgets.QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.addWidget(BodyLabel(title, card))
        content = BodyLabel(description, card)
        content.setWordWrap(True)
        content.setStyleSheet("color: #667085;")
        text_layout.addWidget(content)
        layout.addLayout(text_layout, 1)
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        card._conversion_control_layout = control_layout
        layout.addLayout(control_layout)
        return card

    @staticmethod
    def _setting_control_layout(card: CardWidget) -> QtWidgets.QHBoxLayout:
        return card._conversion_control_layout

    def _on_format_changed(self, output_format: str):
        """根据目标格式切换质量控制与 ICO 多尺寸选项。"""
        is_ico = output_format == "ICO"
        supports_quality = output_format in {"JPEG", "WEBP"}
        self.ico_size_card.setVisible(is_ico)
        self.quality_card.setVisible(supports_quality)
        if is_ico:
            self.info_label.setText("ICO 可包含多个尺寸。勾选需要的尺寸后，程序会在一个 .ico 文件中写入所有选中的图标版本。")
        elif output_format in {"JPEG", "BMP"}:
            self.info_label.setText(f"{output_format} 不保留透明通道；透明区域会填充为白色。转换结果会另存，不覆盖源文件。")
        else:
            self.info_label.setText(f"输出为 {output_format} 图片。转换结果会另存，不覆盖源文件。")

    def open_source(self):
        """选择源图片并只预览源文件，避免格式转换被误认为视觉处理。"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.ico);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            frame = self._read_bgr(path)
        except Exception as exc:
            InfoBar.error("无法打开图片", str(exc), parent=self, position=InfoBarPosition.TOP, duration=4500)
            return
        self._source_path = Path(path)
        self._source_label_text = self._source_path.name
        self._update_source_label()
        self.preview.set_frame(frame, "图片转换预览")
        metadata = self._image_metadata(self._source_path)
        self._append_log(
            f"已打开图片：{self._source_path.name}。\n"
            f"输入属性：{self._format_metadata(metadata)}。"
        )

    def convert_and_save(self):
        """请求目标路径后在后台线程执行编码，源文件始终保持不变。"""
        if self._working:
            return
        if self._source_path is None:
            InfoBar.warning("请先打开图片", "选择一张图片后再进行格式转换。", parent=self, position=InfoBarPosition.TOP, duration=3500)
            return
        output_format = self.format_combo.currentText()
        suffix = self._FORMAT_SUFFIXES[output_format]
        suggested = self._source_path.with_name(f"{self._source_path.stem}_converted{suffix}")
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存转换后的图片",
            str(suggested),
            self._SAVE_FILTERS[output_format],
        )
        if not target:
            return
        target_path = Path(target).with_suffix(suffix)
        ico_sizes = sorted(size for size, check in self._size_checks.items() if check.isChecked())
        if output_format == "ICO" and not ico_sizes:
            InfoBar.warning("请选择 ICO 尺寸", "至少勾选一个图标尺寸。", parent=self, position=InfoBarPosition.TOP, duration=3500)
            return
        quality = self.quality_slider.value()
        self._set_working(True)
        self._append_log(f"开始转换：{self._source_path.name} → {target_path.name}（{output_format}）。")

        def work():
            try:
                source_metadata = self._image_metadata(self._source_path)
                self._convert_file(self._source_path, target_path, output_format, quality, ico_sizes)
                self.conversion_finished.emit(
                    {
                        "path": str(target_path),
                        "format": output_format,
                        "ico_sizes": ico_sizes,
                        "source_metadata": source_metadata,
                        "output_metadata": self._image_metadata(target_path),
                    },
                    None,
                )
            except Exception as exc:  # GUI 线程只负责显示错误，转换线程不直接碰控件。
                self.conversion_finished.emit(None, exc)

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _convert_file(source: Path, target: Path, output_format: str, quality: int, ico_sizes: list[int]):
        """执行一次纯文件转换。

        JPEG/BMP 以白色背景扁平化 alpha；ICO 则保持透明并把所有选中尺寸
        写入同一个文件。该方法不得访问 Qt 控件，因此可安全在线程中调用。
        """
        with Image.open(source) as opened:
            image = opened.copy()
        target.parent.mkdir(parents=True, exist_ok=True)
        rgba = image.convert("RGBA")
        if output_format == "ICO":
            # Pillow 不会自动把较小的源图扩展为更大的 ICO 子图。先制作最大
            # 尺寸的正方形透明画布，再让 Pillow 从它导出所有选中的标准尺寸。
            # 保持原始宽高比，避免把非正方形图片硬拉伸成图标。
            largest = max(ico_sizes)
            icon_canvas = Image.new("RGBA", (largest, largest), (0, 0, 0, 0))
            icon_content = rgba.copy()
            icon_content.thumbnail((largest, largest), Image.Resampling.LANCZOS)
            offset = ((largest - icon_content.width) // 2, (largest - icon_content.height) // 2)
            icon_canvas.alpha_composite(icon_content, dest=offset)
            icon_canvas.save(target, format="ICO", sizes=[(size, size) for size in ico_sizes])
            return
        if output_format == "JPEG":
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            background.save(target, format="JPEG", quality=quality, optimize=True)
            return
        if output_format == "WEBP":
            rgba.save(target, format="WEBP", quality=quality, method=6)
            return
        if output_format == "BMP":
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            background.save(target, format="BMP")
            return
        if output_format == "TIFF":
            rgba.save(target, format="TIFF")
            return
        rgba.save(target, format="PNG")

    @staticmethod
    def _read_bgr(path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return rgb[:, :, ::-1].copy()

    @staticmethod
    def _image_metadata(path: str | Path) -> dict:
        """读取用于日志展示的文件属性，不进行像素修改。"""
        image_path = Path(path)
        with Image.open(image_path) as image:
            sizes = sorted(image.info.get("sizes", set())) if image.format == "ICO" else []
            return {
                "format": image.format or image_path.suffix.lstrip(".").upper(),
                "size": image.size,
                "mode": image.mode,
                "has_alpha": "A" in image.mode or "transparency" in image.info,
                "ico_sizes": sizes,
                "bytes": image_path.stat().st_size,
            }

    @staticmethod
    def _format_metadata(metadata: dict) -> str:
        width, height = metadata["size"]
        size_kib = metadata["bytes"] / 1024
        alpha = "，含透明通道" if metadata["has_alpha"] else ""
        ico_sizes = ""
        if metadata["ico_sizes"]:
            ico_sizes = "，ICO 尺寸：" + "、".join(f"{width}×{height}" for width, height in metadata["ico_sizes"])
        return (
            f"格式 {metadata['format']}，尺寸 {width}×{height}，色彩模式 {metadata['mode']}"
            f"{alpha}，文件 {size_kib:.1f} KiB{ico_sizes}"
        )

    @QtCore.Slot(object, object)
    def _on_conversion_finished(self, result, error):
        """消费后台结果并把可核对的输出属性写入日志。"""
        self._set_working(False)
        if error is not None:
            self._append_log(f"转换失败：{error}")
            InfoBar.error("图片转换失败", str(error), parent=self, position=InfoBarPosition.TOP, duration=5500)
            return
        target = Path(result["path"])
        # 图片格式转换通常没有肉眼可见的内容变化，因此预览继续保留源图片；
        # 将真正可核对的格式、尺寸、色彩模式和文件大小写入日志即可。
        ico_note = f"，包含 {', '.join(f'{size}px' for size in result['ico_sizes'])}" if result["format"] == "ICO" else ""
        alpha_note = "输出格式不保留透明通道，透明区域已填充白色。" if (
            result["format"] in {"JPEG", "BMP"} and result["source_metadata"]["has_alpha"]
        ) else ""
        self._append_log(
            f"转换完成：{target}{ico_note}。\n"
            f"输出属性：{self._format_metadata(result['output_metadata'])}。"
            + (f"\n{alpha_note}" if alpha_note else "")
        )
        InfoBar.success("图片转换完成", str(target), parent=self, position=InfoBarPosition.TOP, duration=4500)

    def _set_working(self, working: bool):
        self._working = working
        self.open_button.setEnabled(not working)
        self.convert_button.setEnabled(not working)
        self.format_combo.setEnabled(not working)
        self.quality_slider.setEnabled(not working)
        for check in self._size_checks.values():
            check.setEnabled(not working)

    def _update_source_label(self):
        """单行展示文件名；完整路径留在 tooltip，避免长路径挤高设置卡。"""
        if self._source_path is None:
            self.source_label.setText(self._source_label_text)
            self.source_label.setToolTip("")
            return
        width = max(1, self.source_label.contentsRect().width())
        text = self.source_label.fontMetrics().elidedText(
            self._source_label_text,
            QtCore.Qt.ElideMiddle,
            width,
        )
        self.source_label.setText(text)
        self.source_label.setToolTip(str(self._source_path))

    def _append_log(self, message: str):
        self.log.append(message)
