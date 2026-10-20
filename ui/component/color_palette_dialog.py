"""A compact visual color picker used by content-repair mask settings.

It deliberately avoids making users type or remember RGB/HEX values.  The hue
and saturation plane is continuous, while the brightness slider keeps black,
white and muted text colours equally accessible.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class HueSaturationPlane(QtWidgets.QWidget):
    """Clickable HSV hue/saturation plane at a caller-controlled brightness."""

    colorChanged = QtCore.Signal(QtGui.QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 170)
        self.setCursor(QtCore.Qt.CrossCursor)
        self._hue = 0
        self._saturation = 0
        self._value = 255
        self._cache_size = QtCore.QSize()
        self._cache_value = -1
        self._image: QtGui.QImage | None = None

    def set_hsv(self, hue: int, saturation: int, value: int):
        self._hue = max(0, min(359, int(hue)))
        self._saturation = max(0, min(255, int(saturation)))
        value = max(0, min(255, int(value)))
        if value != self._value:
            self._value = value
            self._image = None
        self.update()

    def color(self) -> QtGui.QColor:
        return QtGui.QColor.fromHsv(self._hue, self._saturation, self._value)

    def _ensure_image(self):
        size = self.size()
        if size.width() <= 1 or size.height() <= 1:
            return
        if self._image is not None and self._cache_size == size and self._cache_value == self._value:
            return
        image = QtGui.QImage(size, QtGui.QImage.Format_RGB32)
        width = max(1, size.width() - 1)
        height = max(1, size.height() - 1)
        for y in range(size.height()):
            saturation = int(round(255 * (1 - y / height)))
            for x in range(size.width()):
                hue = int(round(359 * x / width))
                image.setPixelColor(x, y, QtGui.QColor.fromHsv(hue, saturation, self._value))
        self._image = image
        self._cache_size = size
        self._cache_value = self._value

    def paintEvent(self, event):
        self._ensure_image()
        painter = QtGui.QPainter(self)
        if self._image is not None:
            painter.drawImage(self.rect(), self._image)
        pen_color = QtGui.QColor("#101010") if self.color().lightness() > 150 else QtGui.QColor("#FFFFFF")
        painter.setPen(QtGui.QPen(pen_color, 2))
        x = round((self._hue / 359) * max(0, self.width() - 1))
        y = round((1 - self._saturation / 255) * max(0, self.height() - 1))
        painter.drawEllipse(QtCore.QPoint(x, y), 6, 6)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._pick(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self._pick(event.position())

    def _pick(self, position: QtCore.QPointF):
        x = min(max(position.x(), 0), max(0, self.width() - 1))
        y = min(max(position.y(), 0), max(0, self.height() - 1))
        self._hue = round(359 * x / max(1, self.width() - 1))
        self._saturation = round(255 * (1 - y / max(1, self.height() - 1)))
        self.update()
        self.colorChanged.emit(self.color())


class ColorPaletteDialog(QtWidgets.QDialog):
    """A visual, continuous color picker returning a ``#RRGGBB`` string."""

    QUICK_COLORS = (
        "#000000", "#4A4A4A", "#808080", "#C8C8C8", "#FFFFFF",
        "#E53935", "#FB8C00", "#FDD835", "#43A047", "#00ACC1",
        "#1E88E5", "#5E35B1", "#8E24AA", "#D81B60",
    )

    def __init__(self, title: str, initial_color: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(395)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self._selected_color = QtGui.QColor(self._normalize(initial_color))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("选择一个颜色", self)
        heading.setStyleSheet("font-weight: 600;")
        layout.addWidget(heading)

        current_row = QtWidgets.QHBoxLayout()
        current_row.setSpacing(10)
        current_row.addWidget(QtWidgets.QLabel("当前选择", self))
        self.current_swatch = QtWidgets.QLabel(self)
        self.current_swatch.setFixedSize(76, 28)
        self.current_swatch.setToolTip("当前选中的颜色")
        current_row.addWidget(self.current_swatch)
        current_row.addStretch(1)
        layout.addLayout(current_row)

        picker_row = QtWidgets.QHBoxLayout()
        picker_row.setSpacing(12)
        self.plane = HueSaturationPlane(self)
        self.plane.colorChanged.connect(self._on_plane_color_changed)
        picker_row.addWidget(self.plane, 1)
        brightness_column = QtWidgets.QVBoxLayout()
        brightness_label = QtWidgets.QLabel("明暗", self)
        brightness_label.setAlignment(QtCore.Qt.AlignHCenter)
        brightness_column.addWidget(brightness_label)
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Vertical, self)
        self.brightness_slider.setRange(0, 255)
        self.brightness_slider.setValue(max(0, self._selected_color.value()))
        self.brightness_slider.setToolTip("向上更亮，向下更暗")
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        brightness_column.addWidget(self.brightness_slider, 1)
        picker_row.addLayout(brightness_column)
        layout.addLayout(picker_row)

        quick_label = QtWidgets.QLabel("常用颜色", self)
        quick_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(quick_label)
        quick_grid = QtWidgets.QGridLayout()
        quick_grid.setHorizontalSpacing(8)
        quick_grid.setVerticalSpacing(7)
        for index, color in enumerate(self.QUICK_COLORS):
            button = QtWidgets.QToolButton(self)
            button.setFixedSize(28, 24)
            button.setToolTip("点击选择此颜色")
            button.setStyleSheet(self._swatch_style(color))
            button.clicked.connect(lambda _checked=False, chosen=color: self._set_color(QtGui.QColor(chosen)))
            quick_grid.addWidget(button, index // 7, index % 7)
        layout.addLayout(quick_grid)

        buttons = QtWidgets.QDialogButtonBox(self)
        buttons.addButton("取消", QtWidgets.QDialogButtonBox.RejectRole)
        buttons.addButton("使用此颜色", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self._set_color(self._selected_color, update_slider=True)

    @staticmethod
    def _normalize(value: str) -> str:
        value = str(value or "#FFFFFF").strip().upper()
        color = QtGui.QColor(value)
        return color.name().upper() if color.isValid() else "#FFFFFF"

    @staticmethod
    def _is_light(color: str | QtGui.QColor) -> bool:
        value = color if isinstance(color, QtGui.QColor) else QtGui.QColor(color)
        return value.lightness() > 175

    @classmethod
    def _swatch_style(cls, color: str) -> str:
        border = "#5A5A5A" if cls._is_light(color) else "#202020"
        return (
            f"QToolButton {{ background: {color}; border: 1px solid {border}; border-radius: 4px; }}"
            "QToolButton:hover { border: 2px solid #0078D4; }"
        )

    def _set_color(self, color: QtGui.QColor, *, update_slider: bool = True):
        if not color.isValid():
            return
        self._selected_color = color
        hue = color.hue()
        self.plane.set_hsv(0 if hue < 0 else hue, color.saturation(), color.value())
        if update_slider:
            self.brightness_slider.blockSignals(True)
            self.brightness_slider.setValue(color.value())
            self.brightness_slider.blockSignals(False)
        self.current_swatch.setStyleSheet(
            f"background: {color.name().upper()}; border: 2px solid #202020; border-radius: 5px;"
        )

    def _on_plane_color_changed(self, color: QtGui.QColor):
        self._set_color(color, update_slider=False)

    def _on_brightness_changed(self, value: int):
        color = self.plane.color()
        color.setHsv(color.hue(), color.saturation(), value)
        self._set_color(color, update_slider=False)

    @classmethod
    def choose(cls, parent, title: str, initial_color: str) -> str | None:
        """Open the visual picker and return a color after explicit confirmation."""
        dialog = cls(title, initial_color, parent)
        return dialog._selected_color.name().upper() if dialog.exec() == QtWidgets.QDialog.Accepted else None
