"""内容修复与画质增强共用的媒体预览卡。"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from qfluentwidgets import BodyLabel, CardWidget, HollowHandleStyle

from ui.component.video_display_component import AspectRatioContainer

# 低饱和石板灰 #94A3B8，OpenCV 使用 BGR 顺序。它能区分两侧画面，
# 又不会像项目主青色那样抢占图片内容。
COMPARISON_DIVIDER_BGR = (184, 163, 148)
COMPARISON_DIVIDER_WIDTH = 4


class MediaPreviewComponent(CardWidget):
    """黑底 16:9 媒体预览，支持输入、输出和输入/输出对比画面。"""

    def __init__(
        self,
        title: str = "媒体预览",
        parent=None,
        *,
        show_title: bool = True,
        placeholder: str = "选择一个任务以开始",
    ):
        super().__init__(parent)
        self._source_pixmap: QtGui.QPixmap | None = None
        self._placeholder = placeholder
        self._show_title = show_title

        layout = QtWidgets.QVBoxLayout(self)
        # 无标题预览用于内容工作台。它必须和内容修复的 VideoDisplayComponent
        # 共享 2px 画布边距；若仍沿用设置卡的 12px 边距，虽然伸缩比例相同，
        # 画质增强/图片转换的预览 size hint 仍会凭空多出 20px，三页无法对齐。
        preview_margin = 12 if show_title else 2
        layout.setContentsMargins(preview_margin, preview_margin, preview_margin, preview_margin)
        layout.setSpacing(8)
        self.title_label = BodyLabel(title, self)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.title_label.setVisible(show_title)
        if show_title:
            layout.addWidget(self.title_label)

        self.black_container = QtWidgets.QWidget(self)
        self.black_container.setStyleSheet(
            "background-color: black; border-radius: 10px; border: 0px solid transparent;"
        )
        black_layout = QtWidgets.QVBoxLayout(self.black_container)
        black_layout.setContentsMargins(0, 0, 0, 0)
        self.aspect_container = AspectRatioContainer(parent=self.black_container)
        self.display = QtWidgets.QLabel(self.aspect_container)
        self.display.setAlignment(QtCore.Qt.AlignCenter)
        self.display.setMinimumSize(1, 1)
        self.display.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.display.setScaledContents(True)
        self.display.setStyleSheet(
            "background-color: black; border-top-left-radius: 10px; border-top-right-radius: 10px;"
        )
        self.aspect_container.setContent(self.display)
        black_layout.addWidget(self.aspect_container)
        # 与内容修复共用同一套视频时间轴视觉规范：黑底控制栏、紧凑边距与
        # 白色空心手柄。两页都会将第一帧显示为 1，而不是暴露 OpenCV 的 0-based
        # 帧下标给用户。
        self.control_container = QtWidgets.QWidget(self.black_container)
        control_layout = QtWidgets.QVBoxLayout(self.control_container)
        control_layout.setContentsMargins(8, 8, 8, 8)
        self.video_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.black_container)
        self.video_slider.setRange(1, 1)
        self.video_slider.setFixedHeight(22)
        self.video_slider.setStyle(HollowHandleStyle({
            "handle.color": QtGui.QColor(255, 255, 255),
            "handle.ring-width": 4,
            "handle.hollow-radius": 6,
            "handle.margin": 1,
        }))
        self.video_slider.valueChanged.connect(self.frame_requested)
        control_layout.addWidget(self.video_slider)
        self.control_container.setStyleSheet(
            "background-color: black; border-bottom-left-radius: 8px; "
            "border-bottom-right-radius: 8px;"
        )
        self.control_container.setVisible(False)
        black_layout.addWidget(self.control_container)
        layout.addWidget(self.black_container, 1)
        self.set_placeholder(placeholder)

    def set_placeholder(self, text: str | None = None):
        """清空媒体像素与视频控件；空状态可选择不显示任何提示文字。"""
        self.clear_media_preview(text)
        self.set_video_controls_visible(False)

    def clear_media_preview(self, text: str | None = None):
        """清空媒体像素，但保留当前任务类型决定的时间轴可见状态。

        切换图片/视频任务时，页面可能还没有同类型任务可选；这时必须移除
        旧媒体画面，同时不能反向覆盖刚由任务类型设置的时间轴状态。
        """
        self._source_pixmap = None
        self._placeholder = self._placeholder if text is None else text
        self.display.setPixmap(QtGui.QPixmap())
        self.display.setText(self._placeholder)
        self.display.setStyleSheet(
            "background-color: black; color: #9aa3ad; border-top-left-radius: 10px; border-top-right-radius: 10px;"
        )

    def set_video_controls_visible(self, visible: bool, frame_count: int = 0, frame_index: int = 0):
        """显示与内容修复一致的 1-based 视频滑块。

        ``frame_index`` 保持 OpenCV 的 0-based 约定；控件显示时统一加一，避免
        用户拖到最左端却看到或读取到第二帧。
        """
        total = max(1, int(frame_count))
        self.video_slider.blockSignals(True)
        self.video_slider.setRange(1, total)
        self.video_slider.setValue(max(1, min(int(frame_index) + 1, total)))
        self.video_slider.blockSignals(False)
        self.control_container.setVisible(visible)

    def set_frame(self, frame_bgr: np.ndarray, title: str):
        """将 BGR 帧转换为 QPixmap，并根据原始宽高比更新显示容器。"""
        if frame_bgr is None:
            self.set_placeholder("无法读取预览媒体")
            return
        frame = frame_bgr[:, :, :3]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self.aspect_container.setAspectRatio(width / height)
        image = QtGui.QImage(rgb.data, width, height, width * channels, QtGui.QImage.Format_RGB888).copy()
        self._source_pixmap = QtGui.QPixmap.fromImage(image)
        self.title_label.setText(title)
        self.display.setText("")
        self.display.setStyleSheet(
            "background-color: black; border-top-left-radius: 10px; border-top-right-radius: 10px;"
        )
        self._refresh_pixmap()

    def set_comparison(self, original_bgr: np.ndarray, enhanced_bgr: np.ndarray, title: str):
        """将输入缩放到输出尺寸后并排显示，避免不同倍率无法拼接。

        分隔线使用所有处理页面共享的中性颜色；它仅用于视觉分区，不代表
        图像内容发生了额外处理。
        """
        if original_bgr is None or enhanced_bgr is None:
            self.set_placeholder("当前任务未提供可用对比预览")
            return
        target_height, target_width = enhanced_bgr.shape[:2]
        source = cv2.resize(original_bgr[:, :, :3], (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        divider = np.full(
            (target_height, COMPARISON_DIVIDER_WIDTH, 3),
            COMPARISON_DIVIDER_BGR,
            dtype=np.uint8,
        )
        comparison = cv2.hconcat([source, divider, enhanced_bgr[:, :, :3]])
        self.set_frame(comparison, title)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._refresh_pixmap)

    def _refresh_pixmap(self):
        if self._source_pixmap is None or self.display.size().isEmpty():
            return
        self.display.setPixmap(
            self._source_pixmap.scaled(
                self.display.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        )
    frame_requested = QtCore.Signal(int)
