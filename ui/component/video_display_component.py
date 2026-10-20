import cv2
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMenu
from PySide6.QtCore import Qt, Signal, QRect, QRectF, QObject, QEvent
from PySide6.QtGui import QAction, QShortcut, QCursor
from PySide6 import QtCore, QtWidgets, QtGui
from qfluentwidgets import qconfig, CardWidget, HollowHandleStyle

from backend.config import config, tr


class AspectRatioContainer(QWidget):
    """让布局系统而不是 resize 回调管理预览区域的 16:9 高度。

    导航栏展开/收起会频繁改变可用宽度。旧实现使用 ``setFixedHeight``
    和 960px 最小宽度，会把一次临时 resize 固化成主窗口的最小尺寸，导致
    页面只能变大不能恢复。本组件通过 Qt 的 height-for-width 协商保持比例，
    同时允许容器在窄窗口中正常收缩。
    """

    def __init__(self, ratio: float = 16 / 9, parent=None):
        super().__init__(parent)
        # 布局比例保持稳定，媒体比例只影响内部内容矩形。若把竖屏比例直接
        # 暴露给 heightForWidth，图片/视频切换会反向改变整个页面的 sizeHint。
        self._layout_ratio = ratio
        self._content_ratio = ratio
        self._content: QWidget | None = None
        self.setMinimumSize(560, 315)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    def setContent(self, widget: QWidget):
        """放入保持比例的内容；多余空间自动成为上下或左右黑边。"""
        self._content = widget
        widget.setParent(self)
        widget.show()
        self._resize_content()

    def hasHeightForWidth(self):
        return True

    def setAspectRatio(self, ratio: float):
        """更新内容比例；用于横屏、竖屏和方形媒体的自适应预览。"""
        if ratio <= 0:
            return
        self._content_ratio = max(0.1, min(float(ratio), 10.0))
        self._resize_content()

    def heightForWidth(self, width):
        return max(self.minimumHeight(), round(max(1, width) / self._layout_ratio))

    def sizeHint(self):
        return QtCore.QSize(960, 540)

    def minimumSizeHint(self):
        return QtCore.QSize(560, 315)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_content()

    def _resize_content(self):
        if self._content is None:
            return
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            return
        content_width = width
        content_height = round(content_width / self._content_ratio)
        if content_height > height:
            content_height = height
            content_width = round(content_height * self._content_ratio)
        left = (width - content_width) // 2
        top = (height - content_height) // 2
        self._content.setGeometry(left, top, content_width, content_height)


class VideoDisplayComponent(QWidget):
    """视频显示组件，包含视频预览和选择框功能"""

    # 定义信号
    selections_changed = Signal(list)  # 选择框变化信号
    brush_strokes_changed = Signal(list)  # 手工涂鸦路径变化信号
    ab_sections_changed = Signal(list)  # AB分区变化信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # 初始化变量
        self.is_drawing = False
        self.selection_rect = (0, 0, 0, 0)  # 当前正在绘制或调整的选区 (ymin, ymax, xmin, xmax)
        self.selection_rects = []  # 存储多个选区，每个元素为 (ymin, ymax, xmin, xmax)
        self.active_selection_index = -1  # 当前活动选区的索引
        self.drag_start_pos = None
        self.resize_edge = None
        self.edge_size = 10  # 调整大小的边缘区域
        self.enable_mouse_events = True  # 控制是否启用鼠标事件
        # 涂鸦以相对源图坐标和“逐笔冻结的半径”保存，和当前预览分辨率、窗口
        # 缩放及之后改动的笔刷粗细无关。它是最终 mask 的直接来源，不复用为
        # OCR 的候选区域。
        self.brush_mode = False
        self.brush_visible = False
        self._painting_brush = False
        self.brush_radius = config.brushRadius.value
        self.brush_strokes = []

        # AB分区标记相关变量
        self.ab_sections = []  # 存储AB分区标记 [range(start, end), ...]
        self.current_ab_start = -1  # 当前AB分区的起点

        # 创建右键菜单
        self.__init_context_menu()

        # 获取屏幕大小
        screen = QtWidgets.QApplication.primaryScreen().size()
        self.screen_width = screen.width()
        self.screen_height = screen.height()

        # 设置视频预览区域大小（根据屏幕宽度动态调整）
        self.video_preview_width = 960
        self.video_preview_height = self.video_preview_width * 9 // 16
        if self.screen_width // 2 < 960:
            self.video_preview_width = 640
            self.video_preview_height = self.video_preview_width * 9 // 16

        # 视频相关参数
        self.frame_width = None
        self.frame_height = None
        self.scaled_width = None
        self.scaled_height = None
        self.border_left = 0
        self.border_top = 0
        self.fps = 30
        self._is_video_media = False

        self.__init_widgets()
        self.__init_shotcuts()

    def set_media_aspect_ratio(self, ratio: float):
        """按媒体实际宽高比设置逻辑预览画布，不强制塞入 16:9。"""
        if ratio <= 0:
            return
        ratio = max(0.1, min(float(ratio), 10.0))
        # 保持长边约 960px，竖屏时逻辑画布自然变为窄而高的比例。
        if ratio >= 1:
            width = 960
            height = max(1, round(width / ratio))
        else:
            height = 960
            width = max(1, round(height * ratio))
        self.video_preview_width = width
        self.video_preview_height = height
        self.ratio_container.setAspectRatio(ratio)

    def __init_widgets(self):
        """初始化组件"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 视频预览区域和进度条容器
        self.video_container = CardWidget(self)
        self.video_container.setObjectName('videoContainer')
        video_layout = QVBoxLayout()
        video_layout.setSpacing(0)
        video_layout.setContentsMargins(2, 2, 2, 2)
        # 不在外层布局上设置 AlignCenter。该对齐会把黑色画布钉死在 16:9
        # 的 size hint，即使内容修复页已经得到更多垂直空间也不会变大；其它
        # 工作台则会让黑底预览填满分配区域。媒体内容本身仍由内部比例容器居中。

        # 创建内部黑色背景容器
        self.black_container = QWidget(self)
        self.black_container.setObjectName('blackContainer')
        self.black_container.setStyleSheet("""
            #blackContainer {
                background-color: black;
                border-radius: 10px;
                border: 0px solid transparent;
            }
        """)
        black_layout = QVBoxLayout()
        black_layout.setContentsMargins(0, 0, 0, 0)
        black_layout.setSpacing(0)
        # 让黑色画布随父布局扩张，内部 QLabel 继续按实际媒体比例留黑边。

        # 视频显示标签
        self.video_display = QtWidgets.QLabel()
        self.video_display.setStyleSheet("""
            background-color: black;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            border: 0px solid transparent;
        """)
        # 预览画布逻辑分辨率仍是 16:9，但控件本身不能把主窗口锁死在 960px。
        self.video_display.setMinimumSize(1, 1)
        self.video_display.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.video_display.setMouseTracking(True)
        self.video_display.setScaledContents(True)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.mousePressEvent = self.selection_mouse_press
        self.video_display.mouseMoveEvent = self.selection_mouse_move
        self.video_display.mouseReleaseEvent = self.selection_mouse_release

        # 视频滑块
        self.video_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.video_slider.setMinimum(1)
        self.video_slider.setFixedHeight(22)
        self.video_slider.setMaximum(100)  # 默认最大值设为100，与进度百分比一致
        self.video_slider.setValue(1)
        self.video_slider.setStyle(HollowHandleStyle({
            "handle.color": QtGui.QColor(255, 255, 255),
            "handle.ring-width": 4,
            "handle.hollow-radius": 6,
            "handle.margin": 1
        }))

        # 视频预览区域
        self.video_display.setObjectName('videoDisplay')
        # black_layout.addWidget(self.video_display, 0, Qt.AlignCenter)
        # 创建一个容器来保持比例
        self.ratio_container = AspectRatioContainer(parent=self)
        self.ratio_container.setContent(self.video_display)

        # 添加到布局
        black_layout.addWidget(self.ratio_container)

        # 进度条和滑块容器
        self.control_container = QWidget(self)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.addWidget(self.video_slider)

        self.control_container.setLayout(control_layout)
        self.control_container.setStyleSheet("""
            background-color: black;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        """)
        black_layout.addWidget(self.control_container)

        self.black_container.setLayout(black_layout)
        video_layout.addWidget(self.black_container)
        self.video_container.setLayout(video_layout)
        main_layout.addWidget(self.video_container)

    def set_video_controls_visible(self, visible: bool):
        """图片没有时间轴，视频才显示帧滑块与 A/B 相关交互。"""
        self._is_video_media = visible
        self.control_container.setVisible(visible)

    def __init_shotcuts(self):
        """初始化快捷键"""
        self.shortcut_ab_start = QShortcut(QtGui.QKeySequence("["), self)
        self.shortcut_ab_start.activated.connect(self.__handle_mark_for_ab_start)
        self.shortcut_ab_start.setContext(Qt.ApplicationShortcut)

        self.shortcut_ab_end = QShortcut(QtGui.QKeySequence("]"), self)
        self.shortcut_ab_end.activated.connect(self.__handle_mark_for_ab_end)
        self.shortcut_ab_end.setContext(Qt.ApplicationShortcut)

        self.shortcut_ab_delete = QShortcut(QtGui.QKeySequence("\\"), self)
        self.shortcut_ab_delete.activated.connect(self.__handle_delete_ab_section)
        self.shortcut_ab_delete.setContext(Qt.ApplicationShortcut)

        self.shortcut_delete_selection = QShortcut(QtGui.QKeySequence.Delete, self)
        self.shortcut_delete_selection.activated.connect(self.__handle_delete_selection)
        self.shortcut_delete_selection.setContext(Qt.ApplicationShortcut)

        # 添加左右键控制slider的快捷键
        self.shortcut_right = QShortcut(QtGui.QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.__adjust_slider_value(self.fps))
        self.shortcut_right.setContext(Qt.ApplicationShortcut)

        self.shortcut_left = QShortcut(QtGui.QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.__adjust_slider_value(-self.fps))
        self.shortcut_left.setContext(Qt.ApplicationShortcut)

        # 添加Ctrl+左右键控制slider的快捷键
        self.shortcut_ctrl_right = QShortcut(QtGui.QKeySequence("Ctrl+Right"), self)
        self.shortcut_ctrl_right.activated.connect(lambda: self.__adjust_slider_value(self.fps*5))
        self.shortcut_ctrl_right.setContext(Qt.ApplicationShortcut)

        self.shortcut_ctrl_left = QShortcut(QtGui.QKeySequence("Ctrl+Left"), self)
        self.shortcut_ctrl_left.activated.connect(lambda: self.__adjust_slider_value(-self.fps*5))
        self.shortcut_ctrl_left.setContext(Qt.ApplicationShortcut)

        # 添加Shift+左右键控制slider的快捷键
        self.shortcut_shift_right = QShortcut(QtGui.QKeySequence("Shift+Right"), self)
        self.shortcut_shift_right.activated.connect(lambda: self.__adjust_slider_value(1))
        self.shortcut_shift_right.setContext(Qt.ApplicationShortcut)

        self.shortcut_shift_left = QShortcut(QtGui.QKeySequence("Shift+Left"), self)
        self.shortcut_shift_left.activated.connect(lambda: self.__adjust_slider_value(-1))
        self.shortcut_shift_left.setContext(Qt.ApplicationShortcut)

    def update_video_display(self, frame, draw_selection=True):
        """更新视频显示"""
        if frame is None:
            return

        # 调整视频帧大小以适应视频预览区域
        frame = cv2.resize(frame, (self.video_preview_width, self.video_preview_height))
        # 将 OpenCV 帧（BGR 格式）转换为 QImage 并显示在 QLabel 上
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(image)

        # 创建带圆角的图像
        rounded_pix = QtGui.QPixmap(pix.size())
        rounded_pix.fill(Qt.transparent)  # 填充透明背景

        painter = QtGui.QPainter(rounded_pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)  # 抗锯齿
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        # 创建圆角路径
        path = QtGui.QPainterPath()
        rect = QRectF(0, 0, pix.width(), pix.height())

        # 手动创建只有左上和右上圆角的路径
        radius = 8
        path.moveTo(radius, 0)
        path.lineTo(pix.width() - radius, 0)
        path.arcTo(pix.width() - radius * 2, 0, radius * 2, radius * 2, 90, -90)
        path.lineTo(pix.width(), pix.height())
        path.lineTo(0, pix.height())
        path.lineTo(0, radius)
        path.arcTo(0, 0, radius * 2, radius * 2, 180, -90)
        path.closeSubpath()

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pix)
        painter.end()

        # 保存当前的pixmap用于绘制选择框
        self.current_pixmap = rounded_pix.copy()

        self.video_display.setPixmap(rounded_pix)

        # 更新视频显示
        self.update_preview_with_rect(draw_selection=draw_selection)

    def update_preview_with_rect(self, rect=None, draw_selection=True):
        """更新带有选择框的预览"""
        if not hasattr(self, 'current_pixmap') or self.current_pixmap is None:
            return

        # 如果提供了新的矩形，使用它
        if rect is not None and self.active_selection_index >= 0:
            self.selection_rects[self.active_selection_index] = rect

        # 创建一个副本用于绘制
        pixmap_copy = self.current_pixmap.copy()
        painter = QtGui.QPainter(pixmap_copy)

        # 绘制所有选区
        if draw_selection:
            # 计算缩放比例
            display_size = self.video_display.size()
            pixmap_size = self.current_pixmap.size()
            scale_x = pixmap_size.width() / display_size.width()
            scale_y = pixmap_size.height() / display_size.height()
            video_display_width = self.video_display.width()
            video_display_height = self.video_display.height()
            for i, rect in enumerate(self.selection_rects):
                # 设置选择框样式
                if i == self.active_selection_index:
                    # 活动选区使用绿色
                    pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
                else:
                    # 非活动选区使用黄色
                    pen = QtGui.QPen(QtGui.QColor(255, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)

                # 将比例坐标转换为像素坐标
                ymin, ymax, xmin, xmax = rect
                pixel_rect = QRect(
                    int(xmin * scale_x * video_display_width),
                    int(ymin * scale_y * video_display_height),
                    int((xmax - xmin) * scale_x * video_display_width),
                    int((ymax - ymin) * scale_y * video_display_height)
                )

                # 绘制选择框
                painter.drawRect(pixel_rect)

            self._draw_brush_strokes(painter, pixmap_copy)

            # 如果正在绘制新选区，也绘制它
            if self.is_drawing and any(self.selection_rect):
                pen = QtGui.QPen(QtGui.QColor(0, 255, 0))  # 绿色
                pen.setWidth(2)
                painter.setPen(pen)

                # 将比例坐标转换为像素坐标
                ymin, ymax, xmin, xmax = self.selection_rect
                pixel_rect = QRect(
                    int(xmin * scale_x * video_display_width),
                    int(ymin * scale_y * video_display_height),
                    int((xmax - xmin) * scale_x * video_display_width),
                    int((ymax - ymin) * scale_y * video_display_height)
                )

                painter.drawRect(pixel_rect)

        # 绘制AB分区标记
        total_frames = self.video_slider.maximum()
        if total_frames > 0 and self.ab_sections:
            # 在视频显示区域下方5像素处绘制AB分区标记
            ab_rect_height = 5
            ab_rect_y = pixmap_copy.height() - ab_rect_height

            # 设置半透明白色画刷
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255, 128))  # 半透明白色

            # 计算可用宽度（考虑左右边距）
            left_margin = 15
            right_margin = 15
            available_width = pixmap_copy.width() - left_margin - right_margin

            for section_range in self.ab_sections:
                # 计算相对位置
                start_x = left_margin + int((section_range.start / total_frames) * available_width)
                end_x = left_margin + int((section_range.stop / total_frames) * available_width)

                # 绘制AB分区矩形
                painter.drawRect(start_x, ab_rect_y, end_x - start_x, ab_rect_height)

        # 绘制current_ab_start的高亮竖线
        if self.current_ab_start >= 0 and total_frames > 0:
            # 计算可用宽度（考虑左右边距）
            left_margin = 15
            right_margin = 15
            available_width = pixmap_copy.width() - left_margin - right_margin

            # 计算current_ab_start的相对位置
            start_x = left_margin + int((self.current_ab_start / total_frames) * available_width)

            # 设置高亮白色画笔
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255))  # 纯白色
            pen.setWidth(2)
            painter.setPen(pen)

            # 绘制高亮竖线，高度为5像素
            ab_line_height = 5
            ab_line_y = pixmap_copy.height() - ab_line_height
            painter.drawLine(start_x, ab_line_y, start_x, pixmap_copy.height())

        painter.end()

        # 更新显示
        self.video_display.setPixmap(pixmap_copy)

    def _draw_brush_strokes(self, painter, pixmap):
        """将正在编辑的涂鸦以项目既有的高对比辅助色叠加到预览中。"""
        if not self.brush_visible or not self.brush_strokes or not self.frame_width:
            return
        for stroke in self.brush_strokes:
            points_value = stroke.get("points", [])
            if not points_value:
                continue
            preview_radius = max(1, round(stroke["radius"] * pixmap.width() / self.frame_width))
            pen = QtGui.QPen(QtGui.QColor(255, 176, 0, 210), preview_radius * 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            points = [
                QtCore.QPointF(
                    (self.border_left + x * self.scaled_width) * pixmap.width(),
                    (self.border_top + y * self.scaled_height) * pixmap.height(),
                )
                for x, y in points_value
            ]
            if len(points) == 1:
                painter.drawPoint(points[0])
                continue
            path = QtGui.QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.drawPath(path)

    def _source_point_from_event(self, event):
        """将 QLabel 鼠标位置映射到未填黑边的原始媒体相对坐标。"""
        width, height = self.video_display.width(), self.video_display.height()
        if width <= 0 or height <= 0 or not self.scaled_width or not self.scaled_height:
            return None
        x_display = event.pos().x() / width
        y_display = event.pos().y() / height
        x = (x_display - self.border_left) / self.scaled_width
        y = (y_display - self.border_top) / self.scaled_height
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def set_brush_mode(self, enabled: bool):
        """启用/退出涂鸦。退出后恢复现有的矩形编辑交互。"""
        self.brush_mode = bool(enabled)
        self._painting_brush = False
        self.is_drawing = False
        self.resize_edge = None
        self.video_display.setCursor(Qt.CrossCursor if self.brush_mode else Qt.ArrowCursor)

    def set_brush_visible(self, visible: bool):
        """仅在手工涂鸦模式显示橙色编辑路径，避免污染其他模式预览。"""
        self.brush_visible = bool(visible)
        if not self.brush_visible:
            self.set_brush_mode(False)
        self.update_preview_with_rect()

    def set_brush_radius(self, radius: int):
        self.brush_radius = max(1, int(radius))
        self.update_preview_with_rect()

    def set_brush_strokes(self, strokes, *, emit=False):
        """加载逐笔笔迹，兼容旧版 ``[[x, y], ...]`` 点列。"""
        normalized = []
        for stroke in strokes or []:
            raw_points = stroke.get("points", []) if isinstance(stroke, dict) else stroke
            radius = stroke.get("radius", self.brush_radius) if isinstance(stroke, dict) else self.brush_radius
            points = [
                (max(0.0, min(1.0, float(point[0]))), max(0.0, min(1.0, float(point[1]))))
                for point in raw_points or []
                if len(point) >= 2
            ]
            if points:
                normalized.append({"points": points, "radius": max(1, int(radius))})
        self.brush_strokes = normalized
        self.update_preview_with_rect()
        if emit:
            self.brush_strokes_changed.emit(self.get_brush_strokes())

    def get_brush_strokes(self):
        return [
            {"points": [list(point) for point in stroke["points"]], "radius": stroke["radius"]}
            for stroke in self.brush_strokes
        ]

    def clear_brush_strokes(self):
        self.brush_strokes = []
        self._painting_brush = False
        self.update_preview_with_rect()
        self.brush_strokes_changed.emit([])

    def clear_media_preview(self):
        """清空未绑定任务时的可视状态，且不向上一层写回任何任务草稿。"""
        self.is_drawing = False
        self._painting_brush = False
        self.selection_rects = []
        self.active_selection_index = -1
        self.brush_strokes = []
        self.ab_sections = []
        self.current_ab_start = -1
        self.current_pixmap = None
        self.frame_width = None
        self.frame_height = None
        self.video_display.clear()

    def selection_mouse_press(self, event):
        """鼠标按下事件处理"""
        if not self.enable_mouse_events:
            return

        if self.brush_mode:
            if event.button() != Qt.LeftButton:
                return
            point = self._source_point_from_event(event)
            if point is None:
                return
            self._painting_brush = True
            # 半径在鼠标按下时冻结；之后调节笔刷只影响下一笔，历史涂鸦的
            # 预览和实际修复 mask 保持不变。
            self.brush_strokes.append({"points": [point], "radius": self.brush_radius})
            self.update_preview_with_rect()
            return

        # 右键点击显示上下文菜单
        if event.button() == Qt.RightButton:
            self.__update_context_menu_actions()
            if not any(action.isVisible() for action in self.context_menu.actions()):
                return
            self.context_menu.exec_(event.globalPos())
            return

        video_display_width = self.video_display.width()
        video_display_height = self.video_display.height()

        # 开始绘制新选区
        if event.modifiers() & Qt.ControlModifier:
            self.is_drawing = True
            pos = event.pos()

            # 转换为比例坐标
            y_ratio = (pos.y() - self.border_top) / video_display_height if video_display_height > 0 else 0
            x_ratio = (pos.x() - self.border_left) / video_display_width if video_display_width > 0 else 0

            # 初始化选区为单点
            self.selection_rect = (y_ratio, y_ratio, x_ratio, x_ratio)
            self.drag_start_pos = (y_ratio, x_ratio)  # 保存起始点的比例坐标
            self.resize_edge = None
            self.active_selection_index = -1
            return

        # 双击重置所有选区
        if event.type() == QEvent.MouseButtonDblClick:
            self.clear_selections()
            return

        # 检查是否点击了已有选区
        pos = event.pos()
        y_ratio = (pos.y() - self.border_top) / video_display_height if video_display_height > 0 else 0
        x_ratio = (pos.x() - self.border_left) / video_display_width if video_display_width > 0 else 0

        clicked_index = -1
        for i, rect in enumerate(self.selection_rects):
            # 将比例坐标转换为像素坐标用于检测
            ymin, ymax, xmin, xmax = rect
            pixel_rect = QRect(
                int(xmin * video_display_width) + self.border_left,
                int(ymin * video_display_height) + self.border_top,
                int((xmax - xmin) * video_display_width),
                int((ymax - ymin) * video_display_height)
            )

            # 检查是否在选区边缘（用于调整大小）
            if self.is_on_rect_edge(pos, pixel_rect):
                clicked_index = i
                self.active_selection_index = i
                self.resize_edge = self.get_resize_edge(pos, pixel_rect)
                self.drag_start_pos = (y_ratio, x_ratio)
                self.update_preview_with_rect()
                return
            # 检查是否在选区内部（用于移动）
            elif pixel_rect.contains(pos):
                clicked_index = i
                self.active_selection_index = i
                self.resize_edge = "move"
                self.drag_start_pos = (y_ratio, x_ratio)
                self.update_preview_with_rect()
                return

        # 如果没有点击任何选区，开始绘制新选区
        if clicked_index == -1:
            self.is_drawing = True
            self.selection_rect = (y_ratio, y_ratio, x_ratio, x_ratio)
            self.drag_start_pos = (y_ratio, x_ratio)
            self.resize_edge = None
            self.active_selection_index = -1

    def is_on_rect_edge(self, pos, pixel_rect):
        """检查点是否在矩形边缘
        注意：这里的pixel_rect是已经转换为像素坐标的QRect对象
        """
        # 右下角
        if abs(pos.x() - pixel_rect.right()) <= self.edge_size and abs(pos.y() - pixel_rect.bottom()) <= self.edge_size:
            return True
        # 右上角
        elif abs(pos.x() - pixel_rect.right()) <= self.edge_size and abs(pos.y() - pixel_rect.top()) <= self.edge_size:
            return True
        # 左下角
        elif abs(pos.x() - pixel_rect.left()) <= self.edge_size and abs(pos.y() - pixel_rect.bottom()) <= self.edge_size:
            return True
        # 左上角
        elif abs(pos.x() - pixel_rect.left()) <= self.edge_size and abs(pos.y() - pixel_rect.top()) <= self.edge_size:
            return True
        # 左边缘
        elif abs(pos.x() - pixel_rect.left()) <= self.edge_size and pixel_rect.top() <= pos.y() <= pixel_rect.bottom():
            return True
        # 右边缘
        elif abs(pos.x() - pixel_rect.right()) <= self.edge_size and pixel_rect.top() <= pos.y() <= pixel_rect.bottom():
            return True
        # 上边缘
        elif abs(pos.y() - pixel_rect.top()) <= self.edge_size and pixel_rect.left() <= pos.x() <= pixel_rect.right():
            return True
        # 下边缘
        elif abs(pos.y() - pixel_rect.bottom()) <= self.edge_size and pixel_rect.left() <= pos.x() <= pixel_rect.right():
            return True
        return False

    def get_resize_edge(self, pos, rect):
        """获取调整大小的边缘类型"""
        # 右下角
        if abs(pos.x() - rect.right()) <= self.edge_size and abs(pos.y() - rect.bottom()) <= self.edge_size:
            return "bottomright"
        # 右上角
        elif abs(pos.x() - rect.right()) <= self.edge_size and abs(pos.y() - rect.top()) <= self.edge_size:
            return "topright"
        # 左下角
        elif abs(pos.x() - rect.left()) <= self.edge_size and abs(pos.y() - rect.bottom()) <= self.edge_size:
            return "bottomleft"
        # 左上角
        elif abs(pos.x() - rect.left()) <= self.edge_size and abs(pos.y() - rect.top()) <= self.edge_size:
            return "topleft"
        # 左边缘
        elif abs(pos.x() - rect.left()) <= self.edge_size and rect.top() <= pos.y() <= rect.bottom():
            return "left"
        # 右边缘
        elif abs(pos.x() - rect.right()) <= self.edge_size and rect.top() <= pos.y() <= rect.bottom():
            return "right"
        # 上边缘
        elif abs(pos.y() - rect.top()) <= self.edge_size and rect.left() <= pos.x() <= rect.right():
            return "top"
        # 下边缘
        elif abs(pos.y() - rect.bottom()) <= self.edge_size and rect.left() <= pos.x() <= rect.right():
            return "bottom"
        return None

    def selection_mouse_move(self, event):
        """鼠标移动事件处理"""
        if not self.enable_mouse_events:
            return

        if self.brush_mode:
            if self._painting_brush and self.brush_strokes:
                point = self._source_point_from_event(event)
                if point is not None:
                    stroke = self.brush_strokes[-1]["points"]
                    if not stroke or point != stroke[-1]:
                        stroke.append(point)
                        self.update_preview_with_rect()
            return

        video_display_width = self.video_display.width()
        video_display_height = self.video_display.height()

        pos = event.pos()
        y_ratio = (pos.y() - self.border_top) / video_display_height if video_display_height > 0 else 0
        x_ratio = (pos.x() - self.border_left) / video_display_width if video_display_width > 0 else 0

        # 限制比例值在0-1范围内
        y_ratio = max(0, min(1, y_ratio))
        x_ratio = max(0, min(1, x_ratio))

        # 根据不同的操作模式处理鼠标移动
        if self.is_drawing:  # 绘制新选择框
            # 更新选择框的右下角，保留原始拖动方向
            start_y, _, start_x, _ = self.selection_rect
            self.selection_rect = (start_y, y_ratio, start_x, x_ratio)
            self.update_preview_with_rect()
        elif self.resize_edge and self.active_selection_index >= 0:  # 调整选择框大小或位置
            ymin, ymax, xmin, xmax = self.selection_rects[self.active_selection_index]
            start_y, start_x = self.drag_start_pos

            if self.resize_edge == "move":
                # 移动整个选择框
                dy = y_ratio - start_y
                dx = x_ratio - start_x

                # 计算新位置，确保不超出边界
                new_ymin = max(0, min(1 - (ymax - ymin), ymin + dy))
                new_ymax = min(1, max(new_ymin + (ymax - ymin), new_ymin))
                new_xmin = max(0, min(1 - (xmax - xmin), xmin + dx))
                new_xmax = min(1, max(new_xmin + (xmax - xmin), new_xmin))

                self.selection_rects[self.active_selection_index] = (new_ymin, new_ymax, new_xmin, new_xmax)
                self.drag_start_pos = (y_ratio, x_ratio)
            else:
                # 调整选择框大小
                if "left" in self.resize_edge:
                    xmin = min(xmax - 0.01, x_ratio)
                if "right" in self.resize_edge:
                    xmax = max(xmin + 0.01, x_ratio)
                if "top" in self.resize_edge:
                    ymin = min(ymax - 0.01, y_ratio)
                if "bottom" in self.resize_edge:
                    ymax = max(ymin + 0.01, y_ratio)

                # 确保选择框在有效范围内
                xmin = max(0, min(xmin, 1))
                xmax = max(0, min(xmax, 1))
                ymin = max(0, min(ymin, 1))
                ymax = max(0, min(ymax, 1))

                # 确保xmin < xmax, ymin < ymax
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin

                self.selection_rects[self.active_selection_index] = (ymin, ymax, xmin, xmax)

            self.update_preview_with_rect()
        else:
            # 更新鼠标光标形状
            self.update_cursor_shape(pos)

    def selection_mouse_release(self, event):
        """鼠标释放事件处理"""
        if not self.enable_mouse_events:
            return

        if self.brush_mode:
            if self._painting_brush:
                self._painting_brush = False
                self.brush_strokes_changed.emit(self.get_brush_strokes())
            return

        # 结束绘制或调整
        if self.is_drawing:
            # 标准化选择框（确保ymin < ymax, xmin < xmax）
            ymin, ymax, xmin, xmax = self.selection_rect
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin > xmax:
                xmin, xmax = xmax, xmin

            # 更新标准化后的选区
            self.selection_rect = (ymin, ymax, xmin, xmax)

            # 如果选择框有效（不是点击），添加到选区列表
            # 使用比例值计算宽度和高度
            width_ratio = abs(xmax - xmin)
            height_ratio = abs(ymax - ymin)

            # 转换为像素大小进行判断
            pixel_width = width_ratio * self.video_display.width()
            pixel_height = height_ratio * self.video_display.height()

            if pixel_width > 5 and pixel_height > 5:
                self.selection_rects.append(self.selection_rect)
                self.active_selection_index = len(self.selection_rects) - 1

                # 发送选择框变化信号
                self.selections_changed.emit(self.selection_rects)

            self.is_drawing = False
            self.selection_rect = (0, 0, 0, 0)  # 重置为空选区
        elif self.resize_edge and self.active_selection_index >= 0:
            # 标准化选择框
            ymin, ymax, xmin, xmax = self.selection_rects[self.active_selection_index]
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin > xmax:
                xmin, xmax = xmax, xmin

            # 更新标准化后的选区
            self.selection_rects[self.active_selection_index] = (ymin, ymax, xmin, xmax)

            # 发送选择框变化信号
            self.selections_changed.emit(self.selection_rects)

            self.resize_edge = None

    def update_cursor_shape(self, pos):
        """根据鼠标位置更新光标形状"""
        video_display_height = self.video_display.height()
        video_display_width = self.video_display.width()

        # 如果有活动选区，优先检查活动选区
        if self.active_selection_index >= 0 and self.active_selection_index < len(self.selection_rects):
            # 获取活动选区
            ymin, ymax, xmin, xmax = self.selection_rects[self.active_selection_index]

            # 确保坐标规范化
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            # 将比例坐标转换为像素坐标
            pixel_rect = QRect(
                round(xmin * video_display_width) + self.border_left,
                round(ymin * video_display_height) + self.border_top,
                round((xmax - xmin) * video_display_width),
                round((ymax - ymin) * video_display_height)
            )

            # 检查鼠标是否在选择框边缘
            if self.is_on_rect_edge(pos, pixel_rect):
                # 根据边缘类型设置光标
                edge_type = self.get_resize_edge(pos, pixel_rect)
                if edge_type == "left" or edge_type == "right":
                    self.video_display.setCursor(Qt.SizeHorCursor)
                    return
                elif edge_type == "top" or edge_type == "bottom":
                    self.video_display.setCursor(Qt.SizeVerCursor)
                    return
                elif edge_type == "topleft" or edge_type == "bottomright":
                    self.video_display.setCursor(Qt.SizeFDiagCursor)
                    return
                elif edge_type == "topright" or edge_type == "bottomleft":
                    self.video_display.setCursor(Qt.SizeBDiagCursor)
                    return
            elif pixel_rect.contains(pos):
                self.video_display.setCursor(Qt.SizeAllCursor)
                return

        # 如果没有活动选区或鼠标不在活动选区上，检查所有其他选区
        for rect in self.selection_rects:
            # 获取选区坐标
            ymin, ymax, xmin, xmax = rect

            # 确保坐标规范化
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            # 将比例坐标转换为像素坐标
            pixel_rect = QRect(
                round(xmin * video_display_width) + self.border_left,
                round(ymin * video_display_height) + self.border_top,
                round((xmax - xmin) * video_display_width),
                round((ymax - ymin) * video_display_height)
            )

            # 检查鼠标是否在选择框边缘
            if self.is_on_rect_edge(pos, pixel_rect):
                # 根据边缘类型设置光标
                edge_type = self.get_resize_edge(pos, pixel_rect)
                if edge_type == "left" or edge_type == "right":
                    self.video_display.setCursor(Qt.SizeHorCursor)
                    return
                elif edge_type == "top" or edge_type == "bottom":
                    self.video_display.setCursor(Qt.SizeVerCursor)
                    return
                elif edge_type == "topleft" or edge_type == "bottomright":
                    self.video_display.setCursor(Qt.SizeFDiagCursor)
                    return
                elif edge_type == "topright" or edge_type == "bottomleft":
                    self.video_display.setCursor(Qt.SizeBDiagCursor)
                    return
            # 检查鼠标是否在选择框内部
            elif pixel_rect.contains(pos):
                self.video_display.setCursor(Qt.SizeAllCursor)
                return

        # 如果鼠标不在任何选区上，设置为默认光标
        self.video_display.setCursor(Qt.ArrowCursor)

    def set_video_parameters(self, frame_width, frame_height,
                             scaled_width=None, scaled_height=None,
                             border_left=0, border_top=0,
                             fps=30):
        """设置视频参数"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height
        self.border_left = border_left
        self.border_top = border_top
        self.fps = fps

    def get_selection_coordinates(self):
        """获取选择框坐标"""
        return self.selection_rect

    def set_selection_rects(self, rects):
        """设置选择框"""
        self.selection_rects = rects
        self.selection_rect = rects[-1] if rects else QRect()
        self.active_selection_index = len(rects) - 1
        self.update_preview_with_rect()

    def load_selections_from_config(self):
        """从配置中加载选择框的相对位置和大小"""
        # 从配置中读取选择框的相对位置和大小
        areas_str = config.ocrSelectionRegions.value

        # 检查配置值是否有效
        if not areas_str:
            return False

        # 清空现有选区
        self.selection_rects = []
        self.selection_ratios = []

        # 解析配置字符串
        areas = areas_str.split(";")
        for area in areas:
            try:
                parts = area.split(",")
                ymin, ymax, xmin, xmax = map(float, parts)
                self.selection_rects.append((ymin, ymax, xmin, xmax))
            except ValueError:
                continue

        # 如果有选区，设置最后一个为活动选区
        if self.selection_rects:
            self.active_selection_index = len(self.selection_rects) - 1
        else:
            self.active_selection_index = -1
        self.selections_changed.emit(self.selection_rects)

        # 更新预览
        self.update_preview_with_rect()

        return len(self.selection_rects) > 0

    def preview_coordinates_to_video_coordinates(self, preview_selection_rects):
        """获取选择框在原始视频中的坐标"""
        selection_rects = []
        video_display_height = self.video_display.height()
        video_display_width = self.video_display.width()
        for rect in preview_selection_rects:
            ymin, ymax, xmin, xmax = rect

            # 调整选择框坐标，考虑黑边偏移
            x_adjusted = max(0, xmin - self.border_left)
            y_adjusted = max(0, ymin - self.border_top)

            # 如果选择框超出了实际视频区域，需要调整宽度和高度
            w_adjusted = min((xmax - xmin), self.scaled_width - x_adjusted)
            h_adjusted = min((ymax - ymin), self.scaled_height - y_adjusted)
            # 转换为原始视频坐标
            scale_x = self.frame_width / (self.scaled_width * video_display_width)
            scale_y = self.frame_height / (self.scaled_height * video_display_height)

            # 使用round代替int，避免精度丢失
            xmin = round(x_adjusted * scale_x * video_display_width)
            xmax = round((x_adjusted + w_adjusted) * scale_x * video_display_width)
            ymin = round(y_adjusted * scale_y * video_display_height)
            ymax = round((y_adjusted + h_adjusted) * scale_y * video_display_height)

            # 确保坐标在有效范围内
            xmin = max(0, min(xmin, self.frame_width))
            xmax = max(0, min(xmax, self.frame_width))
            ymin = max(0, min(ymin, self.frame_height))
            ymax = max(0, min(ymax, self.frame_height))

            # 确保xmin < xmax, ymin < ymax
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            selection_rects.append((ymin, ymax, xmin, xmax))
        return selection_rects

    def video_coordinates_to_preview_coordinates(self, video_rects):
        """将原图 ``(xmin, xmax, ymin, ymax)`` 框转换为可编辑的预览比例坐标。"""
        if not self.frame_width or not self.frame_height:
            return []
        result = []
        for xmin, xmax, ymin, ymax in video_rects:
            preview_xmin = self.border_left + (xmin / self.frame_width) * self.scaled_width
            preview_xmax = self.border_left + (xmax / self.frame_width) * self.scaled_width
            preview_ymin = self.border_top + (ymin / self.frame_height) * self.scaled_height
            preview_ymax = self.border_top + (ymax / self.frame_height) * self.scaled_height
            result.append((
                max(0.0, min(1.0, preview_ymin)), max(0.0, min(1.0, preview_ymax)),
                max(0.0, min(1.0, preview_xmin)), max(0.0, min(1.0, preview_xmax)),
            ))
        return result

    def set_dragger_enabled(self, enabled):
        """设置拖动器是否可用"""
        self.enable_mouse_events = enabled
        self.video_display.setMouseTracking(enabled)
        self.video_display.setCursor(Qt.ArrowCursor)

    def save_selections_to_config(self):
        """保存所有选择框的相对位置和大小"""
        areas_str_parts = []

        for rect in self.selection_rects:
            ymin, ymax, xmin, xmax = rect
            # 直接使用比例值，四舍五入到4位小数
            areas_str_parts.append(f"{round(ymin,4)},{round(ymax,4)},{round(xmin,4)},{round(xmax,4)}")

        # 更新配置
        config.ocrSelectionRegions.value = ";".join(areas_str_parts)
        if len(config.ocrSelectionRegions.value) <= 0:
            config.ocrSelectionRegions.value = config.ocrSelectionRegions.defaultValue
        qconfig.save()

    def get_selection_rects(self):
        """获取所有选区"""
        return self.selection_rects

    def clear_selections(self):
        """清除所有选区"""
        self.selection_rects = []
        self.active_selection_index = -1
        self.update_preview_with_rect()
        self.selections_changed.emit(self.selection_rects)

    def __handle_delete_selection(self):
        """处理删除当前选区的逻辑"""
        if not self.enable_mouse_events:
            return False
        try:
            if self.active_selection_index >= 0 and self.selection_rects:
                # 删除当前活跃选区
                self.selection_rects.pop(self.active_selection_index)

                # 如果还有其他选区，将最后一个选区设为活跃选区
                if self.selection_rects:
                    self.active_selection_index = len(self.selection_rects) - 1
                else:
                    self.active_selection_index = -1

                # 更新显示
                self.update_preview_with_rect()

                # 发送选区变化信号
                self.selections_changed.emit(self.selection_rects)
                return True
            return False
        finally:
            # 获取当前鼠标位置
            global_pos = QCursor.pos()
            pos = self.video_display.mapFromGlobal(global_pos)
            self.update_cursor_shape(pos)

    def __handle_mark_for_ab_start(self):
        """处理标记AB分区起点的逻辑"""
        if not self._is_video_media or not self.enable_mouse_events:
            return False
        current_frame = self.video_slider.value() - 1
        if current_frame >= 0:
            # 起点只是一次尚未完成的区间编辑，不修改任何已确认区间；这样用户
            # 能明确看到“先定起点，再定终点”的状态，不会意外改写旧区段。
            self.current_ab_start = current_frame
            # 复用现有信号刷新外部的区间摘要；列表中保存的仍只有已确认区段。
            self.ab_sections_changed.emit(self.ab_sections)
            self.update_preview_with_rect()
            return True
        return False

    def __handle_mark_for_ab_end(self):
        """处理标记AB分区终点的逻辑"""
        if not self._is_video_media or not self.enable_mouse_events:
            return False
        current_frame = self.video_slider.value() - 1
        if current_frame >= 0 and self.current_ab_start >= 0:
            start, end = sorted((self.current_ab_start, current_frame))
            # range 的 stop 是开区间，因此 +1 才会包含用户选择的终点帧；同一
            # 帧也会生成长度为 1 的合法区间，适合只修一帧的场景。
            self.ab_sections.append(range(start, end + 1))
            self.current_ab_start = -1
            self.ab_sections_changed.emit(self.ab_sections)
            self.update_preview_with_rect()
            return True
        return False

    def __handle_delete_ab_section(self):
        """处理删除当前AB区块的逻辑"""
        if not self._is_video_media or not self.enable_mouse_events:
            return False
        current_frame = self.video_slider.value() - 1
        if current_frame >= 0 and self.ab_sections:
            # 查找当前帧所在的AB区块
            for i, section_range in enumerate(self.ab_sections):
                if current_frame in section_range:
                    # 删除该AB区块
                    self.ab_sections.pop(i)

                    # 如果当前有标记的起点，且在被删除的区块内，重置起点
                    if self.current_ab_start in section_range:
                        self.current_ab_start = -1

                    # 发送AB区块变化信号
                    self.ab_sections_changed.emit(self.ab_sections)

                    # 更新显示
                    self.update_preview_with_rect()
                    return True
        return False

    def __adjust_slider_value(self, delta):
        """调整视频滑块的值"""
        current_value = self.video_slider.value()
        max_value = self.video_slider.maximum()
        new_value = current_value + int(delta)

        # 确保新值在有效范围内
        if new_value < self.video_slider.minimum():
            new_value = self.video_slider.minimum()
        elif new_value > max_value:
            new_value = max_value

        # 设置新值
        self.video_slider.setValue(new_value)

    def eventFilter(self, obj, event):
        """事件过滤器，处理键盘事件"""
        if event.type() == QEvent.KeyPress:
            # 处理退格键和删除键
            if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
                if self.__handle_delete_selection():
                    return True
        # 对于其他事件，继续传递给父类处理
        return super().eventFilter(obj, event)

    def __init_context_menu(self):
        """初始化右键菜单"""
        self.context_menu = QMenu(self)

        # 设定区块起点动作
        self.action_mark_ab_start = QAction(tr['CleanCanvasWindow']['MarkABStart'], self)
        self.action_mark_ab_start.setShortcut("[")
        self.action_mark_ab_start.triggered.connect(self.__handle_mark_for_ab_start)
        self.context_menu.addAction(self.action_mark_ab_start)

        # 设定区块终点动作
        self.action_mark_ab_end = QAction(tr['CleanCanvasWindow']['MarkABEnd'], self)
        self.action_mark_ab_end.setShortcut("]")
        self.action_mark_ab_end.triggered.connect(self.__handle_mark_for_ab_end)
        self.context_menu.addAction(self.action_mark_ab_end)

        self.action_mark_ab_delete = QAction(tr['CleanCanvasWindow']['DeleteABSection'], self)
        self.action_mark_ab_delete.setShortcut("\\")
        self.action_mark_ab_delete.triggered.connect(self.__handle_delete_ab_section)
        self.context_menu.addAction(self.action_mark_ab_delete)

        self.action_clear_ab_sections = QAction("清除所有处理区间", self)
        self.action_clear_ab_sections.setShortcut("Ctrl+\\")
        self.action_clear_ab_sections.triggered.connect(self.clear_ab_sections)
        self.context_menu.addAction(self.action_clear_ab_sections)

        self.action_delete_selection = QAction(tr['CleanCanvasWindow']['DeleteSelection'], self)
        self.action_delete_selection.setShortcut("DELETE")
        self.action_delete_selection.triggered.connect(self.__handle_delete_selection)
        self.context_menu.addAction(self.action_delete_selection)

    def __update_context_menu_actions(self):
        """只展示当前媒体、当前帧和当前选区确实能够执行的右键操作。"""
        is_video = self._is_video_media
        current_frame = self.video_slider.value() - 1
        current_section_exists = any(current_frame in section for section in self.ab_sections)

        self.action_mark_ab_start.setVisible(is_video)
        # 没有起点时设置终点无效；已有区间可重新设置起点后再修改终点。
        self.action_mark_ab_end.setVisible(is_video and self.current_ab_start >= 0)
        self.action_mark_ab_delete.setVisible(is_video and current_section_exists)
        self.action_clear_ab_sections.setVisible(is_video and bool(self.ab_sections))
        self.action_delete_selection.setVisible(self.active_selection_index >= 0 and bool(self.selection_rects))

    def get_ab_sections(self):
        """获取AB分区标记"""
        return self.ab_sections

    def mark_ab_start(self):
        """公开的“设置处理起点”入口，供显式按钮与快捷键共用。"""
        return self.__handle_mark_for_ab_start()

    def mark_ab_end(self):
        """公开的“设置处理终点”入口；只在已有待确认起点时成功。"""
        return self.__handle_mark_for_ab_end()

    def get_pending_ab_start(self):
        """返回尚未配对终点的起始帧，``None`` 表示当前没有编辑中的区间。"""
        return self.current_ab_start if self.current_ab_start >= 0 else None

    def set_ab_sections(self, sections):
        """设置AB分区标记"""
        self.ab_sections = sections
        self.update_preview_with_rect()

    def clear_ab_sections(self):
        """清除所有AB分区标记"""
        had_sections = bool(self.ab_sections) or self.current_ab_start >= 0
        self.ab_sections = []
        self.current_ab_start = -1
        self.update_preview_with_rect()
        if had_sections:
            self.ab_sections_changed.emit(self.ab_sections)

    def closeEvent(self, event):
        """窗口关闭时断开信号连接"""
        try:
            # 断开信号连接
            self.shortcut_ab_start.activated.disconnect(self.__handle_mark_for_ab_start)
            self.shortcut_ab_end.activated.disconnect(self.__handle_mark_for_ab_end)
            self.shortcut_ab_delete.activated.disconnect(self.__handle_delete_ab_section)
            self.action_mark_ab_start.triggered.disconnect(self.__handle_mark_for_ab_start)
            self.action_mark_ab_end.triggered.disconnect(self.__handle_mark_for_ab_end)
            self.action_mark_ab_delete.triggered.disconnect(self.__handle_delete_ab_section)
            self.action_delete_selection.triggered.disconnect(self.__handle_delete_selection)
            self.shortcut_delete_selection.activated.disconnect(self.__handle_delete_selection)
        except Exception as e:
            print(f"Error during close window:", e)
        super().closeEvent(event)
