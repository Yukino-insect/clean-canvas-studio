# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
import os
import cv2
import numpy as np
import threading
import multiprocessing
import time
import traceback
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Slot, QRect, Signal
from PySide6 import QtCore, QtWidgets
from datetime import datetime
from qfluentwidgets import (
    PushButton, CardWidget, TextEdit, FluentIcon, InfoBar, InfoBarPosition,
    BodyLabel, SegmentedWidget,
)
from ui.setting_interface import SettingInterface
from ui.component.video_display_component import VideoDisplayComponent
from ui.component.media_preview_component import COMPARISON_DIVIDER_BGR, COMPARISON_DIVIDER_WIDTH
from ui.component.task_list_component import TaskListComponent, TaskStatus, TaskOptions
from ui.icon.my_fluent_icon import MyFluentIcon
from backend.config import config, tr
from backend.masking import MaskMode, MaskPlan, OcrPolygon, build_mask, filter_polygons_by_regions
from backend.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from backend.tools.process_manager import ProcessManager
from backend.tools.common_tools import get_readable_path, is_image_file, read_image

class HomeInterface(QWidget):
    # Carry the task index with every progress event.  Progress originates in
    # child/worker threads and can be delivered after the worker has already
    # advanced to the next item in the queue.
    progress_signal = Signal(int, int, bool)
    append_log_signal = Signal(list)
    update_preview_with_comp_signal = Signal(list)
    # Error delivery is also asynchronous; keep the originating index for the
    # same reason as progress updates so an old error cannot mark the next task
    # as failed.
    task_error_signal = Signal(int, object)
    toggle_buttons_signal = Signal(bool)  # True=显示运行按钮, False=显示停止按钮
    task_status_signal = Signal(int, object)  # (task_index, TaskStatus)
    select_task_signal = Signal(int)  # task_index
    ocr_detection_finished_signal = Signal(int, object, object)  # task index, regions, error
    def __init__(self, parent=None):
        # 这里的 parent 传的是父组件
        super().__init__(parent=parent)
        self.setObjectName("HomeInterface")
        # 初始化一些变量
        self.video_path = None
        self.video_cap = None
        self.fps = None
        self.frame_count = None
        self.frame_width = None
        self.frame_height = None
        self.se = None  # 后台字幕提取器

        # 字幕区域参数
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        # 添加自动滚动控制标志
        self.auto_scroll = True
        self._stop_event = threading.Event()  # 线程安全的停止信号
        self._worker_thread = None
        self.running_process = None
        self._video_cap_lock = threading.Lock()  # 保护 video_cap 的线程锁

        # 当前正在处理的任务索引
        # -1 代表没有任务正在处理
        self.current_processing_task_index = -1
        self.current_frame = None
        self._applying_ocr_result = False
        # 和画质增强页一致：该状态决定打开文件时的筛选器，以及哪些参数对
        # 当前即将创建的任务有意义。真正加载媒体后会再次按文件类型同步它。
        self._target_media_type = "image"

        self.__init_widgets()
        self.set_target_media_type("image")
        self.video_display_component.set_video_controls_visible(False)
        self.progress_signal.connect(self.update_progress)
        self.append_log_signal.connect(self.append_log)
        self.update_preview_with_comp_signal.connect(self.update_preview_with_comp)
        self.task_error_signal.connect(self.on_task_error)
        self.toggle_buttons_signal.connect(self._toggle_buttons)
        self.task_status_signal.connect(lambda idx, status: self.task_list_component.update_task_status(idx, status))
        self.select_task_signal.connect(self.task_list_component.select_task)
        self.ocr_detection_finished_signal.connect(self._on_ocr_detection_finished)

    def __init_widgets(self):
        """创建主页面"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 左侧视频区域
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        # 创建视频显示组件
        self.video_display_component = VideoDisplayComponent(self)
        self.video_display_component.ab_sections_changed.connect(self.ab_sections_changed)
        self.video_display_component.selections_changed.connect(self.selections_changed)
        left_layout.addWidget(self.video_display_component)

        # 获取视频显示和滑块的引用
        self.video_display = self.video_display_component.video_display
        self.video_slider = self.video_display_component.video_slider
        self.video_slider.valueChanged.connect(self.slider_changed)

        # 输出文本区域
        self.output_text = TextEdit()
        self.output_text.setMinimumHeight(150)
        self.output_text.setReadOnly(True)
        self.output_text.document().setDocumentMargin(10)
        # 连接滚动条值变化信号
        self.output_text.verticalScrollBar().valueChanged.connect(self.on_scroll_change)

        output_container = CardWidget(self)
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_text)
        output_container.setLayout(output_layout)
        left_layout.addWidget(output_container)

        main_layout.addLayout(left_layout, 2)

        # 右侧设置区域
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # 任务类型不再只是隐含在文件扩展名里。用户可先明确选择图片或视频，
        # 再打开对应媒体；视频专属的时间轴策略也只在视频模式中出现。
        media_mode_card = CardWidget(self)
        media_mode_layout = QHBoxLayout(media_mode_card)
        media_mode_layout.setContentsMargins(16, 10, 16, 10)
        media_mode_layout.addWidget(BodyLabel("修复任务类型", media_mode_card))
        media_mode_layout.addStretch(1)
        self.media_type_selector = SegmentedWidget(media_mode_card)
        self.media_type_selector.addItem("image", "图片修复", onClick=lambda: self.set_target_media_type("image"))
        self.media_type_selector.addItem("video", "视频修复", onClick=lambda: self.set_target_media_type("video"))
        self.media_type_selector.setCurrentItem("image")
        media_mode_layout.addWidget(self.media_type_selector)
        right_layout.addWidget(media_mode_card)

        # 设置容器
        settings_container = CardWidget(self)
        self.setting_interface = SettingInterface(settings_container)
        settings_container.setLayout(self.setting_interface)
        right_layout.addWidget(settings_container)

        # 添加任务列表容器
        task_list_container = CardWidget(self)
        task_list_layout = QHBoxLayout()
        task_list_layout.setContentsMargins(0, 0, 0, 0)
        task_list_layout.setSpacing(0)
        self.task_list_component = TaskListComponent(self)
        self.task_list_component.task_selected.connect(self.on_task_selected)
        self.task_list_component.task_deleted.connect(self.on_task_deleted)
        self.task_list_component.task_reset.connect(self.on_task_reset)
        task_list_layout.addWidget(self.task_list_component)
        task_list_container.setLayout(task_list_layout)
        right_layout.addWidget(task_list_container, 1)  # 占满剩余空间

        # 操作按钮容器
        button_container = CardWidget(self)
        # 按钮按自然内容宽度排列。上一版的等宽网格虽然避免了截断，却让短
        # 按钮内部出现大块无意义留白；使用两行自然宽度布局能同时保证文字完整
        # 与视觉比例稳定。
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(8)
        primary_row = QHBoxLayout()
        primary_row.setSpacing(8)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(8)

        self.file_button = PushButton(tr['SubtitleExtractorGUI']['Open'], self)
        self.file_button.setIcon(FluentIcon.FOLDER)
        self.file_button.clicked.connect(self.open_file)
        primary_row.addWidget(self.file_button)

        self.detect_ocr_button = PushButton("检测当前帧 OCR", self)
        self.detect_ocr_button.setIcon(FluentIcon.SEARCH)
        self.detect_ocr_button.setToolTip("检测当前预览帧中的文字位置，并将结果回显为可编辑选框。")
        self.detect_ocr_button.clicked.connect(self.detect_current_frame_ocr)
        primary_row.addWidget(self.detect_ocr_button)

        self.confirm_regions_button = PushButton("确认修复区域", self)
        self.confirm_regions_button.setIcon(FluentIcon.ACCEPT)
        self.confirm_regions_button.setToolTip("确认当前可编辑选框；确认后才能进入批量处理。")
        self.confirm_regions_button.clicked.connect(self.confirm_current_regions)
        primary_row.addWidget(self.confirm_regions_button)
        primary_row.addStretch(1)

        self.clear_regions_button = PushButton("清空选框", self)
        self.clear_regions_button.setIcon(FluentIcon.DELETE)
        self.clear_regions_button.setToolTip("清空所有 OCR/手工选框；之后可重新检测或重新手画。")
        self.clear_regions_button.clicked.connect(self.clear_current_regions)
        secondary_row.addWidget(self.clear_regions_button)

        self.run_button = PushButton(tr['SubtitleExtractorGUI']['Run'], self)
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.clicked.connect(self.run_button_clicked)
        secondary_row.addWidget(self.run_button)

        self.stop_button = PushButton(tr['SubtitleExtractorGUI']['Stop'], self)
        self.stop_button.setIcon(MyFluentIcon.Stop)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_button_clicked)

        secondary_row.addWidget(self.stop_button)
        secondary_row.addStretch(1)
        button_layout.addLayout(primary_row)
        button_layout.addLayout(secondary_row)

        button_container.setLayout(button_layout)
        right_layout.addWidget(button_container)

        main_layout.addLayout(right_layout, 1)

    def set_target_media_type(self, media_type: str):
        """切换待添加任务的媒体类型，并仅展示对此类型有效的配置。"""
        if media_type not in {"image", "video"}:
            raise ValueError(f"不支持的修复任务类型：{media_type}")
        self._target_media_type = media_type
        if hasattr(self, "media_type_selector"):
            self.media_type_selector.setCurrentItem(media_type)
        if hasattr(self, "setting_interface"):
            self.setting_interface.set_media_type(media_type)

    def on_scroll_change(self, value):
        """监控滚动条位置变化"""
        scrollbar = self.output_text.verticalScrollBar()
        # 如果滚动到底部，启用自动滚动
        if value == scrollbar.maximum():
            self.auto_scroll = True
        # 如果用户向上滚动，禁用自动滚动
        elif self.auto_scroll and value < scrollbar.maximum():
            self.auto_scroll = False


    def slider_changed(self, value):
        frame = None
        with self._video_cap_lock:
            if self.video_cap is not None and self.video_cap.isOpened():
                frame_no = self.video_slider.value()
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = self.video_cap.read()
                if not ret:
                    frame = None
        if frame is not None:
            # 更新预览图像
            self.update_preview(frame)
            task_index = self.task_list_component.get_current_task_index()
            raw_plan = self.task_list_component.get_task_option(task_index, TaskOptions.MASK_PLAN, None)
            if raw_plan:
                self._show_mask_preview(MaskPlan.from_dict(raw_plan))

    def ab_sections_changed(self, ab_sections):
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.AB_SECTIONS, ab_sections)

    def selections_changed(self, selections):
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS, selections)
        if not self._applying_ocr_result:
            # 任何移动、缩放、删除或新增框都会使之前的确认失效；否则用户很容易
            # 在编辑后误以为任务仍使用了旧的审核结果。
            self.task_list_component.update_task_option(get_current_task_index, TaskOptions.OCR_REVIEWED, False)
            self.task_list_component.update_task_option(get_current_task_index, TaskOptions.MANUAL_REGIONS, [])
            self.task_list_component.update_task_option(get_current_task_index, TaskOptions.MASK_PLAN, None)

    def detect_current_frame_ocr(self):
        """异步检测当前帧，完成后把 OCR 框回显到现有可编辑选框层。"""
        task_index = self.task_list_component.get_current_task_index()
        if task_index < 0 or self.current_frame is None:
            self.append_output("请先打开并选择一个图片或视频任务，再进行 OCR 检测。")
            return
        task = self.task_list_component.get_task(task_index)
        if not task:
            return
        frame = self.current_frame.copy()
        self.detect_ocr_button.setEnabled(False)
        self.append_output("正在检测当前帧文字位置；检测完成后请在预览中逐框确认、编辑或删除。")

        def detect():
            try:
                from backend.tools.subtitle_detect import SubtitleDetect
                polygons = SubtitleDetect(task.path, []).detect_polygons(frame)
                self.ocr_detection_finished_signal.emit(task_index, polygons, None)
            except Exception as exc:
                self.ocr_detection_finished_signal.emit(task_index, [], exc)

        threading.Thread(target=detect, daemon=True).start()

    @Slot(int, object, object)
    def _on_ocr_detection_finished(self, task_index, polygons, error):
        self.detect_ocr_button.setEnabled(True)
        if error is not None:
            self.append_output(f"OCR 检测失败：{error}")
            InfoBar.error("OCR 检测失败", str(error), parent=self, position=InfoBarPosition.TOP, duration=5000)
            return
        if task_index != self.task_list_component.get_current_task_index():
            self.append_output("OCR 检测已完成；请重新选择对应任务查看结果。")
            return
        regions = [polygon.bbox for polygon in polygons]
        preview_regions = self.video_display_component.video_coordinates_to_preview_coordinates(regions)
        self._applying_ocr_result = True
        try:
            self.video_display_component.set_selection_rects(preview_regions)
            self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, preview_regions)
            self.task_list_component.update_task_option(task_index, TaskOptions.OCR_POLYGONS, [polygon.to_dict() for polygon in polygons])
            self.task_list_component.update_task_option(task_index, TaskOptions.OCR_REVIEWED, False)
            self.task_list_component.update_task_option(task_index, TaskOptions.MANUAL_REGIONS, [])
            self.task_list_component.update_task_option(task_index, TaskOptions.MASK_PLAN, None)
        finally:
            self._applying_ocr_result = False
        self.append_output(
            f"OCR 检测到 {len(regions)} 个候选区域与精确多边形。黄色框仅是候选 ROI；"
            "最终擦除范围将按当前掩膜模式以红色半透明方式预览。可编辑框后再确认。"
        )
        self._show_mask_preview(self._build_mask_plan(task_index, preview_regions, polygons, preview_only=True))
        InfoBar.info("请确认 OCR 结果", f"已回显 {len(regions)} 个候选框；红色区域才是实际掩膜。", parent=self, position=InfoBarPosition.TOP, duration=6000)

    def _build_mask_plan(self, task_index, preview_regions, polygons=None, *, preview_only=False):
        """将 UI 选框作为 OCR 候选 ROI，构造可序列化的最终精确掩膜计划。"""
        source_regions = self.video_display_component.preview_coordinates_to_video_coordinates(preview_regions)
        candidate_regions = [(xmin, xmax, ymin, ymax) for ymin, ymax, xmin, xmax in source_regions]
        if polygons is None:
            raw_polygons = self.task_list_component.get_task_option(task_index, TaskOptions.OCR_POLYGONS, [])
            polygons = [OcrPolygon.from_dict(item) for item in raw_polygons]
        selected_polygons = filter_polygons_by_regions(polygons, candidate_regions)
        mode = MaskMode(config.maskMode.value)
        if mode is MaskMode.RECTANGLE:
            manual_regions = candidate_regions
        elif mode is MaskMode.MIXED:
            # 同一个候选 ROI 命中 OCR 多边形时，优先使用精确多边形；只有未命中
            # OCR 的手工框才被解释为“整块移除的符号/logo”。这样不会因混合模式
            # 又把正常文字框退化成大矩形。
            manual_regions = []
            for region in candidate_regions:
                xmin, xmax, ymin, ymax = region
                intersects_text = any(
                    not (polygon.bbox[1] < xmin or polygon.bbox[0] > xmax or polygon.bbox[3] < ymin or polygon.bbox[2] > ymax)
                    for polygon in selected_polygons
                )
                if not intersects_text:
                    manual_regions.append(region)
        else:
            manual_regions = []
        return MaskPlan(
            mode=mode,
            ocr_polygons=selected_polygons,
            candidate_regions=candidate_regions,
            manual_regions=manual_regions,
            expand_pixels=config.subtitleAreaDeviationPixel.value,
            refine_color_threshold=config.maskRefineColorThreshold.value,
            refine_min_component_area=config.maskRefineMinArea.value,
        )

    def _show_mask_preview(self, plan: MaskPlan):
        """红色半透明层回显最终掩膜，保持 current_frame 为未修改的源帧。"""
        if self.current_frame is None:
            return
        mask = build_mask(self.current_frame, plan)
        overlay = self.current_frame.copy()
        overlay[mask > 0] = (0, 0, 255)
        preview = cv2.addWeighted(self.current_frame, 0.65, overlay, 0.35, 0)
        # 青色线 = PaddleOCR 原始多边形，红色半透明 = 最终会交给修复引擎的掩膜。
        # 二者不同正是本阶段要让用户看见、并能够判断的关键信息。
        for polygon in plan.ocr_polygons:
            points = np.asarray(polygon.points, dtype=np.int32)
            if len(points) >= 3:
                cv2.polylines(preview, [points], isClosed=True, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        self.update_preview(preview, store_current=False)

    def confirm_current_regions(self):
        """冻结当前选框为本任务的直接掩膜区域，而非下一步 OCR 的候选 ROI。"""
        task_index = self.task_list_component.get_current_task_index()
        if task_index < 0:
            self.append_output("没有可确认的任务。")
            return
        # VideoDisplayComponent 的公开接口是 get_selection_rects()。此前使用
        # 了重构前不存在的名称 get_all_selections()，使“确认区域”在 OCR
        # 命中为 0 时也会直接抛 AttributeError。
        preview_regions = self.video_display_component.get_selection_rects()
        if not preview_regions:
            InfoBar.warning("没有修复区域", "请先执行 OCR 检测，或直接在预览上拖动鼠标画框。", parent=self, duration=4000)
            return
        plan = self._build_mask_plan(task_index, preview_regions)
        if plan.mode in {MaskMode.OCR_POLYGON, MaskMode.OCR_REFINED} and not plan.ocr_polygons:
            InfoBar.warning("没有 OCR 精确掩膜", "当前模式不会直接擦除候选框。请先检测 OCR，或切换为手工矩形/混合模式。", parent=self, duration=5000)
            return
        self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, preview_regions)
        self.task_list_component.update_task_option(task_index, TaskOptions.MANUAL_REGIONS, plan.manual_regions)
        self.task_list_component.update_task_option(task_index, TaskOptions.MASK_PLAN, plan.to_dict())
        self.task_list_component.update_task_option(task_index, TaskOptions.USE_OCR_DETECTION, False)
        self.task_list_component.update_task_option(task_index, TaskOptions.OCR_REVIEWED, True)
        # 将模式和设备冻结到任务，避免用户切换下拉框后，已审核任务在不知情的
        # 情况下被另一种算法执行。
        self.task_list_component.update_task_option(task_index, TaskOptions.INPAINT_MODE, config.inpaintMode.value)
        self.task_list_component.update_task_option(task_index, TaskOptions.AI_DEVICE, config.aiDevice.value)
        self.task_list_component.update_task_option(task_index, TaskOptions.MODEL_DIRECTORY, config.modelDirectory.value or None)
        self.task_list_component.update_task_option(task_index, TaskOptions.VIDEO_MASK_STRATEGY, config.videoMaskStrategy.value)
        self._show_mask_preview(plan)
        self.append_output(
            f"已确认精确掩膜：模式={plan.mode.value}，OCR 多边形={len(plan.ocr_polygons)}，"
            f"手工矩形={len(plan.manual_regions)}。运行时只使用该冻结计划。"
        )
        InfoBar.success("精确掩膜已确认", "红色区域是实际擦除范围；编辑候选框或参数后需要重新确认。", parent=self, duration=5000)

    def clear_current_regions(self):
        if self.task_list_component.get_current_task_index() < 0:
            return
        self.video_display_component.clear_selections()
        self.task_list_component.update_task_option(self.task_list_component.get_current_task_index(), TaskOptions.OCR_POLYGONS, [])
        self.task_list_component.update_task_option(self.task_list_component.get_current_task_index(), TaskOptions.MASK_PLAN, None)
        if self.current_frame is not None:
            self.update_preview(self.current_frame)
        self.append_output("已清空当前任务的修复选框；请重新检测 OCR 或手动画框后确认。")

    def on_task_selected(self, index, file_path):
        """处理任务被选中事件

        Args:
            index: 任务索引
            file_path: 文件路径
        """
        # 加载选中的视频进行预览
        self.load_video(file_path)
        ab_sections = self.task_list_component.get_task_option(index, TaskOptions.AB_SECTIONS, [])
        self.video_display_component.set_ab_sections(ab_sections)
        selections = self.task_list_component.get_task_option(index, TaskOptions.SUB_AREAS, [])
        if len(selections) <= 0:
            self.video_display_component.load_selections_from_config()
        else:
            self.video_display_component.set_selection_rects(selections)
        raw_plan = self.task_list_component.get_task_option(index, TaskOptions.MASK_PLAN, None)
        if raw_plan and self.current_frame is not None:
            self._show_mask_preview(MaskPlan.from_dict(raw_plan))

    def on_task_deleted(self, index):
        """处理任务被删除事件

        Args:
            index: 任务索引
        """
        # 如果删除的是正在处理的任务，则需要更新状态
        if index == self.current_processing_task_index:
            self.current_processing_task_index = -1

        task = self.task_list_component.get_task(0)
        if task:
            # 如果还有任务，选中第一个
            self.task_list_component.select_task(0)

    def on_task_reset(self, index):
        """重试任务时采用当前页面模式/设备，而不是沿用可能失败的旧配置。"""
        task = self.task_list_component.get_task(index)
        if task is None:
            return
        task.options[TaskOptions.INPAINT_MODE.value] = config.inpaintMode.value
        task.options[TaskOptions.AI_DEVICE.value] = config.aiDevice.value
        task.options[TaskOptions.MODEL_DIRECTORY.value] = config.modelDirectory.value or None
        self.append_output(f"已重置任务：{task.name}，将使用当前修复模式与设备重新执行。")

    def update_preview(self, frame, *, store_current=True):
        if store_current:
            self.current_frame = frame.copy()
        # 先缩放图像
        resized_frame = self._img_resize(frame)

        # 设置视频参数
        self.video_display_component.set_video_parameters(
            self.frame_width, self.frame_height,
            self.scaled_width if hasattr(self, 'scaled_width') else None,
            self.scaled_height if hasattr(self, 'scaled_height') else None,
            self.border_left if hasattr(self, 'border_left') else 0,
            self.border_top if hasattr(self, 'border_top') else 0,
            self.fps if self.fps is not None else 30,
        )

        # 更新视频显示（这会同时保存current_pixmap）
        self.video_display_component.update_video_display(resized_frame)

    def _img_resize(self, image):
        height, width = image.shape[:2]
        # 逻辑画布跟随源媒体比例。横屏、竖屏和方形图片都以自身比例显示，
        # 而不是先被缩放并填黑边到固定的 16:9 图像中。
        self.video_display_component.set_media_aspect_ratio(width / height)
        video_preview_width = self.video_display_component.video_preview_width
        video_preview_height = self.video_display_component.video_preview_height
        # 计算等比缩放后的尺寸
        target_ratio = video_preview_width / video_preview_height
        image_ratio = width / height

        if image_ratio > target_ratio:
            # 宽度适配，高度按比例缩放
            new_width = video_preview_width
            new_height = int(new_width / image_ratio)
            top_border = (video_preview_height - new_height) // 2
            bottom_border = video_preview_height - new_height - top_border
            left_border = 0
            right_border = 0
        else:
            # 高度适配，宽度按比例缩放
            new_height = video_preview_height
            new_width = int(new_height * image_ratio)
            left_border = (video_preview_width - new_width) // 2
            right_border = video_preview_width - new_width - left_border
            top_border = 0
            bottom_border = 0

        # 先缩放图像
        resized = cv2.resize(image, (new_width, new_height))

        # 添加黑边以填充到目标尺寸
        padded = cv2.copyMakeBorder(
            resized,
            top_border, bottom_border,
            left_border, right_border,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # 保存边框信息，用于坐标转换
        self.border_left = left_border / video_preview_width
        self.border_right = right_border / video_preview_width
        self.border_top = top_border / video_preview_height
        self.border_bottom = bottom_border / video_preview_height
        self.original_width = width
        self.original_height = height
        self.is_vertical = width < height
        self.scaled_width = new_width / video_preview_width
        self.scaled_height = new_height / video_preview_height

        return padded

    def stop_button_clicked(self):
        try:
            self._stop_event.set()
            running_process = self.running_process
            if running_process:
                ProcessManager.instance().terminate_by_process(running_process)
            # 更新任务状态为待处理
            if self.current_processing_task_index >= 0:
                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.PENDING)
        finally:
            self.running_process = None
            self.run_button.setVisible(True)
            self.stop_button.setVisible(False)

    @Slot(bool)
    def _toggle_buttons(self, show_run):
        """线程安全地切换按钮可见性"""
        self.run_button.setVisible(show_run)
        self.stop_button.setVisible(not show_run)

    def run_button_clicked(self):
        # 如果任务队列为空
        if not self.task_list_component.get_pending_tasks():
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return

        # OCR 的角色是辅助定位而非不可见的后台黑盒。每个任务必须先经过用户
        # 确认（OCR 回显后编辑，或直接手工画框后确认）才能真正进入子进程。
        for index, task in self.task_list_component.get_pending_tasks():
            raw_plan = task.options.get(TaskOptions.MASK_PLAN.value)
            if raw_plan:
                plan = MaskPlan.from_dict(raw_plan)
                plan_is_current = (
                    plan.mode.value == config.maskMode.value
                    and plan.expand_pixels == config.subtitleAreaDeviationPixel.value
                    and plan.refine_color_threshold == config.maskRefineColorThreshold.value
                    and plan.refine_min_component_area == config.maskRefineMinArea.value
                )
                if not plan_is_current:
                    self.task_list_component.update_task_option(index, TaskOptions.OCR_REVIEWED, False)
            if not task.options.get(TaskOptions.OCR_REVIEWED.value, False):
                self.task_list_component.select_task(index)
                InfoBar.warning(
                    "需要确认修复区域",
                    "请先检测 OCR 或手动画框，检查选框后点击“确认修复区域”。",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=6000,
                )
                return

        # 处理任务
        try:
            # 获取所有待执行的任务
            pending_task_indices = [index for index, _task in self.task_list_component.get_pending_tasks()]
            if not pending_task_indices:
                return

            self._stop_event.clear()
            self.toggle_buttons_signal.emit(False)
            # 开启后台线程处理视频
            def task():
                try:
                    # TaskStatus is updated in the GUI thread through queued Qt
                    # signals.  A worker must not use that asynchronously updated
                    # state to decide what to run next, otherwise a quick task can
                    # be picked a second time before its completion signal is
                    # handled.  The queue is intentionally fixed at start time;
                    # controls are disabled while it runs.
                    for task_index in pending_task_indices:
                        if self._stop_event.is_set():
                            break
                        try:
                            task_item = self.task_list_component.get_task(task_index)
                            if task_item is None or task_item.status != TaskStatus.PENDING:
                                continue
                            # 更新当前处理的任务索引
                            self.current_processing_task_index = task_index
                            if not self.load_video(task_item.path): # 如果加载视频失败
                                self.append_log_signal.emit([tr['SubtitleExtractorGUI']['OpenVideoFailed'].format(task_item.path)])
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)
                                continue

                            self.video_display_component.save_selections_to_config()

                            # 更新任务状态为运行中
                            self.progress_signal.emit(self.current_processing_task_index, 1, False)

                            # 选中当前任务
                            self.select_task_signal.emit(self.current_processing_task_index)

                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None

                            self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.PROCESSING)
                            options = {}
                            for key in task_item.options:
                                value = task_item.options[key]
                                if key == TaskOptions.SUB_AREAS.value:
                                    value = self.video_display_component.preview_coordinates_to_video_coordinates(value)
                                options[key] = value
                            options.setdefault(TaskOptions.INPAINT_MODE.value, config.inpaintMode.value)
                            options.setdefault(TaskOptions.AI_DEVICE.value, config.aiDevice.value)
                            options.setdefault(TaskOptions.MODEL_DIRECTORY.value, config.modelDirectory.value or None)
                            # 清理缓存, 使用动态路径
                            task_item.output_path = None
                            output_path = task_item.output_path
                            process = self.run_subtitle_remover_process(
                                task_item.path,
                                output_path,
                                options,
                                self.current_processing_task_index,
                            )

                            # 检查是否在处理过程中被停止
                            if self._stop_event.is_set():
                                break

                            # 更新任务状态为已完成
                            if process.exitcode == 0:
                                # 任务完成, 更新输出路径为只读
                                task_item.output_path = output_path
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.COMPLETED)
                                self.progress_signal.emit(self.current_processing_task_index, 100, True)
                            else:
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)

                        except Exception as e:
                            print(e)
                            self.append_log_signal.emit([f"Error: {e}"])
                            # 更新任务状态为失败
                            if self.current_processing_task_index >= 0:
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)
                            break
                        finally:
                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None
                            time.sleep(1)
                finally:
                    self.toggle_buttons_signal.emit(True)

            self._worker_thread = threading.Thread(target=task, daemon=True)
            self._worker_thread.start()
        except Exception as e:
            print(traceback.format_exc())
            self.append_log_signal.emit([f"Error: {e}"])
            self.toggle_buttons_signal.emit(True)

    @staticmethod
    def remover_process(queue, video_path, output_path, options):
        """
        在子进程中执行字幕提取的函数

        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 选项
        """
        sr = None
        try:
            from backend.main import SubtitleRemover
            sr = SubtitleRemover(video_path, True)
            sr.video_out_path = output_path
            for key in options:
                setattr(sr, key, options[key])
            # 这个 queue 通过闭包传递给了 lambda 中
            sr.add_progress_listener(lambda progress, isFinished: SubtitleRemoverRemoteCall.remote_call_update_progress(queue, progress, isFinished))
            sr.append_output = lambda *args: SubtitleRemoverRemoteCall.remote_call_append_log(queue, args)
            sr.manage_process = lambda pid: SubtitleRemoverRemoteCall.remote_call_manage_process(queue, pid)
            sr.update_preview_with_comp = lambda *args: SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(queue, args)
            sr.run()
        except Exception as e:
            traceback.print_exc()
            SubtitleRemoverRemoteCall.remote_call_catch_error(queue, e)
            # The parent process uses the exit code as the authoritative result.
            # Re-raise after reporting the actionable error so a failed repair
            # can never be promoted to COMPLETED by a stale queued UI status.
            raise
        finally:
            if sr:
                sr.isFinished = True
                sr.vsf_running = False
            SubtitleRemoverRemoteCall.remote_call_finish(queue)

    def run_subtitle_remover_process(self, video_path, output_path, options, task_index):
        """
        使用多进程执行字幕提取，并等待进程完成

        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 任务选项
            task_index: 当前队列任务的稳定索引
        """
        subtitle_remover_remote_caller = SubtitleRemoverRemoteCall()
        subtitle_remover_remote_caller.register_update_progress_callback(
            lambda progress, finished: self.progress_signal.emit(task_index, progress, finished)
        )
        subtitle_remover_remote_caller.register_log_callback(self.append_log_signal.emit)
        subtitle_remover_remote_caller.register_update_preview_with_comp_callback(self.update_preview_with_comp_signal.emit)
        subtitle_remover_remote_caller.register_error_callback(
            lambda error: self.task_error_signal.emit(task_index, error)
        )
        process = multiprocessing.Process(
            target=HomeInterface.remover_process,
            args=(subtitle_remover_remote_caller.queue, video_path, output_path, options)
        )
        try:
            if self._stop_event.is_set():
                return process
            process.start()
            ProcessManager.instance().add_process(process)
            self.running_process = process
            process.join()
            print(f"Process exited with code {process.exitcode}")
        finally:
            subtitle_remover_remote_caller.stop()
        return process

    @Slot()
    def processing_finished(self):
        pending_tasks = self.task_list_component.get_pending_tasks()
        if pending_tasks:
            # 还有待执行任务, 忽略
            return
        # 处理完成后恢复界面可用性
        self.run_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.se = None
        # 重置视频滑块
        self.video_slider.setValue(1)
        # 重置当前处理任务索引
        self.current_processing_task_index = -1

    @Slot(int, int, bool)
    def update_progress(self, task_index, progress_total, isFinished):
        try:
            if self.task_list_component.get_task(task_index) is None:
                return
            # 把百分比进度换算成视频滑块的位置
            if task_index == self.current_processing_task_index:
                pos = min(self.frame_count - 1, int(progress_total / 100 * self.frame_count))
                if pos != self.video_slider.value():
                    self.video_slider.blockSignals(True)
                    self.video_slider.setValue(pos)
                    self.video_slider.blockSignals(False)

            # Use the event's stable task index rather than the mutable
            # ``current_processing_task_index``.  This keeps a previous task's
            # final 100% event from being painted onto the following task.
            self.task_list_component.update_task_progress(task_index, progress_total)

            # Completion is decided by the worker after the child process exits,
            # not by an in-process progress callback.  A callback can reach 100%
            # just before teardown, while the final status signal is still queued.
        except Exception as e:
            # 捕获任何异常，防止崩溃
            print(f"更新进度时出错: {str(e)}")

    @Slot(list)
    def append_log(self, log):
        self.append_output(*log)

    def append_output(self, *args):
        """添加文本到输出区域并控制滚动
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        # 将所有参数转换为字符串并用空格连接
        text = ' '.join(str(arg) for arg in args).rstrip()
        timestamp = datetime.now().strftime('%H:%M:%S')
        # 转义HTML特殊字符
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # 根据内容判断消息类型并着色
        if '错误' in text or 'Error' in text or '失败' in text or 'Failed' in text:
            color = '#e74c3c'
        elif '成功' in text or '完成' in text or 'Success' in text or 'Finished' in text:
            color = '#27ae60'
        elif '警告' in text or 'Warning' in text:
            color = '#f39c12'
        else:
            color = '#2980b9'
        html = f'<span style="color:#888;">[{timestamp}]</span> <span style="color:{color};">{escaped}</span><br>'
        self.output_text.append(html)
        print(*args)  # 保持原始的 print 行为
        # 如果启用了自动滚动，则滚动到底部
        if self.auto_scroll:
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(list)
    def update_preview_with_comp(self, args):
        """更新执行时预览"""
        # frame_ori 原始视频帧
        # frame_comp 经过处理后的帧，通常是去字幕、修复之后的结果
        frame_ori, frame_comp = args
        if self.current_processing_task_index >= 0:
            subtitle_areas = self.task_list_component.get_task_option(self.current_processing_task_index, TaskOptions.SUB_AREAS, [])
            if len(subtitle_areas) > 0:
                subtitle_areas = self.video_display_component.preview_coordinates_to_video_coordinates(subtitle_areas)
                if frame_ori is frame_comp:
                    frame_ori = frame_ori.copy()
                for rect in subtitle_areas:
                    ymin, ymax, xmin, xmax = rect
                    cv2.rectangle(frame_ori, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 内容修复和画质增强使用同一条中性分隔线。若两个帧的尺寸因
        # 上游处理不同而不一致，先将处理前帧适配到处理后帧，保持左右可比。
        if frame_ori.shape[:2] != frame_comp.shape[:2]:
            frame_ori = cv2.resize(
                frame_ori,
                frame_comp.shape[1::-1],
                interpolation=cv2.INTER_LANCZOS4,
            )
        divider = np.full(
            (frame_comp.shape[0], COMPARISON_DIVIDER_WIDTH, 3),
            COMPARISON_DIVIDER_BGR,
            dtype=np.uint8,
        )
        preview_frame = cv2.hconcat([frame_ori, divider, frame_comp])
        # 先缩放图像
        resized_frame = self._img_resize(preview_frame)
        # 更新视频显示（这会同时保存current_pixmap）
        self.video_display_component.update_video_display(resized_frame, draw_selection=False)
        self.video_display_component.set_dragger_enabled(False)

    @Slot(int, object)
    def on_task_error(self, task_index, e):
        self.append_output(tr['SubtitleExtractorGUI']['ErrorDuringProcessing'].format(str(e)))
        if self.task_list_component.get_task(task_index) is not None:
            self.task_list_component.update_task_status(task_index, TaskStatus.FAILED)

    def load_video(self, video_path):
        self.video_path = video_path
        with self._video_cap_lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
        # 如果是图片文件，直接走图片加载路径
        if is_image_file(video_path):
            return self.load_as_picture(video_path)
        with self._video_cap_lock:
            self.video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
            if not self.video_cap.isOpened():
                self.video_cap = None
                return self.load_as_picture(video_path)
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.release()
                self.video_cap = None
                return self.load_as_picture(video_path)
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)

        self.update_preview(frame)
        self.set_target_media_type("video")
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(True)
        self.video_display_component.set_video_controls_visible(True)
        return True

    def load_as_picture(self, path):
        if not is_image_file(path):
            return False
        self.video_path = path
        self.video_cap = None
        frame = read_image(get_readable_path(path))
        if frame is None:
            return False
        self.frame_count = 1
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.fps = 1
        self.update_preview(frame)
        self.set_target_media_type("image")
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(True)
        self.video_display_component.set_video_controls_visible(False)
        return True


    def open_file(self):
        is_picture = self._target_media_type == "image"
        media_filter = (
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;所有文件 (*.*)"
            if is_picture else
            "视频文件 (*.mp4 *.flv *.wmv *.avi *.mkv *.mov);;所有文件 (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择需要修复的图片" if is_picture else "选择需要修复的视频",
            "",
            media_filter,
        )
        if files:
            files_loaded = []
            # 倒序打开, 确保第一个视频截图显示在屏幕上
            for path in reversed(files):
                if self.load_video(path):
                    self.append_output(f"成功打开媒体: {path}")
                    files_loaded.append(path)
                else:
                    self.append_output(f"无法打开媒体: {path}")
            # 正序添加, 确保任务列表顺序一致
            for path in reversed(files_loaded):
                # 添加到任务列表
                self.task_list_component.add_task(path)
                index = max(0, self.task_list_component.find_task_index_by_path(path))
                self.task_list_component.select_task(index)

    def closeEvent(self, event):
        """窗口关闭时断开信号连接并清理资源"""
        try:
            # 通知 worker 线程停止
            self._stop_event.set()
            # 终止子进程
            ProcessManager.instance().terminate_all()
            # 等待 worker 线程结束（最多5秒）
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5)

            # 断开信号连接
            self.progress_signal.disconnect(self.update_progress)
            self.append_log_signal.disconnect(self.append_log)
            self.update_preview_with_comp_signal.disconnect(self.update_preview_with_comp)
            self.task_error_signal.disconnect(self.on_task_error)
            self.toggle_buttons_signal.disconnect(self._toggle_buttons)
            self.video_display_component.video_slider.valueChanged.disconnect(self.slider_changed)
            self.video_display_component.ab_sections_changed.disconnect(self.ab_sections_changed)
            self.video_display_component.selections_changed.disconnect(self.selections_changed)
            # 释放视频资源
            with self._video_cap_lock:
                if self.video_cap:
                    self.video_cap.release()
                    self.video_cap = None
        except Exception as e:
            print(f"Error during close window:", e)
        super().closeEvent(event)
