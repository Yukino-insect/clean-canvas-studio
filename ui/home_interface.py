# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
import os
import copy
import cv2
import numpy as np
import threading
import multiprocessing
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from queue import Empty, Queue
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Slot, QRect, Signal
from PySide6 import QtCore, QtGui, QtWidgets
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
from backend.masking import MaskMode, MaskPlan, OcrPolygon, build_mask
from ui.component.color_palette_dialog import ColorPaletteDialog
from backend.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from backend.tools.process_manager import ProcessManager
from backend.tools.common_tools import get_readable_path, is_image_file, read_image


def detect_ocr_in_process(result_queue, task_path, frame):
    """在独立进程中检测单帧 OCR，避免模型推理占用 GUI 进程。

    PaddleOCR 的模型初始化及 CPU 推理会大量占用 Python/原生线程。即使从
    ``threading.Thread`` 调用，它仍可能因 GIL 与 CPU 竞争而使主事件循环
    迟迟得不到调度。进程边界既隔离了该负载，也保证子进程不会直接触碰 Qt
    控件；结果只通过可序列化的数据返回。
    """
    try:
        # OCR 只是交互式预览阶段的后台工作；在 CPU 核心紧张时，让窗口事件
        # 循环优先获得调度时间，避免“进程分开了但拖动/点击仍明显迟滞”。
        try:
            if os.name == "nt":
                import ctypes
                below_normal_priority = 0x00004000
                ctypes.windll.kernel32.SetPriorityClass(
                    ctypes.windll.kernel32.GetCurrentProcess(),
                    below_normal_priority,
                )
            else:
                os.nice(5)
        except OSError:
            # 优先级属于体验优化；系统拒绝调整时仍应正常完成检测。
            pass

        from backend.tools.subtitle_detect import SubtitleDetect

        polygons = SubtitleDetect(task_path, []).detect_polygons(frame)
        result_queue.put({
            "polygons": [polygon.to_dict() for polygon in polygons],
            "error": None,
        })
    except BaseException as exc:
        # Exception 对象本身不一定可 pickle；保留类型可使错误提示仍可诊断。
        result_queue.put({
            "polygons": [],
            "error": f"{type(exc).__name__}: {exc}",
        })


class _ProcessLogWriter:
    """把子进程的 stdout/stderr 按行转发到内容修复日志板。

    第三方库直接 ``print`` 的警告不会经过 ``SubtitleRemover.append_output``。
    这些输出若只留在启动控制台，用户既无法判断初始化是否仍在进行，也无法
    得知真正的失败原因。tqdm 的回车式进度条由独立的百分比控件负责，因此
    不重复转发，避免每帧淹没日志板。
    """

    def __init__(self, queue):
        self.queue = queue
        self._buffer = ""

    def write(self, text):
        text = str(text)
        if not text:
            return 0
        if "\r" in text and "Subtitle Removing:" in text:
            return len(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                SubtitleRemoverRemoteCall.remote_call_append_log(self.queue, [line])
        return len(text)

    def flush(self):
        line, self._buffer = self._buffer.strip(), ""
        if line and "Subtitle Removing:" not in line:
            SubtitleRemoverRemoteCall.remote_call_append_log(self.queue, [line])


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
        self._repair_task_queue = None
        self._repair_queue_lock = threading.Lock()
        self._accepting_repair_tasks = False
        self.running_process = None
        self._video_cap_lock = threading.Lock()  # 保护 video_cap 的线程锁
        # OCR 使用独立进程而不是 Python 线程：模型推理不应抢占 Qt 主线程。
        # 定时器仅在主线程轮询 IPC 队列，因此所有控件更新仍遵守 Qt 线程规则。
        self._ocr_process = None
        self._ocr_process_id = None
        self._ocr_result_queue = None
        self._ocr_task_index = -1
        self._ocr_exit_poll_count = 0
        self._ocr_result_timer = QtCore.QTimer(self)
        self._ocr_result_timer.setInterval(50)
        self._ocr_result_timer.timeout.connect(self._poll_ocr_detection_result)

        # 仅用于进度滑块、预览和停止按钮定位当前活动项；每个任务自己的
        # status/progress/execution_options 保存在 Task 中，不能以此作为状态源。
        # -1 代表没有活动预览任务。
        self.current_processing_task_index = -1
        self.current_frame = None
        self._applying_ocr_result = False
        # 和画质增强页一致：该状态决定打开文件时的筛选器，以及哪些参数对
        # 当前即将创建的任务有意义。真正加载媒体后会再次按文件类型同步它。
        self._target_media_type = "image"

        self.__init_widgets()
        self.set_target_media_type("image")
        self.progress_signal.connect(self.update_progress)
        self.append_log_signal.connect(self.append_log)
        self.update_preview_with_comp_signal.connect(self.update_preview_with_comp)
        self.task_error_signal.connect(self.on_task_error)
        self.toggle_buttons_signal.connect(self._toggle_buttons)
        self.task_status_signal.connect(lambda idx, status: self.task_list_component.update_task_status(idx, status))
        self.select_task_signal.connect(self.task_list_component.select_task)
        self._connect_pending_task_config_sync()
        self._update_mask_mode_controls()

    def _connect_pending_task_config_sync(self):
        """让尚未执行的内容修复任务跟随页面设置，并记录每次同步。"""
        config.inpaintMode.valueChanged.connect(
            lambda value: self._sync_pending_runtime_option(TaskOptions.INPAINT_MODE, value, "修复模式")
        )
        config.aiDevice.valueChanged.connect(
            lambda value: self._sync_pending_runtime_option(TaskOptions.AI_DEVICE, value, "AI 设备")
        )
        config.modelDirectory.valueChanged.connect(
            lambda value: self._sync_pending_runtime_option(TaskOptions.MODEL_DIRECTORY, value or None, "AI 模型目录")
        )
        config.videoMaskStrategy.valueChanged.connect(
            lambda value: self._sync_pending_runtime_option(TaskOptions.VIDEO_MASK_STRATEGY, value, "动态 OCR 检测频率")
        )
        config.videoDynamicOcr.valueChanged.connect(
            lambda value: self._sync_pending_runtime_option(TaskOptions.VIDEO_DYNAMIC_OCR, value, "动态 OCR 字幕定位")
        )
        for config_item, option, label in (
            (config.dynamicOcrRefineMask, TaskOptions.DYNAMIC_OCR_REFINE_MASK, "动态 OCR 字幕像素精细定位"),
            (config.dynamicOcrForeground, TaskOptions.DYNAMIC_OCR_FOREGROUND, "动态 OCR 字幕前景色"),
            (config.dynamicOcrBackground, TaskOptions.DYNAMIC_OCR_BACKGROUND, "动态 OCR 字幕背景色"),
            (config.dynamicOcrRefineTolerance, TaskOptions.DYNAMIC_OCR_REFINE_TOLERANCE, "动态 OCR 颜色容差"),
            (config.dynamicOcrOutlineExpand, TaskOptions.DYNAMIC_OCR_OUTLINE_EXPAND, "动态 OCR 描边扩张"),
        ):
            config_item.valueChanged.connect(
                lambda value, task_option=option, task_label=label: self._sync_pending_runtime_option(
                    task_option, value, task_label,
                )
            )
        config.videoDynamicOcr.valueChanged.connect(lambda _value: self._update_mask_mode_controls())
        config.videoMaskStrategy.valueChanged.connect(lambda _value: self._update_mask_mode_controls())
        for config_item in (
            config.dynamicOcrRefineMask, config.dynamicOcrForeground,
            config.dynamicOcrBackground, config.dynamicOcrRefineTolerance,
            config.dynamicOcrOutlineExpand,
        ):
            config_item.valueChanged.connect(self._refresh_dynamic_ocr_refine_controls)
        config.dynamicOcrRefineMask.valueChanged.connect(lambda _value: self._update_mask_mode_controls())

        # 编辑方式与参数只影响尚未确认的当前编辑区。确认后它们会被写入
        # MaskPlan，运行任务时必须坚持该冻结快照，不能因随后改设置而变形。
        config.maskMode.valueChanged.connect(self._switch_mask_editor)
        for config_item in (
            config.subtitleAreaDeviationPixel, config.brushRadius,
            config.colorMaskForeground, config.colorMaskBackground, config.colorMaskTolerance,
            config.colorMaskOutlineExpand,
        ):
            config_item.valueChanged.connect(self._refresh_mask_editor_after_setting_change)
        config.brushRadius.valueChanged.connect(self.video_display_component.set_brush_radius)

        # OCR 阈值会改变检测出的候选轮廓；保留旧多边形继续确认会造成“设置已
        # 修改、结果仍是旧规则”的假象，因此必须要求重新检测。
        for config_item in (
            config.ocrPixelThreshold,
            config.ocrBoxThreshold,
            config.ocrUnclipRatio,
        ):
            config_item.valueChanged.connect(self._invalidate_pending_ocr_results)

    def _current_runtime_options(self):
        """返回内容修复任务在执行时使用的可序列化配置快照。"""
        return {
            TaskOptions.INPAINT_MODE: config.inpaintMode.value,
            TaskOptions.AI_DEVICE: config.aiDevice.value,
            TaskOptions.MODEL_DIRECTORY: config.modelDirectory.value or None,
            TaskOptions.VIDEO_MASK_STRATEGY: config.videoMaskStrategy.value,
            TaskOptions.VIDEO_DYNAMIC_OCR: config.videoDynamicOcr.value,
            TaskOptions.DYNAMIC_OCR_REFINE_MASK: config.dynamicOcrRefineMask.value,
            TaskOptions.DYNAMIC_OCR_FOREGROUND: config.dynamicOcrForeground.value,
            TaskOptions.DYNAMIC_OCR_BACKGROUND: config.dynamicOcrBackground.value,
            TaskOptions.DYNAMIC_OCR_REFINE_TOLERANCE: config.dynamicOcrRefineTolerance.value,
            TaskOptions.DYNAMIC_OCR_OUTLINE_EXPAND: config.dynamicOcrOutlineExpand.value,
        }

    def _apply_runtime_options(self, task_index):
        for option, value in self._current_runtime_options().items():
            self.task_list_component.update_task_option(task_index, option, value)

    def _enqueue_confirmed_repair_task(self, task_index):
        """将已审核的新任务追加到正在运行的内容修复队列。"""
        with self._repair_queue_lock:
            if self._stop_event.is_set() or not self._accepting_repair_tasks or self._repair_task_queue is None:
                return False
            self._repair_task_queue.put(task_index)
        task = self.task_list_component.get_task(task_index)
        if task is not None:
            self.append_output(f"任务已加入当前内容修复队列：{task.name}；将在当前任务结束后自动执行。")
        return True

    @staticmethod
    def _strip_refine_from_sampled_dynamic_ocr(options: dict, *, is_video: bool) -> bool:
        """Remove an incompatible refine flag from legacy/external task snapshots.

        The visible selector cannot combine sampling with pixel refinement, but
        pending tasks may have been saved by an earlier version or an external
        importer.  Keep the task's sampling strategy authoritative and clear
        only the incompatible refine flag before the immutable execution copy
        is handed to the child process.
        """
        dynamic = bool(options.get(TaskOptions.VIDEO_DYNAMIC_OCR.value, False)) and is_video
        sampled = options.get(TaskOptions.VIDEO_MASK_STRATEGY.value) == "sampled_ocr"
        refine_key = TaskOptions.DYNAMIC_OCR_REFINE_MASK.value
        if dynamic and sampled and bool(options.get(refine_key, False)):
            options[refine_key] = False
            return True
        return False

    def _sync_pending_runtime_option(self, option, value, label):
        affected = 0
        for index, _task in self.task_list_component.get_pending_tasks():
            self.task_list_component.update_task_option(index, option, value)
            affected += 1
        if affected:
            display_value = "默认目录" if option is TaskOptions.MODEL_DIRECTORY and not value else value
            self.append_output(f"配置已同步：{label}={display_value}，已更新 {affected} 个待处理内容修复任务。")

    def _invalidate_pending_mask_plans(self, _value):
        self._update_mask_mode_controls()
        affected = 0
        for index, task in self.task_list_component.get_pending_tasks():
            if task.options.get(TaskOptions.MASK_PLAN.value):
                self.task_list_component.update_task_option(index, TaskOptions.OCR_REVIEWED, False)
                affected += 1
        if affected:
            self.append_output(
                f"最终掩膜配置已变更，{affected} 个待处理任务已标记为需要重新确认修复区域；"
                "这是为了让新的红色实际掩膜经过确认后再执行。"
            )
        self._refresh_editing_mask_preview()

    def _invalidate_pending_ocr_results(self, _value):
        affected = 0
        for index, task in self.task_list_component.get_pending_tasks():
            if task.options.get(TaskOptions.OCR_POLYGONS.value):
                self.task_list_component.update_task_option(index, TaskOptions.OCR_POLYGONS, [])
                affected += 1
        if affected:
            self.append_output(
                f"OCR 检测阈值已变更，{affected} 个待处理任务的 OCR 候选已清空；"
                "已确认的最终掩膜不受影响。"
            )

    def __init_widgets(self):
        """创建主页面"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 左侧视频区域
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        # 三个工作台共用同一条视觉基线：预览占左栏可伸缩空间的 2/3，
        # 日志占 1/3。内容修复此前遗漏 stretch，Qt 会按 size hint 将多余
        # 高度更多分给日志，因而预览明显小于画质增强和图片转换。
        self.preview_log_layout = left_layout

        # 创建视频显示组件
        self.video_display_component = VideoDisplayComponent(self)
        self.video_display_component.ab_sections_changed.connect(self.ab_sections_changed)
        self.video_display_component.selections_changed.connect(self.selections_changed)
        self.video_display_component.brush_strokes_changed.connect(self.brush_strokes_changed)
        left_layout.addWidget(self.video_display_component, 2)

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
        left_layout.addWidget(output_container, 1)

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
        self.media_type_selector.addItem("image", "图片任务", onClick=lambda: self.set_target_media_type("image"))
        self.media_type_selector.addItem("video", "视频任务", onClick=lambda: self.set_target_media_type("video"))
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

        # 全局执行与“当前编辑方式”的操作分开。打开/运行面向整个队列，不能
        # 因为矩形、涂鸦或颜色筛选而改变位置；每种方式的编辑动作则只在下方
        # 独立区域出现，避免用户把“确认”误解为确认所有来源。
        button_container = CardWidget(self)
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(8)

        self.file_button = PushButton(tr['SubtitleExtractorGUI']['Open'], self)
        self.file_button.setIcon(FluentIcon.FOLDER)
        self.file_button.clicked.connect(self.open_file)

        self.run_button = PushButton(tr['SubtitleExtractorGUI']['Run'], self)
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.setToolTip(
            "启动当前页面的整个内容修复队列：仅已确认修复区域的待处理任务会按顺序逐项执行。"
            "一次只运行一个任务；它不是选中任务的单独启动按钮。"
        )
        self.run_button.clicked.connect(self.run_button_clicked)

        self.stop_button = PushButton(tr['SubtitleExtractorGUI']['Stop'], self)
        self.stop_button.setIcon(MyFluentIcon.Stop)
        self.stop_button.setToolTip(
            "停止当前内容修复队列：会终止正在执行的任务；尚未开始的任务保留为待处理，"
            "之后可再次点击“运行”继续。"
        )
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        global_actions = QtWidgets.QWidget(button_container)
        global_layout = QHBoxLayout(global_actions)
        global_layout.setContentsMargins(0, 0, 0, 0)
        global_layout.setSpacing(8)
        global_layout.addWidget(self.file_button)
        global_layout.addWidget(self.run_button)
        global_layout.addWidget(self.stop_button)
        global_layout.addStretch(1)
        button_layout.addWidget(global_actions)

        # 用轻量分隔线将全局队列操作与当前掩膜方式的编辑操作区分开，避免
        # 额外标题占空间，也让“打开 / 运行”不会与局部确认动作混在一起。
        self.mask_editor_separator = QtWidgets.QFrame(button_container)
        self.mask_editor_separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.mask_editor_separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.mask_editor_separator.setStyleSheet("color: rgba(128, 128, 128, 0.35);")
        button_layout.addWidget(self.mask_editor_separator)

        self.mask_editor_widget = QtWidgets.QWidget(button_container)
        # 兼容外部主题/自动化检查的旧属性名：其语义现为“模式专属编辑区”；
        # 动态 OCR 时它会隐藏，而全局操作区仍始终可见。动态标记控件不再承载
        # 按钮，仅保留给旧的可见性集成点。
        self.static_mask_actions_widget = self.mask_editor_widget
        self.dynamic_ocr_actions_widget = QtWidgets.QWidget(button_container)
        editor_layout = QVBoxLayout(self.mask_editor_widget)
        editor_layout.setContentsMargins(0, 4, 0, 0)
        editor_layout.setSpacing(7)
        editor_actions = QHBoxLayout()
        editor_actions.setSpacing(8)
        # OCR 候选框会把“定位”和“最终修复区域”混为一谈，且与已有三种
        # 明确编辑方式重复；保留内部实现兼容，但不再暴露给用户操作。
        self.detect_ocr_button = PushButton("检测当前帧 OCR", self.mask_editor_widget)
        self.detect_ocr_button.setIcon(FluentIcon.SEARCH)
        self.detect_ocr_button.setToolTip("只为当前矩形/颜色筛选方式生成可编辑的候选框。")
        self.detect_ocr_button.clicked.connect(self.detect_current_frame_ocr)
        self.brush_button = PushButton("开始涂鸦", self.mask_editor_widget)
        self.brush_button.setIcon(FluentIcon.EDIT)
        self.brush_button.setCheckable(True)
        self.brush_button.setToolTip("只在涂鸦方式中可用：在预览上直接标记要修复的像素。")
        self.brush_button.toggled.connect(self.toggle_brush_mode)
        editor_actions.addWidget(self.brush_button)
        self.confirm_regions_button = PushButton("确认当前方式区域", self.mask_editor_widget)
        self.confirm_regions_button.setIcon(FluentIcon.ACCEPT)
        self.confirm_regions_button.setToolTip("仅确认当前编辑方式；会与此前已确认的其它方式合并。")
        self.confirm_regions_button.clicked.connect(self.confirm_current_regions)
        editor_actions.addWidget(self.confirm_regions_button)
        self.clear_regions_button = PushButton("清除当前区域", self.mask_editor_widget)
        self.clear_regions_button.setIcon(FluentIcon.DELETE)
        self.clear_regions_button.setToolTip("移除当前编辑方式的预选区及其已确认结果，不影响其它方式。")
        self.clear_regions_button.clicked.connect(self.clear_current_regions)
        editor_actions.addWidget(self.clear_regions_button)
        editor_actions.addStretch(1)
        editor_layout.addLayout(editor_actions)

        # 与编辑方式绑定的工具区独立于全局打开/运行按钮。这样不同模式的
        # 控件不会再挤入一排，也不会在切换时改变全局操作的视觉位置。
        self.mask_tool_widget = QtWidgets.QWidget(self.mask_editor_widget)
        mask_tool_layout = QVBoxLayout(self.mask_tool_widget)
        mask_tool_layout.setContentsMargins(0, 0, 0, 0)
        mask_tool_layout.setSpacing(6)

        self.brush_tool_widget = QtWidgets.QWidget(self.mask_tool_widget)
        brush_tool_row = QHBoxLayout(self.brush_tool_widget)
        brush_tool_row.setContentsMargins(0, 0, 0, 0)
        brush_tool_row.setSpacing(8)

        self.brush_radius_button = PushButton(self.brush_tool_widget)
        self.brush_radius_button.setIcon(FluentIcon.BRUSH)
        self.brush_radius_button.setToolTip("设置本任务使用的涂鸦笔刷半径（原始媒体像素）。")
        self.brush_radius_button.clicked.connect(self.choose_brush_radius)
        brush_tool_row.addWidget(self.brush_radius_button)
        self.clear_brush_button = PushButton("清空涂鸦", self.brush_tool_widget)
        self.clear_brush_button.setIcon(FluentIcon.DELETE)
        self.clear_brush_button.setToolTip("仅清空当前任务的涂鸦预选区和已确认涂鸦来源。")
        self.clear_brush_button.clicked.connect(self.clear_current_brush)
        brush_tool_row.addWidget(self.clear_brush_button)
        brush_tool_row.addStretch(1)
        mask_tool_layout.addWidget(self.brush_tool_widget)

        # 颜色选择需要清楚地回显两种已确认颜色，但一行容纳全部控件会在
        # 非全屏时挤占预览。颜色对照同行，容差另起一行，保持任务卡紧凑。
        self.color_tool_widget = QtWidgets.QWidget(self.mask_tool_widget)
        color_tool_layout = QVBoxLayout(self.color_tool_widget)
        color_tool_layout.setContentsMargins(0, 0, 0, 0)
        color_tool_layout.setSpacing(6)
        color_choice_row = QHBoxLayout()
        color_choice_row.setContentsMargins(0, 0, 0, 0)
        color_choice_row.setSpacing(8)

        self.foreground_swatch = QtWidgets.QLabel(self.color_tool_widget)
        self.foreground_swatch.setFixedSize(18, 18)
        self.foreground_swatch.setToolTip("当前待去除文字/符号的前景色；点击右侧按钮可更改。")
        self.foreground_color_button = PushButton(self.color_tool_widget)
        self.foreground_color_button.setToolTip("选择待去除文字或符号的前景色。请在目标笔画上取色。")
        self.foreground_color_button.clicked.connect(lambda: self.choose_task_mask_color("foreground"))
        color_choice_row.addWidget(self.foreground_swatch)
        color_choice_row.addWidget(self.foreground_color_button)

        self.background_swatch = QtWidgets.QLabel(self.color_tool_widget)
        self.background_swatch.setFixedSize(18, 18)
        self.background_swatch.setToolTip("当前背景参考色；点击右侧按钮可更改。")
        self.background_color_button = PushButton(self.color_tool_widget)
        self.background_color_button.setToolTip("选择目标周围应保留的背景色；它会帮助排除接近背景的像素。")
        self.background_color_button.clicked.connect(lambda: self.choose_task_mask_color("background"))
        color_choice_row.addWidget(self.background_swatch)
        color_choice_row.addWidget(self.background_color_button)
        color_choice_row.addStretch(1)
        color_tool_layout.addLayout(color_choice_row)

        color_parameter_row = QHBoxLayout()
        color_parameter_row.setContentsMargins(0, 0, 0, 0)
        color_parameter_row.setSpacing(8)
        self.color_tolerance_button = PushButton(self.color_tool_widget)
        self.color_tolerance_button.setIcon(FluentIcon.FILTER)
        self.color_tolerance_button.setToolTip(
            "颜色筛选的 LAB 容差（0–255）：0 为严格匹配；值越大，越能覆盖抗锯齿和压缩色差，"
            "也越可能选到邻近背景。"
        )
        self.color_tolerance_button.clicked.connect(self.choose_color_tolerance)
        color_parameter_row.addWidget(self.color_tolerance_button)
        self.color_outline_expand_button = PushButton(self.color_tool_widget)
        self.color_outline_expand_button.setIcon(FluentIcon.ZOOM_IN)
        self.color_outline_expand_button.setToolTip(
            "只对颜色筛出的字形小范围扩张，用于补足描边和抗锯齿；0 可关闭。"
        )
        self.color_outline_expand_button.clicked.connect(self.choose_color_outline_expand)
        color_parameter_row.addWidget(self.color_outline_expand_button)
        color_parameter_row.addStretch(1)
        color_tool_layout.addLayout(color_parameter_row)
        mask_tool_layout.addWidget(self.color_tool_widget)
        editor_layout.addWidget(self.mask_tool_widget)
        button_layout.addWidget(self.mask_editor_widget)

        # 开关位于上方“动态 OCR 检测频率”卡片旁，由“逐帧检测”控制。此处只
        # 承载其已开启后的局部参数，避免再产生第二份开关状态或标题分区。
        dynamic_layout = QVBoxLayout(self.dynamic_ocr_actions_widget)
        dynamic_layout.setContentsMargins(0, 4, 0, 0)
        dynamic_layout.setSpacing(7)
        self.dynamic_refine_options_widget = QtWidgets.QWidget(self.dynamic_ocr_actions_widget)
        dynamic_options_layout = QVBoxLayout(self.dynamic_refine_options_widget)
        dynamic_options_layout.setContentsMargins(0, 0, 0, 0)
        dynamic_options_layout.setSpacing(6)
        dynamic_color_row = QHBoxLayout()
        dynamic_color_row.setContentsMargins(0, 0, 0, 0)
        dynamic_color_row.setSpacing(8)
        self.dynamic_foreground_swatch = QtWidgets.QLabel(self.dynamic_refine_options_widget)
        self.dynamic_foreground_swatch.setFixedSize(18, 18)
        self.dynamic_foreground_swatch.setToolTip("当前动态 OCR 字幕前景色。")
        self.dynamic_foreground_button = PushButton("选择字幕前景色", self.dynamic_refine_options_widget)
        self.dynamic_foreground_button.setToolTip("选择动态 OCR 要保留为字幕字形的前景颜色。")
        self.dynamic_foreground_button.clicked.connect(lambda: self.choose_dynamic_ocr_color("foreground"))
        dynamic_color_row.addWidget(self.dynamic_foreground_swatch)
        dynamic_color_row.addWidget(self.dynamic_foreground_button)
        self.dynamic_background_swatch = QtWidgets.QLabel(self.dynamic_refine_options_widget)
        self.dynamic_background_swatch.setFixedSize(18, 18)
        self.dynamic_background_swatch.setToolTip("当前动态 OCR 字幕背景参考色。")
        self.dynamic_background_button = PushButton("选择背景参考色", self.dynamic_refine_options_widget)
        self.dynamic_background_button.setToolTip("选择字幕周围应保留的背景参考色，用于排除相近背景。")
        self.dynamic_background_button.clicked.connect(lambda: self.choose_dynamic_ocr_color("background"))
        dynamic_color_row.addWidget(self.dynamic_background_swatch)
        dynamic_color_row.addWidget(self.dynamic_background_button)
        dynamic_color_row.addStretch(1)
        dynamic_options_layout.addLayout(dynamic_color_row)
        dynamic_parameter_row = QHBoxLayout()
        dynamic_parameter_row.setContentsMargins(0, 0, 0, 0)
        dynamic_parameter_row.setSpacing(8)
        self.dynamic_refine_tolerance_button = PushButton(self.dynamic_refine_options_widget)
        self.dynamic_refine_tolerance_button.setIcon(FluentIcon.FILTER)
        self.dynamic_refine_tolerance_button.setToolTip(
            "前景颜色容差（0–255）：越小越精确，越大越能覆盖抗锯齿和压缩色差。"
        )
        self.dynamic_refine_tolerance_button.clicked.connect(self.choose_dynamic_ocr_tolerance)
        dynamic_parameter_row.addWidget(self.dynamic_refine_tolerance_button)
        self.dynamic_outline_expand_button = PushButton(self.dynamic_refine_options_widget)
        self.dynamic_outline_expand_button.setIcon(FluentIcon.ZOOM_IN)
        self.dynamic_outline_expand_button.setToolTip(
            "只对筛出的字幕字形做小范围扩张，用于包含白字周围的黑色描边；0 可关闭。"
        )
        self.dynamic_outline_expand_button.clicked.connect(self.choose_dynamic_ocr_outline_expand)
        dynamic_parameter_row.addWidget(self.dynamic_outline_expand_button)
        dynamic_parameter_row.addStretch(1)
        dynamic_options_layout.addLayout(dynamic_parameter_row)
        dynamic_layout.addWidget(self.dynamic_refine_options_widget)
        button_layout.addWidget(self.dynamic_ocr_actions_widget)

        # 保留别名可让运行/停止状态切换保持单一逻辑；这三个控件已是全局
        # 控件，不再为动态 OCR 复制出第二套按钮。
        self.dynamic_open_button = self.file_button
        self.dynamic_run_button = self.run_button
        self.dynamic_stop_button = self.stop_button

        button_container.setLayout(button_layout)
        right_layout.addWidget(button_container)

        main_layout.addLayout(right_layout, 1)

    def set_target_media_type(self, media_type: str, *, select_matching_task: bool = True):
        """切换任务类型，并阻断不匹配媒体遗留的预选区/预览状态。"""
        if media_type not in {"image", "video"}:
            raise ValueError(f"不支持的修复任务类型：{media_type}")
        changed = media_type != self._target_media_type
        self._target_media_type = media_type
        if hasattr(self, "media_type_selector"):
            self.media_type_selector.setCurrentItem(media_type)
        if hasattr(self, "setting_interface"):
            self.setting_interface.set_media_type(media_type)
        if hasattr(self, "video_display_component"):
            # 图片模式没有“当前帧区间”的概念。同步关闭时间轴与 A/B 右键操作，
            # 避免用户在图片模式下看到无法产生有效结果的菜单项。
            self.video_display_component.set_video_controls_visible(media_type == "video")
        # 媒体类型与动态 OCR 开关共同决定局部编辑区。不能只等配置变化信号：
        # 若用户已保存“动态 OCR 开启”的偏好后首次切到视频，静态颜色筛选控件
        # 否则会保留创建时的默认可见状态，造成错误的操作面板。
        self._update_mask_mode_controls()
        if not changed or not select_matching_task or not hasattr(self, "task_list_component"):
            return
        current_index = self.task_list_component.get_current_task_index()
        current_task = self.task_list_component.get_task(current_index)
        wants_image = media_type == "image"
        if current_task is not None and current_task.is_image == wants_image:
            return
        matching_index = next(
            (index for index, task in enumerate(self.task_list_component.get_all_tasks()) if task.is_image == wants_image),
            -1,
        )
        if matching_index >= 0:
            self.task_list_component.select_task(matching_index)
            return
        # 当前队列没有目标类型任务时，不能继续展示旧任务的选框、笔迹或
        # A/B 区间；否则用户会误以为它们属于即将添加的视频/图片。
        self.task_list_component.clear_current_task()
        self.current_frame = None
        self.video_path = None
        with self._video_cap_lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
        self.video_display_component.clear_media_preview()
        self.append_output("已切换为图片修复；当前没有图片任务，预览已清空。" if wants_image else "已切换为视频修复；当前没有视频任务，预览已清空。")

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
        # 时间轴面向用户从第 1 帧开始；OpenCV 的帧读取位置从 0 开始。
        frame_index = max(0, int(value) - 1)
        frame = None
        with self._video_cap_lock:
            if self.video_cap is not None and self.video_cap.isOpened():
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
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
        """仅保存当前矩形类编辑方式的草稿，不破坏其它已确认来源。"""
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS, selections)
        mode = self._current_mask_mode()
        if mode in {MaskMode.RECTANGLE, MaskMode.COLOR}:
            drafts = self._mask_drafts(get_current_task_index)
            drafts[mode.value] = {"regions": [tuple(region) for region in selections]}
            self._save_mask_drafts(get_current_task_index, drafts)
        if not self._applying_ocr_result:
            self._refresh_editing_mask_preview()

    def brush_strokes_changed(self, strokes):
        """手工涂鸦是独立的最终掩膜来源，编辑后同样必须重新确认。"""
        task_index = self.task_list_component.get_current_task_index()
        if task_index < 0:
            return
        self.task_list_component.update_task_option(task_index, TaskOptions.BRUSH_STROKES, strokes)
        drafts = self._mask_drafts(task_index)
        drafts[MaskMode.BRUSH.value] = {"strokes": strokes}
        self._save_mask_drafts(task_index, drafts)
        self._refresh_editing_mask_preview()

    def _current_mask_mode(self):
        return MaskMode(config.maskMode.value)

    def _mask_drafts(self, task_index: int) -> dict:
        """读取任务级草稿；草稿和冻结计划分开，切换方式才不会丢内容。"""
        raw = self.task_list_component.get_task_option(task_index, TaskOptions.MASK_DRAFTS, {}) or {}
        return copy.deepcopy(raw)

    def _save_mask_drafts(self, task_index: int, drafts: dict):
        self.task_list_component.update_task_option(task_index, TaskOptions.MASK_DRAFTS, copy.deepcopy(drafts))

    def _active_draft_regions(self, task_index: int, mode: MaskMode | None = None) -> list:
        mode = mode or self._current_mask_mode()
        return list(self._mask_drafts(task_index).get(mode.value, {}).get("regions", []))

    def _active_draft_strokes(self, task_index: int) -> list:
        return list(self._mask_drafts(task_index).get(MaskMode.BRUSH.value, {}).get("strokes", []))

    def _switch_mask_editor(self, _value):
        """切换编辑方式时恢复该方式自己的草稿，绝不清除组合计划。"""
        self._update_mask_mode_controls()
        task_index = self.task_list_component.get_current_task_index()
        if task_index >= 0:
            self._load_mask_editor_draft(task_index)
        self._refresh_editing_mask_preview()

    def _refresh_mask_editor_after_setting_change(self, _value):
        """参数变化只刷新未确认预览；已确认计划保持其序列化快照。"""
        self._update_mask_mode_controls()
        self._refresh_editing_mask_preview()

    def _load_mask_editor_draft(self, task_index: int):
        mode = self._current_mask_mode()
        if mode in {MaskMode.RECTANGLE, MaskMode.COLOR}:
            self.video_display_component.set_selection_rects(self._active_draft_regions(task_index, mode))
            self.video_display_component.set_brush_strokes([])
        else:
            self.video_display_component.clear_selections()
            self.video_display_component.set_brush_strokes(self._active_draft_strokes(task_index))

    def _set_color_swatch(self, widget, color: str):
        widget.setStyleSheet(
            f"background-color: {color}; border: 1px solid #8a8a8a; border-radius: 4px;"
        )

    def _update_mask_mode_controls(self):
        """让任务页面只显示当前掩膜来源实际需要的交互。"""
        if not hasattr(self, "mask_tool_widget"):
            return
        mode = self._current_mask_mode()
        is_brush = mode is MaskMode.BRUSH
        is_color = mode is MaskMode.COLOR
        is_rectangle = mode is MaskMode.RECTANGLE
        dynamic_video = self._target_media_type == "video" and config.videoDynamicOcr.value
        every_frame = dynamic_video and config.videoMaskStrategy.value == "every_frame_ocr"
        precise_dynamic_ocr = every_frame and config.dynamicOcrRefineMask.value

        self.mask_editor_widget.setVisible(not dynamic_video)
        # 参数区仅在精细定位已启用时出现；普通动态 OCR 没有任何静态编辑按钮。
        self.mask_editor_separator.setVisible(not dynamic_video or precise_dynamic_ocr)
        self.dynamic_ocr_actions_widget.setVisible(precise_dynamic_ocr)
        self.dynamic_refine_options_widget.setVisible(precise_dynamic_ocr)

        # 不再在矩形/颜色筛选中提供 OCR 检测入口：这两种方式均由用户明确
        # 画出 ROI，避免候选定位与最终修复区域混淆。
        self.detect_ocr_button.setVisible(False)
        self.clear_regions_button.setVisible((is_rectangle or is_color) and not dynamic_video)
        self.brush_button.setVisible(is_brush and not dynamic_video)
        self.clear_brush_button.setVisible(is_brush and not dynamic_video)
        self.confirm_regions_button.setVisible(not dynamic_video)
        if not is_brush and self.brush_button.isChecked():
            self.brush_button.blockSignals(True)
            self.brush_button.setChecked(False)
            self.brush_button.blockSignals(False)
        self.video_display_component.set_brush_visible(is_brush and not dynamic_video)

        self.brush_radius_button.setVisible(is_brush and not dynamic_video)
        self.brush_tool_widget.setVisible(is_brush and not dynamic_video)
        self.color_tool_widget.setVisible(is_color and not dynamic_video)
        self.mask_tool_widget.setVisible((is_brush or is_color) and not dynamic_video)

        action_text = {
            MaskMode.RECTANGLE: ("确认手工矩形", "清除手工矩形"),
            MaskMode.BRUSH: ("确认涂鸦笔迹", "清除当前涂鸦"),
            MaskMode.COLOR: ("确认颜色筛选", "清除颜色筛选"),
        }[mode]
        self.confirm_regions_button.setText(action_text[0])
        self.clear_regions_button.setText(action_text[1])

        self.brush_radius_button.setText(f"笔刷 {config.brushRadius.value}px")
        foreground = config.colorMaskForeground.value.upper()
        background = config.colorMaskBackground.value.upper()
        self.foreground_color_button.setText("选择前景色")
        self.background_color_button.setText("选择背景色")
        self.color_tolerance_button.setText(f"容差 {config.colorMaskTolerance.value}")
        self.color_outline_expand_button.setText(f"描边扩张 {config.colorMaskOutlineExpand.value}px")
        self._set_color_swatch(self.foreground_swatch, foreground)
        self._set_color_swatch(self.background_swatch, background)
        self._refresh_dynamic_ocr_refine_controls()

    def _refresh_dynamic_ocr_refine_controls(self, *_args):
        """同步精细定位局部参数的色块和紧凑操作按钮。"""
        if not hasattr(self, "dynamic_refine_options_widget"):
            return
        self._set_color_swatch(self.dynamic_foreground_swatch, config.dynamicOcrForeground.value.upper())
        self._set_color_swatch(self.dynamic_background_swatch, config.dynamicOcrBackground.value.upper())
        self.dynamic_refine_tolerance_button.setText(
            f"颜色容差 {config.dynamicOcrRefineTolerance.value}"
        )
        self.dynamic_outline_expand_button.setText(
            f"描边扩张 {config.dynamicOcrOutlineExpand.value}px"
        )

    def _refresh_editing_mask_preview(self):
        """编辑态不伪装成已确认的红色最终掩膜。

        颜色筛选必须让用户在确认前核对阈值效果，因此以橙色显示临时 mask；
        矩形与涂鸦本身已有绿色选框或橙色笔迹，确认前仅还原源帧。
        """
        task_index = self.task_list_component.get_current_task_index()
        if task_index < 0 or self.current_frame is None:
            return
        raw_plan = self.task_list_component.get_task_option(task_index, TaskOptions.MASK_PLAN, None)
        confirmed = MaskPlan.from_dict(raw_plan) if raw_plan else None
        if self._current_mask_mode() is MaskMode.COLOR:
            draft = self._build_mask_plan(task_index, self.video_display_component.get_selection_rects())
            if confirmed is None:
                # ``confirmed=False`` 明确表示橙色只是尚未提交的颜色筛选
                # 预览，绝不能被当成最终修复区域。
                self._show_mask_preview(draft, confirmed=False)
            else:
                self._show_mask_preview(confirmed, draft_plan=draft)
        elif confirmed:
            self._show_mask_preview(confirmed)
        else:
            self.update_preview(self.current_frame, store_current=False)

    @Slot(bool)
    def toggle_brush_mode(self, enabled):
        self.video_display_component.set_brush_mode(enabled)
        self.brush_button.setText("退出涂鸦" if enabled else "涂鸦")
        if enabled:
            self.append_output(
                f"已进入涂鸦掩膜模式（笔刷半径 {config.brushRadius.value}px）。"
                "请直接涂抹需要修复的像素；完成后点击“确认修复区域”。"
            )

    def choose_brush_radius(self):
        value, accepted = QtWidgets.QInputDialog.getInt(
            self, "涂鸦笔刷半径", "原始媒体像素：", config.brushRadius.value, 1, 100,
        )
        if accepted:
            config.set(config.brushRadius, value)

    def choose_task_mask_color(self, kind: str):
        """从可见色板选择颜色；主编辑区与高级设置保持一致的交互。"""
        item = config.colorMaskForeground if kind == "foreground" else config.colorMaskBackground
        title = "选择待去除符号颜色" if kind == "foreground" else "选择背景参考色"
        value = ColorPaletteDialog.choose(self, title, item.value)
        if value is None:
            return
        config.set(item, value)

    def choose_dynamic_ocr_color(self, kind: str):
        """从连续色相面板选择动态 OCR 的前景或背景参考色。"""
        item = config.dynamicOcrForeground if kind == "foreground" else config.dynamicOcrBackground
        title = "选择动态 OCR 字幕前景色" if kind == "foreground" else "选择动态 OCR 背景参考色"
        value = ColorPaletteDialog.choose(self, title, item.value)
        if value is not None:
            config.set(item, value)

    def choose_dynamic_ocr_tolerance(self):
        value, accepted = QtWidgets.QInputDialog.getInt(
            self, "动态 OCR 前景颜色容差", "LAB 距离（0–255）：",
            config.dynamicOcrRefineTolerance.value, 0, 255,
        )
        if accepted:
            config.set(config.dynamicOcrRefineTolerance, value)

    def choose_dynamic_ocr_outline_expand(self):
        value, accepted = QtWidgets.QInputDialog.getInt(
            self, "动态 OCR 字幕描边扩张", "原始视频像素（0–32）：",
            config.dynamicOcrOutlineExpand.value, 0, 32,
        )
        if accepted:
            config.set(config.dynamicOcrOutlineExpand, value)

    def choose_color_tolerance(self):
        value, accepted = QtWidgets.QInputDialog.getInt(
            self, "颜色筛选容差", "Lab 距离（0–255）：", config.colorMaskTolerance.value, 0, 255,
        )
        if accepted:
            config.set(config.colorMaskTolerance, value)

    def choose_color_outline_expand(self):
        """设置颜色字形的后处理扩张，不会修改矩形 ROI 的边界。"""
        value, accepted = QtWidgets.QInputDialog.getInt(
            self, "颜色筛选描边扩张", "原始图像像素（0–32）：",
            config.colorMaskOutlineExpand.value, 0, 32,
        )
        if accepted:
            config.set(config.colorMaskOutlineExpand, value)

    def clear_current_brush(self):
        if self.task_list_component.get_current_task_index() < 0:
            return
        # 涂鸦清空和其它方式同样必须同步移除其已确认贡献；否则用户会看到
        # 笔迹消失，运行时却仍修复旧笔迹，属于最难排查的一类界面/任务偏差。
        self.clear_current_regions()

    def detect_current_frame_ocr(self):
        """在独立进程中检测当前帧，并在完成后回显可编辑选框。"""
        if self._ocr_process is not None:
            # 按钮通常已禁用；此保护也覆盖快捷键或外部代码的重复调用。
            self.append_output("当前 OCR 检测尚未完成，请稍候。")
            return
        task_index = self.task_list_component.get_current_task_index()
        if task_index < 0 or self.current_frame is None:
            self.append_output("请先打开并选择一个图片或视频任务，再进行 OCR 检测。")
            return
        task = self.task_list_component.get_task(task_index)
        if not task:
            return
        frame = self.current_frame.copy()
        self.detect_ocr_button.setEnabled(False)
        self.append_output("正在检测当前帧文字位置；可继续操作界面，检测完成后请在预览中逐框确认、编辑或删除。")

        self._ocr_task_index = task_index
        self._ocr_exit_poll_count = 0
        self._ocr_result_queue = multiprocessing.Queue()
        self._ocr_process = multiprocessing.Process(
            target=detect_ocr_in_process,
            args=(self._ocr_result_queue, task.path, frame),
            name="CleanCanvasOcrDetection",
        )
        try:
            self._ocr_process.start()
            self._ocr_process_id = ProcessManager.instance().add_process(self._ocr_process)
            self._ocr_result_timer.start()
        except Exception as exc:
            self._finish_ocr_detection([], f"无法启动 OCR 检测进程：{exc}")

    @Slot()
    def _poll_ocr_detection_result(self):
        """非阻塞读取 OCR 子进程结果；本方法始终运行在 GUI 线程。"""
        result_queue = self._ocr_result_queue
        process = self._ocr_process
        if result_queue is None or process is None:
            return
        try:
            result = result_queue.get_nowait()
        except Empty:
            # 子进程退出和 multiprocessing 队列的后台传输之间可能有短暂间隔。
            # 给它最多 500ms，避免将一个刚写入的正常结果误判为失败。
            if not process.is_alive():
                self._ocr_exit_poll_count += 1
                if self._ocr_exit_poll_count >= 10:
                    exit_code = process.exitcode
                    self._finish_ocr_detection([], f"OCR 检测进程异常结束（退出码 {exit_code}）。")
            return

        error = result.get("error")
        polygons = [OcrPolygon.from_dict(item) for item in result.get("polygons", [])]
        self._finish_ocr_detection(polygons, error)

    def _finish_ocr_detection(self, polygons, error):
        """收尾 IPC 资源，然后将结果交给既有的主线程 UI 更新逻辑。"""
        self._ocr_result_timer.stop()
        process = self._ocr_process
        process_id = self._ocr_process_id
        task_index = self._ocr_task_index
        result_queue = self._ocr_result_queue

        self._ocr_process = None
        self._ocr_process_id = None
        self._ocr_result_queue = None
        self._ocr_task_index = -1
        self._ocr_exit_poll_count = 0

        if process is not None:
            # ``start()`` 本身失败时，Process 尚未拥有 pid，不能调用 join().
            if process.pid is not None:
                # 已收到结果后无需在 GUI 线程等待子进程收尾；timeout=0 只回收
                # 已退出的进程，避免最后一次回显也产生可感知的卡顿。
                process.join(timeout=0)
        if process_id is not None:
            ProcessManager.instance().remove_process(process_id)
        if result_queue is not None:
            result_queue.close()
            result_queue.join_thread()

        self._on_ocr_detection_finished(task_index, polygons, error)

    @Slot(int, object, object)
    def _on_ocr_detection_finished(self, task_index, polygons, error):
        self.detect_ocr_button.setEnabled(True)
        if error is not None:
            self.append_output(f"OCR 检测失败：{error}")
            InfoBar.error("OCR 检测失败", str(error), parent=self, position=InfoBarPosition.TOP, duration=5000)
            return
        if task_index != self.task_list_component.get_current_task_index():
            # 检测期间允许切换任务，结果不能因为用户正常操作而白白丢弃。
            # 不修改当前预览的选框，只持久化到发起检测的任务，待用户返回时再加载。
            if self.task_list_component.get_task(task_index) is not None:
                self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, [])
                self.task_list_component.update_task_option(task_index, TaskOptions.OCR_POLYGONS, [polygon.to_dict() for polygon in polygons])
                drafts = self._mask_drafts(task_index)
                drafts[self._current_mask_mode().value] = {"regions": []}
                self._save_mask_drafts(task_index, drafts)
                self.append_output("OCR 检测已完成，结果已保存；重新选择对应任务即可查看。")
            return
        regions = [polygon.bbox for polygon in polygons]
        preview_regions = self.video_display_component.video_coordinates_to_preview_coordinates(regions)
        self._applying_ocr_result = True
        try:
            self.video_display_component.set_selection_rects(preview_regions)
            self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, preview_regions)
            self.task_list_component.update_task_option(task_index, TaskOptions.OCR_POLYGONS, [polygon.to_dict() for polygon in polygons])
            drafts = self._mask_drafts(task_index)
            drafts[self._current_mask_mode().value] = {"regions": [tuple(region) for region in preview_regions]}
            self._save_mask_drafts(task_index, drafts)
        finally:
            self._applying_ocr_result = False
        color_hint = (
            "颜色筛选模式会以橙色显示临时颜色 mask；确认后才会显示红色最终 mask。"
            if self._current_mask_mode() is MaskMode.COLOR
            else "请编辑绿色候选框；只有点击“确认修复区域”后才会显示红色最终 mask。"
        )
        self.append_output(
            f"OCR 检测到 {len(regions)} 个候选区域与候选多边形。黄色框仅是候选 ROI；{color_hint}"
        )
        self._refresh_editing_mask_preview()
        InfoBar.info("请确认 OCR 结果", f"已回显 {len(regions)} 个候选框；{color_hint}", parent=self, position=InfoBarPosition.TOP, duration=6000)

    def _build_mask_plan(self, task_index, preview_regions):
        """将当前编辑方式的草稿转换为可序列化来源计划。"""
        source_regions = self.video_display_component.preview_coordinates_to_video_coordinates(preview_regions)
        source_regions = [(xmin, xmax, ymin, ymax) for ymin, ymax, xmin, xmax in source_regions]
        mode = MaskMode(config.maskMode.value)
        brush_strokes = self.video_display_component.get_brush_strokes() if mode is MaskMode.BRUSH else []
        # 一个草稿只携带当前方式自己的数据。若让涂鸦草稿顺手带上旧选框，
        # 对“当前方式是否为空”的验证和确认语义都会变得含混。
        candidate_regions = source_regions if mode is MaskMode.COLOR else []
        manual_regions = source_regions if mode is MaskMode.RECTANGLE else []
        return MaskPlan(
            mode=mode,
            candidate_regions=candidate_regions,
            manual_regions=manual_regions,
            brush_strokes=brush_strokes,
            brush_radius=config.brushRadius.value,
            color_foreground=config.colorMaskForeground.value,
            color_background=config.colorMaskBackground.value,
            color_tolerance=config.colorMaskTolerance.value,
            color_outline_expand_pixels=config.colorMaskOutlineExpand.value,
            expand_pixels=config.subtitleAreaDeviationPixel.value,
        )

    def _show_mask_preview(
        self,
        plan: MaskPlan | None,
        *,
        draft_plan: MaskPlan | None = None,
        confirmed: bool = True,
    ):
        """回显已确认组合（红）与当前颜色草稿（橙）。

        矩形与涂鸦草稿由视频组件的绿色框/橙色笔迹表达；颜色筛选需要像素级
        临时结果，因而叠加为橙色。红色永远表示已经进入冻结组合计划的部分。
        """
        if self.current_frame is None:
            return
        if not confirmed:
            draft_plan, plan = plan, None
        overlay = self.current_frame.copy()
        if plan is not None:
            confirmed_mask = build_mask(self.current_frame, plan)
            overlay[confirmed_mask > 0] = (0, 0, 255)
        if draft_plan is not None:
            draft_mask = build_mask(self.current_frame, draft_plan)
            overlay[draft_mask > 0] = (0, 165, 255)
        # 黑色细字在 35% 叠加下几乎仍像原始黑色，用户会误以为冻结 mask
        # 没有出现。提高显示层红/橙覆盖度只影响预览，不会改变实际修复 mask。
        preview = cv2.addWeighted(self.current_frame, 0.30, overlay, 0.70, 0)
        self.update_preview(preview, store_current=False)

    def confirm_current_regions(self):
        """只确认当前编辑方式，并将其替换/合并进任务的冻结组合计划。"""
        task_index = self.task_list_component.get_current_task_index()
        if task_index < 0:
            self.append_output("没有可确认的任务。")
            return
        preview_regions = self.video_display_component.get_selection_rects()
        strokes = self.video_display_component.get_brush_strokes()
        mode = self._current_mask_mode()
        if mode is MaskMode.BRUSH and not strokes:
            InfoBar.warning("没有涂鸦区域", "请先进入“涂鸦”模式，在预览上直接涂抹需要修复的像素。", parent=self, duration=4000)
            return
        if mode in {MaskMode.RECTANGLE, MaskMode.COLOR} and not preview_regions:
            InfoBar.warning("没有修复区域", "请先手动画框，或先使用 OCR 候选定位后编辑选框。", parent=self, duration=4000)
            return
        source_plan = self._build_mask_plan(task_index, preview_regions)
        if source_plan.mode is MaskMode.COLOR and not source_plan.candidate_regions:
            InfoBar.warning("颜色筛选需要 ROI", "请先手动画一个包含目标符号的矩形框，再确认颜色筛选掩膜。", parent=self, duration=5000)
            return
        source_mask = build_mask(self.current_frame, source_plan)
        if not np.any(source_mask):
            InfoBar.warning("最终掩膜为空", "当前颜色、容差或来源没有选中任何像素。请检查红色预览后调整设置。", parent=self, duration=5000)
            return

        raw_plan = self.task_list_component.get_task_option(task_index, TaskOptions.MASK_PLAN, None)
        # 新确认当前方式时只替换该方式的贡献。其它已确认来源保留，因此一个
        # 任务能稳定组合“矩形 + 涂鸦 + 颜色筛选”，并在子进程中复现。
        plan = MaskPlan.from_dict(raw_plan) if raw_plan else MaskPlan(mode=mode)
        plan.mode = mode
        if mode is MaskMode.RECTANGLE:
            plan.manual_regions = source_plan.manual_regions
            plan.expand_pixels = source_plan.expand_pixels
        elif mode is MaskMode.BRUSH:
            plan.brush_strokes = source_plan.brush_strokes
            plan.brush_radius = source_plan.brush_radius
        else:
            plan.candidate_regions = source_plan.candidate_regions
            plan.color_foreground = source_plan.color_foreground
            plan.color_background = source_plan.color_background
            plan.color_tolerance = source_plan.color_tolerance
            plan.color_outline_expand_pixels = source_plan.color_outline_expand_pixels

        self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, preview_regions)
        self.task_list_component.update_task_option(task_index, TaskOptions.BRUSH_STROKES, source_plan.brush_strokes)
        self.task_list_component.update_task_option(task_index, TaskOptions.MASK_PLAN, plan.to_dict())
        self.task_list_component.update_task_option(task_index, TaskOptions.OCR_REVIEWED, bool(plan.source_modes))
        # 确认时写入当前快照；之后若用户还未执行任务又修改这些设置，配置监听器
        # 会同步更新待处理任务，并在日志中留下记录。
        self._apply_runtime_options(task_index)
        self._enqueue_confirmed_repair_task(task_index)
        # 确认后的来源已经存入冻结计划。清空对应的“未确认草稿”及绿色选框，
        # 让红色最终 mask 成为唯一的可见反馈；切换回来时仍可从任务计划看到
        # 已确认结果，而不会被同一份草稿的橙色预览覆盖。
        self._clear_confirmed_draft(task_index, mode)
        self._show_mask_preview(plan)
        final_mask_pixels = int(np.count_nonzero(build_mask(self.current_frame, plan)))
        source_names = "、".join({
            MaskMode.RECTANGLE: "手工矩形", MaskMode.BRUSH: "手工涂鸦", MaskMode.COLOR: "颜色筛选",
        }[source] for source in plan.source_modes)
        self.append_output(
            f"已确认当前方式并更新修复区域组合：{source_names}。手工矩形={len(plan.manual_regions)}，"
            f"涂鸦路径={len(plan.brush_strokes)}，颜色 ROI={len(plan.candidate_regions)}；"
            f"冻结 mask 像素={final_mask_pixels}；关闭动态 OCR 时，运行时只使用这份冻结计划。"
        )
        InfoBar.success(
            "当前方式已确认",
            f"红色区域是全部已确认方式的合并结果（当前冻结 {final_mask_pixels} 个像素）。",
            parent=self,
            duration=5000,
        )

    def _clear_confirmed_draft(self, task_index: int, mode: MaskMode):
        """提交来源后只清理该方式的临时编辑态，不触碰冻结组合计划。"""
        drafts = self._mask_drafts(task_index)
        if mode is MaskMode.BRUSH:
            drafts[mode.value] = {"strokes": []}
            self.video_display_component.set_brush_strokes([])
            self.task_list_component.update_task_option(task_index, TaskOptions.BRUSH_STROKES, [])
        else:
            drafts[mode.value] = {"regions": []}
            self.video_display_component.clear_selections()
            self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, [])
        self._save_mask_drafts(task_index, drafts)

    def clear_current_regions(self):
        if self.task_list_component.get_current_task_index() < 0:
            return
        task_index = self.task_list_component.get_current_task_index()
        mode = self._current_mask_mode()
        drafts = self._mask_drafts(task_index)
        raw_plan = self.task_list_component.get_task_option(task_index, TaskOptions.MASK_PLAN, None)
        plan = MaskPlan.from_dict(raw_plan) if raw_plan else None
        if mode is MaskMode.BRUSH:
            self.video_display_component.clear_brush_strokes()
            self.task_list_component.update_task_option(task_index, TaskOptions.BRUSH_STROKES, [])
            drafts[mode.value] = {"strokes": []}
            if plan:
                plan.brush_strokes = []
            message = "已清空当前任务的涂鸦预选区及已确认涂鸦来源。"
        else:
            self.video_display_component.clear_selections()
            drafts[mode.value] = {"regions": []}
            if plan:
                if mode is MaskMode.RECTANGLE:
                    plan.manual_regions = []
                else:
                    plan.candidate_regions = []
            message = "已清空当前方式的预选区及已确认来源；其它方式的结果仍会保留。"
        self._save_mask_drafts(task_index, drafts)
        if plan and plan.source_modes:
            self.task_list_component.update_task_option(task_index, TaskOptions.MASK_PLAN, plan.to_dict())
            self.task_list_component.update_task_option(task_index, TaskOptions.OCR_REVIEWED, True)
        else:
            self.task_list_component.update_task_option(task_index, TaskOptions.MASK_PLAN, None)
            self.task_list_component.update_task_option(task_index, TaskOptions.OCR_REVIEWED, False)
        self._refresh_editing_mask_preview()
        self.append_output(message)

    def on_task_selected(self, index, file_path):
        """处理任务被选中事件

        Args:
            index: 任务索引
            file_path: 文件路径
        """
        task = self.task_list_component.get_task(index)
        if task is None:
            return
        # 从任务队列切换时，任务自身的真实媒体类型优先于此前的“待添加
        # 类型”选择。这样图片预选区不会借由旧预览泄漏到视频任务，反之亦然。
        task_media_type = "image" if task.is_image else "video"
        if task_media_type != self._target_media_type:
            self.set_target_media_type(task_media_type, select_matching_task=False)
            self.append_output(f"已按所选任务切换为{'图片' if task.is_image else '视频'}修复预览。")
        # 加载选中的视频进行预览
        self.load_video(file_path)
        ab_sections = self.task_list_component.get_task_option(index, TaskOptions.AB_SECTIONS, [])
        self.video_display_component.set_ab_sections(ab_sections)
        raw_plan = self.task_list_component.get_task_option(index, TaskOptions.MASK_PLAN, None)
        plan = MaskPlan.from_dict(raw_plan) if raw_plan else None
        drafts = self._mask_drafts(index)
        # 兼容升级前的任务：将旧的单一选框/涂鸦迁入当前方式的草稿，再由
        # 新的组合计划继续管理。不会改写已确认的历史 MaskPlan。
        if not drafts:
            legacy_regions = self.task_list_component.get_task_option(index, TaskOptions.SUB_AREAS, [])
            if legacy_regions:
                drafts[self._current_mask_mode().value] = {"regions": legacy_regions}
            legacy_strokes = self.task_list_component.get_task_option(index, TaskOptions.BRUSH_STROKES, [])
            if legacy_strokes:
                drafts[MaskMode.BRUSH.value] = {"strokes": legacy_strokes}
            self._save_mask_drafts(index, drafts)
        self._load_mask_editor_draft(index)
        if raw_plan and self.current_frame is not None:
            self._show_mask_preview(plan)
        else:
            self._refresh_editing_mask_preview()

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
        self._apply_runtime_options(index)
        self.append_output(
            f"已重置任务：{task.name}，将使用当前设置重新执行（修复模式={config.inpaintMode.value}，"
            f"AI 设备={config.aiDevice.value}，动态 OCR={'开启' if config.videoDynamicOcr.value else '关闭'}，"
            f"检测频率={config.videoMaskStrategy.value}）。"
        )

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
            self._toggle_buttons(True)

    @Slot(bool)
    def _toggle_buttons(self, show_run):
        """线程安全地切换按钮可见性"""
        for button in (self.run_button, self.dynamic_run_button):
            button.setVisible(show_run)
        for button in (self.stop_button, self.dynamic_stop_button):
            button.setVisible(not show_run)

    def run_button_clicked(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        # 如果任务队列为空
        if not self.task_list_component.get_pending_tasks():
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return

        # OCR 的角色是辅助定位而非不可见的后台黑盒。每个任务必须先经过用户
        # 确认（OCR 回显后编辑，或直接手工画框后确认）才能真正进入子进程。
        for index, task in self.task_list_component.get_pending_tasks():
            raw_plan = task.options.get(TaskOptions.MASK_PLAN.value)
            dynamic_ocr = bool(task.options.get(TaskOptions.VIDEO_DYNAMIC_OCR.value, False)) and not task.is_image
            if dynamic_ocr and raw_plan:
                self.task_list_component.select_task(index)
                InfoBar.warning(
                    "请选择一种视频处理策略",
                    "当前任务已确认静态最终掩膜，不能同时开启动态 OCR。请关闭动态 OCR 执行静态掩膜；"
                    "或清空静态掩膜后执行动态 OCR。若两种方式都需要，请先完成一次处理，再将输出视频作为新任务处理。",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=8000,
                )
                return
            if raw_plan and not MaskPlan.from_dict(raw_plan).source_modes:
                self.task_list_component.update_task_option(index, TaskOptions.OCR_REVIEWED, False)
            if not dynamic_ocr and not task.options.get(TaskOptions.OCR_REVIEWED.value, False):
                self.task_list_component.select_task(index)
                InfoBar.warning(
                    "需要确认修复区域",
                    "请先手动画框或使用 OCR 候选定位，检查红色最终 mask 后点击“确认修复区域”；"
                    "视频也可以开启动态 OCR 字幕定位后直接运行。",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=6000,
                )
                return

        # 处理任务
        try:
            # 启动时将已审核任务加入线程安全队列；运行中，新确认的任务可以
            # 继续追加到同一队列，而不会因为初始索引快照而被遗漏。
            pending_task_indices = [index for index, _task in self.task_list_component.get_pending_tasks()]
            if not pending_task_indices:
                return

            self._stop_event.clear()
            with self._repair_queue_lock:
                self._repair_task_queue = Queue()
                for task_index in pending_task_indices:
                    self._repair_task_queue.put(task_index)
                self._accepting_repair_tasks = True
            self.toggle_buttons_signal.emit(False)
            # 开启后台线程处理视频
            def task():
                try:
                    # TaskStatus is updated in the GUI thread through queued Qt
                    # signals.  A worker must not use that asynchronously updated
                    # state to decide what to run next, otherwise a quick task can
                    # be picked a second time before its completion signal is
                    # handled.  Indices come from a thread-safe queue, so a task
                    # confirmed while another task runs can safely join this batch.
                    while True:
                        if self._stop_event.is_set():
                            break
                        task_queue = self._repair_task_queue
                        if task_queue is None:
                            break
                        try:
                            task_index = task_queue.get(timeout=0.1)
                        except Empty:
                            # 与确认任务使用同一把锁，避免队列刚判空时新增任务
                            # 被遗漏，并确保它仍保持“待处理”而非“处理中”。
                            with self._repair_queue_lock:
                                try:
                                    task_index = task_queue.get_nowait()
                                except Empty:
                                    self._accepting_repair_tasks = False
                                    break
                        try:
                            task_item = self.task_list_component.get_task(task_index)
                            if task_item is None or task_item.status != TaskStatus.PENDING:
                                continue
                            # 在页面级“当前预览任务”发生变化前冻结本任务的执行
                            # 选项。之后的配置修改只能作用于其它待处理任务。
                            options = copy.deepcopy(task_item.options)
                            options.setdefault(TaskOptions.INPAINT_MODE.value, config.inpaintMode.value)
                            options.setdefault(TaskOptions.AI_DEVICE.value, config.aiDevice.value)
                            options.setdefault(TaskOptions.MODEL_DIRECTORY.value, config.modelDirectory.value or None)
                            if self._strip_refine_from_sampled_dynamic_ocr(
                                options, is_video=not task_item.is_image,
                            ):
                                self.task_list_component.update_task_option(
                                    task_index, TaskOptions.DYNAMIC_OCR_REFINE_MASK, False,
                                )
                                self.append_log_signal.emit([
                                    "任务使用采样检测：已清除旧快照中的动态 OCR 字幕像素精细定位；"
                                    "采样检测不会跨帧复用精细字形 mask。"
                                ])
                            task_item.execution_options = copy.deepcopy(options)
                            # 更新当前处理的任务索引
                            self.current_processing_task_index = task_index
                            if not self.load_video(task_item.path): # 如果加载视频失败
                                self.append_log_signal.emit([tr['SubtitleExtractorGUI']['OpenVideoFailed'].format(task_item.path)])
                                self.task_status_signal.emit(task_index, TaskStatus.FAILED)
                                continue

                            self.video_display_component.save_selections_to_config()

                            # 更新任务状态为运行中
                            self.progress_signal.emit(task_index, 1, False)

                            # 选中当前任务
                            self.select_task_signal.emit(task_index)

                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None

                            self.task_status_signal.emit(task_index, TaskStatus.PROCESSING)
                            for key in options:
                                value = options[key]
                                if key == TaskOptions.SUB_AREAS.value:
                                    value = self.video_display_component.preview_coordinates_to_video_coordinates(value)
                                options[key] = value
                            # 清理缓存, 使用动态路径
                            task_item.output_path = None
                            output_path = task_item.output_path
                            process = self.run_subtitle_remover_process(
                                task_item.path,
                                output_path,
                                options,
                                task_index,
                            )

                            # 检查是否在处理过程中被停止
                            if self._stop_event.is_set():
                                break

                            # 更新任务状态为已完成
                            if process.exitcode == 0:
                                # 任务完成, 更新输出路径为只读
                                task_item.output_path = output_path
                                self.task_status_signal.emit(task_index, TaskStatus.COMPLETED)
                                self.progress_signal.emit(task_index, 100, True)
                            else:
                                self.task_status_signal.emit(task_index, TaskStatus.FAILED)

                        except Exception as e:
                            print(e)
                            self.append_log_signal.emit([f"Error: {e}"])
                            # 更新任务状态为失败
                            if task_index >= 0:
                                self.task_status_signal.emit(task_index, TaskStatus.FAILED)
                            break
                        finally:
                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None
                            task_queue.task_done()
                            time.sleep(1)
                finally:
                    with self._repair_queue_lock:
                        self._accepting_repair_tasks = False
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
        process_log = _ProcessLogWriter(queue)
        try:
            # 模型框架会把加载警告和初始化错误直接写到 stdout/stderr；统一接入
            # 进程消息桥后，GUI 日志板与控制台可以看到相同的诊断信息。
            with redirect_stdout(process_log), redirect_stderr(process_log):
                from backend.main import SubtitleRemover
                sr = SubtitleRemover(video_path, True)
                sr.video_out_path = output_path
                for key in options:
                    setattr(sr, key, options[key])
                sr.add_progress_listener(
                    lambda progress, isFinished: SubtitleRemoverRemoteCall.remote_call_update_progress(
                        queue, progress, isFinished
                    )
                )
                sr.append_output = lambda *args: SubtitleRemoverRemoteCall.remote_call_append_log(queue, list(args))
                sr.manage_process = lambda pid: SubtitleRemoverRemoteCall.remote_call_manage_process(queue, pid)
                sr.update_preview_with_comp = lambda *args: SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(queue, args)
                sr.run()
        except Exception as e:
            # 保留子进程完整堆栈到 GUI 日志；仅显示异常字符串不足以定位模型、
            # 文件或编码阶段的实际失败位置。
            process_log.write(traceback.format_exc())
            process_log.flush()
            SubtitleRemoverRemoteCall.remote_call_catch_error(queue, e)
            # The parent process uses the exit code as the authoritative result.
            # Re-raise after reporting the actionable error so a failed repair
            # can never be promoted to COMPLETED by a stale queued UI status.
            raise
        finally:
            process_log.flush()
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
        process_started = False
        try:
            if self._stop_event.is_set():
                return process
            process.start()
            process_started = True
            ProcessManager.instance().add_process(process)
            self.running_process = process
            process.join()
            print(f"Process exited with code {process.exitcode}")
        finally:
            if not process_started:
                # 进程尚未启动时没有发送方能够投递 FINISH；直接唤醒并回收本地
                # 消费线程，避免启动失败或提前取消时进行无意义的等待。
                subtitle_remover_remote_caller.stop()
            else:
                # 子进程在 finally 中把 FINISH 放在所有进度/日志/预览消息之后。
                # 必须等消费者读到这个哨兵，不能在 process.join() 后立即 stop；
                # 后一种做法会丢弃队列尾部的进度事件，使任务停在启动时的 1%。
                wait_timeout = 0.5 if self._stop_event.is_set() else 15
                if not subtitle_remover_remote_caller.wait_for_finish(timeout=wait_timeout):
                    subtitle_remover_remote_caller.stop()
        return process

    @Slot()
    def processing_finished(self):
        pending_tasks = self.task_list_component.get_pending_tasks()
        if pending_tasks:
            # 还有待执行任务, 忽略
            return
        # 处理完成后恢复界面可用性
        self._toggle_buttons(True)
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
                pos = 1 + int(progress_total / 100 * max(0, self.frame_count - 1))
                pos = min(self.frame_count, max(1, pos))
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
                self._apply_runtime_options(index)
                self.append_output(
                    f"已保存任务配置快照：{path}（修复模式={config.inpaintMode.value}，"
                    f"AI 设备={config.aiDevice.value}，动态 OCR={'开启' if config.videoDynamicOcr.value else '关闭'}，"
                    f"检测频率={config.videoMaskStrategy.value}）。"
                )
                if self._worker_thread and self._worker_thread.is_alive():
                    self.append_output(
                        "新添加的内容修复任务保持“待处理”；完成 OCR 检测并确认修复区域后，"
                        "会自动加入当前队列。"
                    )
                self.task_list_component.select_task(index)

    def closeEvent(self, event):
        """窗口关闭时断开信号连接并清理资源"""
        try:
            # 通知 worker 线程停止
            self._stop_event.set()
            self._cancel_ocr_detection()
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

    def _cancel_ocr_detection(self):
        """在关闭窗口时终止尚未完成的 OCR 子进程并释放 IPC 资源。"""
        self._ocr_result_timer.stop()
        process = self._ocr_process
        process_id = self._ocr_process_id
        result_queue = self._ocr_result_queue
        self._ocr_process = None
        self._ocr_process_id = None
        self._ocr_result_queue = None
        self._ocr_task_index = -1
        self._ocr_exit_poll_count = 0
        if process is not None and process.is_alive():
            ProcessManager.instance().terminate_by_process(process)
        if process_id is not None:
            ProcessManager.instance().remove_process(process_id)
        if result_queue is not None:
            result_queue.close()
            result_queue.join_thread()
