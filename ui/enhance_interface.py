"""与内容修复页一致的画质增强任务队列界面。"""

from __future__ import annotations

import threading
from datetime import datetime
from queue import Empty, Queue

import cv2
from PySide6 import QtCore, QtWidgets
from qfluentwidgets import BodyLabel, CardWidget, ComboBoxSettingCard, FluentIcon, InfoBar, InfoBarPosition, PushButton, SegmentedWidget, TextEdit

from backend.config import config
from backend.enhance.media_enhancer import EnhanceCancelled, MediaEnhancer
from backend.enhance.realesrgan import EnhanceMode
from backend.tools.common_tools import is_image_file
from ui.component.enhance_task_list_component import EnhanceTaskListComponent, EnhanceTaskStatus
from ui.component.media_preview_component import MediaPreviewComponent


class EnhanceSettingInterface(QtWidgets.QVBoxLayout):
    """使用与内容修复页一致的 SettingCard 呈现增强参数。

    这些选项在添加任务时写入任务快照；队列开始前的设置变更会同步待处理
    任务，而已经进入处理中的任务继续使用启动时的快照。
    """

    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(16, 16, 16, 16)
        self.model_combo = ComboBoxSettingCard(
            configItem=config.enhanceMode, icon=FluentIcon.ZOOM_IN, title="增强模型",
            content="动漫或通用图像模型。", parent=parent,
            texts={"Anime": EnhanceMode.REAL_ESRGAN_ANIME.value, "General": EnhanceMode.REAL_ESRGAN_GENERAL.value},
        )
        self._compact(self.model_combo, "Anime 适合二次元插画；General 更适合普通照片与通用图像。", 185)
        self.addWidget(self.model_combo)
        self.scale_combo = ComboBoxSettingCard(
            configItem=config.enhanceScale, icon=FluentIcon.ZOOM, title="输出倍率",
            content="决定输出尺寸；不会覆盖原文件。", parent=parent,
            texts={"1×": "1", "1.5×": "1.5", "2×": "2", "3×": "3", "4×": "4"},
        )
        self._compact(
            self.scale_combo,
            "选择输出的放大倍数：1×保持原尺寸；2×、3×、4×会让宽和高约变为原来的 2、3、4 倍。"
            "倍率越大，处理时间、显存占用和输出文件也越大；4×视频的像素量约为原来的 16 倍。",
            125,
        )
        self.addWidget(self.scale_combo)
        self.device_combo = ComboBoxSettingCard(
            configItem=config.aiDevice, icon=FluentIcon.DEVELOPER_TOOLS, title="AI 运算设备",
            content="自动依次尝试 CUDA、CPU。", parent=parent,
            texts={"自动": "auto", "CUDA": "cuda", "CPU": "cpu"},
        )
        self._compact(self.device_combo, "自动会依次完整初始化 CUDA、CPU，使用首个可用设备。", 125)
        self.addWidget(self.device_combo)
        self.addStretch(1)

    @staticmethod
    def _compact(card: ComboBoxSettingCard, full_description: str, combo_width: int):
        # 当前设置摘要必须可完整阅读；完整的使用建议仍保留在 tooltip 中。
        card.contentLabel.setWordWrap(False)
        card.contentLabel.setMaximumWidth(16777215)
        card.contentLabel.setToolTip(full_description)
        card.titleLabel.setToolTip(full_description)
        card.comboBox.setToolTip(full_description)
        card.setToolTip(full_description)
        # 设置卡保持左对齐：标题与说明共用左边界，阅读扫视路径更稳定。
        for label in (card.titleLabel, card.contentLabel):
            label.setAlignment(label.alignment() | QtCore.Qt.AlignLeft)
        # 保留 Fluent SettingCard 原有的弹性空间：说明在左、选择框贴右，
        # 与参考界面的阅读顺序一致。
        card.hBoxLayout.setAlignment(QtCore.Qt.AlignVCenter)
        def resize_to_current_text(*_args):
            # 从 Qt 控件本身获取自然宽度，避免长模型名被截断，亦不为短选项
            # 保留没有意义的大块空白。
            natural_width = max(
                card.comboBox.minimumSizeHint().width(),
                card.comboBox.sizeHint().width(),
            )
            card.comboBox.setFixedWidth(min(combo_width, natural_width))

        card.comboBox.currentIndexChanged.connect(resize_to_current_text)
        resize_to_current_text()

    def current_options(self):
        """返回可序列化的任务级模型、倍率与设备快照。"""
        return (
            config.enhanceMode.value,
            float(config.enhanceScale.value),
            config.aiDevice.value,
            config.modelDirectory.value or None,
        )

class EnhanceInterface(QtWidgets.QWidget):
    """多媒体增强队列：左侧预览/日志，右侧设置/任务/操作。

    后台线程只运行增强器；所有 Qt 控件更新都通过信号回到 GUI 线程完成。
    ``_preview_capture`` 保存输出媒体，``_preview_source_capture`` 只在任务
    完成后用于输入/输出逐帧对比，二者需要在切换任务时一起释放。
    """

    progress_signal = QtCore.Signal(int, int, bool)
    log_signal = QtCore.Signal(str)
    preview_signal = QtCore.Signal(int, object, object)
    complete_signal = QtCore.Signal(int, str)
    error_signal = QtCore.Signal(int, object)
    status_signal = QtCore.Signal(int, object)
    queue_finished_signal = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("EnhanceInterface")
        self._queue_thread: threading.Thread | None = None
        self._queued_task_indices: Queue[int] | None = None
        self._queue_lock = threading.Lock()
        self._accepting_queued_tasks = False
        self._active_enhancer: MediaEnhancer | None = None
        self._stop_event = threading.Event()
        # 仅用于把实时预览定位到正在渲染的任务；任务状态和执行配置均由
        # EnhanceTask 自身持有，不能以这个页面级索引作为状态来源。
        self._active_preview_task_index = -1
        # 兼容既有扩展与回归测试中的预览定位字段；生产执行路径不读取它
        # 决定任务状态或模型配置。
        self._current_processing_index = -1
        self._auto_scroll = True
        self._target_media_type = "image"
        self._preview_capture = None
        self._preview_source_capture = None
        self._preview_media_path = None
        self._preview_is_comparison = False
        self._create_widgets()
        self.progress_signal.connect(self._on_progress)
        self.log_signal.connect(self.append_log)
        self.preview_signal.connect(self._on_preview)
        self.complete_signal.connect(self._on_complete)
        self.error_signal.connect(self._on_error)
        self.status_signal.connect(self.task_list.update_task_status)
        self.queue_finished_signal.connect(self._on_queue_finished)
        self._connect_pending_task_config_sync()

    def _connect_pending_task_config_sync(self):
        """配置改变时同步尚未执行的增强任务，避免队列保留过期快照。"""
        config.enhanceMode.valueChanged.connect(
            lambda value: self._sync_pending_task_option("mode", value, "增强模型")
        )
        config.enhanceScale.valueChanged.connect(
            lambda value: self._sync_pending_task_option("scale", float(value), "输出倍率", suffix="×")
        )
        config.aiDevice.valueChanged.connect(
            lambda value: self._sync_pending_task_option("device", value, "AI 设备")
        )
        config.modelDirectory.valueChanged.connect(
            lambda value: self._sync_pending_task_option("model_dir", value or None, "AI 模型目录")
        )

    def _sync_pending_task_option(self, attribute: str, value, label: str, *, suffix: str = ""):
        affected = 0
        for _index, task in self.task_list.get_pending_tasks():
            setattr(task, attribute, value)
            affected += 1
        if affected:
            display_value = "默认目录" if attribute == "model_dir" and not value else f"{value}{suffix}"
            self.append_log(f"配置已同步：{label}={display_value}，已更新 {affected} 个待处理画质增强任务。")

    def _create_widgets(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(8)
        # 与内容修复、图片转换保持同一纵向比例：预览 2 份，日志 1 份。
        self.preview_log_layout = left_layout
        # 与内容修复页一致：空预览区保持干净，不额外显示标题或引导占位文字。
        self.preview_component = MediaPreviewComponent(
            parent=self,
            show_title=False,
            placeholder="",
        )
        left_layout.addWidget(self.preview_component, 2)
        self.output_text = TextEdit(self)
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(150)
        self.output_text.document().setDocumentMargin(10)
        self.output_text.verticalScrollBar().valueChanged.connect(self._on_log_scroll)
        log_card = CardWidget(self)
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self.output_text)
        left_layout.addWidget(log_card, 1)
        layout.addLayout(left_layout, 2)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(10)
        media_mode_card = CardWidget(self)
        media_mode_layout = QtWidgets.QHBoxLayout(media_mode_card)
        media_mode_layout.setContentsMargins(16, 10, 16, 10)
        media_mode_layout.addWidget(BodyLabel("添加任务类型", media_mode_card))
        media_mode_layout.addStretch(1)
        self.media_type_selector = SegmentedWidget(media_mode_card)
        self.media_type_selector.addItem("image", "图片任务", onClick=lambda: self.set_target_media_type("image"))
        self.media_type_selector.addItem("video", "视频任务", onClick=lambda: self.set_target_media_type("video"))
        self.media_type_selector.setCurrentItem("image")
        media_mode_layout.addWidget(self.media_type_selector)
        right_layout.addWidget(media_mode_card)
        settings_card = CardWidget(self)
        self.setting_interface = EnhanceSettingInterface(settings_card)
        settings_card.setLayout(self.setting_interface)
        right_layout.addWidget(settings_card)
        task_card = CardWidget(self)
        task_layout = QtWidgets.QVBoxLayout(task_card)
        task_layout.setContentsMargins(0, 0, 0, 0)
        self.task_list = EnhanceTaskListComponent(task_card)
        self.task_list.task_selected.connect(self._on_task_selected)
        self.task_list.task_deleted.connect(self._on_task_deleted)
        self.task_list.task_reset.connect(self._on_task_reset)
        task_layout.addWidget(self.task_list)
        right_layout.addWidget(task_card, 1)
        action_card = CardWidget(self)
        actions = QtWidgets.QHBoxLayout(action_card)
        actions.setContentsMargins(16, 16, 16, 16)
        actions.setSpacing(8)
        self.add_button = PushButton("打开", action_card)
        self.add_button.setIcon(FluentIcon.FOLDER)
        self.add_button.clicked.connect(self.add_media_tasks)
        actions.addWidget(self.add_button)
        self.run_button = PushButton("运行", action_card)
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.setToolTip(
            "启动当前页面的整个画质增强队列：所有待处理任务按顺序逐项执行，"
            "一次只运行一个任务；它不是选中任务的单独启动按钮。"
        )
        self.run_button.clicked.connect(self.start_queue)
        actions.addWidget(self.run_button)
        self.stop_button = PushButton("停止", action_card)
        self.stop_button.setIcon(FluentIcon.CLOSE)
        self.stop_button.setToolTip(
            "停止当前画质增强队列：当前任务在安全取消点结束，尚未开始的任务保留为待处理，"
            "之后可再次点击“运行”继续。"
        )
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_queue)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        right_layout.addWidget(action_card)
        layout.addLayout(right_layout, 1)
        self.preview_component.frame_requested.connect(self._on_preview_frame_requested)

    def _on_log_scroll(self, value):
        self._auto_scroll = value == self.output_text.verticalScrollBar().maximum()

    def append_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        raw = str(message)
        escaped = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        color = "#e74c3c" if any(word in raw for word in ("失败", "错误", "Error", "failed")) else "#2980b9"
        self.output_text.append(f'<span style="color:#888;">[{timestamp}]</span> <span style="color:{color};">{escaped}</span><br>')
        if self._auto_scroll:
            self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

    @staticmethod
    def _read_preview(path: str, is_picture: bool):
        if is_picture:
            data = np_from_file(path)
            return cv2.imdecode(data, cv2.IMREAD_COLOR) if data is not None else None
        capture = cv2.VideoCapture(path)
        try:
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            return frame if ok else None
        finally:
            capture.release()

    def add_media_tasks(self):
        is_picture = self._target_media_type == "image"
        media_filter = (
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;所有文件 (*.*)"
            if is_picture else "视频文件 (*.mp4 *.mov *.mkv *.avi *.wmv *.flv);;所有文件 (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "添加需要增强的图片" if is_picture else "添加需要增强的视频",
            "",
            media_filter,
        )
        if not files:
            return
        mode, scale, device, model_dir = self.setting_interface.current_options()
        last_index = -1
        for path in files:
            actual_is_picture = is_image_file(path)
            if actual_is_picture != is_picture:
                self.append_log(f"跳过与当前任务类型不符的媒体：{path}")
                continue
            if self._read_preview(path, actual_is_picture) is None:
                self.append_log(f"跳过无法读取的媒体：{path}")
                continue
            last_index = self.task_list.add_task(path, mode, scale, device, model_dir)
            model_dir_label = model_dir or "默认目录"
            self.append_log(
                f"已添加增强任务：{path}（模型={mode}，倍率={scale:g}×，AI 设备={device}，"
                f"模型目录={model_dir_label}）。"
            )
            if self._enqueue_task_while_running(last_index):
                self.append_log(f"任务已加入当前增强队列：{path}；将在正在运行的任务结束后自动执行。")
        if last_index >= 0:
            self.task_list.select_task(last_index)

    def set_target_media_type(self, media_type: str):
        if media_type not in {"image", "video"}:
            raise ValueError(f"不支持的画质增强任务类型：{media_type}")
        changed = media_type != self._target_media_type
        self._target_media_type = media_type
        self.media_type_selector.setCurrentItem(media_type)
        # 与内容修复一致：选择视频任务类型即显示同款时间轴。尚未添加文件时
        # 使用默认 100 格作为视觉占位；加载实际视频后会立即替换为真实帧数。
        self.preview_component.set_video_controls_visible(
            media_type == "video",
            frame_count=100 if media_type == "video" else 0,
            frame_index=0,
        )
        if not changed:
            return
        if media_type == "image":
            self.append_log("已切换为图片任务模式。")
        else:
            # 视频限制属于一次性的模式切换事件，而非可配置参数。写入日志既让
            # 用户在需要时能回看，也不长期挤占右侧设置和任务列表空间。
            self.append_log(
                "已切换为视频任务模式：按帧增强并尝试回封装原音频；"
                "处理速度更慢，可能出现轻微闪烁。添加文件后可使用帧滑块预览。"
            )

    def _on_task_selected(self, index: int, _path: str):
        """加载选中任务的预览；已完成媒体优先进入输入/输出对比模式。"""
        task = self.task_list.get_task(index)
        if task is None:
            return
        is_completed = task.status is EnhanceTaskStatus.COMPLETED and bool(task.output_path)
        preview_path = task.output_path if is_completed else task.path
        self._release_preview_capture()
        frame = self._read_preview(preview_path, task.is_picture)
        if frame is None:
            self.preview_component.set_placeholder("无法读取选中任务的预览媒体")
            return
        if task.is_picture:
            self.preview_component.set_video_controls_visible(False)
            # 已完成的图片始终并排展示输入和输出。原图按输出分辨率缩放，
            # 因而左右两侧在界面上拥有完全相同的显示尺寸。
            original = self._read_preview(task.path, True) if is_completed else None
            if original is not None:
                self.preview_component.set_comparison(original, frame, "")
            else:
                self.preview_component.set_frame(frame, "")
        else:
            self._preview_capture = cv2.VideoCapture(preview_path)
            frame_count = int(self._preview_capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self._preview_capture.isOpened() else 0
            self._preview_media_path = preview_path
            self.preview_component.set_video_controls_visible(True, frame_count, 0)
            if is_completed:
                self._preview_source_capture = cv2.VideoCapture(task.path)
                source_frame = self._read_preview(task.path, False)
                if source_frame is not None and self._preview_source_capture.isOpened():
                    self._preview_is_comparison = True
                    self.preview_component.set_comparison(source_frame, frame, "")
                else:
                    self.preview_component.set_frame(frame, "")
            else:
                self.preview_component.set_frame(frame, "")

    def _on_task_deleted(self, _index: int):
        if not self.task_list.tasks:
            self._release_preview_capture()
            self.preview_component.set_placeholder()

    def _on_task_reset(self, index: int):
        """重试时读取当前设置，避免保留旧任务的 CUDA/模型/倍率。"""
        task = self.task_list.get_task(index)
        if task is None:
            return
        task.mode, task.scale, task.device, task.model_dir = self.setting_interface.current_options()
        self.append_log(
            f"已重置任务：{task.name}，将使用当前设置（模型={task.mode}，倍率={task.scale:g}×，"
            f"AI 设备={task.device}，模型目录={task.model_dir or '默认目录'}）。"
        )

    def _on_preview_frame_requested(self, displayed_frame_no: int):
        """按 1-based 滑块位置读取同一帧的输入/输出视频并保持同步。"""
        capture = self._preview_capture
        if capture is None or not capture.isOpened():
            return
        frame_index = max(0, int(displayed_frame_no) - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            return
        if self._preview_is_comparison and self._preview_source_capture is not None and self._preview_source_capture.isOpened():
            self._preview_source_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            source_ok, source_frame = self._preview_source_capture.read()
            if source_ok:
                self.preview_component.set_comparison(source_frame, frame, "")
                return
        if ok:
            self.preview_component.set_frame(frame, "")

    def _release_preview_capture(self):
        """释放当前任务的输出/输入 VideoCapture，避免 Windows 独占文件句柄。"""
        if self._preview_capture is not None:
            self._preview_capture.release()
        if self._preview_source_capture is not None:
            self._preview_source_capture.release()
        self._preview_capture = None
        self._preview_source_capture = None
        self._preview_media_path = None
        self._preview_is_comparison = False

    def _enqueue_task_while_running(self, index: int) -> bool:
        """将新建任务安全追加到正在执行的增强队列。"""
        with self._queue_lock:
            if self._stop_event.is_set() or not self._accepting_queued_tasks or self._queued_task_indices is None:
                return False
            self._queued_task_indices.put(index)
            return True

    def start_queue(self):
        """启动本轮队列。

        初始待处理任务先进入线程安全队列；运行中新增任务可继续追加，且每个
        索引只消费一次，避免异步状态信号尚未落地时重复执行同一任务。
        """
        if self._queue_thread and self._queue_thread.is_alive():
            return
        pending_task_indices = [index for index, _task in self.task_list.get_pending_tasks()]
        if not pending_task_indices:
            InfoBar.warning("没有待处理任务", "请先添加媒体，或将失败/完成任务重置为待处理。", parent=self, duration=3500)
            return
        self._stop_event.clear()
        self.run_button.setVisible(False)
        self.stop_button.setVisible(True)
        with self._queue_lock:
            self._queued_task_indices = Queue()
            for index in pending_task_indices:
                self._queued_task_indices.put(index)
            self._accepting_queued_tasks = True
        self.append_log("开始按队列顺序执行画质增强任务。")

        def work():
            cancelled = False
            # Status changes are delivered to the GUI through queued Qt signals.
            # Do not use the GUI model as the worker loop's control flow: a fast
            # task could otherwise still look PENDING here and run repeatedly.
            while True:
                if self._stop_event.is_set():
                    cancelled = True
                    break
                task_queue = self._queued_task_indices
                if task_queue is None:
                    break
                try:
                    index = task_queue.get(timeout=0.1)
                except Empty:
                    # 与添加任务共享同一把锁，确保“队列判空”和“追加任务”之间
                    # 不会发生竞态，避免刚添加的任务被遗漏到下一轮运行。
                    with self._queue_lock:
                        try:
                            index = task_queue.get_nowait()
                        except Empty:
                            self._accepting_queued_tasks = False
                            break
                task = self.task_list.get_task(index)
                if task is None or task.status is not EnhanceTaskStatus.PENDING:
                    task_queue.task_done()
                    continue
                try:
                    # 状态信号回到 GUI 线程是异步的。先冻结本任务的局部执行
                    # 快照，再发出“处理中”，避免设置变更把这项任务误当作
                    # 待处理并改写其模型/设备。
                    execution_options = (task.mode, task.scale, task.device, task.model_dir)
                    task.execution_options = execution_options
                    mode, scale, device, model_dir = execution_options
                    self._active_preview_task_index = index
                    self.status_signal.emit(index, EnhanceTaskStatus.PROCESSING)
                    self.progress_signal.emit(index, 0, False)
                    self.log_signal.emit(
                        f"开始任务：{task.name}（模型={mode}，倍率={scale:g}×，AI 设备={device}，"
                        f"模型目录={model_dir or '默认目录'}）。"
                    )
                    enhancer = MediaEnhancer(
                        task.path,
                        mode=mode,
                        scale=scale,
                        device=device,
                        model_dir=model_dir,
                    )
                    self._active_enhancer = enhancer
                    enhancer.add_progress_listener(lambda progress, done, i=index: self.progress_signal.emit(i, progress, done))
                    enhancer.add_log_listener(self.log_signal.emit)
                    enhancer.add_preview_listener(lambda original, enhanced, i=index: self.preview_signal.emit(i, original, enhanced))
                    output = enhancer.run()
                    if self._stop_event.is_set():
                        cancelled = True
                        self.status_signal.emit(index, EnhanceTaskStatus.PENDING)
                        self.progress_signal.emit(index, 0, False)
                        self.log_signal.emit(f"任务已停止，将保留待处理状态：{task.name}")
                        break
                    self.complete_signal.emit(index, output)
                except EnhanceCancelled:
                    cancelled = True
                    self.status_signal.emit(index, EnhanceTaskStatus.PENDING)
                    self.progress_signal.emit(index, 0, False)
                    self.log_signal.emit(f"任务已取消：{task.name}")
                    break
                except Exception as exc:
                    self.error_signal.emit(index, exc)
                finally:
                    self._active_enhancer = None
                    task_queue.task_done()
            with self._queue_lock:
                self._accepting_queued_tasks = False
            self.queue_finished_signal.emit(cancelled)

        self._queue_thread = threading.Thread(target=work, daemon=True)
        self._queue_thread.start()

    def stop_queue(self):
        self._stop_event.set()
        if self._active_enhancer:
            self._active_enhancer.cancel()
        self.stop_button.setEnabled(False)
        self.append_log("已请求停止；当前帧/当前图片完成后将安全退出队列。")

    @QtCore.Slot(int, int, bool)
    def _on_progress(self, index: int, progress: int, _done: bool):
        self.task_list.update_task_progress(index, progress)
        task = self.task_list.get_task(index)
        if task is None or task.is_picture or index != self._active_preview_task_index:
            return
        # 与内容修复一样，视频任务执行期间沿用预览时间轴显示当前进度；
        # 滑块使用 1-based 用户帧号，内部任务进度则是 0–100 百分比。
        total_frames = self.preview_component.video_slider.maximum()
        if total_frames <= 1:
            return
        displayed_frame_no = 1 + int(progress / 100 * (total_frames - 1))
        self.preview_component.video_slider.blockSignals(True)
        self.preview_component.video_slider.setValue(
            max(1, min(total_frames, displayed_frame_no))
        )
        self.preview_component.video_slider.blockSignals(False)
        self.preview_component.control_container.setVisible(True)

    @QtCore.Slot(int, object, object)
    def _on_preview(self, index: int, original, enhanced):
        active_index = self._active_preview_task_index
        if active_index < 0:
            active_index = self._current_processing_index
        if index != active_index:
            return
        self.task_list.current_task_index = index
        self.task_list.table.selectRow(index)
        task = self.task_list.get_task(index)
        # 图片没有帧序列；视频在增强中则继续展示同款时间轴并由进度更新手柄。
        if task is None or task.is_picture:
            self.preview_component.set_video_controls_visible(False)
        self.preview_component.set_comparison(original, enhanced, "")

    @QtCore.Slot(int, str)
    def _on_complete(self, index: int, output_path: str):
        """在 GUI 线程记录输出，并切换到完成后的前后对比预览。"""
        task = self.task_list.get_task(index)
        if task is None:
            return
        task.output_path = output_path
        self.task_list.update_task_progress(index, 100)
        self.task_list.update_task_status(index, EnhanceTaskStatus.COMPLETED)
        self.log_signal.emit(f"任务完成：{output_path}")
        active_index = self._active_preview_task_index
        if active_index < 0:
            active_index = self._current_processing_index
        if index == active_index:
            self._on_task_selected(index, task.path)

    @QtCore.Slot(int, object)
    def _on_error(self, index: int, error: object):
        task = self.task_list.get_task(index)
        if task is None:
            return
        self.task_list.update_task_status(index, EnhanceTaskStatus.FAILED)
        self.log_signal.emit(f"任务失败：{task.name}；{error}")

    @QtCore.Slot(bool)
    def _on_queue_finished(self, cancelled: bool):
        self._active_enhancer = None
        self._active_preview_task_index = -1
        self._current_processing_index = -1
        self.run_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.stop_button.setEnabled(True)
        self.append_log("画质增强队列已停止。" if cancelled else "画质增强队列已完成。")
        InfoBar.info("画质增强队列已停止" if cancelled else "画质增强队列已完成", "未开始的任务保留待处理状态。" if cancelled else "可在任务列表中选择任务查看输入或输出预览。", parent=self, position=InfoBarPosition.TOP, duration=4500)

    def closeEvent(self, event):
        self.stop_queue()
        self._release_preview_capture()
        super().closeEvent(event)


def np_from_file(path: str):
    try:
        import numpy as np
        return np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
