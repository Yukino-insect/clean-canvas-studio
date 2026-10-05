"""画质增强任务队列组件。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QBrush
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon, InfoBar, TableWidget
from showinfm import show_in_file_manager

from backend.tools.common_tools import is_image_file


@unique
class EnhanceTaskStatus(Enum):
    PENDING = "待处理"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    FAILED = "失败"


@dataclass
class EnhanceTask:
    path: str
    name: str
    mode: str
    scale: float
    device: str
    progress: int = 0
    status: EnhanceTaskStatus = EnhanceTaskStatus.PENDING
    output_path: str | None = None

    @property
    def is_picture(self):
        return is_image_file(self.path)

    @property
    def expected_output_path(self):
        suffix = "png" if self.is_picture else "mp4"
        return str(Path(self.path).with_name(f"{Path(self.path).stem}_enhanced_x{self.scale:g}.{suffix}"))


class EnhanceTaskListComponent(QWidget):
    task_selected = Signal(int, str)
    task_deleted = Signal(int)
    task_reset = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tasks: list[EnhanceTask] = []
        self.current_task_index = -1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = TableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["名称", "进度", "状态"])
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setTextElideMode(Qt.ElideMiddle)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.clicked.connect(self._on_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.table)

    def add_task(self, path: str, mode: str, scale: float, device: str):
        existing = self.find_task_index_by_path(path)
        if existing >= 0:
            self.delete_task(existing)
        task = EnhanceTask(path=path, name=os.path.basename(path), mode=mode, scale=scale, device=device)
        self.tasks.append(task)
        row = len(self.tasks) - 1
        self.table.insertRow(row)
        name = QTableWidgetItem(task.name)
        name.setToolTip(path)
        name.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        progress = QTableWidgetItem("0%")
        progress.setTextAlignment(Qt.AlignCenter)
        status = QTableWidgetItem(task.status.value)
        status.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, name)
        self.table.setItem(row, 1, progress)
        self.table.setItem(row, 2, status)
        self.table.scrollToBottom()
        return row

    def get_task(self, index: int):
        return self.tasks[index] if 0 <= index < len(self.tasks) else None

    def get_pending_tasks(self):
        return [(index, task) for index, task in enumerate(self.tasks) if task.status is EnhanceTaskStatus.PENDING]

    def find_task_index_by_path(self, path: str):
        return next((index for index, task in enumerate(self.tasks) if task.path == path), -1)

    def select_task(self, index: int):
        if not 0 <= index < len(self.tasks):
            return
        self.current_task_index = index
        self.table.selectRow(index)
        self.table.scrollTo(self.table.model().index(index, 0))
        self.task_selected.emit(index, self.tasks[index].path)

    def update_task_progress(self, index: int, progress: int):
        task = self.get_task(index)
        if task is None:
            return
        task.progress = max(0, min(100, int(progress)))
        self.table.item(index, 1).setText(f"{task.progress}%")

    def update_task_status(self, index: int, status: EnhanceTaskStatus):
        task = self.get_task(index)
        if task is None:
            return
        task.status = status
        item = self.table.item(index, 2)
        item.setText(status.value)
        colors = {
            EnhanceTaskStatus.COMPLETED: "#2ecc71",
            EnhanceTaskStatus.PROCESSING: "#3498db",
            EnhanceTaskStatus.FAILED: "#e74c3c",
            EnhanceTaskStatus.PENDING: "#8a8a8a",
        }
        item.setForeground(QBrush(QColor(colors[status])))
        self.table.selectRow(index)

    def reset_task(self, index: int):
        task = self.get_task(index)
        if task is None:
            return
        task.output_path = None
        self.update_task_progress(index, 0)
        self.update_task_status(index, EnhanceTaskStatus.PENDING)
        self.task_reset.emit(index)

    def delete_task(self, index: int):
        if not 0 <= index < len(self.tasks):
            return
        del self.tasks[index]
        self.table.removeRow(index)
        if index == self.current_task_index:
            self.current_task_index = -1
        elif index < self.current_task_index:
            self.current_task_index -= 1
        self.task_deleted.emit(index)

    def _on_clicked(self, model_index):
        self.select_task(model_index.row())

    def _show_context_menu(self, position):
        model_index = self.table.indexAt(position)
        if not model_index.isValid():
            return
        row = model_index.row()
        task = self.tasks[row]
        self.select_task(row)
        menu = QMenu(self)
        source = QAction("打开源文件位置", self)
        source.triggered.connect(lambda: self._open_location(task.path))
        menu.addAction(source)
        output = QAction("打开输出文件位置", self)
        output.triggered.connect(lambda: self._open_output(row))
        output.setVisible(
            task.status is EnhanceTaskStatus.COMPLETED and bool(task.output_path) and os.path.exists(task.output_path)
        )
        menu.addAction(output)
        reset = QAction("重置为待处理", self)
        reset.triggered.connect(lambda: self.reset_task(row))
        queue_running = any(item.status is EnhanceTaskStatus.PROCESSING for item in self.tasks)
        reset.setVisible(not queue_running and task.status in {EnhanceTaskStatus.COMPLETED, EnhanceTaskStatus.FAILED})
        menu.addAction(reset)
        remove = QAction("从队列移除", self)
        remove.triggered.connect(lambda: self.delete_task(row))
        remove.setVisible(not queue_running)
        menu.addAction(remove)
        menu.exec(self.table.viewport().mapToGlobal(position))

    def _open_output(self, index):
        task = self.get_task(index)
        if task is None or task.status is not EnhanceTaskStatus.COMPLETED or not task.output_path:
            InfoBar.warning("输出不可用", "任务尚未完成，或输出文件不存在。", parent=self.window(), duration=3000)
            return
        self._open_location(task.output_path)

    def _open_location(self, path: str):
        if not os.path.exists(path):
            InfoBar.warning("文件不存在", path, parent=self.window(), duration=3000)
            return
        show_in_file_manager(os.path.abspath(path))
