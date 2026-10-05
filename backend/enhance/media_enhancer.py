"""图片与实验性视频逐帧超分任务。"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend.enhance.realesrgan import EnhanceMode, create_enhancer
from backend.inpaint.engine import InpaintError
from backend.tools.common_tools import get_readable_path, is_image_file, read_image
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.video_io import FFmpegVideoWriter, FramePrefetcher


ProgressCallback = Callable[[int, bool], None]
LogCallback = Callable[[str], None]
PreviewCallback = Callable[[np.ndarray, np.ndarray], None]


class EnhanceCancelled(InpaintError):
    """用户请求停止增强任务。"""


class MediaEnhancer:
    """不覆盖源文件的超分任务；视频输出保留原音频（若存在）。"""

    def __init__(
        self,
        media_path: str,
        *,
        mode: str = EnhanceMode.REAL_ESRGAN_ANIME.value,
        scale: float = 2.0,
        device: str = "auto",
        model_dir: str | None = None,
        tile_size: int = 512,
        preview_interval_frames: int = 15,
    ) -> None:
        self.media_path = media_path
        self.is_picture = is_image_file(media_path)
        self.mode = mode
        self.scale = float(scale)
        self.device = device
        self.model_dir = model_dir
        self.tile_size = int(tile_size)
        self.progress_listeners: list[ProgressCallback] = []
        self.log_listeners: list[LogCallback] = []
        self.preview_listeners: list[PreviewCallback] = []
        self.preview_interval_frames = max(1, int(preview_interval_frames))
        self.stop_event = threading.Event()
        self.is_finished = False
        self.progress = 0
        stem = Path(media_path).stem
        parent = Path(media_path).parent
        self.output_path = str(parent / (f"{stem}_enhanced_x{self.scale:g}.png" if self.is_picture else f"{stem}_enhanced_x{self.scale:g}.mp4"))

    def add_progress_listener(self, listener: ProgressCallback) -> None:
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)

    def add_log_listener(self, listener: LogCallback) -> None:
        if listener not in self.log_listeners:
            self.log_listeners.append(listener)

    def add_preview_listener(self, listener: PreviewCallback) -> None:
        if listener not in self.preview_listeners:
            self.preview_listeners.append(listener)

    def append_log(self, message: str) -> None:
        for listener in self.log_listeners:
            listener(message)

    def _notify_preview(self, original: np.ndarray, enhanced: np.ndarray) -> None:
        # GUI 预览只需要判断处理趋势，不需要跨线程传递 4× 原始分辨率帧。
        # 将两张图缩小到统一的预览尺寸，可显著降低视频增强期间的内存复制和
        # Qt 信号队列压力。
        height, width = enhanced.shape[:2]
        max_side = 960
        if max(height, width) > max_side:
            ratio = max_side / max(height, width)
            preview_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
            enhanced = cv2.resize(enhanced, preview_size, interpolation=cv2.INTER_AREA)
            original = cv2.resize(original, preview_size, interpolation=cv2.INTER_AREA)
        for listener in self.preview_listeners:
            listener(original, enhanced)

    def cancel(self) -> None:
        """在当前帧推理结束后安全停止；不会写入正式输出文件。"""
        self.stop_event.set()

    def _check_cancelled(self) -> None:
        if self.stop_event.is_set():
            raise EnhanceCancelled("用户已取消画质增强任务。")

    def _notify(self, progress: int, finished: bool = False) -> None:
        self.progress = max(0, min(100, int(progress)))
        for listener in self.progress_listeners:
            listener(self.progress, finished)

    @staticmethod
    def _even(value: int) -> int:
        return value if value % 2 == 0 else value - 1

    def _enhancer(self):
        return create_enhancer(self.mode, device=self.device, model_dir=self.model_dir, tile_size=self.tile_size)

    def _write_image(self, image: np.ndarray) -> None:
        encoded, data = cv2.imencode(".png", image)
        if not encoded:
            raise InpaintError("无法编码增强后的 PNG 图片。")
        data.tofile(self.output_path)

    def _run_image(self, enhancer) -> None:
        self._check_cancelled()
        image = read_image(self.media_path)
        if image is None:
            raise InpaintError(f"无法读取图片：{self.media_path}")
        self.append_log(f"读取图片：{self.media_path}")
        self._notify(5)
        self.append_log(f"正在增强图片，倍率 {self.scale:g}×。")
        result = enhancer.enhance(image, self.scale)
        self._check_cancelled()
        self._notify(95)
        self._write_image(result)
        self._notify_preview(image, result)

    def _merge_audio(self, temp_video: str) -> None:
        command = [
            FFmpegCLI.instance().ffmpeg_path, "-y", "-i", temp_video, "-i", self.media_path,
            "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-shortest", "-loglevel", "error", self.output_path,
        ]
        try:
            subprocess.check_output(command, stdin=subprocess.DEVNULL, timeout=600)
        except Exception:
            # 无音频或输入编码不兼容时，仍保留已成功编码的画面。
            shutil.copy2(temp_video, self.output_path)

    def _run_video(self, enhancer) -> None:
        capture = cv2.VideoCapture(get_readable_path(self.media_path))
        if not capture.isOpened():
            raise InpaintError(f"无法读取视频：{self.media_path}")
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_size = (self._even(round(width * self.scale)), self._even(round(height * self.scale)))
        if min(output_size) <= 0:
            capture.release()
            raise InpaintError("增强后的分辨率无效。")
        self.append_log(
            f"开始逐帧增强视频：{frame_count} 帧，{width}×{height} -> {output_size[0]}×{output_size[1]}。"
        )
        temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp.close()
        writer = FFmpegVideoWriter(get_readable_path(temp.name), fps, output_size)
        reader = FramePrefetcher(capture)
        processed = 0
        try:
            while True:
                self._check_cancelled()
                ret, frame = reader.read()
                if not ret:
                    break
                result = enhancer.enhance(frame, self.scale)
                self._check_cancelled()
                if result.shape[1::-1] != output_size:
                    result = cv2.resize(result, output_size, interpolation=cv2.INTER_LANCZOS4)
                writer.write(result)
                processed += 1
                if processed == 1 or processed % self.preview_interval_frames == 0:
                    self._notify_preview(frame, result)
                self._notify(processed * 95 / frame_count)
        finally:
            reader.stop()
            writer.release()
            capture.release()
        try:
            self.append_log("视频帧增强完成，正在回封装原始音频。")
            self._merge_audio(temp.name)
        finally:
            Path(temp.name).unlink(missing_ok=True)

    def run(self) -> str:
        self._check_cancelled()
        self.append_log(f"加载增强模型：{self.mode}，设备请求：{self.device}。")
        enhancer = self._enhancer()
        if enhancer.device_fallback_reason:
            self.append_log(f"自动设备初始化失败，已切换至 {enhancer.device.upper()}：{enhancer.device_fallback_reason}")
        self.append_log(f"增强模型已就绪，实际设备：{enhancer.device}。")
        if self.is_picture:
            self._run_image(enhancer)
        else:
            self._run_video(enhancer)
        self.is_finished = True
        self._notify(100, True)
        self.append_log(f"增强完成：{self.output_path}")
        return self.output_path
