# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
"""图片/视频文字与符号去除主流程。

修复区域可以是用户确认的 OCR 框或手工框；区域会生成同一张掩膜，再由
OpenCV、LaMa、Anime-LaMa 或 Manga 模式处理。视频仍通过 FFmpeg 回封装音频。
"""

import gc
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, TypeAlias

import cv2
import numpy as np
from tqdm import tqdm

# 将指定目录动态添加到 Python 的模块搜索路径的最前端
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import config, tr
from backend.inpaint.engine import InpaintError, InpaintMode, create_inpaint_engine
from backend.tools.common_tools import get_readable_path, is_image_file, is_video_or_image, read_image
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.masking import (
    MaskMode, MaskPlan, OcrPolygon, VideoMaskStrategy, build_mask, polygon_mask,
    refine_ocr_polygon_mask,
)
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.video_io import FramePrefetcher, FFmpegVideoWriter


SubtitleArea: TypeAlias = tuple[int, int, int, int]
ProgressListener: TypeAlias = Callable[[int, bool], None]


class SubtitleRemover:
    """内容修复主编排器。

    负责统一图片与视频的输入、最终掩膜、修复引擎、进度通知和输出封装；
    不在此类中决定 GUI 交互，GUI 只通过选项和回调与它交换数据。
    """

    def __init__(self, vd_path: str, gui_mode: bool = False) -> None:
        self.lock = threading.RLock()
        self.sub_areas: list[SubtitleArea] = []
        self.gui_mode = gui_mode
        self.is_picture = is_image_file(str(vd_path))
        self.video_path = vd_path
        self.vd_name = Path(self.video_path).stem
        if self.is_picture:
            image = read_image(self.video_path)
            if image is None:
                self.size = (0, 0)
            else:
                self.size = (image.shape[1], image.shape[0])
            self.video_cap = None
            self.frame_count = 1
            self.fps = 1.0
        else:
            self.video_cap = cv2.VideoCapture(get_readable_path(vd_path))
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.size = (
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        self.mask_size = (self.size[1], self.size[0])
        self.frame_height, self.frame_width = self.mask_size

        # 图片不需要视频写入器；视频的临时文件由 FFmpeg 编码后再回封装原音频。
        self.video_temp_file = None
        self.video_writer = None
        if not self.is_picture:
            # Windows 下 NamedTemporaryFile(delete=True) 会阻止 FFmpeg 打开文件。
            self.video_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            self.video_writer = FFmpegVideoWriter(
                get_readable_path(self.video_temp_file.name), self.fps, self.size
            )
        self.video_out_path = os.path.abspath(
            os.path.join(os.path.dirname(self.video_path), f"{self.vd_name}_no_sub.mp4")
        )
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), "no_sub")
            os.makedirs(pic_dir, exist_ok=True)
            self.video_out_path = os.path.join(pic_dir, f"{self.vd_name}{self.ext}")

        # 这两个值既供命令行进度条使用，也会通过进度监听器同步给 GUI。
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        self.is_successful_merged = False
        self.progress_listeners: list[ProgressListener] = []
        self.ab_sections = None
        # 动态 OCR 只在视频中定位每帧字幕；它不属于任何冻结的最终 mask 计划。
        self.video_dynamic_ocr = False
        # 与任务快照一同从 GUI 传入。默认关闭，因而历史任务仍使用原始
        # OCR 多边形掩膜；开启后才按字幕前景色精细定位字形像素。
        self.dynamic_ocr_refine_mask = config.dynamicOcrRefineMask.value
        self.dynamic_ocr_foreground = config.dynamicOcrForeground.value
        self.dynamic_ocr_background = config.dynamicOcrBackground.value
        self.dynamic_ocr_refine_tolerance = config.dynamicOcrRefineTolerance.value
        self.dynamic_ocr_outline_expand = config.dynamicOcrOutlineExpand.value
        self.inpaint_mode = InpaintMode.OPENCV.value
        self.ai_device = "auto"
        self.model_dir: str | None = None
        # 修复模型采用惰性加载：OpenCV 无需模型，AI 权重仅在确实需要修复
        # 第一帧/图片时才初始化，避免用户打开媒体即占用显存或触发下载。
        self._inpaint_engine = None
        self.mask_plan: dict | None = None
        self.video_mask_strategy = VideoMaskStrategy.SAMPLED_OCR.value

    def update_progress(self, tbar: tqdm, increment: int) -> None:
        """推进统一百分比并通知所有前端监听器。"""
        tbar.update(increment)
        # 任务进度与预览时间轴只能表示已修复并写出的实际视频帧。动态 OCR
        # 的定位扫描会在日志中单独报告，不能借用任务百分比，否则滑块会在
        # 画面尚未处理时提前跳动。
        current_percentage = (tbar.n / tbar.total) * 100 if tbar.total else 100
        if tbar.n < tbar.total:
            current_percentage = max(1, current_percentage)
        self._set_progress_percentage(current_percentage)

    def _set_progress_percentage(self, percentage: float) -> None:
        """发布单调的任务总进度，并避免为同一整数百分比重复发送 IPC 消息。"""
        value = max(0, min(100, int(percentage)))
        if value == self.progress_total:
            return
        self.progress_remover = value
        self.progress_total = value
        self.notify_progress_listeners()

    def begin_dynamic_ocr_scan(self, strategy: VideoMaskStrategy) -> None:
        """在日志中标记动态 OCR 的预处理开始，不改变任务/时间轴进度。"""
        self.append_output(
            f"正在初始化动态 OCR 并扫描字幕位置（{strategy.value}）。"
            "定位阶段仅输出日志；任务进度将在开始逐帧修复后更新。"
        )

    def append_output(self, *args: object) -> None:
        print(*args)

    def add_progress_listener(self, listener: ProgressListener) -> None:
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)

    def remove_progress_listener(self, listener: ProgressListener) -> None:
        if listener in self.progress_listeners:
            self.progress_listeners.remove(listener)

    def notify_progress_listeners(self) -> None:
        for listener in self.progress_listeners:
            try:
                listener(self.progress_total, self.isFinished)
            except Exception:
                traceback.print_exc()

    def update_preview_with_comp(self, frame_ori: np.ndarray, frame_comp: np.ndarray) -> None:
        """供 GUI 覆写，用于显示原始帧与处理后帧。"""

    def _engine(self):
        """按当前模式创建并缓存修复引擎，整个任务复用同一模型实例。"""
        if self._inpaint_engine is None:
            try:
                self._inpaint_engine = create_inpaint_engine(
                    self.inpaint_mode,
                    device=self.ai_device,
                    model_dir=self.model_dir,
                )
            except InpaintError:
                raise
            except Exception as exc:
                raise InpaintError(f"初始化修复模式 {self.inpaint_mode} 失败：{exc}") from exc
        return self._inpaint_engine

    def _plan(self) -> MaskPlan | None:
        """将跨进程传递的字典恢复为不可依赖 GUI 对象的掩膜计划。"""
        return MaskPlan.from_dict(self.mask_plan) if self.mask_plan else None

    def _mask(
        self,
        frame: np.ndarray,
        polygons: list[OcrPolygon] | None = None,
    ) -> np.ndarray:
        if polygons is not None:
            if getattr(self, "dynamic_ocr_refine_mask", False):
                return refine_ocr_polygon_mask(
                    frame,
                    polygons,
                    getattr(self, "dynamic_ocr_foreground", config.dynamicOcrForeground.value),
                    getattr(self, "dynamic_ocr_background", config.dynamicOcrBackground.value),
                    getattr(self, "dynamic_ocr_refine_tolerance", config.dynamicOcrRefineTolerance.value),
                    getattr(self, "dynamic_ocr_outline_expand", config.dynamicOcrOutlineExpand.value),
                )
            return polygon_mask(frame.shape, polygons, config.subtitleAreaDeviationPixel.value)
        plan = self._plan()
        if plan:
            return build_mask(frame, plan)
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    def _repair(
        self,
        frame: np.ndarray,
        polygons: list[OcrPolygon] | None = None,
    ) -> np.ndarray:
        """为单帧构造 mask、执行修复，并向 GUI 发送可视化对比。"""
        mask = self._mask(frame, polygons)
        if not np.any(mask):
            return frame
        repaired = self._engine().inpaint(frame, mask)
        self.update_preview_with_comp(
            np.clip(frame + mask[:, :, np.newaxis] * 0.3, 0, 255).astype(np.uint8),
            repaired,
        )
        return repaired

    def _effective_dynamic_ocr_strategy(self) -> VideoMaskStrategy:
        """返回一个安全的动态 OCR 策略，并拒绝陈旧的精细化（refine）快照。

        像素精细化会从每个当前帧中推导出一个字形掩码（glyph mask）。而采样的 OCR 映射
        则有意在采样帧之间重用检测结果，因此，包含这两种选项的陈旧任务/配置绝不能在采样时静默应用精细化。
        主界面会阻止这种组合；此运行时防护机制还可以保护旧的任务快照、CLI 集成以及绕过界面的未来调用者。
        """
        strategy = VideoMaskStrategy(self.video_mask_strategy)
        if strategy is VideoMaskStrategy.SAMPLED_OCR and self.dynamic_ocr_refine_mask:
            self.dynamic_ocr_refine_mask = False
            self.append_output(
                "动态 OCR 当前使用采样检测：已忽略任务快照中的字幕像素精细定位，"
                "以避免跨帧复用精细字形 mask。"
            )
        return strategy

    def remove_video_subtitles(self, tbar: tqdm) -> None:
        """按视频策略生成每帧掩膜，并使用当前选择的修复引擎写出结果。"""
        polygon_frames: dict[int, list[OcrPolygon]] = {}
        plan = self._plan()
        strategy = self._effective_dynamic_ocr_strategy()
        if self.video_dynamic_ocr:
            refine_hint = "，字幕像素精细定位已开启" if self.dynamic_ocr_refine_mask else ""
            self.append_output(f"动态 OCR 字幕定位已启用：{strategy.value}{refine_hint}。")
            sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
            self.begin_dynamic_ocr_scan(strategy)
            polygon_frames = sub_detector.find_polygon_frame_map(
                strategy,
                sub_remover=self,
                sample_step=config.ocrSampleStep.value,
            )
            if not polygon_frames:
                raise RuntimeError(tr["Main"]["NoSubtitleDetected"].format(self.video_path))
            self.append_output(
                f"动态 OCR 定位完成：已为 {len(polygon_frames)} 帧生成字幕区域；"
                "现在开始逐帧修复与编码，任务进度将随实际输出帧更新。"
            )
        elif plan is None:
            raise RuntimeError(tr["Main"]["NoSubtitleDetected"].format(self.video_path))
        else:
            self.append_output(f"使用已确认的最终掩膜计划：{plan.mode.value}。")

        self.append_output(tr["Main"]["ProcessingStartRemovingSubtitles"])
        reader = FramePrefetcher(self.video_cap)
        frame_no = 0
        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                frame_no += 1
                polygons = polygon_frames.get(frame_no)
                if self.video_dynamic_ocr and polygons:
                    repaired = self._repair(frame, polygons=polygons or [])
                elif not self.video_dynamic_ocr and is_frame_number_in_ab_sections(frame_no - 1, self.ab_sections):
                    repaired = self._repair(frame)
                else:
                    repaired = frame
                    self.update_preview_with_comp(frame, repaired)
                self.video_writer.write(repaired)
                self.update_progress(tbar, increment=1)
        finally:
            reader.stop()
            if self.video_dynamic_ocr:
                del sub_detector
            gc.collect()

    def remove_image_subtitles(self) -> None:
        """处理单张图片；复用与视频相同的最终 mask 和修复引擎。"""
        original_frame = read_image(self.video_path)
        if original_frame is None:
            raise RuntimeError(tr["Main"]["ReadImageFailed"].format(self.video_path))
        plan = self._plan()
        if plan is None:
            raise RuntimeError("图片修复需要先确认最终掩膜。")
        repaired = self._repair(original_frame)
        output_ext = Path(self.video_out_path).suffix.lower() or self.ext.lower() or ".png"
        if output_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            output_ext = ".png"
        encoded, data = cv2.imencode(output_ext, repaired)
        if not encoded:
            raise RuntimeError(f"无法编码图片输出：{self.video_out_path}")
        data.tofile(self.video_out_path)
        gc.collect()

    def merge_audio_to_video(self) -> None:
        """把输入视频的原始音频无损回封装到处理结果。

        音频流不兼容或源视频无音频时，保留已成功编码的无音频画面文件，
        使视频修复结果仍可交付。
        """
        temp_audio = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
        use_shell = os.name == "nt"
        try:
            subprocess.check_output(
                [
                    FFmpegCLI.instance().ffmpeg_path,
                    "-y", "-i", self.video_path, "-acodec", "copy", "-vn", "-loglevel", "error", temp_audio.name,
                ],
                stdin=subprocess.DEVNULL,
                shell=use_shell,
                timeout=600,
            )
            subprocess.check_output(
                [
                    FFmpegCLI.instance().ffmpeg_path,
                    "-y", "-i", self.video_temp_file.name, "-i", temp_audio.name,
                    "-vcodec", "copy", "-acodec", "copy", "-loglevel", "error", self.video_out_path,
                ],
                stdin=subprocess.DEVNULL,
                shell=use_shell,
                timeout=600,
            )
            self.is_successful_merged = True
        except Exception as exc:
            traceback.print_exc()
            self.append_output(tr["Main"]["FailToMergeAudio"].format(str(exc)))
        finally:
            temp_audio.close()
            if os.path.exists(temp_audio.name):
                try:
                    os.remove(temp_audio.name)
                except OSError:
                    pass
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_path)
                except OSError as exc:
                    self.append_output(
                        tr["Main"]["CopyFileFailed"].format(self.video_temp_file.name, self.video_out_path, str(exc))
                    )
            self.video_temp_file.close()

    def log_model(self) -> None:
        mode_info = {
            InpaintMode.OPENCV.value: "OpenCV Telea (CPU)",
            InpaintMode.LAMA.value: f"LaMa ({self.ai_device})",
            InpaintMode.ANIME_LAMA.value: f"Anime-LaMa ({self.ai_device})",
            InpaintMode.MANGA.value: f"Manga ({self.ai_device})",
        }
        self.append_output(tr["Main"]["SubtitleRemoverModel"].format(mode_info.get(self.inpaint_mode, self.inpaint_mode)))
        plan = self._plan()
        if self.video_dynamic_ocr and not self.is_picture:
            self.append_output(f"修复区域来源: 动态 OCR 字幕定位（{self.video_mask_strategy}）。")
        elif plan:
            sources = "、".join(mode.value for mode in plan.source_modes) or plan.mode.value
            self.append_output(f"修复区域来源: 已确认的组合掩膜计划（{sources}）。")

    def _remove_temporary_video(self) -> None:
        """关闭并删除视频处理的临时文件；图片流程不会创建它。"""
        if self.video_temp_file is None:
            return
        try:
            self.video_temp_file.close()
        except OSError:
            pass
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except OSError:
                pass

    def run(self) -> None:
        """执行完整任务，并确保视频句柄、写入器和临时文件最终得到清理。"""
        start_time = time.time()
        if self.video_dynamic_ocr and not self.is_picture and self.mask_plan:
            raise RuntimeError(
                "动态 OCR 字幕定位与静态最终掩膜不能在同一视频任务中同时执行。"
                "请只保留一种策略；若两种方式都需要，请将第一次输出作为新任务处理。"
            )
        if not self.mask_plan and not (self.video_dynamic_ocr and not self.is_picture):
            raise RuntimeError("没有可修复区域。请先确认 OCR 结果或手动画框后确认。")
        if self.video_dynamic_ocr and not self.is_picture:
            self._effective_dynamic_ocr_strategy()
        self.append_output(tr["Main"]["SubtitleArea"].format(self.sub_areas))
        self.append_output(
            tr["Main"]["ABSection"].format(
                str(self.ab_sections).replace("range", "")
                if self.ab_sections else tr["Main"]["ABSectionAll"]
            )
        )
        os.makedirs(os.path.dirname(self.video_out_path), exist_ok=True)
        self.progress_total = 0
        tbar = tqdm(
            total=max(self.frame_count, 1), unit="frame", position=0,
            # 使用当前 stdout：GUI 子进程会把 stdout/stderr 接入消息桥，CLI
            # 则仍自然输出到控制台。此前强制 sys.__stdout__ 会绕过 GUI 日志。
            file=sys.stdout, desc="Subtitle Removing",
        )
        completed = False
        try:
            self.log_model()
            if self.is_picture:
                self.remove_image_subtitles()
                self.update_progress(tbar, increment=1)
            else:
                self.remove_video_subtitles(tbar)
            completed = True
        finally:
            if self.video_cap is not None:
                self.video_cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            tbar.close()
            if not completed:
                self._remove_temporary_video()

        if not self.is_picture:
            self.merge_audio_to_video()
        self.append_output(tr["Main"]["FinishedProcessing"].format(self.video_out_path))
        self.append_output(tr["Main"]["ProcessingTime"].format(round(time.time() - start_time)))
        self.isFinished = True
        self.progress_total = 100
        self.notify_progress_listeners()
        self._remove_temporary_video()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    from backend.tools.args_handler import parse_args

    args = parse_args()
    config.set(config.interface, "en")
    tr.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface", "en.ini"), encoding="utf-8")
    remover = SubtitleRemover(args.input)
    if not is_video_or_image(args.input):
        remover.append_output(f"Error: {remover.video_path} is not supported or is corrupted.")
        raise SystemExit(1)
    remover.sub_areas = args.subtitle_area_coords
    if args.manual_area:
        remover.mask_plan = MaskPlan(
            mode=MaskMode.RECTANGLE,
            manual_regions=[tuple(region) for region in args.manual_area],
            expand_pixels=config.subtitleAreaDeviationPixel.value,
        ).to_dict()
    remover.video_dynamic_ocr = args.dynamic_ocr
    remover.video_mask_strategy = args.ocr_strategy
    remover.inpaint_mode = args.mode
    remover.ai_device = args.device
    remover.model_dir = args.model_dir
    if args.output:
        remover.video_out_path = args.output
    remover.run()
