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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import config, tr
from backend.inpaint.engine import InpaintError, InpaintMode, create_inpaint_engine
from backend.tools.common_tools import get_readable_path, is_image_file, is_video_or_image, read_image
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.inpaint.mask import regions_to_mask
from backend.masking import MaskMode, MaskPlan, OcrPolygon, VideoMaskStrategy, build_mask
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.video_io import FramePrefetcher, FFmpegVideoWriter


SubtitleArea: TypeAlias = tuple[int, int, int, int]
ProgressListener: TypeAlias = Callable[[int, bool], None]


class SubtitleRemover:
    """检测硬字幕/文本水印，并以 OpenCV 对相应区域逐帧修复。"""

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

        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        self.is_successful_merged = False
        self.progress_listeners: list[ProgressListener] = []
        self.ab_sections = None
        # GUI 在 OCR 审核后把确认/手工框传入 manual_regions；它们是直接擦除
        # 区域，而不是旧流程中“仅供 OCR 过滤”的候选 ROI。
        self.manual_regions: list[tuple[int, int, int, int]] = []
        self.use_ocr_detection = True
        self.inpaint_mode = InpaintMode.OPENCV.value
        self.ai_device = "auto"
        self.model_dir: str | None = None
        self._inpaint_engine = None
        self.mask_plan: dict | None = None
        self.video_mask_strategy = VideoMaskStrategy.STATIC.value

    def update_progress(self, tbar: tqdm, increment: int) -> None:
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100 if tbar.total else 100
        self.progress_remover = int(current_percentage)
        self.progress_total = self.progress_remover
        self.notify_progress_listeners()

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

    @staticmethod
    def _valid_regions(regions: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """过滤竖向长条等明显不属于字幕/水印的检测框。"""
        return [
            region
            for region in regions
            if (region[3] - region[2]) - (region[1] - region[0])
            <= config.subtitleYXAxisDifferencePixel.value
        ]

    def _engine(self):
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
        return MaskPlan.from_dict(self.mask_plan) if self.mask_plan else None

    def _mask(
        self,
        frame: np.ndarray,
        regions: list[tuple[int, int, int, int]] | None = None,
        polygons: list[OcrPolygon] | None = None,
    ) -> np.ndarray:
        plan = self._plan()
        if plan:
            if polygons is not None:
                plan = plan.with_polygons(polygons)
            return build_mask(frame, plan)
        return regions_to_mask(
            frame.shape,
            regions or [],
            expand_pixels=config.subtitleAreaDeviationPixel.value,
        )

    def _repair(
        self,
        frame: np.ndarray,
        regions: list[tuple[int, int, int, int]] | None = None,
        polygons: list[OcrPolygon] | None = None,
    ) -> np.ndarray:
        mask = self._mask(frame, regions, polygons)
        if not np.any(mask):
            return frame
        repaired = self._engine().inpaint(frame, mask)
        self.update_preview_with_comp(
            np.clip(frame + mask[:, :, np.newaxis] * 0.3, 0, 255).astype(np.uint8),
            repaired,
        )
        return repaired

    def remove_video_subtitles(self, tbar: tqdm) -> None:
        """检测字幕帧，并仅对包含文字的帧执行 OpenCV 修复。"""
        subtitle_frames: dict[int, list[tuple[int, int, int, int]]] = {}
        polygon_frames: dict[int, list[OcrPolygon]] = {}
        sub_detector = None
        plan = self._plan()
        strategy = VideoMaskStrategy(self.video_mask_strategy)
        static_plan = plan is not None and strategy is VideoMaskStrategy.STATIC
        if static_plan:
            self.append_output(f"使用已确认的精确掩膜计划：{plan.mode.value}（静态复用）。")
        elif plan is not None and strategy is not VideoMaskStrategy.STATIC:
            sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
            polygon_frames = sub_detector.find_polygon_frame_map(
                strategy,
                sub_remover=self,
                sample_step=config.ocrSampleStep.value,
            )
            if not polygon_frames and plan.mode is not MaskMode.MIXED:
                raise RuntimeError(tr["Main"]["NoSubtitleDetected"].format(self.video_path))
        elif self.manual_regions:
            # 经人工确认或直接手画的区域，在处理区间中按静态掩膜应用；这使 OCR
            # 漏检的艺术字、符号和 logo 终于可以被真正移除。
            subtitle_frames = {
                frame_no: list(self.manual_regions)
                for frame_no in range(1, self.frame_count + 1)
                if is_frame_number_in_ab_sections(frame_no - 1, self.ab_sections)
            }
            self.append_output(f"使用 {len(self.manual_regions)} 个已确认/手工修复区域。")
        elif self.use_ocr_detection:
            sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
            subtitle_frames = sub_detector.find_subtitle_frame_no(sub_remover=self)
        if not static_plan and not polygon_frames and not subtitle_frames:
            raise RuntimeError(tr["Main"]["NoSubtitleDetected"].format(self.video_path))

        self.append_output(tr["Main"]["ProcessingStartRemovingSubtitles"])
        reader = FramePrefetcher(self.video_cap)
        frame_no = 0
        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                frame_no += 1
                raw_regions = subtitle_frames.get(frame_no, [])
                polygons = polygon_frames.get(frame_no)
                regions = raw_regions if self.manual_regions else self._valid_regions(raw_regions)
                if static_plan and is_frame_number_in_ab_sections(frame_no - 1, self.ab_sections):
                    repaired = self._repair(frame)
                elif polygons or (plan is not None and plan.mode is MaskMode.MIXED and is_frame_number_in_ab_sections(frame_no - 1, self.ab_sections)):
                    repaired = self._repair(frame, polygons=polygons or [])
                elif regions:
                    repaired = self._repair(frame, regions)
                else:
                    repaired = frame
                    self.update_preview_with_comp(frame, repaired)
                self.video_writer.write(repaired)
                self.update_progress(tbar, increment=1)
        finally:
            reader.stop()
            del sub_detector
            gc.collect()

    def remove_image_subtitles(self) -> None:
        original_frame = read_image(self.video_path)
        if original_frame is None:
            raise RuntimeError(tr["Main"]["ReadImageFailed"].format(self.video_path))
        sub_detector = None
        plan = self._plan()
        if plan is not None:
            repaired = self._repair(original_frame)
            regions = []
        elif self.manual_regions:
            regions = list(self.manual_regions)
        elif self.use_ocr_detection:
            sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
            regions = self._valid_regions(sub_detector.detect_subtitle(original_frame))
        else:
            regions = []
        if plan is not None:
            pass
        elif regions:
            repaired = self._repair(original_frame, regions)
        else:
            repaired = original_frame
            self.update_preview_with_comp(original_frame, repaired)
        output_ext = Path(self.video_out_path).suffix.lower() or self.ext.lower() or ".png"
        if output_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            output_ext = ".png"
        encoded, data = cv2.imencode(output_ext, repaired)
        if not encoded:
            raise RuntimeError(f"无法编码图片输出：{self.video_out_path}")
        data.tofile(self.video_out_path)
        del sub_detector
        gc.collect()

    def merge_audio_to_video(self) -> None:
        """把输入视频的原始音频无损回封装到处理结果。"""
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
        if plan:
            self.append_output(f"修复区域来源: 已确认的 {plan.mode.value} 精确掩膜计划。")
        elif self.manual_regions:
            self.append_output("修复区域来源: 用户确认的 OCR/手工框（不再进行隐藏的二次 OCR）。")
        elif self.use_ocr_detection:
            self.append_output(tr["Main"]["SubtitleDetectionModel"].format("PP-OCRv5 Server (CPU)"))

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
        start_time = time.time()
        if not self.mask_plan and not self.manual_regions and not self.use_ocr_detection:
            raise RuntimeError("没有可修复区域。请先确认 OCR 结果或手动画框后确认。")
        self.append_output(tr["Main"]["SubtitleArea"].format(self.manual_regions or self.sub_areas))
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
            file=sys.__stdout__, desc="Subtitle Removing",
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
    remover.manual_regions = [tuple(region) for region in args.manual_area]
    remover.use_ocr_detection = not args.no_ocr
    remover.inpaint_mode = args.mode
    remover.ai_device = args.device
    remover.model_dir = args.model_dir
    if args.output:
        remover.video_out_path = args.output
    remover.run()
