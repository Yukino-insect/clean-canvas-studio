# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
import sys
import os
from bisect import bisect_right
from functools import cached_property
from typing import Any

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .common_tools import get_readable_path, is_image_file
from backend.config import config, tr
from backend.masking import OcrPolygon, VideoMaskStrategy, filter_polygons_by_regions
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections

Rectangle = tuple[int, int, int, int]
Interval = tuple[int, int]
# 记录每帧图片上的字幕文本框区域
FrameBoxMap = dict[int, list[Rectangle]]


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    # 采样间隔，根据视频帧率在 _init_sample_step 中自适应设置
    SAMPLE_STEP = 3

    def __init__(self, video_path: str, sub_areas: list[Rectangle] | None = None) -> None:
        """
        初始化字幕检测器。

        Args:
            video_path: 待检测的视频或图片路径。
            sub_areas: 用户限定的字幕候选区域列表，坐标格式为
                ``(ymin, ymax, xmin, xmax)``。为空时表示不过滤区域。
        """
        self.video_path = video_path
        self.sub_areas = sub_areas or []
        # 采样间隔
        self.SAMPLE_STEP: int
        self._init_sample_step()

    def _init_sample_step(self) -> None:
        """
        根据视频帧率自适应设置采样间隔。

        Returns:
            None。方法会直接更新实例上的 ``SAMPLE_STEP``，尽量保持每秒至少
            采样 8 帧左右，以平衡检测速度与时间分辨率。
        """
        if is_image_file(self.video_path):
            self.SAMPLE_STEP = 1
            return
        cap = cv2.VideoCapture(get_readable_path(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps >= 60:
            self.SAMPLE_STEP = 4
        elif fps >= 30:
            self.SAMPLE_STEP = 3
        else:
            self.SAMPLE_STEP = 2

    @cached_property
    def text_detector(self) -> Any:
        """
        延迟初始化 PaddleOCR 文本检测器。

        Returns:
            PaddleOCR 的 ``TextDetection`` 实例，用于输出文本区域多边形框。
        """
        # 使用项目内的模型文件，跳过 PaddleX 对远程模型站点的连通性检查。
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TextDetection
        model_config = ModelConfig()
        return TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device="cpu",
            enable_hpi=False,
            thresh=config.ocrPixelThreshold.value / 100,
            box_thresh=config.ocrBoxThreshold.value / 100,
            unclip_ratio=config.ocrUnclipRatio.value / 100,
        )

    def detect_polygons(self, img: Any) -> list[OcrPolygon]:
        """返回 PaddleOCR 原始文字轮廓及分数，不把它降级为矩形。"""
        polygons: list[OcrPolygon] = []
        for result in self.text_detector.predict(img):
            dt_polys = result.get("dt_polys")
            if dt_polys is None or len(dt_polys) == 0:
                continue
            scores = result.get("dt_scores") or []
            for index, points in enumerate(dt_polys):
                parsed = tuple((int(point[0]), int(point[1])) for point in points)
                if len(parsed) >= 3:
                    score = float(scores[index]) if index < len(scores) else 1.0
                    polygons.append(OcrPolygon(points=parsed, score=score))

        # 此处 sub_areas 是界面中用户画出的候选 ROI，只负责过滤 OCR 结果，
        # 绝不在精确掩膜模式里被直接填充成最终 mask。
        roi_regions = [(xmin, xmax, ymin, ymax) for ymin, ymax, xmin, xmax in self.sub_areas]
        polygons = filter_polygons_by_regions(polygons, roi_regions)
        return sorted(polygons, key=lambda polygon: (polygon.bbox[2], polygon.bbox[0], -polygon.score))

    def detect_subtitle(self, img: Any) -> list[Rectangle]:
        """
        检测单帧图像中的字幕文本框。

        Args:
            img: OpenCV 读取出的图像帧，通常是 ``numpy.ndarray`` 格式的 BGR 图像。

        Returns:
            检测到的字幕矩形框列表，坐标格式为 ``(xmin, xmax, ymin, ymax)``。
            如果当前帧未检测到字幕，则返回空列表。
        """
        return self.sort_regions([polygon.bbox for polygon in self.detect_polygons(img)])

    @staticmethod
    def _merge_polygon_groups(current: list[OcrPolygon], following: list[OcrPolygon]) -> list[OcrPolygon]:
        """采样策略的保守填补：中间帧使用两侧采样轮廓的去重并集。"""
        result: list[OcrPolygon] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for polygon in [*current, *following]:
            if polygon.points not in seen:
                seen.add(polygon.points)
                result.append(polygon)
        return result

    def find_polygon_frame_map(
        self,
        strategy: VideoMaskStrategy | str,
        sub_remover: Any | None = None,
        sample_step: int | None = None,
    ) -> dict[int, list[OcrPolygon]]:
        """按视频策略收集逐帧或采样帧的精确 OCR 多边形。"""
        selected = strategy if isinstance(strategy, VideoMaskStrategy) else VideoMaskStrategy(strategy)
        step = 1 if selected is VideoMaskStrategy.EVERY_FRAME_OCR else max(1, sample_step or self.SAMPLE_STEP)
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = max(1, int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        sampled: dict[int, list[OcrPolygon]] = {}
        frame_no = 0
        while video_cap.isOpened():
            ok, frame = video_cap.read()
            if not ok:
                break
            frame_no += 1
            ab_sections = getattr(sub_remover, "ab_sections", None)
            if not is_frame_number_in_ab_sections(frame_no - 1, ab_sections):
                continue
            if (frame_no - 1) % step == 0:
                polygons = self.detect_polygons(frame)
                if polygons:
                    sampled[frame_no] = polygons
            if sub_remover:
                sub_remover.progress_total = int(50 * frame_no / frame_count)
        video_cap.release()
        if selected is VideoMaskStrategy.EVERY_FRAME_OCR:
            return sampled

        completed: dict[int, list[OcrPolygon]] = {}
        frame_numbers = sorted(sampled)
        for current_no, next_no in zip(frame_numbers, frame_numbers[1:]):
            completed[current_no] = sampled[current_no]
            if next_no - current_no <= step * 2:
                fill = self._merge_polygon_groups(sampled[current_no], sampled[next_no])
                for fill_no in range(current_no + 1, next_no):
                    completed[fill_no] = fill
        if frame_numbers:
            completed[frame_numbers[-1]] = sampled[frame_numbers[-1]]
        return completed

    @staticmethod
    def sort_regions(regions: list[Rectangle]) -> list[Rectangle]:
        """
        对字幕框做稳定排序并去重，减少 OCR 返回顺序波动带来的后续抖动。

        Args:
            regions: 字幕矩形框列表。

        Returns:
            经过排序和去重后的字幕矩形框列表。
        """
        return sorted(set(regions), key=lambda region: (region[2], region[0], region[3], region[1]))

    @staticmethod
    def merge_region(region1: Rectangle, region2: Rectangle) -> Rectangle:
        """
        将两个相近字幕框合并为一个并集框。

        Args:
            region1: 第一个字幕矩形框。
            region2: 第二个字幕矩形框。

        Returns:
            覆盖两个输入区域的并集矩形框。
        """
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2
        return (
            min(xmin1, xmin2),
            max(xmax1, xmax2),
            min(ymin1, ymin2),
            max(ymax1, ymax2),
        )

    def find_best_region_match(
        self,
        region: Rectangle,
        candidate_regions: list[Rectangle],
        used_indices: set[int] | None = None,
    ) -> int | None:
        """
        在候选区域中为当前区域寻找最接近的相似框。

        Args:
            region: 当前待匹配字幕框。
            candidate_regions: 候选字幕框列表。
            used_indices: 已经匹配过的候选索引集合。

        Returns:
            最佳匹配候选框的索引；若不存在足够相似的候选框，则返回 ``None``。
        """
        used_indices = used_indices or set()
        best_index = None
        best_score = None
        xmin, xmax, ymin, ymax = region
        for index, candidate in enumerate(candidate_regions):
            if index in used_indices or not self.are_similar(region, candidate):
                continue
            c_xmin, c_xmax, c_ymin, c_ymax = candidate
            score = abs(xmin - c_xmin) + abs(xmax - c_xmax) + abs(ymin - c_ymin) + abs(ymax - c_ymax)
            if best_score is None or score < best_score:
                best_index = index
                best_score = score
        return best_index

    def merge_region_groups(self, regions1: list[Rectangle], regions2: list[Rectangle]) -> list[Rectangle]:
        """
        将两帧中的字幕框做稳定并集，用于补足采样帧之间的边界位移。

        Args:
            regions1: 前一采样帧的字幕框列表。
            regions2: 后一采样帧的字幕框列表。

        Returns:
            合并后的字幕框列表。相似框会扩展成并集，不相似框则保留。
        """
        base_regions = self.sort_regions(regions1)
        next_regions = self.sort_regions(regions2)
        merged_regions: list[Rectangle] = []
        used_indices: set[int] = set()

        for region in base_regions:
            matched_index = self.find_best_region_match(region, next_regions, used_indices)
            if matched_index is None:
                merged_regions.append(region)
                continue
            used_indices.add(matched_index)
            merged_regions.append(self.merge_region(region, next_regions[matched_index]))

        for index, region in enumerate(next_regions):
            if index not in used_indices:
                merged_regions.append(region)

        return self.sort_regions(merged_regions)

    def find_subtitle_frame_no(self, sub_remover: Any | None = None) -> FrameBoxMap:
        """
        扫描视频并收集含字幕的帧号及其对应检测框。

        该方法分为两个阶段：
        1. 按 ``SAMPLE_STEP`` 对视频进行采样检测；
        2. 在相邻采样帧均检测到字幕时，对中间帧进行时间轴补全。

        Args:
            sub_remover: 可选的字幕移除器实例，用于输出日志、读取 A/B 区间和
                回写检测阶段进度。

        Returns:
            一个字典，键为 1-based 帧号，值为该帧对应的字幕矩形框列表。
        """
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 字幕收集区域的进度条
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        # 阶段1：采样检测，仅对每隔 sample_step 帧执行 OCR
        sampled_results = {}  # frame_no -> temp_list
        if sub_remover:
            sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            # 判断当前帧是否在给定的区间
            if not is_frame_number_in_ab_sections(current_frame_no - 1, sub_remover.ab_sections):
                tbar.update(1)
                continue
            # 仅对采样帧执行 OCR 推理
            if (current_frame_no - 1) % self.SAMPLE_STEP == 0 or self.SAMPLE_STEP <= 1:
                # 进行 OCR 推理，获取检测框
                temp_list = self.detect_subtitle(frame)
                if len(temp_list) > 0:
                    sampled_results[current_frame_no] = self.sort_regions(temp_list)
            # 没处理一帧，跟进进度
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        video_cap.release()
        # 阶段2：插值填充 — 两个采样帧之间都有字幕时，中间帧也标记为有字幕
        subtitle_frame_no_box_dict = {}
        detected_nos = sorted(sampled_results.keys())
        max_gap = self.SAMPLE_STEP * 2
        # 遍历两帧之间，最大间隔为 max_gap
        for f, next_f in zip(detected_nos, detected_nos[1:]):
            # 因为首尾两帧都存在字幕，所以认为该区间也存在字幕
            # 将 max_gap 距离中的的帧都填充上 sampled_results[f] 的字幕框
            subtitle_frame_no_box_dict[f] = sampled_results[f]
            if next_f - f <= max_gap:
                fill_mask = self.merge_region_groups(sampled_results[f], sampled_results[next_f])
                for fill_f in range(f + 1, next_f):
                    subtitle_frame_no_box_dict[fill_f] = fill_mask
        # 添加最后一个检测帧
        if detected_nos:
            subtitle_frame_no_box_dict[detected_nos[-1]] = sampled_results[detected_nos[-1]]
        # 统一具有相似的字幕区域的相邻帧
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals: list[Interval], points: list[int]) -> list[Interval]:
        """
        按场景切分点拆分字幕连续区间。

        Args:
            intervals: 原始连续帧区间列表，格式为 ``[(start, end), ...]``。
            points: 场景切分发生的帧号列表。

        Returns:
            按场景边界拆分后的区间列表。
        """
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 获取在当前连续帧区间的场景分割点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            # 按场景分割点重新分割连续帧区间
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def expand_ranges_by_scene(
        intervals: list[Interval],
        backward_frame_count: int,
        forward_frame_count: int,
        scene_div_points: list[int],
        frame_count: int,
    ) -> list[Interval]:
        """
        在场景边界内扩展字幕区间，补足字幕淡入淡出等边界帧。

        Args:
            intervals: 已按场景切分后的字幕连续区间。
            backward_frame_count: 向前扩展帧数。
            forward_frame_count: 向后扩展帧数。
            scene_div_points: 场景切分点列表，1-based。
            frame_count: 视频总帧数。

        Returns:
            经过边界约束和去重后的扩展区间列表。
        """
        if not intervals:
            return []

        # 场景分割点，每一个点就是场景的开始帧
        sorted_points = sorted(point for point in scene_div_points if 1 < point <= frame_count)
        # 场景起始点
        scene_starts = [1] + sorted_points
        # 场景结束位置
        scene_ends = [point - 1 for point in sorted_points] + [frame_count]

        def get_scene_bounds(frame_no: int) -> tuple[int, int]:
            scene_index = bisect_right(scene_starts, frame_no) - 1
            scene_index = max(scene_index, 0)
            return scene_starts[scene_index], scene_ends[scene_index]

        sorted_intervals = sorted(intervals)
        expanded_intervals: list[Interval] = []

        for index, (start, end) in enumerate(sorted_intervals):
            scene_start, scene_end = get_scene_bounds(start)
            # 防止扩展超出场景边界
            new_start = max(scene_start, start - backward_frame_count)
            new_end = min(scene_end, end + forward_frame_count)

            # 检查和后一个区间是否有重复区间
            if index < len(sorted_intervals) - 1:
                next_start, _ = sorted_intervals[index + 1]
                next_scene_start, _ = get_scene_bounds(next_start)
                # 原来的区间，是不可能有相交区域的
                # 如果当前区间和下一个区间在同一个场景中，并且当前区间扩展后的 new_end >= next_start
                # 说明当前区间扩展后和下一个区间发生了相交
                if next_scene_start == scene_start and new_end >= next_start:
                    # 如果原来当前区间和下一个区间相邻
                    if next_start - end == 1:
                        # new_end 回退到 end
                        new_end = end
                    # 如果不相邻
                    else:
                        new_end = min(new_end, next_start - 1)
            # 检查前一个 new_start 边界是否和已经处理好的前一个区间的 end 重叠
            if expanded_intervals:
                prev_start, prev_end = expanded_intervals[-1]
                prev_scene_start, _ = get_scene_bounds(prev_start)
                # 同一个场景中发生了重叠
                if prev_scene_start == scene_start and new_start <= prev_end:
                    # new_start 回退
                    new_start = prev_end + 1

            if new_start <= new_end:
                expanded_intervals.append((new_start, new_end))
            else:
                expanded_intervals.append((start, end))

        return expanded_intervals

    @staticmethod
    def expand_frame_box_map(
        frame_box_map: FrameBoxMap,
        original_intervals: list[Interval],
        expanded_intervals: list[Interval],
    ) -> FrameBoxMap:
        """
        根据扩展后的区间，把边界帧补上对应字幕框。

        Args:
            frame_box_map: 原始字幕帧到字幕框的映射。
            original_intervals: 扩展前区间列表。
            expanded_intervals: 扩展后区间列表。

        Returns:
            新的字幕帧映射。新增帧会沿用各自区间的代表性字幕框。
        """
        expanded_frame_box_map = dict(frame_box_map)
        for (origin_start, _), (expanded_start, expanded_end) in zip(original_intervals, expanded_intervals):
            reference_regions = frame_box_map.get(origin_start)
            if not reference_regions:
                continue
            for frame_no in range(expanded_start, expanded_end + 1):
                expanded_frame_box_map.setdefault(frame_no, reference_regions)
        return expanded_frame_box_map

    @staticmethod
    def get_scene_div_frame_no(v_path: str) -> list[int]:
        """
        获取发生场景切换的帧号。

        Args:
            v_path: 输入视频路径。

        Returns:
            场景切分点对应的 1-based 帧号列表。
        """
        scene_div_frame_no_list = []
        # 找到所有场景发生变更的时间节点
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            # 跳过第一帧
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1: Rectangle, region2: Rectangle) -> bool:
        """
        判断两个字幕区域是否足够相似。

        Args:
            region1: 第一个字幕矩形框，格式为 ``(xmin, xmax, ymin, ymax)``。
            region2: 第二个字幕矩形框，格式为 ``(xmin, xmax, ymin, ymax)``。

        Returns:
            如果两个区域在 X/Y 方向上的偏差均小于配置阈值，则返回 ``True``，
            否则返回 ``False``。
        """
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions: FrameBoxMap) -> FrameBoxMap:
        """
        统一相邻帧中相似的字幕区域，减少检测框抖动。

        Args:
            raw_regions: 原始检测结果映射，键为帧号，值为该帧的字幕框列表。

        Returns:
            经过统一后的检测结果映射。对于连续帧中足够相似的区域，会复用上一帧
            的标准框坐标。
        """
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: self.sort_regions(raw_regions[last_key])}

            for key in keys[1:]:
                current_regions = self.sort_regions(raw_regions[key])
                previous_regions = self.sort_regions(unify_value_map[last_key])

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []
                used_indices: set[int] = set()

                for region in current_regions:
                    matched_index = self.find_best_region_match(region, previous_regions, used_indices)
                    if matched_index is None:
                        new_unify_values.append(region)
                        continue
                    used_indices.add(matched_index)
                    new_unify_values.append(previous_regions[matched_index])

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = self.sort_regions(new_unify_values)
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict: FrameBoxMap) -> list[Interval]:
        """
        将离散字幕帧号合并为连续区间。

        Args:
            subtitle_frame_no_box_dict: 含字幕帧号到字幕框列表的映射。

        Returns:
            连续帧区间列表，格式为 ``[(start, end), ...]``。
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict: FrameBoxMap) -> list[Interval]:
        """
        将字幕帧号合并为连续且遮罩一致的区间。

        Args:
            subtitle_frame_no_box_dict: 含字幕帧号到字幕框列表的映射。

        Returns:
            连续区间列表。若相邻帧虽然连续但字幕框列表不同，也会被拆分为新的区间。
        """
        # 存在字幕的帧集
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        # 起始帧
        start = numbers[0]  # 初始区间开始值
        # 遍历 [1,len(numbers)]
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                # 区间不一致的，不能当作连续区间，因此需要截断
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def filter_and_merge_intervals(intervals: list[Interval], target_length: int) -> list[Interval]:
        """
        扩展并合并字幕区间，使区间长度尽量满足参考帧要求。

        Args:
            intervals: 原始字幕连续区间列表。
            target_length: 目标区间最小长度。

        Returns:
            处理后的区间列表。该结果会尽量扩展单点区间，并合并相邻的短区间。

        Notes:
            该方法整体复杂度约为 ``O(n log n)``，主要来自排序开销。
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        # 一次遍历：扩展单点区间，利用排序后的相邻关系 O(n)
        expanded = []
        for i, (start, end) in enumerate(intervals):
            if start == end:  # 单点区间
                # 获取 extended 列表最后一个节点的终点
                prev_end = expanded[-1][1] if expanded else float('-inf')
                # 获取 intervals 中的下一个节点的起点
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else float('inf')
                half = (target_length - 1) // 2
                new_start = max(start - half, prev_end + 1)
                new_end = min(start + half, next_start - 1)
                if new_end < new_start:
                    new_start, new_end = start, start
                # 扩展单节点
                expanded.append((new_start, new_end))
            else:
                # 不是单节点保持区间不变
                expanded.append((start, end))
        # 一次遍历：合并重叠或相邻的短区间 O(n)
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            last_len = last_end - last_start + 1
            cur_len = end - start + 1
            # 相交、相邻或者长度不满足 target_length 的话，合并区间
            if (start <= last_end or start == last_end + 1) and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
