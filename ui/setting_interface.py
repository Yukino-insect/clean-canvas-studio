# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
"""去文字/符号页面的基础设置。"""

from PySide6 import QtCore, QtWidgets
from qfluentwidgets import ComboBoxSettingCard, FluentIcon, SwitchSettingCard

from backend.config import config


class SettingInterface(QtWidgets.QVBoxLayout):
    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(16, 16, 16, 16)
        self.inpaint_mode_combo = ComboBoxSettingCard(
            configItem=config.inpaintMode,
            icon=FluentIcon.BRUSH,
            title="修复模式",
            content="快速或 AI 细节优先。",
            parent=parent,
            texts={
                "OpenCV": "opencv",
                "LaMa": "lama",
                "Anime-LaMa": "anime_lama",
                "Manga": "manga",
            },
        )
        self._compact_combo_card(
            self.inpaint_mode_combo,
            "OpenCV 速度最快；Anime-LaMa 更适合二次元图片；AI 图像模型逐帧修复视频时可能出现闪烁。",
            220,
        )
        self.addWidget(self.inpaint_mode_combo)
        self.device_combo = ComboBoxSettingCard(
            configItem=config.aiDevice,
            icon=FluentIcon.DEVELOPER_TOOLS,
            title="AI 运算设备",
            content="自动依次尝试 CUDA、CPU。",
            parent=parent,
            texts={"自动": "auto", "CUDA": "cuda", "CPU": "cpu"},
        )
        self._compact_combo_card(
            self.device_combo,
            "自动会依次完整初始化 CUDA、CPU，使用首个可用设备；OpenCV 模式不受此项影响。",
            120,
        )
        self.addWidget(self.device_combo)
        self.mask_mode_combo = ComboBoxSettingCard(
            configItem=config.maskMode,
            icon=FluentIcon.EDIT,
            title="修复区域编辑方式",
            content="确认后的多种方式会合并为红色修复区域。",
            parent=parent,
            texts={
                "手工矩形": "rectangle",
                "手工涂鸦": "brush",
                "颜色筛选": "color",
            },
        )
        self._compact_combo_card(
            self.mask_mode_combo,
            "选择当前要编辑的方式，而不是唯一来源。矩形、涂鸦和颜色筛选可分别确认，"
            "红色预览为全部已确认方式的合并结果。OCR 仅用于定位候选框。",
            175,
        )
        self.addWidget(self.mask_mode_combo)

        self.video_dynamic_ocr_switch = SwitchSettingCard(
            configItem=config.videoDynamicOcr,
            icon=FluentIcon.SEARCH,
            title="动态 OCR 字幕定位",
            content="与静态最终掩膜二选一。",
            parent=parent,
        )
        self.video_dynamic_ocr_switch.setToolTip(
            "开启后，处理时会在视频中重新检测字幕多边形；它直接决定动态 OCR 路径的当帧 mask，"
            "不会复用静态最终掩膜。同一视频任务只能选择其中一种策略；若两者都需要，"
            "请先完成一次处理，再将输出视频作为新任务处理。"
        )
        self.addWidget(self.video_dynamic_ocr_switch)
        self.video_mask_strategy_combo = ComboBoxSettingCard(
            configItem=config.videoMaskStrategy,
            icon=FluentIcon.MOVIE,
            title="动态 OCR 检测频率",
            content="采样较快；逐帧更跟随字幕。",
            parent=parent,
            texts={
                "采样检测": "sampled_ocr",
                "逐帧检测": "every_frame_ocr",
            },
        )
        self._compact_combo_card(
            self.video_mask_strategy_combo,
            "只在视频且开启“动态 OCR 字幕定位”后生效。采样检测按间隔重新 OCR 并填补相邻帧；"
            "逐帧检测每帧重新 OCR。字幕像素精细定位只在逐帧检测下可选，避免将上一帧的"
            "字形像素错误复用到下一帧。关闭动态 OCR 时，视频只使用已确认的最终 mask。",
            175,
        )
        self.addWidget(self.video_mask_strategy_combo)
        # 精细定位依赖当帧 OCR 的 ROI，故开关紧跟“逐帧检测”这一前提；采样模式
        # 不显示它，并主动清除旧值，避免该参数穿透到采样任务快照中。
        self.dynamic_ocr_refine_switch = SwitchSettingCard(
            configItem=config.dynamicOcrRefineMask,
            icon=FluentIcon.SEARCH,
            title="字幕像素精细定位",
            content="仅逐帧检测可用",
            parent=parent,
        )
        self.dynamic_ocr_refine_switch.setToolTip(
            "仅在“视频修复 + 动态 OCR 字幕定位 + 逐帧检测”时显示。开启后会在每帧 OCR "
            "文本区域内按字幕前景、背景参考色和容差提取字形像素。切换为采样检测时，"
            "此开关会自动关闭，避免跨帧复用精细字形 mask。"
        )
        self.addWidget(self.dynamic_ocr_refine_switch)

        config.videoDynamicOcr.valueChanged.connect(self._update_video_ocr_controls_visibility)
        config.videoMaskStrategy.valueChanged.connect(self._update_video_ocr_controls_visibility)
        self.addStretch(1)

    def set_media_type(self, media_type: str | None):
        """仅在视频任务中展示动态 OCR 相关控件。"""
        self._media_type = media_type
        self._update_video_ocr_controls_visibility()

    def _update_video_ocr_controls_visibility(self, *_args):
        media_type = getattr(self, "_media_type", None)
        is_video = media_type == "video"
        dynamic_video = is_video and config.videoDynamicOcr.value
        every_frame = dynamic_video and config.videoMaskStrategy.value == "every_frame_ocr"
        self.video_dynamic_ocr_switch.setVisible(is_video)
        self.video_mask_strategy_combo.setVisible(dynamic_video)
        self.video_mask_strategy_combo.comboBox.setEnabled(True)
        self.video_mask_strategy_combo.setContent("采样较快；逐帧更跟随字幕。")
        self.dynamic_ocr_refine_switch.setVisible(every_frame)
        if dynamic_video and not every_frame and config.dynamicOcrRefineMask.value:
            # 采样路径只能使用 OCR 行级区域；清除旧值同时会同步未运行任务快照。
            config.set(config.dynamicOcrRefineMask, False)
        # 动态 OCR 是另一条完整的视频处理路径，而不是“被禁用的静态掩膜”。
        # 隐藏无关卡片可避免一张灰掉的下拉框既占空间又暗示用户仍需选择它。
        self.mask_mode_combo.setVisible(not dynamic_video)
        self.video_dynamic_ocr_switch.setContent(
            "运行时自动定位字幕。" if dynamic_video else "与静态最终掩膜二选一。"
        )

    def set_precise_dynamic_ocr_active(self, enabled: bool):
        """兼容旧调用；可见性现在完全由检测频率决定。"""
        self._update_video_ocr_controls_visibility()
    @staticmethod
    def _compact_combo_card(card: ComboBoxSettingCard, full_description: str, combo_width: int):
        """让设置卡展示决策摘要，完整说明保留在 tooltip 中。

        SettingCard 的 content 会参与 size hint。若把整段帮助文字直接放入卡片，
        它会抢占视频预览的宽度；限制摘要文本和下拉框宽度后，主页面可在非全屏
        时优先保留预览区域，用户仍可通过悬停查看完整说明。
        """
        # 说明是当前选项的必要摘要，不应因为人为的最大宽度而被截断。
        # 卡片本身仍受右侧面板宽度约束，较长的背景说明继续放在 tooltip 中。
        card.contentLabel.setWordWrap(False)
        card.contentLabel.setMaximumWidth(16777215)
        card.contentLabel.setToolTip(full_description)
        card.titleLabel.setToolTip(full_description)
        card.comboBox.setToolTip(full_description)
        card.setToolTip(full_description)
        # 设置卡保持左对齐：标题与说明共用左边界，阅读扫视路径更稳定。
        for label in (card.titleLabel, card.contentLabel):
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            card.vBoxLayout.setAlignment(label, QtCore.Qt.AlignLeft)
        # 保留 Fluent SettingCard 原有的弹性空间：说明区固定在左，选择框贴在
        # 右侧。这正是设置页最容易扫读的“说明 → 选择”布局。
        card.hBoxLayout.setAlignment(QtCore.Qt.AlignVCenter)
        # 固定宽度会让“自动”等短选项右侧出现大片空白。使用内容自适应宽度，
        # 仅设置上限以防将来某个翻译或模型名异常长而挤压预览区域。
        def resize_to_current_text(*_args):
            # 使用控件实际的 size hint，而非手工猜测箭头和内边距。后一种做法
            # 会截断 Anime-LaMa，也会把“自动”这类短项拉成一块空白按钮。
            natural_width = max(
                card.comboBox.minimumSizeHint().width(),
                card.comboBox.sizeHint().width(),
            )
            card.comboBox.setFixedWidth(min(combo_width, natural_width))

        card.comboBox.currentIndexChanged.connect(resize_to_current_text)
        resize_to_current_text()

    def reset_setting(self):
        """预留给设置页的统一重置入口。"""
