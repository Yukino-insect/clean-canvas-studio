# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
"""去文字/符号页面的基础设置。"""

from PySide6 import QtCore, QtWidgets
from qfluentwidgets import ComboBoxSettingCard, FluentIcon

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
            title="最终掩膜模式",
            content="红色区域才会修复。",
            parent=parent,
            texts={
                "OCR": "ocr_polygon",
                "OCR 细化": "ocr_refined",
                "OCR 手工": "mixed",
                "手工矩形": "rectangle",
            },
        )
        self._compact_combo_card(
            self.mask_mode_combo,
            "精确 OCR 模式将用户画框视为候选区域，不会直接擦掉整块框；请以红色最终 mask 预览为准。",
            175,
        )
        self.addWidget(self.mask_mode_combo)
        self.video_mask_strategy_combo = ComboBoxSettingCard(
            configItem=config.videoMaskStrategy,
            icon=FluentIcon.MOVIE,
            title="视频精确掩膜策略",
            content="静态最快；逐帧最慢。",
            parent=parent,
            texts={
                "静态": "static",
                "采样 OCR": "sampled_ocr",
                "逐帧 OCR": "every_frame_ocr",
            },
        )
        self._compact_combo_card(
            self.video_mask_strategy_combo,
            "静态掩膜最快；采样 OCR 平衡速度；逐帧 OCR 最精确也最慢。图片不受此项影响。",
            175,
        )
        self.addWidget(self.video_mask_strategy_combo)
        self.addStretch(1)

    def set_media_type(self, media_type: str | None):
        """仅在视频任务中展示会实际生效的时间轴/掩膜策略。"""
        is_video = media_type == "video"
        self.video_mask_strategy_combo.setVisible(is_video)

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
