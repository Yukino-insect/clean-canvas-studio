# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
@desc: 高级设置页面
"""

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (ScrollArea, ExpandLayout, CardWidget, SubtitleLabel, PrimaryPushButton,
                           FluentIcon, NavigationWidget, NavigationItemPosition,
                           SettingCardGroup, RangeSettingCard, SwitchSettingCard,
                           ComboBoxSettingCard, HyperlinkCard, PrimaryPushSettingCard, PushSettingCard,
                           MessageBox)
from backend.config import config, tr, VERSION, PROJECT_HOME_URL, PROJECT_ISSUES_URL, PROJECT_RELEASES_URL
from backend.enhance.realesrgan import ANIME_ASSET, GENERAL_ASSET
from backend.inpaint.engine import ANIME_LAMA_ASSET, LAMA_ASSET, MANGA_INPAINT_ASSET, MANGA_LINE_ASSET, ModelStore
from backend.tools.version_service import VersionService
from backend.tools.concurrent import TaskExecutor

class AdvancedSettingInterface(ScrollArea):
    """高级设置页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.version_manager = VersionService()
        self.__init_widgets()

    def __init_widgets(self):
        # 创建滚动内容的容器
        self.scrollWidget = QtWidgets.QWidget(self)
        self.expandLayout = ExpandLayout(self.scrollWidget)

        # 设置滚动区域属性
        self.setWidget(self.scrollWidget)
        self.enableTransparentBackground()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # 设置滚动区域样式以适应主题
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        # 设置UI
        self.setup_ui()
        self.setup_layout()

    def setup_layout(self):
        """按使用顺序放置帮助、OCR、掩膜、视频、通用与模型管理分组。"""
        self.help_group.addSettingCard(self.project_help)
        self.expandLayout.addWidget(self.help_group)

        self.ocr_group.addSettingCard(self.ocr_pixel_threshold)
        self.ocr_group.addSettingCard(self.ocr_box_threshold)
        self.ocr_group.addSettingCard(self.ocr_unclip_ratio)
        self.expandLayout.addWidget(self.ocr_group)

        self.mask_group.addSettingCard(self.mask_expand_pixels)
        self.expandLayout.addWidget(self.mask_group)

        self.video_group.addSettingCard(self.ocr_sample_step)
        self.expandLayout.addWidget(self.video_group)

        self.advanced_group.addSettingCard(self.interface_language)
        self.advanced_group.addSettingCard(self.save_directory)
        self.advanced_group.addSettingCard(self.check_update_on_startup)
        self.expandLayout.addWidget(self.advanced_group)

        self.ai_group.addSettingCard(self.model_directory)
        self.ai_group.addSettingCard(self.model_status)
        self.ai_group.addSettingCard(self.clear_model_cache)
        self.expandLayout.addWidget(self.ai_group)

        self.about_group.addSettingCard(self.feedback)
        self.about_group.addSettingCard(self.copyright)
        self.about_group.addSettingCard(self.project_link)

        self.expandLayout.addWidget(self.about_group)
        self.expandLayout.setSpacing(16)
        self.expandLayout.setContentsMargins(16, 16, 16, 48)

    def setup_ui(self):
        """设置UI"""
        self.help_group = SettingCardGroup("帮助", self.scrollWidget)
        self.ocr_group = SettingCardGroup("OCR 候选检测", self.scrollWidget)
        self.mask_group = SettingCardGroup("精确掩膜", self.scrollWidget)
        self.video_group = SettingCardGroup("视频采样与时序", self.scrollWidget)
        # 界面语言、输出目录和更新策略与处理媒体无关，归入通用项。
        self.advanced_group = SettingCardGroup("通用与输出", self.scrollWidget)
        # 关于设置组
        self.about_group = SettingCardGroup(tr["Setting"]["AboutSetting"], self.scrollWidget)
        self.ai_group = SettingCardGroup("AI 模型与缓存", self.scrollWidget)

        self.project_help = PushSettingCard(
            text="打开帮助",
            icon=FluentIcon.HELP,
            title="项目使用与参数指南",
            content="了解内容修复、画质增强、掩膜确认、视频策略和高级参数的作用。",
            parent=self.help_group,
        )
        self.project_help.clicked.connect(self.show_parameter_help)

        self.interface_language = ComboBoxSettingCard(
            configItem=config.interface,
            icon=FluentIcon.LANGUAGE,
            title=tr["CleanCanvasWindow"]["InterfaceLanguage"],
            content="切换语言后，部分界面文本需重启应用才能完全生效。",
            parent=self.advanced_group,
            texts=config.intefaceTexts.keys(),
        )

        self.mask_expand_pixels = RangeSettingCard(
            configItem=config.ocrMaskPadding,
            icon=FluentIcon.ZOOM_IN,
            title="最终掩膜扩边（px）",
            content="用于已确认手工矩形；0 px 可追求最小区域，过大可能误伤背景。",
            parent=self.mask_group,
        )

        self.ocr_pixel_threshold = RangeSettingCard(
            configItem=config.ocrPixelThreshold,
            icon=FluentIcon.ZOOM_IN,
            title="OCR 二值化阈值（%）",
            content="降低可提高召回率，也更容易把纹理识别成文字。",
            parent=self.ocr_group,
        )
        self.ocr_box_threshold = RangeSettingCard(
            configItem=config.ocrBoxThreshold,
            icon=FluentIcon.ACCEPT,
            title="OCR 候选置信度阈值（%）",
            content="降低会保留更多候选多边形；确认前请检查误检。",
            parent=self.ocr_group,
        )
        self.ocr_unclip_ratio = RangeSettingCard(
            configItem=config.ocrUnclipRatio,
            icon=FluentIcon.ZOOM,
            title="OCR 轮廓外扩比例（%）",
            content="PaddleOCR 的 unclip_ratio；过大将削弱精确掩膜优势。",
            parent=self.ocr_group,
        )
        self.ocr_sample_step = RangeSettingCard(
            configItem=config.ocrSampleStep,
            icon=FluentIcon.MOVIE,
            title="动态 OCR 采样间隔（帧）",
            content="仅开启动态 OCR 且选择“采样检测”后生效；越小越准确，也越慢。",
            parent=self.video_group,
        )

        # 视频保存路径
        self.save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"],
            icon=FluentIcon.DOWNLOAD,
            title="输出保存目录",
            content="默认与输入媒体同目录；画质增强和内容修复均使用此目录。" if not config.saveDirectory.value else config.saveDirectory.value,
            parent=self.advanced_group
        )
        self.save_directory.clicked.connect(self.choose_save_directory)

        self.check_update_on_startup = SwitchSettingCard(
            configItem=config.checkUpdateOnStartup,
            icon=FluentIcon.UPDATE,
            title=tr["Setting"]["CheckUpdateOnStartup"],
            content=tr["Setting"]["CheckUpdateOnStartupDesc"],
            parent=self.advanced_group
        )

        self.model_directory = PushSettingCard(
            text="选择缓存目录",
            icon=FluentIcon.FOLDER,
            title="AI 模型缓存目录",
            content=(
                "默认：backend/models/ai（首次选择 AI 模式时按需下载并校验）"
                if not config.modelDirectory.value else config.modelDirectory.value
            ),
            parent=self.ai_group,
        )
        self.model_directory.clicked.connect(self.choose_model_directory)

        self.model_status = PushSettingCard(
            text="刷新状态",
            icon=FluentIcon.SYNC,
            title="AI 模型状态",
            content="正在读取模型缓存……",
            parent=self.ai_group,
        )
        self.model_status.clicked.connect(self.refresh_model_status)
        self.refresh_model_status()

        self.clear_model_cache = PushSettingCard(
            text="删除已下载模型",
            icon=FluentIcon.DELETE,
            title="清理 AI 模型缓存",
            content="只删除本程序按需下载的 AI 权重；不会删除 OCR 模型或原始媒体文件。",
            parent=self.ai_group,
        )
        self.clear_model_cache.clicked.connect(self.clear_ai_model_cache)

        # 添加反馈链接
        self.feedback = PrimaryPushSettingCard(
            text=tr["Setting"]["FeedbackButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["FeedbackTitle"],
            content=tr["Setting"]["FeedbackDesc"],
            parent=self.about_group
        )
        self.feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl(PROJECT_ISSUES_URL)
        ))
        # SettingCard 的 content 使用辅助字号；版本不再与产品名并列在窗口标题栏。
        self.copyright = PrimaryPushSettingCard(
            text=tr["Setting"]["CopyrightButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["CopyrightTitle"],
            content=tr["Setting"]["CopyrightDesc"].format(VERSION),
            parent=self.about_group
        )
        self.copyright.clicked.connect(lambda: self.check_update())
        # 添加项目链接
        self.project_link = HyperlinkCard(
            url=PROJECT_HOME_URL,
            text=PROJECT_HOME_URL,
            icon=FluentIcon.GITHUB,
            title=tr["Setting"]["ProjectLinkTitle"],
            content=tr["Setting"]["ProjectLinkDesc"],
            parent=self.about_group
        )

    @staticmethod
    def _help_document() -> str:
        """帮助窗口正文：内容必须与当前界面和后端行为严格一致。"""
        return (
            "<p class='lead'>先按默认设置完成一次预览；只有文字残留、误修背景或画面变化较快时，"
            "再针对性调整参数。这样更容易判断每一项设置带来的影响。</p>"
            "<h3>内容修复</h3>"
            "<p class='section-note'>用于移除图片或视频中的字幕、文字、符号和水印区域。</p>"
            "<ol>"
            "<li>添加任务：选择图片修复或视频修复，点击“打开”添加文件。</li>"
            "<li>选择“修复区域编辑方式”：可直接手动画框、逐笔涂鸦，或以手工框作为颜色筛选 ROI。它不是最终来源三选一：每种方式各自有预选区，点击其“确认…区域”只提交当前方式，红色区域显示所有已确认方式的组合。矩形/涂鸦编辑时只显示绿色框或橙色笔迹；颜色筛选以橙色显示临时结果。</li>"
            "<li>开始处理：红色半透明区域才是实际送入模型的掩膜；确认无误后点击“运行”。视频的静态最终掩膜与“动态 OCR 字幕定位”只能二选一；若两种处理都需要，请先完成一次，再把输出视频作为新任务处理。 “运行”控制当前页面整个队列，所有符合条件的待处理任务会按顺序逐项执行，而非只运行当前选中行。</li>"
            "<li>停止队列：点击“停止”会终止当前正在执行的内容修复任务；尚未开始的任务保留“待处理”，之后可再次点击“运行”继续。运行中新增媒体仍需先检测/手画并确认修复区域，才会加入当前队列。</li>"
            "</ol>"
            "<h4>修复模式</h4>"
            "<p class='option'>OpenCV Telea<br><span>速度最快的传统修复，优先试用于小文字、近纯色气泡/底板和连续纹理；不需要 AI 模型，也不使用 AI 设备设置。人物、密集线稿或大缺失区不适合它。</span></p>"
            "<p class='option'>LaMa<br><span>通用 AI 修复，适合照片、自然背景和复杂纹理中的文字/水印。它生成的是合理猜测，不会恢复被遮挡的真实历史像素。</span></p>"
            "<p class='option'>Anime-LaMa<br><span>优先用于彩色二次元插画、动画帧与线条清晰的内容。深色文字位于平坦气泡底色时，程序会在严格条件下用真实邻域底色清理 AI 留下的浅灰残笔；复杂结构仍需人工复核。</span></p>"
            "<p class='option'>Manga<br><span>仅面向黑白/灰度漫画和线稿。它会先把 mask 内输入转为灰度再输出，彩色画面会失去色相与饱和度而发白、发灰；不要把它当作彩色 Anime-LaMa。</span></p>"
            "<h4>先判断问题，再调参数</h4>"
            "<p class='tip'>红色 mask 正确，只说明送入修复器的区域正确，不保证生成模型一定补得自然。白色气泡内有残字时，先在颜色筛选中让 ROI 与气泡边框保持约 2–3 px 距离，描边扩张从 1 px（必要时 2 px）开始，再比较 Telea 与 Anime-LaMa；不要靠无限扩大容差或框选整块气泡解决。文字压住人物、发丝、边框或密集线稿时，应缩小 mask 并比较 LaMa/Anime-LaMa。所有调整先在输出副本或 5–10 秒视频片段验证。</p>"
            "<h4>掩膜来源与视频策略</h4>"
            "<p class='tip'>OCR 候选只定位文本行，不承诺字符笔画级分割。手工矩形会修复框内区域；手工涂鸦直接决定像素 mask；"
            "颜色筛选会在用户画出的 ROI 内，保留接近目标色且更接近目标色而非背景参考色的像素；“描边扩张”只在筛选后小幅扩大这些字形，用于补抗锯齿和描边。矩形、涂鸦和颜色筛选可分别确认，后台只合并这些明确确认过的来源；切换方式不会混用编辑中的草稿。视频中的“动态 OCR 字幕定位”是独立开关，且与静态组合掩膜二选一：开启后按采样或逐帧重新检测字幕；关闭后才复用已确认的组合 mask。需要两种方法时，请将第一次输出作为下一次的新输入。</p>"
            "<h3>画质增强</h3>"
            "<p class='section-note'>用于提升图片或视频的输出尺寸与清晰度，结果会另存，不会覆盖源文件。</p>"
            "<ol><li>选择图片任务或视频任务，设置模型、倍率和设备后，点击“打开”添加任务。</li>"
            "<li>点击“运行”按队列顺序逐项执行全部待处理任务；一次只运行一个任务，不是当前选中行的单独启动按钮。运行中仍可点击“打开”追加新任务，它会在当前任务结束后自动接续。</li>"
            "<li>点击“停止”会请求当前增强任务在安全取消点结束；尚未开始的任务保持“待处理”，之后可再次点击“运行”继续。完成或失败的任务可以右键重置，再按当前设置重新处理。</li></ol>"
            "<h4>模型与倍率</h4>"
            "<p class='option'>Anime / General<br><span>Anime 适合二次元插画和动画帧；General 更适合普通照片。</span></p>"
            "<p class='option'>输出倍率<br><span>1×保持原尺寸；2×、3×、4×会让宽和高约变为原来的 2、3、4 倍。倍率越大，处理时间、显存占用和输出文件也越大。4×视频的像素量约为原来的 16 倍。</span></p>"
            "<h4>视频增强</h4>"
            "<p class='tip'>视频会按帧增强并尝试回封装原音频。耗时、显存占用和输出体积都会明显增加，也可能有轻微闪烁。</p>"
            "<h3>AI 设备与高级参数</h3>"
            "<h4>自动设备</h4>"
            "<p class='tip'>自动会先完整尝试 CUDA；若显卡不可用、驱动或显存异常、模型初始化失败，"
            "再尝试 CPU，并使用第一个可用设备。显式选择 CUDA 或 CPU 时不会自动切换。PaddleOCR 使用独立的 Paddle CPU/GPU profile，"
            "会优先尝试 GPU，初始化失败后自动重建为 CPU，不跟随这里的 PyTorch 设备选择。</p>"
            "<ul>"
            "<li>OCR 二值化阈值：降低可找回较弱文字或符号，但也更容易误识别纹理。</li>"
            "<li>OCR 候选置信度阈值：降低会保留更多候选多边形；必须在预览中人工确认。</li>"
            "<li>OCR 轮廓外扩比例：控制检测轮廓的初始外扩；过大可能侵占背景。</li>"
            "<li>最终掩膜扩边：用于已确认的手工矩形；范围可设为 0–300 px。0 可追求最小区域，红色区域覆盖背景时应减小。</li>"
            "<li>涂鸦笔刷半径：选择“手工涂鸦”后，任务页会显示笔刷按钮；按原始媒体像素计算。每一笔会冻结鼠标按下时的半径，之后调整粗细只影响下一笔。预览中的橙色线表示手工路径，确认后红色区域才是最终 mask。</li>"
            "<li>颜色筛选：选择“颜色筛选”后，任务页会显示前景色、背景色、容差和描边扩张按钮；先用手工框限定 ROI，再检查橙色临时结果。描边扩张只会扩大筛到的字形，默认 1 px；它用于补抗锯齿/描边，过大仍可能吞入背景。点击“确认颜色筛选”后，它会并入红色组合 mask，而不是静默执行。容差范围为 0–255；渐变、透明、发光或多色文字通常更适合涂鸦。</li>"
            "<li>动态 OCR 字幕定位：仅视频可用。开启后忽略已确认的组合 mask，改为在处理时自动检测每段字幕；关闭后视频使用已确认的矩形、涂鸦和/或颜色筛选组合。</li>"
            "<li>动态 OCR 字幕像素精细定位：默认关闭。它紧跟内容修复页的“动态 OCR 检测频率”，仅在“视频修复 + 动态 OCR + 逐帧检测”时显示；切换为采样检测会隐藏并自动关闭它。开启后，页面底部才无标题地展开前景、背景、容差和描边扩张按钮，各按钮可悬浮查看用途。程序在每个处理帧重新计算字形 mask，以适应动态背景，绝不复用上一帧字形；但它是颜色启发式，不保证渐变、透明、发光、多色或与背景近似的字幕效果。旧任务若仍保存“采样 + 精细定位”，运行前会自动清除精细定位。</li>"
            "<li>动态 OCR 检测频率：采样检测每隔若干帧 OCR，并以相邻检测结果覆盖中间帧，较快；逐帧检测每帧 OCR，更能跟随移动字幕但最慢。字幕像素精细定位只在逐帧检测下可选。</li>"
            "<li>动态 OCR 采样间隔：仅“采样检测”使用。数值越小检测越密、画面变化时越稳，但处理更慢。</li>"
            "<li>视频处理区间：仅视频可用，决定处理哪些帧，不决定修复哪些像素。在视频预览上右键选择“将当前帧设为处理起点”（或按 [），移动后再右键选择“将当前帧设为处理终点”（或按 ]）；起止帧均包含，支持多个区间。未设置区间时处理整段视频；右键“清除所有处理区间”会恢复全片处理。</li>"
            "</ul>"
            "<p class='note'>输出目录留空时保存到输入媒体所在位置。清理 AI 模型缓存只会删除本程序下载的模型，不会删除原始图片、视频或 OCR 模型。模型、掩膜、全图输入与视频编码的技术细节请参阅仓库 docs/REPAIR_AND_ENHANCEMENT_TECHNICAL_REFERENCE.md。</p>"
        )

    def show_parameter_help(self):
        """用与主界面相同的卡片和按钮展示可滚动帮助。"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("CleanCanvas Studio 帮助与使用指南")
        dialog.setWindowIcon(self.window().windowIcon())
        dialog.setObjectName("ProjectHelpDialog")
        dialog.setAutoFillBackground(True)
        # 主应用当前使用浅色白底。不要从无边框窗口的 palette 推断主题——该
        # palette 在部分 Windows 环境会返回深色底，导致半透明 CardWidget 叠成灰黑。
        # 以下颜色取自当前高级设置页的实际渲染层级：页面底色 #f7f9fc，
        # 内容卡片叠在其上呈现近白色。帮助窗口因此与主界面连续一致。
        dialog.setStyleSheet("QDialog#ProjectHelpDialog { background-color: #f7f9fc; }")
        dialog.setMinimumSize(820, 680)
        dialog.resize(900, 760)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # 与浅色主界面保持同一层级：深色正文、低对比度辅助文字，以及几乎白色
        # 的说明块。这里使用稳定的应用色而非对话框 palette，避免平台 palette
        # 在深色系统偏好下反转帮助内容的可读性。
        text_color = "#202124"
        muted_color = "#667085"
        surface_color = "#f7f9fc"

        guide_card = CardWidget(dialog)
        guide_layout = QtWidgets.QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(20, 16, 20, 16)
        guide_layout.setSpacing(10)
        guide_layout.addWidget(SubtitleLabel("使用指南", guide_card))
        browser = QtWidgets.QTextBrowser(guide_card)
        browser.setFont(dialog.font())
        browser.document().setDefaultFont(dialog.font())
        browser.setOpenExternalLinks(True)
        browser.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        browser.setFrameShape(QtWidgets.QFrame.NoFrame)
        browser.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; padding: 0; }"
            "QScrollBar:vertical { margin: 2px 0; }"
        )
        browser.document().setDefaultStyleSheet(
            f"body {{ font-size: 14px; line-height: 1.65; color: {text_color}; }}"
            f"h3 {{ color: {text_color}; font-size: 16px; font-weight: 600; margin: 18px 0 6px; }}"
            "h4 { font-size: 14px; font-weight: 600; margin: 12px 0 4px; }"
            f".lead, .section-note, .note {{ color: {muted_color}; }}"
            ".lead { margin: 0 0 12px; } .section-note { margin: 0 0 6px; }"
            f".option {{ margin: 6px 0; padding: 8px 10px; background: {surface_color}; border: 1px solid #edf1f5; }}"
            f".option span {{ color: {muted_color}; }}"
            f".tip {{ margin: 10px 0; padding: 10px 11px; background: {surface_color}; border: 1px solid #edf1f5; }}"
            "ol, ul { margin: 5px 0 8px 18px; padding: 0; } li { margin: 4px 0; }"
        )
        browser.setHtml(self._help_document())
        guide_layout.addWidget(browser)

        close_button = PrimaryPushButton("关闭", dialog)
        close_button.setMinimumWidth(96)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(guide_card, 1)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignRight)
        dialog.exec()

    def show_message_box(self, title: str, content: str, showYesButton=False, yesSlot=None):
        """ show message box """
        w = MessageBox(title, content, self)
        if not showYesButton:
            w.cancelButton.setText(self.tr('Close'))
            w.yesButton.hide()
            w.buttonLayout.insertStretch(0, 1)

        if w.exec() and yesSlot is not None:
            yesSlot()

    def check_update(self, ignore=False):
        """ check software update

        Parameters
        ----------
        ignore: bool
            ignore message box when no updates are available
        """
        (TaskExecutor.runTask(self.version_manager.has_new_version)
        .then(lambda success: self.on_version_info_fetched(success, ignore)))

    def on_version_info_fetched(self, success, ignore=False):
        """将后台更新检查结果转换为主线程中的用户提示。"""
        if success:
            self.show_message_box(
                tr["Setting"]["UpdatesAvailableTitle"],
                tr["Setting"]["UpdatesAvailableDesc"].format(self.version_manager.lastest_version),
                True,
                lambda: QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl(PROJECT_RELEASES_URL)
                )
            )
        elif not ignore:
            self.show_message_box(
                tr["Setting"]["NoUpdatesAvailableTitle"],
                tr["Setting"]["NoUpdatesAvailableDesc"],
            )

    def choose_save_directory(self):
        """选择保存目录"""
        last_save_directory = "./" if not config.saveDirectory.value else config.saveDirectory.value
        folder = QFileDialog.getExistingDirectory(
            self, tr['Setting']['ChooseDirectory'], last_save_directory)
        if not folder:
            folder = ""

        config.set(config.saveDirectory, folder)
        self.save_directory.setContent(
            "默认与输入媒体同目录；画质增强和内容修复均使用此目录。"
            if not config.saveDirectory.value else config.saveDirectory.value
        )

    def choose_model_directory(self):
        """设置可迁移的 AI 权重缓存目录；留空即使用项目默认目录。"""
        initial = config.modelDirectory.value or "./backend/models/ai"
        folder = QFileDialog.getExistingDirectory(self, "选择 AI 模型缓存目录", initial)
        if not folder:
            return
        config.set(config.modelDirectory, folder)
        self.model_directory.setContent(folder)

    def clear_ai_model_cache(self):
        """仅删除注册表中的按需模型，防止把用户自定义缓存目录整棵清空。"""
        store = ModelStore(config.modelDirectory.value or None)
        assets = (LAMA_ASSET, ANIME_LAMA_ASSET, MANGA_INPAINT_ASSET, MANGA_LINE_ASSET, ANIME_ASSET, GENERAL_ASSET)
        targets = [store.path_for(asset) for asset in assets if store.path_for(asset).is_file()]
        if not targets:
            self.show_message_box("没有可清理的模型", "当前缓存目录中没有本程序下载的 AI 模型。")
            return
        names = "\n".join(f"• {target.name}" for target in targets)
        dialog = MessageBox("确认删除 AI 模型", f"将删除以下已下载权重：\n{names}\n\nOCR 模型、原始图片和视频不会受到影响。", self)
        dialog.yesButton.setText("删除")
        dialog.cancelButton.setText("取消")
        if not dialog.exec():
            return
        failures = []
        for target in targets:
            try:
                target.unlink()
            except OSError as exc:
                failures.append(f"{target.name}: {exc}")
        if failures:
            self.show_message_box("部分模型未删除", "\n".join(failures))
        else:
            self.show_message_box("清理完成", "已删除选择的 AI 模型；下次使用相应模式时会重新下载并校验。")
        self.refresh_model_status()

    def refresh_model_status(self):
        """显示缓存中可被当前模式发现的权重；推理时仍会再做哈希校验。"""
        store = ModelStore(config.modelDirectory.value or None)
        assets = (LAMA_ASSET, ANIME_LAMA_ASSET, MANGA_INPAINT_ASSET, MANGA_LINE_ASSET, ANIME_ASSET, GENERAL_ASSET)
        available = []
        for asset in assets:
            if store.path_for(asset).is_file() or (asset.bundled_path and asset.bundled_path.is_file()):
                available.append(asset.filename)
        if available:
            self.model_status.setContent(f"已检测到 {len(available)}/{len(assets)} 个：" + "、".join(available))
        else:
            self.model_status.setContent("尚未下载 AI 权重；首次选择相应模式时会自动下载并校验。")
