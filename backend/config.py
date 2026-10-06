
# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
import os
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
# qfluentwidgets emits a promotional line while importing its configuration
# module. This shared backend config is also imported by CLI/tests, so suppress
# only that import-time third-party output without changing normal application logs.
with redirect_stdout(StringIO()):
    from qfluentwidgets import (qconfig, ConfigItem, QConfig, OptionsValidator, OptionsConfigItem, BoolValidator,
                                RangeValidator, RangeConfigItem, ConfigValidator)
import configparser

# PaddleOCR/PaddleX 会在首次导入时读取该变量；模型已随项目提供，
# 因此无需进行远程模型源连通性检查。
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# 项目版本号由 CleanCanvas Studio 独立维护，不与任何上游项目比较或继承版本序列。
VERSION = "0.1.0"
PROJECT_HOME_URL = "https://github.com/Yukino-insect/clean-canvas-studio"
PROJECT_ISSUES_URL = PROJECT_HOME_URL + "/issues"
PROJECT_RELEASES_URL = PROJECT_HOME_URL + "/releases"
PROJECT_UPDATE_URLS = [
    "https://api.github.com/repos/Yukino-insect/clean-canvas-studio/releases/latest",
]

class Config(QConfig):
    """应用持久化配置。

    页面级选项通过 qfluentwidgets 的 ConfigItem 自动保存；任务开始时会把
    与媒体处理相关的配置复制到任务对象，避免运行中修改设置影响队列快照。
    """
    # 界面语言设置
    intefaceTexts = {
        '简体中文': 'ch',
        '繁體中文': 'chinese_cht',
        'English': 'en',
        '한국어': 'ko',
        '日本語': 'japan',
        'Tiếng Việt': 'vi',
        'Español': 'es'
    }
    interface = OptionsConfigItem("Window", "Interface", "ChineseSimplified", OptionsValidator(intefaceTexts.values()), restart = True)

    # 窗口位置和大小
    windowX = ConfigItem("Window", "X", None)
    windowY = ConfigItem("Window", "Y", None)
    windowW = ConfigItem("Window", "Width", 1368)
    windowH = ConfigItem("Window", "Height", 758)

    # 使用一个配置项存储所有选区
    # 默认值为一个选区，格式为："ymin,ymax,xmin,xmax;ymin,ymax,xmin,xmax;..."，分号分隔不同选区
    subtitleSelectionAreas = ConfigItem("Main", "SubtitleSelectionAreas", "0.88,0.99,0.15,0.85")

    # 【设置像素点偏差】
    # 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
    subtitleYXAxisDifferencePixel = RangeConfigItem("Main", "SubtitleYXAxisDifferencePixel", 10, RangeValidator(0, 300))
    # 用于放大mask大小，防止自动检测的文本框过小，inpaint阶段出现文字边，有残留
    subtitleAreaDeviationPixel = RangeConfigItem("Main", "SubtitleAreaDeviationPixel", 10, RangeValidator(1, 300))
    # 用于判断两个文本框是否为同一行字幕，高度差距指定像素点以内认为是同一行
    subtitleAreaYAxisDifferencePixel = RangeConfigItem("Main", "SubtitleAreaYAxisDifferencePixel", 20, RangeValidator(0, 300))
    # 用于判断两个字幕文本的矩形框是否相似，如果X轴和Y轴偏差都在指定阈值内，则认为时同一个文本框
    subtitleAreaPixelToleranceYPixel = RangeConfigItem("Main", "SubtitleAreaPixelToleranceYPixel", 20, RangeValidator(0, 300))
    subtitleAreaPixelToleranceXPixel = RangeConfigItem("Main", "SubtitleAreaPixelToleranceXPixel", 20, RangeValidator(0, 300))
    # 用于扩展帧数
    subtitleTimelineBackwardFrameCount = RangeConfigItem("Main", "SubtitleTimelineBackwardFrameCount", 3, RangeValidator(0, 300))
    subtitleTimelineForwardFrameCount = RangeConfigItem("Main", "subtitleTimelineForwardFrameCount", 3, RangeValidator(0, 300))
    # 启动时检查应用更新
    # 独立仓库首次发布 Release 后，用户可在设置中自行开启。
    checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", False, BoolValidator())

    # 视频保存目录
    saveDirectory = ConfigItem("Main", "SaveDirectory", "", ConfigValidator())

    # AI 修复/增强运行设置。OpenCV 始终无需 Torch，AI 权重在真正选择模式后才加载。
    inpaintMode = OptionsConfigItem(
        "AI", "InpaintMode", "opencv",
        OptionsValidator(["opencv", "lama", "anime_lama", "manga"]),
    )
    aiDevice = OptionsConfigItem(
        "AI", "Device", "auto", OptionsValidator(["auto", "cuda", "cpu"]),
    )
    modelDirectory = ConfigItem("AI", "ModelDirectory", "", ConfigValidator())

    # 画质增强页独立保存模型和倍率；设备与修复页共用，避免同一块 GPU 出现
    # 两套互相矛盾的设置。
    enhanceMode = OptionsConfigItem(
        "Enhance", "Mode", "realesrgan_anime",
        OptionsValidator(["realesrgan_anime", "realesrgan_general"]),
    )
    enhanceScale = OptionsConfigItem(
        "Enhance", "Scale", "2",
        OptionsValidator(["1", "1.5", "2", "3", "4"]),
    )

    # 精确掩膜：OCR 多边形与视频更新策略。阈值采用百分数，避免浮点配置编辑器
    # 在不同版本的 QFluentWidgets 中表现不一致。
    maskMode = OptionsConfigItem(
        "Mask", "Mode", "ocr_polygon",
        OptionsValidator(["rectangle", "ocr_polygon", "ocr_refined", "mixed"]),
    )
    videoMaskStrategy = OptionsConfigItem(
        "Mask", "VideoStrategy", "static",
        OptionsValidator(["static", "sampled_ocr", "every_frame_ocr"]),
    )
    ocrPixelThreshold = RangeConfigItem("Mask", "OcrPixelThreshold", 30, RangeValidator(1, 99))
    ocrBoxThreshold = RangeConfigItem("Mask", "OcrBoxThreshold", 60, RangeValidator(1, 99))
    ocrUnclipRatio = RangeConfigItem("Mask", "OcrUnclipRatio", 120, RangeValidator(50, 300))
    maskRefineColorThreshold = RangeConfigItem("Mask", "RefineColorThreshold", 22, RangeValidator(1, 128))
    maskRefineMinArea = RangeConfigItem("Mask", "RefineMinArea", 8, RangeValidator(1, 1000))
    ocrSampleStep = RangeConfigItem("Mask", "OcrSampleStep", 3, RangeValidator(1, 30))

CONFIG_FILE = 'config/config.json'
config = Config()
qconfig.load(CONFIG_FILE, config)

# 读取界面语言配置
tr = configparser.ConfigParser()

TRANSLATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', f"{config.interface.value}.ini")
tr.read(TRANSLATION_FILE, encoding='utf-8')

# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
