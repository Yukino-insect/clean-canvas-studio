
# Portions derived from YaoFANGUK/video-subtitle-remover (Apache-2.0).
# Modified by CleanCanvas Studio contributors; see LICENSE, NOTICE and MODIFICATIONS.md.
import json
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

    页面级选项通过 qfluentwidgets 的 ConfigItem 自动保存；待处理任务会在
    配置修改时同步其运行快照，而已经开始运行的任务保持启动时的快照不变。
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

    # 默认 OCR 检测区域以比例坐标持久化，格式为
    # "ymin,ymax,xmin,xmax;ymin,ymax,xmin,xmax;..."。
    ocrSelectionRegions = ConfigItem("Main", "OcrSelectionRegions", "0.88,0.99,0.15,0.85")

    # OCR 候选多边形向外扩张的像素数。允许 0 px，以便精细控制最终掩膜；
    # 默认 2 px 用于覆盖常见的抗锯齿边缘，避免把过大的背景区域交给修复模型。
    ocrMaskPadding = RangeConfigItem("Main", "OcrMaskPadding", 2, RangeValidator(0, 300))
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

    # 最终 mask 由用户明确选择的手工来源构建。视频动态 OCR 是独立的字幕
    # 定位开关，不属于最终 mask 来源。
    maskMode = OptionsConfigItem(
        "Mask", "Mode", "rectangle",
        OptionsValidator(["rectangle", "brush", "color"]),
    )
    videoDynamicOcr = ConfigItem("Mask", "VideoDynamicOcr", False, BoolValidator())
    videoMaskStrategy = OptionsConfigItem(
        "Mask", "VideoStrategy", "sampled_ocr",
        OptionsValidator(["sampled_ocr", "every_frame_ocr"]),
    )
    ocrPixelThreshold = RangeConfigItem("Mask", "OcrPixelThreshold", 30, RangeValidator(1, 99))
    ocrBoxThreshold = RangeConfigItem("Mask", "OcrBoxThreshold", 60, RangeValidator(1, 99))
    ocrUnclipRatio = RangeConfigItem("Mask", "OcrUnclipRatio", 120, RangeValidator(1, 500))
    brushRadius = RangeConfigItem("Mask", "BrushRadius", 12, RangeValidator(1, 100))
    colorMaskForeground = ConfigItem("Mask", "ColorForeground", "#FFFFFF", ConfigValidator())
    colorMaskBackground = ConfigItem("Mask", "ColorBackground", "#000000", ConfigValidator())
    colorMaskTolerance = RangeConfigItem("Mask", "ColorTolerance", 30, RangeValidator(0, 255))
    # 颜色筛选先精确提取前景像素，再只扩张这些像素以覆盖抗锯齿和描边。
    # 这不是 ROI 外扩：ROI 仍然只是用户限定的取样范围。
    colorMaskOutlineExpand = RangeConfigItem("Mask", "ColorOutlineExpand", 1, RangeValidator(0, 32))
    ocrSampleStep = RangeConfigItem("Mask", "OcrSampleStep", 3, RangeValidator(1, 30))

    # 动态 OCR 的文字检测只给出文本行区域。该可选步骤在每个多边形 ROI
    # 中按颜色提取真实字幕像素；默认关闭以保持既有任务结果完全不变。
    dynamicOcrRefineMask = ConfigItem("Mask", "DynamicOcrRefineMask", False, BoolValidator())
    dynamicOcrForeground = ConfigItem("Mask", "DynamicOcrForeground", "#FFFFFF", ConfigValidator())
    dynamicOcrBackground = ConfigItem("Mask", "DynamicOcrBackground", "#182030", ConfigValidator())
    dynamicOcrRefineTolerance = RangeConfigItem("Mask", "DynamicOcrRefineTolerance", 55, RangeValidator(0, 255))
    dynamicOcrOutlineExpand = RangeConfigItem("Mask", "DynamicOcrOutlineExpand", 1, RangeValidator(0, 32))

CONFIG_FILE = 'config/config.json'


def _migrate_legacy_main_config(config_file: str) -> None:
    """将旧版“字幕”配置键一次性迁移为当前内容修复术语。

    两个仍会影响现有行为的值被无损搬迁；其余键属于已经删除的旧检测流程，
    移除后不会改变当前应用的有效配置。迁移在加载 qfluentwidgets 配置之前
    执行，因此用户不会因为内部键重命名而丢失选区或掩膜扩边设置。
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        return
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        main = data.get("Main")
        if not isinstance(main, dict):
            return
    except (OSError, ValueError, TypeError):
        # 配置读取错误仍交由 qconfig 的正常加载流程处理；启动时不因迁移失败中断。
        return

    renamed_keys = {
        "SubtitleSelectionAreas": "OcrSelectionRegions",
        "SubtitleAreaDeviationPixel": "OcrMaskPadding",
    }
    removed_keys = {
        "SubtitleYXAxisDifferencePixel",
        "SubtitleAreaYAxisDifferencePixel",
        "SubtitleAreaPixelToleranceYPixel",
        "SubtitleAreaPixelToleranceXPixel",
        "SubtitleTimelineBackwardFrameCount",
        "subtitleTimelineForwardFrameCount",
    }
    changed = False
    for legacy_key, current_key in renamed_keys.items():
        if legacy_key in main:
            main.setdefault(current_key, main[legacy_key])
            del main[legacy_key]
            changed = True
    for legacy_key in removed_keys:
        if legacy_key in main:
            del main[legacy_key]
            changed = True

    if changed:
        try:
            config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=4) + "\n",
                encoding="utf-8",
            )
        except OSError:
            # 只读配置文件不应阻止应用使用默认值或 qconfig 可加载的已有值。
            pass


_migrate_legacy_main_config(CONFIG_FILE)
config = Config()
qconfig.load(CONFIG_FILE, config)

# 读取界面语言配置
tr = configparser.ConfigParser()

TRANSLATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', f"{config.interface.value}.ini")
tr.read(TRANSLATION_FILE, encoding='utf-8')

# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
