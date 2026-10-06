"""统一的本地图片修复引擎。

本模块刻意只包含掩膜修复所需的薄封装，而不是嵌入 IOPaint 的 Web/
Diffusion 全套依赖。LaMa、Anime-LaMa 与 Manga 的输入约定及预处理方式
参考 IOPaint（Apache-2.0）：RGB uint8 图像 + 白色掩膜，输出 BGR uint8。
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from backend.inpaint.opencv_inpaint import OpenCVInpaint


class InpaintMode(str, Enum):
    OPENCV = "opencv"
    LAMA = "lama"
    ANIME_LAMA = "anime_lama"
    MANGA = "manga"


@dataclass(frozen=True)
class InpaintModeInfo:
    mode: InpaintMode
    title: str
    description: str
    video_note: str
    requires_model: bool


MODE_INFOS: dict[InpaintMode, InpaintModeInfo] = {
    InpaintMode.OPENCV: InpaintModeInfo(
        InpaintMode.OPENCV,
        "OpenCV Telea（快速）",
        "适合小字幕、纯色或连续背景；速度最快。",
        "逐帧稳定且快，复杂纹理修复能力有限。",
        False,
    ),
    InpaintMode.LAMA: InpaintModeInfo(
        InpaintMode.LAMA,
        "LaMa（通用 AI）",
        "适合照片和复杂背景中的文字、水印或小物体。",
        "逐帧 AI 修复可能产生轻微闪烁；建议先用短片段验证。",
        True,
    ),
    InpaintMode.ANIME_LAMA: InpaintModeInfo(
        InpaintMode.ANIME_LAMA,
        "Anime-LaMa（二次元 AI）",
        "优先用于彩色动漫插画、动画画面中的文字和碍眼符号。",
        "没有时序约束；长视频或复杂运动场景可能闪烁。",
        True,
    ),
    InpaintMode.MANGA: InpaintModeInfo(
        InpaintMode.MANGA,
        "Manga（黑白漫画 AI）",
        "面向黑白线稿、漫画气泡和拟声文字；不推荐彩色动画画面。",
        "逐帧 AI 修复可能闪烁，且对彩色画面效果不稳定。",
        True,
    ),
}


class InpaintError(RuntimeError):
    """修复模式无法准备或推理失败时抛出的可读错误。"""


class InpaintEngine(Protocol):
    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    url: str
    md5: str
    bundled_path: Path | None = None
    bundled_md5: str | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "backend" / "models" / "ai"

LAMA_ASSET = ModelAsset(
    filename="big-lama.pt",
    url="https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    md5="e3aa4aaa15225a33ec84f9f4bc47e500",
    # 当前仓库的历史权重可用但哈希与 IOPaint 当前发布版本不同；仅作为离线兼容候选。
    bundled_path=PROJECT_ROOT / "backend" / "models" / "big-lama" / "big-lama.pt",
    bundled_md5="19970cd5673cd1326c70885ea2b5dbb8",
)
ANIME_LAMA_ASSET = ModelAsset(
    filename="anime-manga-big-lama.pt",
    url="https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
    md5="29f284f36a0a510bcacf39ecf4c4d54f",
)
MANGA_INPAINT_ASSET = ModelAsset(
    filename="manga_inpaintor.jit",
    url="https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit",
    md5="7d8b269c4613b6b3768af714610da86c",
)
MANGA_LINE_ASSET = ModelAsset(
    filename="erika.jit",
    url="https://github.com/Sanster/models/releases/download/manga/erika.jit",
    md5="0c926d5a4af8450b0d00bc5b9a095644",
)


class ModelStore:
    """模型缓存、完整性校验与原子下载。

    下载永远先落到临时文件，哈希匹配后再替换正式文件，避免取消/断网后把
    半个权重当成有效模型。缓存目录可由 ``VSR_MODEL_DIR`` 或高级设置覆盖。
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir or os.environ.get("VSR_MODEL_DIR") or DEFAULT_MODEL_DIR)
        self._lock = threading.Lock()

    @staticmethod
    def _md5(path: Path) -> str:
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def path_for(self, asset: ModelAsset) -> Path:
        return self.model_dir / asset.filename

    def is_ready(self, asset: ModelAsset) -> bool:
        target = self.path_for(asset)
        return target.is_file() and self._md5(target) == asset.md5

    def ensure(self, asset: ModelAsset, *, allow_bundled_fallback: bool = False) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        target = self.path_for(asset)
        with self._lock:
            if self.is_ready(asset):
                return target

            # 仅 LaMa 的历史随包权重走此分支：其来源版本较旧，无法以当前远程
            # 发布哈希验证，但在离线环境下仍优先保证已有安装可用。
            if allow_bundled_fallback and asset.bundled_path and asset.bundled_path.is_file():
                legacy_md5 = self._md5(asset.bundled_path)
                if legacy_md5 == asset.bundled_md5:
                    return asset.bundled_path
                raise InpaintError(
                    f"项目内置的历史模型 {asset.bundled_path.name} 校验失败（{legacy_md5}）；"
                    "请删除该文件后重新下载官方模型。"
                )

            temp_fd, temp_name = tempfile.mkstemp(
                prefix=f".{asset.filename}.", suffix=".download", dir=self.model_dir
            )
            os.close(temp_fd)
            temp_path = Path(temp_name)
            try:
                urllib.request.urlretrieve(asset.url, temp_path)
                actual_md5 = self._md5(temp_path)
                if actual_md5 != asset.md5:
                    raise InpaintError(
                        f"模型校验失败：{asset.filename} 的 MD5 为 {actual_md5}，"
                        f"预期为 {asset.md5}。已丢弃不完整或错误文件。"
                    )
                shutil.move(str(temp_path), str(target))
                return target
            except InpaintError:
                raise
            except Exception as exc:
                raise InpaintError(
                    f"无法下载模型 {asset.filename}。请检查网络、代理和磁盘空间，"
                    f"然后重试。原始错误：{exc}"
                ) from exc
            finally:
                temp_path.unlink(missing_ok=True)


def device_candidates(preference: str = "auto") -> tuple[str, ...]:
    """返回设备初始化的候选顺序。

    ``auto`` 不是简单地依据 ``torch.cuda.is_available()`` 作一次判断：驱动、
    显存或特定模型的 CUDA 初始化仍可能失败。因此调用方必须按此顺序真正尝试
    初始化模型，CUDA 失败后才尝试 CPU。显式选择 CPU/CUDA 时不作自动降级，
    这样用户的选择仍然可预测。
    """
    normalized = (preference or "auto").lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise InpaintError(f"未知设备设置：{preference}。可用值为 auto、cuda、cpu。")
    return ("cuda", "cpu") if normalized == "auto" else (normalized,)


def resolve_device(preference: str = "auto") -> str:
    """校验单个 Torch 设备并返回其规范名称。

    自动模式的完整回退由 ``create_inpaint_engine`` / Real-ESRGAN 在模型加载
    阶段执行；这里仅校验传入的单个候选设备。自动模式传入时会取候选链中的
    首项 CUDA，并在上层初始化失败后继续尝试 CPU。
    """
    normalized = device_candidates(preference)[0]
    if normalized == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise InpaintError("AI 修复模式需要 PyTorch。请按照 README 安装与显卡匹配的 torch 版本。") from exc
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise InpaintError("已选择 CUDA，但当前 PyTorch 未检测到可用的 NVIDIA CUDA 设备。")
        return "cuda"
    return "cuda"


def _pad_to_modulo(image: np.ndarray, modulo: int, *, constant: bool = False) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    height, width = image.shape[:2]
    padded_height = (height + modulo - 1) // modulo * modulo
    padded_width = (width + modulo - 1) // modulo * modulo
    padding = ((0, padded_height - height), (0, padded_width - width), (0, 0))
    return np.pad(image, padding, mode="constant" if constant else "symmetric")


class OpenCVEngine:
    """传统 Telea 修复适配器，保持与 AI 引擎相同的 inpaint 接口。"""
    def __init__(self) -> None:
        self._engine = OpenCVInpaint()

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._engine.inpaint(image_bgr, mask)


class TorchScriptLaMaEngine:
    """LaMa/Anime-LaMa 的轻量 TorchScript 运行器。"""

    def __init__(self, asset: ModelAsset, device_preference: str = "auto", model_dir: str | Path | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise InpaintError("AI 修复模式需要 PyTorch。") from exc
        self.torch = torch
        self.device = resolve_device(device_preference)
        path = ModelStore(model_dir).ensure(asset, allow_bundled_fallback=asset is LAMA_ASSET)
        try:
            self.model = torch.jit.load(str(path), map_location="cpu").to(self.device).eval()
        except Exception as exc:
            raise InpaintError(f"无法加载 {asset.filename}：{exc}") from exc

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """执行 TorchScript 修复，并强制恢复 mask 外的原始像素。"""
        if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
            raise InpaintError("AI 修复只接受三通道彩色图像。")
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        binary_mask = (mask > 0).astype(np.uint8)
        padded_image = _pad_to_modulo(rgb, 8)
        padded_mask = _pad_to_modulo(binary_mask, 8, constant=True)[:, :, 0]
        image_tensor = self.torch.from_numpy(padded_image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask_tensor = self.torch.from_numpy(padded_mask[np.newaxis].astype(np.float32))
        try:
            with self.torch.inference_mode():
                result = self.model(image_tensor.unsqueeze(0).to(self.device), mask_tensor.unsqueeze(0).to(self.device))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                raise InpaintError("显存不足。请在高级设置中切换 CPU，或缩小选区/分辨率后重试。") from exc
            raise InpaintError(f"LaMa 推理失败：{exc}") from exc
        result_rgb = result[0].permute(1, 2, 0).detach().cpu().numpy()
        result_bgr = cv2.cvtColor(np.clip(result_rgb[:height, :width] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # 网络偶尔会轻微修改选区外像素；保持未遮罩区域完全不变。
        result_bgr[binary_mask == 0] = image_bgr[:, :, :3][binary_mask == 0]
        return result_bgr


class MangaEngine:
    """IOPaint Manga 模式的最小运行器，仅用于黑白漫画内容。"""

    def __init__(self, device_preference: str = "auto", model_dir: str | Path | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise InpaintError("Manga 模式需要 PyTorch。") from exc
        self.torch = torch
        self.device = resolve_device(device_preference)
        store = ModelStore(model_dir)
        try:
            self.inpaintor = torch.jit.load(str(store.ensure(MANGA_INPAINT_ASSET)), map_location="cpu").to(self.device).eval()
            self.line = torch.jit.load(str(store.ensure(MANGA_LINE_ASSET)), map_location="cpu").to(self.device).eval()
        except InpaintError:
            raise
        except Exception as exc:
            raise InpaintError(f"无法加载 Manga 模型：{exc}") from exc

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """以灰度与线稿先验执行漫画修复，再转换回统一的 BGR 输出。"""
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
        binary_mask = (mask > 0).astype(np.float32)
        padded_gray = _pad_to_modulo(gray, 16)[:, :, 0]
        padded_mask = _pad_to_modulo(binary_mask, 16, constant=True)[:, :, 0]
        torch = self.torch
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        gray_tensor = torch.from_numpy(padded_gray[np.newaxis, np.newaxis].astype(np.float32)).to(self.device)
        mask_tensor = torch.from_numpy(padded_mask[np.newaxis, np.newaxis]).to(self.device)
        try:
            with torch.inference_mode():
                lines = torch.clamp(self.line(gray_tensor), 0, 255)
                normalized_gray = gray_tensor / 255 * 2 - 1.0
                normalized_lines = lines / 255 * 2 - 1.0
                mask_tensor = (mask_tensor > 0.5).float()
                result = self.inpaintor(
                    normalized_gray, normalized_lines, mask_tensor,
                    torch.randn_like(mask_tensor), torch.ones_like(mask_tensor),
                )
        except RuntimeError as exc:
            raise InpaintError(f"Manga 推理失败：{exc}") from exc
        result_gray = np.clip((result[0, 0].detach().cpu().numpy() * 127.5 + 127.5), 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_gray[:height, :width], cv2.COLOR_GRAY2BGR)
        binary_mask_bool = binary_mask.astype(bool)
        result_bgr[~binary_mask_bool] = image_bgr[:, :, :3][~binary_mask_bool]
        return result_bgr


def create_inpaint_engine(
    mode: InpaintMode | str,
    *,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> InpaintEngine:
    """按模式创建惰性修复引擎；调用方不需要了解模型细节。

    自动设备模式会真实初始化 CUDA，再在失败后初始化 CPU；不能只依赖
    ``torch.cuda.is_available()``，因为模型加载、驱动或显存仍可能失败。
    """
    try:
        selected = mode if isinstance(mode, InpaintMode) else InpaintMode(mode)
    except ValueError as exc:
        choices = ", ".join(item.value for item in InpaintMode)
        raise InpaintError(f"未知修复模式：{mode}。可用模式：{choices}。") from exc
    if selected is InpaintMode.OPENCV:
        return OpenCVEngine()

    def build(candidate: str) -> InpaintEngine:
        if selected is InpaintMode.LAMA:
            return TorchScriptLaMaEngine(LAMA_ASSET, candidate, model_dir)
        if selected is InpaintMode.ANIME_LAMA:
            return TorchScriptLaMaEngine(ANIME_LAMA_ASSET, candidate, model_dir)
        return MangaEngine(candidate, model_dir)

    errors: list[tuple[str, InpaintError]] = []
    for candidate in device_candidates(device):
        try:
            return build(candidate)
        except InpaintError as exc:
            errors.append((candidate, exc))

    # 显式设备只会经过一次循环；保留原始、可定位的错误信息。自动模式则将
    # 两次真实初始化的失败原因一起呈现，用户无需猜测是否已测试过 CUDA。
    if len(errors) == 1:
        raise errors[0][1]
    details = "；".join(f"{candidate.upper()}：{error}" for candidate, error in errors)
    raise InpaintError(f"自动设备初始化失败（已依次尝试 CUDA、CPU）：{details}") from errors[-1][1]
