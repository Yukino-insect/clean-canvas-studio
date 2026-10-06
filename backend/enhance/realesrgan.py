"""面向图片与视频帧的 Real-ESRGAN 超分辨率。

本文件的 RRDBNet、SRVGGNetCompact 与分块推理结构参考 IOPaint
(`plugins/realesrgan.py`，Apache-2.0) 及其随附的 BasicSR 实现；Real-ESRGAN
模型本身遵循上游 BSD-3-Clause，具体归属见 THIRD_PARTY_NOTICES.md。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from backend.inpaint.engine import InpaintError, ModelAsset, ModelStore, device_candidates, resolve_device


class EnhanceMode(str, Enum):
    REAL_ESRGAN_ANIME = "realesrgan_anime"
    REAL_ESRGAN_GENERAL = "realesrgan_general"


@dataclass(frozen=True)
class EnhanceModeInfo:
    mode: EnhanceMode
    title: str
    description: str


MODE_INFOS = {
    EnhanceMode.REAL_ESRGAN_ANIME: EnhanceModeInfo(
        EnhanceMode.REAL_ESRGAN_ANIME,
        "Real-ESRGAN Anime",
        "二次元插画/动漫画面优先；会推测细节，需人工复核线条与文字。",
    ),
    EnhanceMode.REAL_ESRGAN_GENERAL: EnhanceModeInfo(
        EnhanceMode.REAL_ESRGAN_GENERAL,
        "Real-ESRGAN General",
        "通用照片与普通图像优先；不等于恢复真实丢失像素。",
    ),
}


ANIME_ASSET = ModelAsset(
    filename="RealESRGAN_x4plus_anime_6B.pth",
    url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    md5="d58ce384064ec1591c2ea7b79dbf47ba",
)
GENERAL_ASSET = ModelAsset(
    filename="realesr-general-x4v3.pth",
    url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    md5="91a7644643c884ee00737db24e478156",
)


def _make_layer(block, count: int, **kwargs):
    import torch.nn as nn
    return nn.Sequential(*(block(**kwargs) for _ in range(count)))


def _pixel_unshuffle(x, scale: int):
    import torch
    batch, channels, height, width = x.size()
    return x.view(batch, channels, height // scale, scale, width // scale, scale).permute(
        0, 1, 3, 5, 2, 4
    ).reshape(batch, channels * (scale**2), height // scale, width // scale)


def _build_rrdb_net(num_block: int):
    """延迟定义网络，避免 OpenCV 模式导入 GUI 时加载 Torch。"""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
            self.body = _make_layer(RRDB, num_block, num_feat=64, num_grow_ch=32)
            self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(functional.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(functional.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def _build_srvgg_net():
    from torch import nn
    from torch.nn import functional as functional

    class SRVGGNetCompact(nn.Module):
        def __init__(self):
            super().__init__()
            body = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(num_parameters=64)]
            for _ in range(32):
                body.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(num_parameters=64)])
            body.append(nn.Conv2d(64, 3 * 4 * 4, 3, 1, 1))
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(4)

        def forward(self, x):
            output = x
            for layer in self.body:
                output = layer(output)
            return self.upsampler(output) + functional.interpolate(x, scale_factor=4, mode="nearest")

    return SRVGGNetCompact()


class RealESRGANEnhancer:
    """Real-ESRGAN 推理器。

    网络固定产生原生 4×结果；调用方可请求 1×至4×，非 4×倍率会在推理后
    通过高质量缩放得到目标尺寸。大图按 tile 分块以限制显存峰值。
    """

    native_scale = 4

    def __init__(
        self,
        mode: EnhanceMode | str = EnhanceMode.REAL_ESRGAN_ANIME,
        *,
        device: str = "auto",
        model_dir: str | Path | None = None,
        tile_size: int = 512,
        tile_pad: int = 10,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise InpaintError("画质增强需要 PyTorch。请按照 README 安装与设备匹配的 torch 版本。") from exc
        self.torch = torch
        self.mode = EnhanceMode(mode)
        self.requested_device = (device or "auto").lower()
        candidates = device_candidates(self.requested_device)
        self.device_fallback_reason: str | None = None
        self.tile_size = max(0, int(tile_size))
        self.tile_pad = max(0, int(tile_pad))
        asset, model = self._asset_and_model()
        path = ModelStore(model_dir).ensure(asset)
        errors: list[tuple[str, Exception]] = []
        for candidate in candidates:
            try:
                # 对显式 CUDA 也先做可用性校验，避免将底层异常包装成难懂的
                # ``model.to`` 错误。自动模式会把此失败继续交给下一候选设备。
                selected_device = resolve_device(candidate)
                candidate_model = model if not errors else self._asset_and_model()[1]
                self.model, self.use_half = self._load_model(candidate_model, path, selected_device)
                self.device = selected_device
                if errors:
                    self.device_fallback_reason = str(errors[-1][1])
                break
            except Exception as exc:
                errors.append((candidate, exc))
        else:
            if len(errors) == 1:
                raise InpaintError(f"无法加载超分模型 {asset.filename}：{errors[0][1]}") from errors[0][1]
            details = "；".join(f"{candidate.upper()}：{error}" for candidate, error in errors)
            raise InpaintError(
                f"无法加载超分模型 {asset.filename}（已依次尝试 CUDA、CPU）：{details}"
            ) from errors[-1][1]

    def _load_model(self, model, path: Path, device: str):
        """在 CPU 加载权重后再迁移设备，避免 checkpoint 绑定原始设备。"""
        checkpoint = self.torch.load(str(path), map_location="cpu", weights_only=True)
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        model.load_state_dict(state, strict=True)
        loaded = model.to(device).eval()
        use_half = device == "cuda"
        if use_half:
            loaded = loaded.half()
        return loaded, use_half

    def _asset_and_model(self):
        if self.mode is EnhanceMode.REAL_ESRGAN_ANIME:
            return ANIME_ASSET, _build_rrdb_net(num_block=6)
        return GENERAL_ASSET, _build_srvgg_net()

    def _infer(self, image: np.ndarray):
        torch = self.torch
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        if self.use_half:
            tensor = tensor.half()
        with torch.inference_mode():
            return self.model(tensor)

    def _infer_tiled(self, image: np.ndarray):
        """对大图执行带重叠边缘的 4×分块推理，减少单次显存占用。"""
        torch = self.torch
        height, width = image.shape[:2]
        if not self.tile_size or max(height, width) <= self.tile_size:
            return self._infer(image)
        channels = 3
        dtype = torch.float16 if self.use_half else torch.float32
        output = torch.zeros((1, channels, height * 4, width * 4), dtype=dtype, device=self.device)
        tiles_x, tiles_y = math.ceil(width / self.tile_size), math.ceil(height / self.tile_size)
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                left, top = tile_x * self.tile_size, tile_y * self.tile_size
                right, bottom = min(left + self.tile_size, width), min(top + self.tile_size, height)
                pad_left, pad_top = max(left - self.tile_pad, 0), max(top - self.tile_pad, 0)
                pad_right, pad_bottom = min(right + self.tile_pad, width), min(bottom + self.tile_pad, height)
                tile_output = self._infer(image[pad_top:pad_bottom, pad_left:pad_right])
                crop_left, crop_top = (left - pad_left) * 4, (top - pad_top) * 4
                crop_right, crop_bottom = crop_left + (right - left) * 4, crop_top + (bottom - top) * 4
                output[:, :, top * 4:bottom * 4, left * 4:right * 4] = tile_output[:, :, crop_top:crop_bottom, crop_left:crop_right]
        return output

    def enhance(self, image_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """输入/输出为 BGR uint8；模型原生 4×，其他倍率以 Lanczos 缩放。

        所有张量后处理保持在 ``inference_mode`` 内，避免对 inference tensor
        进行非法原地修改；这对视频逐帧处理尤为重要。
        """
        if not 1.0 <= float(scale) <= 4.0:
            raise InpaintError("增强倍率必须在 1× 到 4×之间。")
        if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
            raise InpaintError("画质增强只接受三通道彩色图像。")
        input_height, input_width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        try:
            # `_infer()` 产生的结果是 inference tensor。PyTorch 不允许在
            # inference_mode 之外对它执行 clamp_、copy_ 等原地更新；视频逐帧
            # 场景会稳定触发该限制。将整段张量后处理留在同一上下文，并使用
            # 非原地 clamp，既避免错误也不修改模型输出本身。
            with self.torch.inference_mode():
                output = self._infer_tiled(rgb)
                result_rgb = (
                    output.squeeze(0)
                    .float()
                    .cpu()
                    .clamp(0, 1)
                    .numpy()
                    .transpose(1, 2, 0)
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                raise InpaintError("超分显存不足。请降低分块大小、缩小图像或切换 CPU。") from exc
            raise InpaintError(f"Real-ESRGAN 推理失败：{exc}") from exc
        result_bgr = cv2.cvtColor((result_rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        if float(scale) != self.native_scale:
            result_bgr = cv2.resize(
                result_bgr,
                (round(input_width * float(scale)), round(input_height * float(scale))),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return result_bgr


def create_enhancer(mode: EnhanceMode | str, *, device: str = "auto", model_dir: str | Path | None = None, tile_size: int = 512) -> RealESRGANEnhancer:
    return RealESRGANEnhancer(mode, device=device, model_dir=model_dir, tile_size=tile_size)
