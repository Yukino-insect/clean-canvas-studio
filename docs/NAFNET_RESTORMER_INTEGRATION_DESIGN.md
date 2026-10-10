# NAFNet 与 Restormer 全图复原接入设计

> **文档状态：设计提案，尚未实现。**
>
> 本文约束将来把 NAFNet 和 Restormer 接入 CleanCanvas Studio 的方式，包含模型职责、输入输出契约、预处理与后处理、拟新增代码、任务快照、视频边界、依赖/权重管理与验证标准。它不是当前功能清单；当前仓库实际已接入的模型见 [当前局部修复与全图增强：技术实现参考](REPAIR_AND_ENHANCEMENT_TECHNICAL_REFERENCE.md)。

## 1. 目标与非目标

### 1.1 目标

新增“全图清晰复原”能力，处理整张图片或视频帧中已经存在、但被以下退化破坏的内容：

- 运动模糊；
- 失焦/柔焦；
- 传感器或暗部噪声；
- JPEG/视频压缩造成的块状伪影、振铃和脏边。

该能力应与现有 Real-ESRGAN 协同：先减轻退化、再按需放大，而非把模糊和压缩块一起放大。

### 1.2 非目标

以下事情不属于 NAFNet / Restormer 的职责，不能以“清晰复原”名义承诺：

- 去除文字、水印、符号或 logo；这需要局部修复页中的二值 mask 与 Telea/LaMa/Anime-LaMa/Manga；
- 修复第一次去水印后不自然的某个小补丁；应在“局部修复”中对该补丁和接缝再次画 mask；
- 恢复已经永久丢失的真实历史像素、文字或身份信息；
- 保证视频帧间完全一致；第一阶段的视频支持仍是逐帧复原；
- 自动准确判断所有图片到底属于运动模糊、失焦、噪声还是压缩伪影。

## 2. 产品位置与处理顺序

新能力应放入拟更名后的“**全图复原与增强**”页面，不新建一级导航。该页面和“局部修复”的语义边界如下：

```text
局部修复：用户指定哪里需要被擦除或重绘
    整图/整帧 + 二值 mask → Telea / LaMa / Anime-LaMa / Manga

全图复原与增强：程序改善整张图片或整段视频的整体退化
    整图/整帧 → NAFNet / Restormer → （可选）Real-ESRGAN
```

页面仅提供三种处理目标：

| 处理目标 | 实际管线 | 默认输出尺寸 | 适用场景 |
| --- | --- | --- | --- |
| 清晰复原 | `NAFNet` 或 `Restormer` | 与输入一致 | 运动模糊、失焦、噪声、压缩伪影 |
| 放大增强 | 现有 `Real-ESRGAN` | 1×–4× | 已经较干净，主要需要更大分辨率/视觉细节 |
| 清晰复原后放大 | `NAFNet/Restormer → Real-ESRGAN` | 由最终倍率决定 | 低清且同时有模糊、噪声或压缩损伤 |

组合顺序固定为“复原 → 放大”。不在普通界面提供自由拖拽顺序；先放大退化会放大噪声、压缩块和模糊边缘，通常增加后续恢复难度。

## 3. 模型调研与接入范围

### 3.1 NAFNet

NAFNet（Nonlinear Activation Free Network）是面向低层图像恢复的高效网络。官方项目提供了去模糊和去噪等配置/预训练权重；其单图演示分别以 `NAFNet-width64.yml` 处理噪声或模糊图片。官方说明其实现建立在 BasicSR/HINet 工具链上。[NAFNet 官方仓库](https://github.com/megvii-research/NAFNet)

建议第一批支持的**明确权重任务**：

| 内部模式 ID（建议） | 任务 | 外部模型/配置选择 | 用户可见名称 | 是否改变尺寸 |
| --- | --- | --- | --- | --- |
| `nafnet_deblur` | 单图运动模糊 | NAFNet 去模糊权重 | NAFNet：运动模糊 | 否 |
| `nafnet_denoise` | 真实/相机噪声 | NAFNet 去噪权重 | NAFNet：图像噪声 | 否 |
| `nafnet_jpeg` | 模糊加压缩伪影 | 仅在有经过验证的相应权重时启用 | NAFNet：压缩伪影（实验） | 否 |

不应把“NAFNet”作为一个没有任务语义的单选项。权重是按退化任务训练的；拿去噪权重处理运动模糊、或拿运动去模糊权重处理强 JPEG 块效应，不应被宣传为稳定可行。

### 3.2 Restormer

Restormer 是高分辨率图像恢复 Transformer。官方实现分别提供单图运动去模糊、失焦去模糊、真实/高斯去噪和去雨等预训练模型与推理入口。[Restormer 官方仓库](https://github.com/swz30/Restormer)

建议第一批支持的**明确权重任务**：

| 内部模式 ID（建议） | 任务 | 用户可见名称 | 是否改变尺寸 | 说明 |
| --- | --- | --- | --- | --- |
| `restormer_motion_deblur` | 单图运动模糊 | Restormer：运动模糊 | 否 | 面对拉丝/抖动类模糊的可选高质量后端 |
| `restormer_defocus_deblur` | 单图失焦模糊 | Restormer：失焦/柔焦 | 否 | 不等同于运动模糊，应显式选择 |
| `restormer_real_denoise` | 真实图像去噪 | Restormer：真实噪声 | 否 | 适合相机/暗光噪声，不应承诺去除水印残片 |

Restormer 在模型结构上适合较大分辨率的像素交互，但并不意味着无视显存。必须让引擎在 CUDA OOM 时报告清晰错误、允许降低分块大小或回退 CPU；不能静默把图缩小后再放大，因为那会改变“保持原尺寸复原”的承诺。

### 3.3 为什么第一期不接入生成式恢复模型

DiffBIR、SUPIR 一类扩散式盲恢复在严重退化和缺失纹理上可能更具视觉冲击力，但它们能生成并不存在的细节；此外依赖、显存、许可证和可复现性成本显著更高。它们应在 NAFNet/Restormer 的保真复原路径稳定后，作为独立的“实验性生成式复原”评估，不得伪装成无损还原。

## 4. 对现有项目的影响边界

| 现有模块 | 是否直接修改 | 设计要求 |
| --- | --- | --- |
| `backend/masking.py` | 否 | 全图复原不读取 mask，也不改变局部修复语义 |
| `backend/inpaint/` | 否 | 不把 NAFNet/Restormer 塞进 `InpaintMode`；二者不是 mask inpainting 模型 |
| `backend/enhance/realesrgan.py` | 尽量不改 | Real-ESRGAN 保持单一“放大增强”职责；仅由新编排层按需调用 |
| `backend/enhance/media_enhancer.py` | 是 | 从“只增强”扩展为可选的复原、放大或两阶段管线 |
| 任务组件与队列 | 是 | 每个待处理任务冻结复原阶段和增强阶段的完整快照 |
| `backend/config.py` | 是 | 增加枚举配置与安全默认值，不能复用含义不同的 `enhanceMode` 字段 |
| 高级设置 / 模型缓存 | 是 | 复原权重纳入状态、下载校验与可控删除范围 |

## 5. 建议的本地代码结构

为避免让 `realesrgan.py` 演变成所有模型的容器，建议新建独立包：

```text
backend/restoration/
├── __init__.py
├── engine.py             # 公共协议、枚举、资产登记、工厂与设备回退
├── nafnet.py             # NAFNet 网络构建、权重加载与推理适配器
├── restormer.py          # Restormer 网络构建、权重加载与推理适配器
├── pipeline.py           # 复原 → 可选 Real-ESRGAN 的阶段编排
└── tiling.py             # 经过验证的重叠分块、拼接和边界裁剪（如确有必要）
```

### 5.1 公共协议与枚举

`backend/restoration/engine.py` 应定义独立于局部修复的接口：

```python
class RestorationMode(str, Enum):
    NAFNET_DEBLUR = "nafnet_deblur"
    NAFNET_DENOISE = "nafnet_denoise"
    RESTORMER_MOTION_DEBLUR = "restormer_motion_deblur"
    RESTORMER_DEFOCUS_DEBLUR = "restormer_defocus_deblur"
    RESTORMER_REAL_DENOISE = "restormer_real_denoise"


class RestorationEngine(Protocol):
    """整图复原接口：不接受 mask，输入输出保持 BGR uint8。"""

    def restore(self, image_bgr: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RestorationModelSpec:
    mode: RestorationMode
    title: str
    task: str
    asset: ModelAsset
    spatial_multiple: int
    supports_video: bool = False
```

`spatial_multiple` 不能写成全局常量。不同网络、不同层数或权重配置对高宽可整除性的要求不同；适配器须基于实际网络配置设置并用测试验证。预期初始值可从官方配置中读取，例如常见 NAFNet 编码器层数对应 16 的倍数、常见 Restormer 多尺度下采样对应 8 的倍数，但最终以纳入项目的权重配置和集成测试为准。

### 5.2 资产登记与模型缓存

复用现有 [ModelStore](../backend/inpaint/engine.py) 的原则：

1. 每个权重使用文件名、官方 URL、哈希和可读模型标题进行登记；
2. 下载先写临时文件，校验后原子替换；
3. 高级设置只清理白名单中已登记的权重，绝不递归删除用户选定的缓存目录；
4. 把权重来源、许可、哈希、模型结构参数和适用任务写入本仓库的第三方说明；
5. 发布二进制前重新核验**代码许可、权重许可和下载来源**，不能仅根据论文或 GitHub 仓库名称推断可再分发性。

为避免破坏当前哈希机制，第一期可沿用 `ModelAsset` 的 MD5 字段；后续可统一升级到 SHA-256，但这是横向安全改造，不应在 NAFNet/Restormer 接入时只改一半。

## 6. 统一的输入、预处理、推理与输出契约

### 6.1 处理对象：完整图片/完整视频帧，不是局部框

NAFNet 与 Restormer 的输入应为完整的 BGR 图片或当前视频帧；它们不读取 `MaskPlan`、OCR 多边形、手工 ROI 或颜色筛选结果。输入输出的空间语义如下：

```text
H × W × 3 BGR uint8
    → BGR 转 RGB
    → float32 / 255，范围 [0, 1]
    → 为网络可整除尺寸作反射填充
    → [1, 3, padded_H, padded_W] PyTorch tensor
    → NAFNet 或 Restormer 推理
    → 裁回原始 H × W
    → 限制到 [0, 1]、RGB 转 BGR、uint8
    → H × W × 3 BGR uint8
```

复原阶段默认必须保持 `H × W` 不变。只有其后显式启用 Real-ESRGAN 时，最终尺寸才按用户选择的 1×–4×变化。

### 6.2 建议的共享预后处理函数

以下是设计级伪代码；实际实现需用单元测试固定 dtype、颜色通道、尺寸和 mask 无关性：

```python
def restore_full_frame(image_bgr: np.ndarray, model, *, multiple: int, device: str) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise RestorationError("全图复原只接受三通道彩色图像。")

    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    padded, padding = pad_reflect_to_multiple(normalized, multiple)
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        prediction = model(tensor)

    restored_rgb = prediction[0].float().detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    restored_rgb = crop_padding(restored_rgb, padding, height, width)
    return cv2.cvtColor((restored_rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
```

这里的约束是：

- 必须使用反射/镜像类填充，而不是黑边常数填充，以避免边缘出现深色条带；
- 必须保存并精确裁回填充范围，不能按比例 resize 回原尺寸；
- 输出必须显式 `clamp`，防止模型浮点预测稍越出合法范围后在 `uint8` 转换时回绕；
- `torch.inference_mode()` 覆盖推理和张量后处理，避免当前 Real-ESRGAN 曾遇到的 inference tensor 原地修改问题；
- 不要让模型静默改变输入的 alpha；第一期只接收三通道 BGR，带 alpha 的文件应由统一媒体读写层先明确处理。

### 6.3 CUDA、CPU、半精度与 OOM

设备语义必须与现有修复/增强一致：

| 用户选择 | 实际行为 |
| --- | --- |
| `auto` | 完整尝试 CUDA 初始化和权重加载；失败后重新创建 CPU 模型，并记录失败原因 |
| `cuda` | 只尝试 CUDA；失败时明确报错，不隐式切 CPU |
| `cpu` | 只在 CPU 推理 |

CUDA 可在经过权重验证后启用 FP16；CPU 始终用 FP32。出现 OOM 时，第一期应报出“可降低复原分块大小、缩小图像或改用 CPU”的可读错误。不能捕获 OOM 后偷偷降低图像分辨率，因为这会造成用户不可见的质量和尺寸变化。

## 7. 分块推理策略

Real-ESRGAN 已有 512 px tile + 10 px 重叠的专用拼接实现，但不能不加验证地复制给 NAFNet 或 Restormer。不同网络的感受野、边界行为和输出尺寸语义不同；错误分块会留下棋盘接缝或亮度跳变。

推荐实施顺序：

1. **第一期**：只实现整图推理与清晰 OOM 回退；以可复现质量为优先。
2. **第二期**：仅在固定回归集证明整图/分块差异可接受后，增加模型专属的重叠分块。分块输出应只保留中心无边界区域，或使用经过验证的加权融合；不能简单覆盖拼接。
3. 分块大小、重叠像素只出现在“高级模型选项”，默认值应由模型规格决定；不在普通页面向用户暴露。

## 8. 与现有 Real-ESRGAN 的组合管线

建议新增 `RestorationPipeline`，而不是让 UI 分别调两个互不认识的类：

```python
@dataclass(frozen=True)
class FullFramePlan:
    restoration_mode: RestorationMode | None  # None = 仅放大增强
    enhance_mode: EnhanceMode | None          # None = 仅清晰复原
    enhance_scale: float = 1.0
    device: str = "auto"
    model_dir: str | None = None


def process_frame(frame_bgr: np.ndarray, plan: FullFramePlan) -> np.ndarray:
    restored = frame_bgr
    if plan.restoration_mode is not None:
        restored = create_restoration_engine(plan.restoration_mode, ...).restore(restored)
    if plan.enhance_mode is not None:
        restored = create_enhancer(plan.enhance_mode, ...).enhance(restored, plan.enhance_scale)
    return restored
```

三种 UI 目标到计划的映射应固定：

| UI 目标 | `restoration_mode` | `enhance_mode` | `enhance_scale` |
| --- | --- | --- | --- |
| 清晰复原 | 非空 | `None` | `1.0` |
| 放大增强 | `None` | Real-ESRGAN Anime / General | 1×–4× |
| 清晰复原后放大 | 非空 | Real-ESRGAN Anime / General | 1×–4× |

现有局部修复不应被偷偷插进此管线。用户若要“去水印 → 二次局部修复 → 全图复原 → 放大”，每一步都必须产生独立可检查的输出并作为下一任务的输入，这比后台自动串联更可追溯。

## 9. 图片与视频任务编排

### 9.1 图片

图片任务可直接复用现有读取与 PNG 输出原则：

```text
read_image → FullFramePlan.process_frame → cv2.imencode(output suffix) → 独立输出文件
```

若仅做清晰复原，建议输出名为：

```text
原文件_restored_<mode>.png
```

若包含放大，建议输出名为：

```text
原文件_restored_<mode>_enhanced_x2.png
```

### 9.2 视频

第一期视频支持应明确标为“逐帧复原（实验）”：

```text
OpenCV 解码 → FramePrefetcher → process_frame(每一帧) → FFmpegVideoWriter → 尝试回封装原音频
```

必须沿用当前视频输出约束：

- 保留输入 FPS；
- 输出宽高为偶数以适配 `yuv420p`；
- 单帧预览按固定间隔节流；
- 任务可在帧间安全取消；
- 音频回封装失败时，保留成功编码的无音频画面；
- 清晰复原、Real-ESRGAN 以及二者组合都**没有时序模型**，可能闪烁。

在没有固定视频回归集之前，不提供“自动批量处理全片”的强推荐文案；界面应提示用户先测 5–10 秒有运动、切镜和字幕变化的片段。

## 10. 任务快照、配置与 UI

### 10.1 配置项建议

`backend/config.py` 建议新增独立命名空间，而不是挪用 `Enhance.Mode`：

```text
Restoration.Pipeline = restore_only | enhance_only | restore_then_enhance
Restoration.Mode = nafnet_deblur | nafnet_denoise | restormer_motion_deblur | ...
Restoration.TileSize = 0 或经验证的默认值
Restoration.TilePad = 经验证的默认值
```

`AI.Device` 和 `AI.ModelDirectory` 继续共用，避免同一 GPU 被页面各自记录为不同设置。待处理任务应同步新的全图计划；任务进入“处理中”后冻结，不受之后 UI 修改影响。

### 10.2 任务数据结构建议

不要只给现有 `EnhanceTask` 增加大量可选字段并让含义变模糊。可将其重命名为 `FullFrameTask`，或新增兼容的执行快照：

```python
execution_options = {
    "pipeline": "restore_then_enhance",
    "restoration_mode": "nafnet_deblur",
    "enhance_mode": "realesrgan_anime",
    "enhance_scale": 2.0,
    "device": "auto",
    "model_dir": "...",
    "tile_size": 0,
}
```

日志必须记录：请求设备、实际设备、复原模型和权重、可选增强模型和倍率、输入/输出分辨率、视频帧数、总耗时，以及自动设备回退原因。

### 10.3 UI 只暴露任务语义

普通页面应让用户选：

```text
处理目标：清晰复原 / 放大增强 / 清晰复原后放大
复原问题：运动模糊 / 失焦与柔焦 / 噪声与压缩伪影
素材类型：二次元插画与动画 / 照片与通用图片（仅放大阶段）
输出倍率：1×–4×（仅放大阶段）
```

模型名、tile、权重文件、FP16、缓存目录和实验性参数应折叠到高级区。第一期不提供“自动判断退化类型”；没有足够可靠的分类证据时，错误的自动选择比要求用户选择“看起来像哪种问题”更糟。

## 11. 依赖、许可证与安全边界

1. 不直接在主 UI 线程导入或下载新模型；沿用现有惰性初始化和工作线程/子进程任务机制。
2. 不直接把上游 `setup.py develop` 的全部依赖塞入基础 requirements；先创建独立、锁定的复原 profile，例如 `requirements/restoration.in`，再验证其与当前 PyTorch、Python 版本和 Windows 环境兼容。
3. 若需复用上游网络定义，应保留来源注释、许可证文本、NOTICE 和本项目修改说明；模型权重的再分发许可必须逐项核验。
4. NAFNet 官方实现依赖 BasicSR/HINet 工具链；Restormer 官方实现同样基于既有恢复工具链。应优先抽取最小、可审计的推理依赖或以独立运行环境调用，避免无约束地引入训练依赖。
5. 所有模型下载必须使用可验证的官方来源与哈希；下载失败、哈希不符、CUDA OOM、CPU 不支持和权重结构不匹配都应给出可读错误，不得生成空白/半成品输出。

## 12. 验收与测试矩阵

### 12.1 单元测试（不下载权重）

- `RestorationMode`、计划序列化、旧任务兼容与 UI 映射；
- `auto` 真正按 CUDA → CPU 顺序重建引擎，显式 CPU/CUDA 不自动越权切换；
- BGR/RGB 转换、0–1 归一化、反射填充与精确裁回；
- 输出尺寸与 dtype 保持为原尺寸 `uint8 BGR`；
- 组合管线顺序固定为“复原 → 增强”；
- 任务日志包含阶段与实际设备；
- 视频奇数尺寸输出向下调整为偶数、音频回封装失败回退仍可交付。

### 12.2 可选真实权重集成测试

- 每个登记权重的哈希、结构加载和一张固定样本的烟雾推理；
- 输入为非整除尺寸，验证输出准确裁回原尺寸且没有黑边；
- CUDA 与 CPU 各至少一条可重复路径；
- 大图 OOM 时错误可读，且不会留下正式输出文件；
- 组合流程输出尺寸等于 `round(input × scale)`；
- 动漫、照片、噪声、运动模糊、失焦、JPEG 伪影分别建立固定视觉回归样本。

### 12.3 人工质量验收

每个模型/权重都须与“原图、仅 Real-ESRGAN、复原后 Real-ESRGAN”三组结果做 100% 像素对比，重点检查：

- 文字、线稿、细发丝是否粘连或被抹掉；
- 平坦色块是否出现涂抹感；
- JPEG 块效应是否减少而非被锐化；
- 皮肤、纹理、重复图案是否产生不真实细节；
- 视频的相邻帧是否闪烁、亮度漂移或出现分块接缝。

只有固定回归集和人工检查都通过的模式，才可以从“实验”提升为默认推荐。

## 13. 分阶段实施建议

| 阶段 | 范围 | 交付标准 |
| --- | --- | --- |
| A | 重命名页面语义、增加 `FullFramePlan` 和任务快照，但不改变现有 Real-ESRGAN 行为 | 旧增强任务完全兼容，队列/日志/预览无回归 |
| B | NAFNet 图片去模糊、去噪；完整缓存/设备/测试链路 | 整图输入输出契约、CPU/GPU 回退、非整除尺寸测试通过 |
| C | Restormer 图片任务（运动、失焦、真实去噪） | 每个权重明确任务语义，质量回归集通过 |
| D | “复原后放大”组合模式 | 阶段顺序、输出命名、快照、预览和取消正确 |
| E | 视频逐帧实验模式 | 短片段输出可解码、音频回封装与闪烁边界说明完整 |
| F | 经验证的模型专属 tile、生成式/时序模型评估 | 不出现拼接接缝，不误把实验能力设为默认 |

## 14. 结论

NAFNet 与 Restormer 应作为全图、原尺寸、任务特异的“清晰复原”后端接入；Real-ESRGAN 保持放大与视觉增强职责；Telea/LaMa/Anime-LaMa/Manga 继续负责用户用 mask 指定的局部重绘。三者服务的输入语义不同，不能塞进同一个模型下拉框，更不能让 Real-ESRGAN 替代对去水印错误补丁的二次局部修复。
