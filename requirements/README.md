# 依赖 profile 与锁文件

此目录将平台无关依赖与 PyTorch、PaddlePaddle 二进制构建分开管理。CPU 与 CUDA wheel 是不同发行物，不能用一条模糊的版本范围同时承诺两者。尤其是 PaddlePaddle：同一环境只能安装 CPU 或 GPU profile 之一。

## 当前 profile

| 文件 | 目标 | 用途 |
| --- | --- | --- |
| `base.in` / `base.lock` | Windows x86_64 + CPython 3.12 | Qt、OpenCV、PaddleOCR 等不含 PyTorch/PaddlePaddle 二进制运行时的基础依赖。 |
| `paddle-cpu.in` / `paddle-cpu.lock` | Windows x86_64 + CPython 3.12 | PaddleOCR 的 CPU PaddlePaddle profile。 |
| `paddle-gpu-cu118.in` / `paddle-gpu-cu118.lock` | Windows x86_64 + CPython 3.12 + NVIDIA CUDA 11.8 runtime | PaddleOCR 的 NVIDIA GPU PaddlePaddle profile。OCR 会优先尝试 GPU；失败时自动回退 CPU。 |
| `torch-cpu.in` / `torch-cpu.lock` | Windows x86_64 + CPython 3.12 | 锁定的 CPU PyTorch wheel 及其依赖。 |
| `torch-cu126.in` / `torch-cu126.lock` | Windows x86_64 + CPython 3.12 + NVIDIA CUDA 12.6 | 锁定的 `torch ... +cu126` wheel 及其依赖。 |
| `base-linux.lock` | Linux x86_64 + CPython 3.12 | 面向 Linux 的基础运行时 lock；Docker CPU 镜像使用它。 |
| `paddle-cpu-linux.lock` / `paddle-gpu-cu118-linux.lock` | Linux x86_64 + CPython 3.12 | 上述 Paddle CPU/GPU profile 的 Linux lock。 |
| `torch-cpu-linux.lock` | Linux x86_64 + CPython 3.12 | 锁定的 Linux CPU PyTorch wheel 及其依赖。 |
| `torch-cu126-linux.lock` | Linux x86_64 + CPython 3.12 + NVIDIA CUDA 12.6 | 锁定的 Linux `torch ... +cu126` wheel 及其依赖。 |

不要直接将这些 Windows lock 用于 Linux、macOS、不同 Python 版本或其他 CUDA 构建。应在目标环境解析新的 lock，并将其按平台/后端命名后提交。

## 安装

从项目根目录运行：

```powershell
.\scripts\install-runtime.ps1 -TorchBackend cpu -PaddleBackend cpu -Python .\.venv\Scripts\python.exe
.\scripts\install-runtime.ps1 -TorchBackend cu126 -PaddleBackend cu118 -Python .\.venv\Scripts\python.exe
```

Git Bash on Windows 或 Linux x86_64 + CPython 3.12 可使用等价的 Bash 脚本：

```bash
bash scripts/install-runtime.sh --torch-backend cpu --paddle-backend cpu --python .venv/Scripts/python.exe
bash scripts/install-runtime.sh --torch-backend cu126 --paddle-backend cu118 --python .venv/Scripts/python.exe
```

The Bash script selects the matching Windows/Linux lock by the selected interpreter's platform, architecture, and Python version. It intentionally refuses macOS, non-x86_64, and non-3.12 environments because no committed binary profile exists for them.

脚本先安装基础依赖，再选择一个 Paddle profile，最后安装 Torch profile。切换 Paddle profile 时会先移除旧的 CPU/GPU wheel，避免同名 `paddle` 导入包冲突；若 GPU profile 安装失败，脚本自动安装锁定的 CPU profile。GPU Paddle wheel 额外查询 Paddle 官方 CUDA 11.8 索引；Torch 则查询 PyTorch 官方 wheel 索引。

## 刷新锁文件

安装 [uv](https://docs.astral.sh/uv/) 后，在 Windows + CPython 3.12 环境执行：

```powershell
uv pip compile requirements/base.in --python .\.venv\Scripts\python.exe --python-platform windows --generate-hashes --output-file requirements/base.lock
uv pip compile requirements/torch-cpu.in --python .\.venv\Scripts\python.exe --python-platform windows --torch-backend cpu --generate-hashes --output-file requirements/torch-cpu.lock
uv pip compile requirements/torch-cu126.in --python .\.venv\Scripts\python.exe --python-platform windows --torch-backend cu126 --generate-hashes --output-file requirements/torch-cu126.lock
uv pip compile requirements/paddle-cpu.in --constraint requirements/base.lock --python .\.venv\Scripts\python.exe --python-platform windows --generate-hashes --output-file requirements/paddle-cpu.lock
uv pip compile requirements/paddle-gpu-cu118.in --constraint requirements/base.lock --index-strategy unsafe-best-match --python .\.venv\Scripts\python.exe --python-platform windows --generate-hashes --output-file requirements/paddle-gpu-cu118.lock
```

Linux x86_64 profiles are generated separately:

```bash
uv pip compile requirements/base.in --python .venv/bin/python --python-platform linux --generate-hashes --output-file requirements/base-linux.lock
uv pip compile requirements/torch-cpu.in --python .venv/bin/python --python-platform linux --torch-backend cpu --generate-hashes --output-file requirements/torch-cpu-linux.lock
uv pip compile requirements/torch-cu126.in --python .venv/bin/python --python-platform linux --torch-backend cu126 --generate-hashes --output-file requirements/torch-cu126-linux.lock
uv pip compile requirements/paddle-cpu.in --constraint requirements/base-linux.lock --python .venv/bin/python --python-platform linux --generate-hashes --output-file requirements/paddle-cpu-linux.lock
uv pip compile requirements/paddle-gpu-cu118.in --constraint requirements/base-linux.lock --index-strategy unsafe-best-match --python .venv/bin/python --python-platform linux --generate-hashes --output-file requirements/paddle-gpu-cu118-linux.lock
```

刷新 lock 后，应在新的空虚拟环境内分别安装 CPU 和 CUDA profile；Paddle GPU profile 应验证 `paddle.is_compiled_with_cuda()` 与设备数量，Torch CUDA profile 还必须验证 `torch.version.cuda` 非空且 `torch.cuda.is_available()` 为 `True`。GPU lock 使用 `unsafe-best-match` 是为了让官方 Paddle 索引的 wheel 与 PyPI 上同版本的公共依赖一同受基础锁约束；它只用于受控的锁文件生成，不应随意替换为未知第三方索引。
