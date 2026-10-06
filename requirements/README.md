# 依赖 profile 与锁文件

此目录将平台无关依赖与 PyTorch 二进制构建分开管理。原因很简单：CPU 和 CUDA 的 `torch` 不是同一个发行物，不能用一条模糊的 `torch>=...` 同时承诺两者。

## 当前 profile

| 文件 | 目标 | 用途 |
| --- | --- | --- |
| `base.in` / `base.lock` | Windows x86_64 + CPython 3.12 | Qt、OpenCV、PaddleOCR 等不含 PyTorch 的基础运行时依赖。 |
| `torch-cpu.in` / `torch-cpu.lock` | Windows x86_64 + CPython 3.12 | 锁定的 CPU PyTorch wheel 及其依赖。 |
| `torch-cu126.in` / `torch-cu126.lock` | Windows x86_64 + CPython 3.12 + NVIDIA CUDA 12.6 | 锁定的 `torch ... +cu126` wheel 及其依赖。 |
| `base-linux.lock` | Linux x86_64 + CPython 3.12 | 面向 Linux 的基础运行时 lock；Docker CPU 镜像使用它。 |
| `torch-cpu-linux.lock` | Linux x86_64 + CPython 3.12 | 锁定的 Linux CPU PyTorch wheel 及其依赖。 |
| `torch-cu126-linux.lock` | Linux x86_64 + CPython 3.12 + NVIDIA CUDA 12.6 | 锁定的 Linux `torch ... +cu126` wheel 及其依赖。 |

不要直接将这些 Windows lock 用于 Linux、macOS、不同 Python 版本或其他 CUDA 构建。应在目标环境解析新的 lock，并将其按平台/后端命名后提交。

## 安装

从项目根目录运行：

```powershell
.\scripts\install-runtime.ps1 -TorchBackend cpu -Python .\.venv\Scripts\python.exe
.\scripts\install-runtime.ps1 -TorchBackend cu126 -Python .\.venv\Scripts\python.exe
```

Git Bash on Windows 或 Linux x86_64 + CPython 3.12 可使用等价的 Bash 脚本：

```bash
bash scripts/install-runtime.sh --torch-backend cpu --python .venv/Scripts/python.exe
bash scripts/install-runtime.sh --torch-backend cu126 --python .venv/Scripts/python.exe
```

The Bash script selects the matching Windows/Linux lock by the selected interpreter's platform, architecture, and Python version. It intentionally refuses macOS, non-x86_64, and non-3.12 environments because no committed binary profile exists for them.

脚本分两次执行 pip：基础依赖来自 PyPI，Torch 则额外查询 PyTorch 官方 wheel 索引。这能让 `torch==...+cu126` 精确选中 CUDA wheel，同时不会错误地把普通项目依赖限制到 PyTorch wheel 索引。

## 刷新锁文件

安装 [uv](https://docs.astral.sh/uv/) 后，在 Windows + CPython 3.12 环境执行：

```powershell
uv pip compile requirements/base.in --python .\.venv\Scripts\python.exe --python-platform windows --generate-hashes --output-file requirements/base.lock
uv pip compile requirements/torch-cpu.in --python .\.venv\Scripts\python.exe --python-platform windows --torch-backend cpu --generate-hashes --output-file requirements/torch-cpu.lock
uv pip compile requirements/torch-cu126.in --python .\.venv\Scripts\python.exe --python-platform windows --torch-backend cu126 --generate-hashes --output-file requirements/torch-cu126.lock
```

Linux x86_64 profiles are generated separately:

```bash
uv pip compile requirements/base.in --python .venv/bin/python --python-platform linux --generate-hashes --output-file requirements/base-linux.lock
uv pip compile requirements/torch-cpu.in --python .venv/bin/python --python-platform linux --torch-backend cpu --generate-hashes --output-file requirements/torch-cpu-linux.lock
uv pip compile requirements/torch-cu126.in --python .venv/bin/python --python-platform linux --torch-backend cu126 --generate-hashes --output-file requirements/torch-cu126-linux.lock
```

刷新 lock 后，应在新的空虚拟环境内分别安装 CPU 和 CUDA profile；CUDA profile 还必须验证 `torch.version.cuda` 非空且 `torch.cuda.is_available()` 为 `True`。
