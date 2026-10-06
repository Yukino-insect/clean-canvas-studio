FROM python:3.12

WORKDIR /vsr
COPY requirements ./requirements

# CPU 基线镜像：安装经哈希锁定的基础依赖，但刻意不安装 PyTorch。
# 若要运行 LaMa、Anime-LaMa、Manga 或 Real-ESRGAN，请在主机上使用
# Windows CPU/CUDA profile，或另行维护与 NVIDIA Container Toolkit 匹配的
# GPU 专用镜像；不要把本 Dockerfile 误解成 CUDA 运行时镜像。
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libegl1 libxkbcommon0 libdbus-1-3 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --require-hashes -r requirements/base-linux.lock

COPY . .

CMD ["python", "gui.py"]
