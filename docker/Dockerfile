FROM python:3.12

WORKDIR /vsr
COPY requirements.txt ./

# OpenCV/Qt 运行所需的系统库。项目的修复流程固定为 CPU OpenCV，
# 不再安装 CUDA、DirectML、PyTorch 或时序补全模型。
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libegl1 libxkbcommon0 libdbus-1-3 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "gui.py"]
