"""字幕检测模型的单一配置入口。"""

import os

from backend.config import BASE_DIR

class ModelConfig:
    """只保留准确度优先的 PP-OCRv5 Server 文本检测模型。"""

    DET_MODEL_DIR = os.path.join(BASE_DIR, "models", "V5", "ch_det")
    DET_MODEL_NAME = "PP-OCRv5_server_det"
