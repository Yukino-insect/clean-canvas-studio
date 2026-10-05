"""耗时 AI 集成测试；默认不下载权重也不执行。"""

from __future__ import annotations

import os
import unittest

import numpy as np

from backend.enhance.realesrgan import EnhanceMode, create_enhancer
from backend.inpaint.engine import InpaintMode, create_inpaint_engine
from backend.inpaint.mask import regions_to_mask


@unittest.skipUnless(
    os.environ.get("VSR_RUN_AI_INTEGRATION") == "1",
    "set VSR_RUN_AI_INTEGRATION=1 only after preparing local AI model weights",
)
class OptionalAIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = os.environ.get("VSR_MODEL_DIR")
        if not cls.model_dir:
            raise unittest.SkipTest("VSR_MODEL_DIR must point to the prepared model cache")

    def test_anime_lama_repairs_a_masked_region(self):
        image = np.full((64, 96, 3), (90, 120, 160), dtype=np.uint8)
        image[20:40, 30:60] = 255
        mask = regions_to_mask(image.shape, [(30, 60, 20, 40)], expand_pixels=2)
        result = create_inpaint_engine(InpaintMode.ANIME_LAMA, device="cpu", model_dir=self.model_dir).inpaint(image, mask)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.array_equal(result[mask == 0], image[mask == 0]))

    def test_realesrgan_anime_changes_output_resolution(self):
        image = np.full((32, 48, 3), 128, dtype=np.uint8)
        result = create_enhancer(EnhanceMode.REAL_ESRGAN_ANIME, device="cpu", model_dir=self.model_dir, tile_size=0).enhance(image, 2)
        self.assertEqual(result.shape[:2], (64, 96))


if __name__ == "__main__":
    unittest.main()
