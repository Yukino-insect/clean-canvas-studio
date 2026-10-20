import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="本地图片/视频文字、符号与水印去除工具"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="输入图片或视频路径"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None, help="输出路径（可选）"
    )
    parser.add_argument(
        "--ocr-region-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="OCR 候选区域，可重复指定：YMIN YMAX XMIN XMAX"
    )
    parser.add_argument(
        "--manual-area", "-m", action="append", nargs=4, type=int, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="直接修复区域，可重复指定；适用于 OCR 漏检的文字/符号"
    )
    parser.add_argument(
        "--mode", choices=("opencv", "lama", "anime_lama", "manga"), default="opencv",
        help="修复模式（默认：opencv）"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="AI 模式运算设备")
    parser.add_argument("--model-dir", default=None, help="AI 模型缓存目录（可选）")
    parser.add_argument("--dynamic-ocr", action="store_true", help="视频中动态检测字幕位置")
    parser.add_argument(
        "--ocr-strategy", choices=("sampled_ocr", "every_frame_ocr"), default="sampled_ocr",
        help="动态 OCR 的检测频率（默认：采样检测）",
    )
    args = parser.parse_args()
    args.ocr_region_coords = args.ocr_region_coords or []
    args.manual_area = args.manual_area or []
    return args
