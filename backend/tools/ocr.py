def get_coordinates(dt_box):
    """兼容旧调用：从 OCR 多边形得到其**外接**矩形。

    新处理链路应直接使用 ``backend.masking.OcrPolygon``，不要调用本函数来
    生成精确掩膜。历史实现使用内接矩形，会在倾斜文字上丢失笔画并且语义不清。
    """
    coordinate_list = []
    if isinstance(dt_box, list):
        for points in dt_box:
            points = list(points)
            if len(points) < 3:
                continue
            xs = [int(point[0]) for point in points]
            ys = [int(point[1]) for point in points]
            coordinate_list.append((min(xs), max(xs), min(ys), max(ys)))
    return coordinate_list
