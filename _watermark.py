from PIL import Image, ImageDraw
import torch
import random
import numpy as np
import pandas as pd
from simple_lama_inpainting import SimpleLama
import os
import shutil
from _image import load_image
from _model import models_path

def detect_watermark_from_img_result(img, res, err_ratio=0.05, threshold=0.1):
    res: pd.DataFrame = res.sort_values(by='confidence', ascending=False)
    img_np = np.array(img)
    # 以最高置信度为主，假如有其他大小相当的检测框则合并
    width, height = None, None
    for i, box in res.iterrows():
        w, h = box['xmax'] - box['xmin'], box['ymax'] - box['ymin']
        if width is None:  # first run
            width, height = w, h
            continue
        if w > width * (1 + err_ratio) or w < width * (1 - err_ratio) \
                or h > height * (1 + err_ratio) or h < height * (1 - err_ratio):
            res.loc[i, 'class'] = 1
        if box['confidence'] < threshold:
            res.loc[i, 'class'] = 1
    res_less = res
    #res_less = res.drop(index=res[res['class'] == 1].index)
    boxes = [list(map(int, i[1:5])) for i in res_less.itertuples()]
    # 假如少于等于5个，直接返回，否则根据多幅图像提取水印
    if len(res)==0:
        return None, []
    if len(res) <= 5:
        # w1, h1, w2, h2 = boxes[0]
        w1, h1, w2, h2 = random.choice(boxes)
        return img_np[h1:h2, w1:w2], boxes
    else:
        # 把所有子图都resize到相同大小
        wms = []  # watermarks
        for w1, h1, w2, h2 in boxes:
            i = img_np[h1:h2, w1:w2]
            i = Image.fromarray(i).resize((int(width), int(height)))
            wms.append(np.array(i))
        return [list(map(int, i[1:5])) for i in res.itertuples()]

yolo_path = os.path.join(models_path,'yolov5')
yolo_best_path = os.path.join(models_path,'yolov5','best.pt')
yolo_model = torch.hub.load(yolo_path, 'custom', yolo_best_path, source='local')
yolo_model = yolo_model.cpu()
from PIL import Image, ImageDraw
import torch
import random
import numpy as np
import pandas as pd
from simple_lama_inpainting import SimpleLama
import os
from _image import load_image
from _model import models_path

def detect_watermark_from_img_result(img, res, err_ratio=0.05, threshold=0.1):
    res: pd.DataFrame = res.sort_values(by='confidence', ascending=False)
    img_np = np.array(img)
    # 以最高置信度为主，假如有其他大小相当的检测框则合并
    width, height = None, None
    for i, box in res.iterrows():
        w, h = box['xmax'] - box['xmin'], box['ymax'] - box['ymin']
        if width is None:  # first run
            width, height = w, h
            continue
        if w > width * (1 + err_ratio) or w < width * (1 - err_ratio) \
                or h > height * (1 + err_ratio) or h < height * (1 - err_ratio):
            res.loc[i, 'class'] = 1
        if box['confidence'] < threshold:
            res.loc[i, 'class'] = 1
    res_less = res
    #res_less = res.drop(index=res[res['class'] == 1].index)
    boxes = [list(map(int, i[1:5])) for i in res_less.itertuples()]
    # 假如少于等于5个，直接返回，否则根据多幅图像提取水印
    if len(res)==0:
        return None, []
    if len(res) <= 5:
        # w1, h1, w2, h2 = boxes[0]
        w1, h1, w2, h2 = random.choice(boxes)
        return img_np[h1:h2, w1:w2], boxes
    else:
        # 把所有子图都resize到相同大小
        wms = []  # watermarks
        for w1, h1, w2, h2 in boxes:
            i = img_np[h1:h2, w1:w2]
            i = Image.fromarray(i).resize((int(width), int(height)))
            wms.append(np.array(i))
        return [list(map(int, i[1:5])) for i in res.itertuples()]

yolo_path = os.path.join(models_path,'yolov5')
yolo_best_path = os.path.join(models_path,'yolov5','best.pt')
yolo_model = torch.hub.load(yolo_path, 'custom', yolo_best_path, source='local')
yolo_model = yolo_model.cpu()

def convert_coordinates_to_percentage(width, height, regions):
    converted_regions = []
    for region in regions:
        x1, y1, x2, y2 = region
        converted_region = [x1/width, y1/height, x2/width, y2/height]
        converted_regions.append(converted_region)
    return converted_regions

def get_watermark(image):
    imgs = [load_image(image=image,mode='RGB')]
    width, height = imgs[0].size
    results = yolo_model(imgs)
    results = results.pandas().xyxy
    if len(imgs) > 0 and len(results) > 0:
        _, box = detect_watermark_from_img_result(imgs[0], results[0])
    else:
        return []
    #result = ['.'.join(map(str, sub_list)) for sub_list in box]
    return convert_coordinates_to_percentage(width, height, box)

def romove_watermarks(input_image,output_image="inpainted.png",watermarks=[]):
    if watermarks==[]:
        watermarks = get_watermark(input_image)
    if len(watermarks) != 0:
        simple_lama = SimpleLama()
        image = load_image(image)
        width, height = image.size
        mask = Image.new('L', (width, height))
        draw = ImageDraw.Draw(mask)
        for watermark in watermarks:
            x1, y1, x2, y2 = watermark
            draw.rectangle([x1, y1, x2, y2], fill=255)

        result = simple_lama(image, mask)
        result.save(output_image)