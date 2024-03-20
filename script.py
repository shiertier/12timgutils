import os
#os.environ['IMGUTILS_MODELS_PATH'] = "/gemini/pretrain"
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

from _image import *
from _model import *
from _hsv import get_hsv
from _phash import get_phash
from _bad import load_bad_model,get_bad_score
#from _watermark import get_watermark
from _ram import get_ram_tags, load_ram_model
from _detect import is_have_object_censors_hand
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
import json
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
#yolo_model = yolo_model.cpu()
yolo_model = yolo_model.to('cuda')
def convert_coordinates_to_percentage(width, height, regions):
    converted_regions = []
    for region in regions:
        x1, y1, x2, y2 = region
        converted_region = [round(x1/width,4), round(y1/height,4), round(x2/width,4), round(y2/height,4)]
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
## 分类

### 审美
def aesthetic_scores(image):
    #AESTHETIC_LABELS = ['masterpiece', 'best', 'great', 'good', 'normal', 'low', 'worst']
    AESTHETIC_LABELS = [6, 5, 4, 3, 2, 1, 0]
    scores = classify_predict_score(image,"aesthetic_448.onnx",AESTHETIC_LABELS,448)
    return scores

def get_aesthetic(image: ImageTyping):
    def get_highest_group_dict(scores):
        AESTHETIC_LABELS = [6, 5, 4, 3, 2, 1, 0]
        group_sums = []
        for i in range(len(AESTHETIC_LABELS) - 2):
            group_sum = sum(scores[label] for label in AESTHETIC_LABELS[i:i+3])
            group_sums.append(group_sum)
        max_index = group_sums.index(max(group_sums))
        highest_group_labels = AESTHETIC_LABELS[max_index:max_index+3]
        highest_group_dict = {label: scores[label] for label in highest_group_labels}
        return highest_group_dict
    def convert_score(highest_group_dict):
        if 6 in highest_group_dict and highest_group_dict[6] > 0.2:
            return 6
        for label, score in highest_group_dict.items():
            if score > 0.5:
                highest_label = label
                return highest_label
        max_score = 0
        max_label = None
        max_use = False
        for label, score in highest_group_dict.items():
            if score > max_score:
                max_score = score
                max_label = label
            if score < 0.1:
                max_use = True
        if max_use is True:
            return max_label
        mid_key = list(highest_group_dict.keys())[1]
        return mid_key
    scores = aesthetic_scores(image)
    highest_group_dict = get_highest_group_dict(scores)
    aesthetic = convert_score(highest_group_dict)

    return aesthetic, dict_2_list(scores) # 返回结果和置信度

def dict_2_list(_dict, reverse = True):
    score_list = list(_dict.values())
    if reverse:
        score_list.reverse()
    return score_list

### 身体 {2:'full-body', 1:'half-body', 0:'headshot'}
def get_bodypart(image):
    LABELS = [2,1,0]
    scores = classify_predict_score(image,"portrait_caformer_s36_v0.onnx",LABELS,384)

    return max(scores, key=scores.get), dict_2_list(scores)

### 完成度         2:"polished",1:"rough",0:"monochrome"
def get_completeness(image):
    LABELS = [2,1,0]
    scores = classify_predict_score(image,"completeness_mobilenetv3_v2.2_dist.onnx",LABELS,384)
    return max(scores, key=scores.get), dict_2_list(scores)

### 动画得分 0:动漫 1:真实
def get_anime_real(image):
    LABELS = [0,1]
    scores = classify_predict_score(image,"anime_real_mobilenetv3_v1.2_dist.onnx",LABELS,384)
    return max(scores, key=scores.get), dict_2_list(scores,False)

### nsfw safe,r15,r18
def get_nsfw(image):
    LABELS = [0,1,2]
    scores = classify_predict_score(image,"nsfw_mobilenetv3_v1_pruned_ls0.1.onnx",LABELS,384)
    return max(scores, key=scores.get), dict_2_list(scores,False)

# 汇总
def multiply_and_int_list(lst):
    return [int(num * 100) for num in lst]

def get_all(image: str):
    id = int(os.path.splitext(os.path.basename(image))[0])
    ram_tags = get_ram_tags(image)
    phash = get_phash(image)
    hsv = get_hsv(image)
    img = load_image(image)
    width, height = img.size
    have_object,have_censors,have_hand = is_have_object_censors_hand(img)
    if have_object:
        bodypart, bodypart_score = get_bodypart(img)
        bodypart_score = multiply_and_int_list(bodypart_score)
    else:
        bodypart_score=[0,0,0]
        bodypart = None
    if have_censors:
        nsfw_score = [0,0,1]
        nsfw = 2
    else:
        nsfw,nsfw_score = get_nsfw(img)
        nsfw_score = multiply_and_int_list(nsfw_score)
    bad_score = get_bad_score(img)
    bad_score = round(bad_score* 100)
    aesthetic, aesthetic_score = get_aesthetic(img)
    aesthetic_score = multiply_and_int_list(aesthetic_score)
    completeness,completeness_scores = get_completeness(img)
    completeness_scores = multiply_and_int_list(completeness_scores)
    anime_real, anime_real_score = get_anime_real(img)
    anime_real_score = multiply_and_int_list(anime_real_score)
    watermarks = get_watermark(image)
    all = {
        '_id': id,
        'width': width,
        'height': height,
        'aesthetic': aesthetic,
        'aesthetic_score': aesthetic_score,
        'bodypart': bodypart,
        'bodypart_score': bodypart_score,
        'completeness': completeness,
        'completeness_scores': completeness_scores,
        'anime_real': anime_real,
        'anime_real_score': anime_real_score,
        'nsfw': nsfw,
        "nsfw_score": nsfw_score,
        "ram_tags": ram_tags,
        "bad_score":bad_score,
        "have_object": have_object,
        "have_censors": have_censors,
        'have_hands':have_hand,
        "hsv": hsv,
        "phash":phash,
        'watermarks': watermarks,
        'download': 99,
    }
    return all

load_ram_model()
load_bad_model()

# 解压路径
import re
import shutil
import tarfile
import os
def get_webp_file_paths(dest_path):
    webp_file_paths = []
    for root, dirs, files in os.walk(dest_path):
        for file in files:
            if file.endswith(".webp"):
                file_path = os.path.join(root, file)
                webp_file_paths.append(file_path)
    return webp_file_paths
def copy_and_extract_tarfile(tar_path, dest_root='/pics'):
    global dest_path
    shutil.copy2(tar_path, '/home')
    tar_file_home = '/home/' + os.path.basename(tar_path)
    file_name = os.path.splitext(os.path.basename(tar_path))[0]
    number = re.findall(r'\d+', file_name)[0]
    dest_path = os.path.join(dest_root, number)
    os.makedirs(dest_path, exist_ok=True)
    with tarfile.open(tar_file_home, 'r') as tar:
        tar.extractall(dest_path)
    os.remove(tar_file_home)
    return dest_path
def pad_int(number,num=4):
    number_str = str(number)
    zeros_to_add = num - len(number_str)
    padded_number = '0' * zeros_to_add + number_str
    return padded_number  
def main(webp_file):
    if webp_file.endswith(".webp"):
        file_name = os.path.basename(webp_file)
        id = int(os.path.splitext(file_name)[0])
    else:
        return
    add_file_path = os.path.join("/gemini/output","dan",f"{str(id)}.add")
    if os.path.exists(add_file_path):
        return
    try:
        add_data = get_all(webp_file)
        with open(add_file_path, 'w') as file:
            json.dump(add_data, file)
    except Exception as e:
        print(e)
        return

    #pics_collection.update_one({"_id": id}, {"$set": {"download": 99}})
    #pics_danbooru.insert_one(new_data)

import os
import json

def doit(id):
    dest_path = f'/pics/{pad_int(id)}'
    if os.path.exists(dest_path):
        print("已解压")
    else:
        tar_file = f'/gemini/data-1/dan/{pad_int(id)}.tar'
        copy_and_extract_tarfile(tar_file)
    print(dest_path)
    os.makedirs("/gemini/output/dan", exist_ok=True)
    for file_name in os.listdir(dest_path):
        if file_name.endswith(".webp"):
            webp_file = os.path.join(dest_path, file_name)
            main(webp_file)
    
    def find_files_with_json_error(directory):
        error_files = []
    
        # 遍历目录下的所有 .add 文件
        for filename in os.listdir(directory):
            if filename.endswith(".add"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    try:
                        json.load(file)
                    except json.JSONDecodeError:
                        error_files.append(os.path.splitext(filename)[0])
        for file in error_files:
            add_file = os.path.join("/gemini/output/dan",f"{file}.add")
            os.remove(add_file)
            webp_file = os.path.join(dest_path, f"{file}.webp")
            main(webp_file)
    
    error_files = find_files_with_json_error('/gemini/output/dan')
    error_files = find_files_with_json_error('/gemini/output/dan')
    
    def extract_number_from_filename(filename):
        return int(''.join(filter(str.isdigit, filename)))
    
    def merge_add_files(directory, output_file):
        json_data_list = []
        error_count = 0
        files = [f for f in os.listdir(directory) if f.endswith(".add")]
        files.sort(key=extract_number_from_filename)
    
        for filename in files:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    json_data_list.append(json_data)
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Error decoding JSON in file: {file_path}. Error: {e}")
    
        with open(output_file, 'w') as outfile:
            json.dump(json_data_list, outfile)
    
        print(f"Total number of errors encountered: {error_count}")
    
    merge_add_files('/gemini/output/dan', f'/gemini/output/{id}.json')

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理通过命令行传入的数字。')
    parser.add_argument('--id', type=int, help='一个整数参数')
    args = parser.parse_args()
    doit(args.id)
