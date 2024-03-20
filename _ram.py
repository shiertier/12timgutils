### ram
from ram.models import ram_plus
from ram import inference_ram,get_transform
import torch
from _image import load_image
from _model import models_path
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型加载函数
def load_ram_model():
    model_path = os.path.join(models_path, "ram_plus_swin_large_14m.pth")
    print("加载ram+模型")
    global ram_model
    ram_model = ram_plus(pretrained=model_path, image_size=384, vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)
    print("加载完成")

def get_ram_tags(image):
    image = load_image(image)
    transform = get_transform(image_size=384)
    image = transform(image).unsqueeze(0).to(device)
    res = inference_ram(image, ram_model)
    tag_list = res[0].split('|')
    tag_list = [tag.strip() for tag in tag_list]
    return tag_list