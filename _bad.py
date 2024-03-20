import timm
from torchvision import transforms
import torch
from PIL import Image
import os
from _model import models_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# bad score
def get_backbone_model(name, labels, **kwargs):
    model = timm.create_model(name, num_classes=len(labels), **kwargs)
    model.__arguments__ = {'name': name, 'labels': labels, **kwargs}
    model.__info__ = {}
    return model.float()

def load_model_from_ckpt(file):
    data = torch.load(file, map_location='cpu')
    arguments = data['arguments'].copy()
    name = arguments.pop('name')
    labels = arguments.pop('labels')
    model = get_backbone_model(name, labels, **arguments)
    existing_keys = set(model.state_dict())
    state_dict = {key: value for key, value in data['state_dict'].items() if key in existing_keys}
    model.load_state_dict(state_dict)
    model.__info__ = data['info']

    return model

def load_bad_model():
    global bad_model
    print("加载二分模型")
    model_path = os.path.join(models_path, "class_2.ckpt")
    bad_model = load_model_from_ckpt(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bad_model.to(device)
    bad_model.eval()
    print("加载完成")

def get_bad_score(img):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    size = (512,512)
    fill_color=(0, 0, 0)
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1:  # Width > Height
        new_width = size[0]
        new_height = round(new_width / aspect_ratio)
    else:  # Height >= Width
        new_height = size[1]
        new_width = round(new_height * aspect_ratio)
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    new_img = Image.new("RGB", size, fill_color)
    paste_position = ((size[0] - new_width) // 2, (size[1] - new_height) // 2)
    new_img.paste(resized_img, paste_position)
    img = new_img.convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = bad_model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    label_scores = [{'label': 'good', 'score': probabilities[0, 0].item()},
                    {'label': 'bad', 'score': probabilities[0, 1].item()}]
    _, predicted_class = torch.max(output, 1)
    predicted_label = 'good' if predicted_class.item() == 0 else 'bad'
    return probabilities[0, 1].item()