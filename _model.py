from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import logging
from functools import lru_cache
import os
from _image import *
from _image import _img_encode
from typing import Tuple, Dict
import tarfile
import shutil

models_path = os.getenv('IMGUTILS_MODELS_PATH')
if models_path:
    pass
else:
    models_path = "/root/.cache/ImgutilsModels/"

#mobilenetv3_large_100_ra-f55367f5.pth
def copy_model(src):
    model_name = os.path.basename(src)
    if os.name == 'posix':
        dst = f"/root/.cache/torch/hub/checkpoints/{model_name}"
    elif os.name == 'nt':
        dst = os.path.expanduser(f"~/.cache/torch/hub/checkpoints/{model_name}")
    else:
        print("Unsupported OS.")
        return
    if os.path.exists(dst):
        #print(f"Error: Destination file '{dst}' already exists.")
        return
    if not os.path.exists(src):
        print(f"Error: Source file '{src}' does not exist.")
        return
    if not os.path.exists(os.path.dirname(dst)):
        try:
            os.makedirs(os.path.dirname(dst))
            print(f"Created directory: {os.path.dirname(dst)}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            return
    try:
        shutil.copy2(src, dst)
        print(f"File copied from '{src}' to '{dst}'.")
    except IOError as e:
        print(f"Error copying file: {e}")
lama_model_path = os.path.join(models_path,'big-lama.pt')
mobilenetv3_model_path = os.path.join(models_path,'mobilenetv3_large_100_ra-f55367f5.pth')
copy_model(lama_model_path)
copy_model(mobilenetv3_model_path)

if os.name == 'posix':
    if os.path.exists('/root/.cache/huggingface/hub/models--bert-base-uncased'):
        pass
    else:
        with tarfile.open(os.path.join(models_path,"bert-base-uncased.tar"), 'r') as tar:
            tar.extractall(path='/')

def _open_onnx_model(ckpt: str, provider: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    logging.info(f'使用提供者 {provider!r} 加载模型 {ckpt!r}')
    return InferenceSession(ckpt, options, providers=[provider])

@lru_cache()
def open_onnx_model(ckpt: str) -> InferenceSession:
    return _open_onnx_model(os.path.join(models_path,ckpt), 'CUDAExecutionProvider')

class ClassifyModel:
    def __init__(self, model_name, labels, img_size: int = 384):
        self.model_name = model_name
        self._model = None
        self._labels = labels
        self.img_size = img_size

    def _open_model(self):
        if self._model is None:
            self._model = open_onnx_model(self.model_name)
        return self._model

    def _raw_predict(self, image: ImageTyping):
        image = load_image(image, force_background='white', mode='RGB')
        input_ = _img_encode(image,(self.img_size,self.img_size))[None, ...]
        output, = self._open_model().run(['output'], {'input': input_})
        del image,input_
        return output # 获得原始输出

    def predict_score(self, image: ImageTyping) -> Dict[str, float]:
        output = self._raw_predict(image)[0]
        values = dict(zip(self._labels, map(lambda x: x.item(), output)))
        del output,image
        return values # 字典

    def predict(self, image: ImageTyping) -> Tuple[str, float]:
        output = self._raw_predict(image)[0]
        max_id = np.argmax(output)
        return self._labels[max_id], output[max_id].item()

    def clear(self):
        self._models.clear()
        self._labels.clear()

@lru_cache()
def _open_models(model_name, labels, img_size):
    return ClassifyModel(model_name, tuple(labels), img_size)

def classify_predict_score(image: ImageTyping, model_name, labels, img_size):
    return _open_models(model_name, tuple(labels), img_size).predict_score(image)

def classify_predict(image: ImageTyping, model_name, labels, img_size):
    return _open_models(model_name, tuple(labels), img_size).predict(image)