from _image import *
from _model import *
from _yolo import _data_postprocess
from _image import _image_preprocess

def detect_pic(image: ImageTyping, max_infer_size=640):
    new_image, old_size, new_size = _image_preprocess(image, max_infer_size)
    data = rgb_encode(new_image)[None, ...]
    return data, old_size, new_size

person_detect = open_onnx_model("person_detect_plus_v1.1_best_m.onnx")
hand_detect = open_onnx_model("hand_detect_v1.0_s.onnx")
censor_detect = open_onnx_model("censor_detect_v1.0_s.onnx")


# 人物检查
def detect_person(data, old_size, new_size, conf_threshold: float = 0.3, iou_threshold: float = 0.5):
    output, = person_detect.run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ['person'])

# 手检查
def detect_hand(data, old_size, new_size, conf_threshold: float = 0.3, iou_threshold: float = 0.5):
    output, = hand_detect.run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ['hand'])

## 三点检查
def detect_censors(data, old_size, new_size, conf_threshold: float = 0.3, iou_threshold: float = 0.5):
    output, = censor_detect.run(['output0'], {'images': data})
    return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ["nipple_f", "penis", "pussy"])

# 检查 all
def is_have_object_censors_hand(image):
    image = load_image(image, mode='RGB')
    image, old_size, new_size, = detect_pic(image)
    result_p = detect_person(image,old_size,new_size)
    if len(result_p) == 0:
        have_person=False
        have_censors=False
        have_hand=False
    else:
        have_person=True
        result_censors = detect_censors(image,old_size,new_size)
        if len(result_censors) == 0:
            have_censors=False
        else:
            have_censors=True
        result_hand = detect_hand(image,old_size,new_size)
        if len(result_hand) == 0:
            have_hand=False
        else:
            have_hand=True
    return int(have_person), int(have_censors), int(have_hand)