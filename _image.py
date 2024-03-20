from typing import Union
from os import PathLike
from typing import Union, BinaryIO, List, Tuple, Optional
from PIL import ImageColor, Image
import numpy as np
import math

_AlphaTyping = Union[float, np.ndarray]
ImageTyping = Union[str, PathLike, bytes, bytearray, BinaryIO, Image.Image]

def _is_readable(obj):
    return hasattr(obj, 'read') and hasattr(obj, 'seek')

def load_image(image: ImageTyping, mode=None, force_background: Optional[str] = 'white'):
    if isinstance(image, (str, PathLike, bytes, bytearray, BinaryIO)) or _is_readable(image):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        pass  # just do nothing
    else:
        raise TypeError(f'Unknown image type - {image!r}.')
    if force_background is not None:
        image = add_background_for_rgba(image, force_background)
    if mode is not None and image.mode != mode:
        image = image.convert(mode)
    return image

def _load_image_or_color(image) -> Union[str, Image.Image]:
    if isinstance(image, str):
        try:
            _ = ImageColor.getrgb(image)
        except ValueError:
            pass
        else:
            return image
    return load_image(image, mode='RGBA', force_background=None)

def _process(item):
    if isinstance(item, tuple):
        image, alpha = item
    else:
        image, alpha = item, 1
    return _load_image_or_color(image), alpha

def _add_alpha(image: Image.Image, alpha: _AlphaTyping) -> Image.Image:
    data = np.array(image.convert('RGBA')).astype(np.float32)
    data[:, :, 3] = (data[:, :, 3] * alpha).clip(0, 255)
    return Image.fromarray(data.astype(np.uint8), mode='RGBA')

def istack(*items: Union[ImageTyping, str, Tuple[ImageTyping, _AlphaTyping], Tuple[str, _AlphaTyping]],
           size: Optional[Tuple[int, int]] = None) -> Image.Image:
    if size is None:
        height, width = None, None
        items = list(map(_process, items))
        for item, alpha in items:
            if isinstance(item, Image.Image):
                height, width = item.height, item.width
                break
    else:
        width, height = size
    if height is None:
        raise ValueError('Unable to determine image size, please make sure '
                         'you have provided at least one image object (image path or PIL object).')

    retval = Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8), mode='RGBA')
    for item, alpha in items:
        if isinstance(item, str):
            current = Image.new("RGBA", (width, height), item)
        elif isinstance(item, Image.Image):
            current = item
        else:
            assert False, f'Invalid type - {item!r}. If you encounter this situation, ' \
                          f'it means there is a bug in the code. Please contact the developer.'  # pragma: no cover

        current = _add_alpha(current, alpha)
        retval.paste(current, mask=current)

    return retval

def add_background_for_rgba(image: ImageTyping, background: str = 'white'):
    return istack(background, image).convert('RGB')

_DEFAULT_ORDER = 'HWC'

def _get_hwc_map(order_):
    return tuple(_DEFAULT_ORDER.index(c) for c in order_.upper())

def rgb_encode(image: ImageTyping, order_: str = 'CHW', use_float: bool = True) -> np.ndarray:
    image = load_image(image, mode='RGB')
    array = np.asarray(image)
    array = np.transpose(array, _get_hwc_map(order_))
    if use_float:
        array = (array / 255.0).astype(np.float32)
        assert array.dtype == np.float32
    else:
        assert array.dtype == np.uint8
    return array

def _image_preprocess(image: Image.Image, max_infer_size: int = 1216, align: int = 32):
    old_width, old_height = image.width, image.height
    new_width, new_height = old_width, old_height
    r = max_infer_size / max(new_width, new_height)
    if r < 1:
        new_width, new_height = new_width * r, new_height * r
    new_width = int(math.ceil(new_width / align) * align)
    new_height = int(math.ceil(new_height / align) * align)
    image = image.resize((new_width, new_height))
    return image, (old_width, old_height), (new_width, new_height)

def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')
    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)

