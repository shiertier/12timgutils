# pHash
from bson.binary import Binary
import cv2
import numpy as np

def pHash(img_path, leng=32, wid=32):
    # Read the image from the file path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (leng, wid))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:16, 0:16]
    avreage = np.mean(dct_roi)
    phash_01 = (dct_roi > avreage) + 0
    phash_list = phash_01.reshape(1, -1)[0].tolist()
    hash = ''.join([str(x) for x in phash_list])
    return hash

def get_phash(img_path):
    binary_string = pHash(img_path)
    def binary_string_to_bytes(binary_string):
        return bytes(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
    binary = binary_string_to_bytes(binary_string)
    binary_data = Binary(binary, subtype=0)
    return binary_string