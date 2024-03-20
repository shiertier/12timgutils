import cv2
import numpy as np
# hsv
def get_hsv(a):
    image = cv2.imread(a)
    # 将图像转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 提取亮度分量
    value_channel = hsv_image[:,:,2]

    # 计算亮度比例
    lightness_levels = np.linspace(0, 255, 11)  # 10个亮度级别
    lightness_counts, _ = np.histogram(value_channel, bins=lightness_levels)
    lightness_proportions = lightness_counts / np.sum(lightness_counts)

    # 以字典形式返回亮度比例
    lightness_values = []
    for proportion in lightness_proportions:
        lightness_value = int(round(proportion, 2)*100)
        lightness_values.append(lightness_value)

    # 计算平均亮度
    average_lightness = round(np.mean(value_channel) / 255, 3)
    v_key = round(average_lightness*10)

    saturation_channel = hsv_image[:,:,1]

    saturation_levels = np.linspace(0, 255, 11)  # 10个饱和度级别
    saturation_counts, _ = np.histogram(saturation_channel, bins=saturation_levels)
    saturation_proportions = saturation_counts / np.sum(saturation_counts)

    saturation_values = []
    for proportion in saturation_proportions:
        saturation_value = int(round(proportion, 2)*100)
        saturation_values.append(saturation_value)

    average_saturation = round(np.mean(saturation_channel) / 255, 3)
    s_key = round(average_saturation * 10)

    # 定义颜色范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([360, 255, 45])
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([360, 221, 255])
    lower_grey = np.array([0, 0, 45])
    upper_grey = np.array([360, 42, 220])
    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([311, 43, 46])
    upper_red2 = np.array([360, 255, 255])
    lower_orange = np.array([21, 43, 46])
    upper_orange = np.array([50, 255, 255])
    lower_yellow = np.array([51, 43, 46])
    upper_yellow = np.array([68, 255, 255])
    lower_green = np.array([69, 43, 46])
    upper_green = np.array([156, 255, 255])
    lower_cyan = np.array([157, 43, 46])
    upper_cyan = np.array([198, 255, 255])
    lower_blue = np.array([199, 43, 46])
    upper_blue = np.array([248, 255, 255])
    lower_purple = np.array([249, 43, 46])
    upper_purple = np.array([310, 255, 255])

    # 创建掩模
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    mask_grey = cv2.inRange(hsv_image, lower_grey, upper_grey)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_cyan = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # 计算各颜色像素占比
    total_pixels = np.sum(saturation_counts)
    black_percentage = np.sum(mask_black > 0) / total_pixels
    white_percentage = np.sum(mask_white > 0) / total_pixels
    grey_percentage = np.sum(mask_grey > 0) / total_pixels
    red_percentage = np.sum(mask_red > 0) / total_pixels
    orange_percentage = np.sum(mask_orange > 0) / total_pixels
    yellow_percentage = np.sum(mask_yellow > 0) / total_pixels
    green_percentage = np.sum(mask_green > 0) / total_pixels
    cyan_percentage = np.sum(mask_cyan > 0) / total_pixels
    blue_percentage = np.sum(mask_blue > 0) / total_pixels
    purple_percentage = np.sum(mask_purple > 0) / total_pixels
    # 黑白灰红橙黄绿青蓝紫
    color_percentages = [black_percentage,white_percentage,grey_percentage,red_percentage,orange_percentage,yellow_percentage,green_percentage,cyan_percentage,blue_percentage,purple_percentage]
    final_color = []
    for color in color_percentages:
        per = int(round(color, 2)*100)
        final_color.append(per)

    result = {
        "v_key": v_key,
        "v_scores": lightness_values,
        "s_key": s_key,
        "s_scores": saturation_values,
        "h_scores": final_color
    }
    return result