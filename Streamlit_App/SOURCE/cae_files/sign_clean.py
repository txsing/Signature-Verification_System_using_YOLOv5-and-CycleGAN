import numpy as np
from torchvision import transforms
from PIL import Image, ImageChops
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F

GAN_OP = 'results/gan/gan_signdata_kaggle/test_latest/images/'

def replace_bg(img):
    """
    Replace background with white.
    
    :param img: Input image path or PIL Image
    :return: Image with replaced background
    """
    if isinstance(img, str):
        img = Image.open(img)
    
    # Convert to NumPy for processing
    img_array = np.array(img)
    
    # Define background color range
    lower_bg_color = np.array([240, 240, 240], dtype=np.uint8)
    upper_bg_color = np.array([254, 254, 254], dtype=np.uint8)

    # Create mask for background
    bg_mask = ((img_array >= lower_bg_color) & (img_array <= upper_bg_color)).all(axis=2)
    
    # Replace background with white
    img_array[bg_mask] = [255, 255, 255]
    return Image.fromarray(img_array)

class ReplaceBackground:
    def __call__(self, img):
        return replace_bg(img)

def clean(input_img_path):
    device = torch.device('cuda')
    model = torch.load('SOURCE/cae_files/CAE-latest.pth', map_location=device)
    model.eval()
    input_img = Image.open(input_img_path).convert("RGB")

    transformTest = transforms.Compose([
        ReplaceBackground(),  # 替换背景颜色
#         transforms.Resize(256),  # 调整大小
        transforms.ToTensor()  # 转换为 Tensor
    ])

    input_tensor = transformTest(input_img).unsqueeze(0).to(device)
    sign_img_with_chop = replace_bg(input_img)
    with torch.no_grad():
        reconstructed = model(input_tensor)
        squeezed_tensor = reconstructed.to('cpu').squeeze()
        squeezed_tensor = (squeezed_tensor - squeezed_tensor.min()) / (squeezed_tensor.max() - squeezed_tensor.min()) * 255
        squeezed_tensor = squeezed_tensor.byte()
        img_array = squeezed_tensor.numpy()

        threshold = 190
        filtered_array = np.where(img_array < threshold, img_array, 255)  # 较黑的像素保留，其他设置为白色
        # 转换回 PIL 图像
        filtered_img = Image.fromarray(filtered_array.astype("uint8"))
        chop_img = get_chop_image(sign_img_with_chop, filtered_img)
        print(chop_img.size, crop_non_white_area(chop_img).size)
        return crop_non_white_area(filtered_img), crop_non_white_area(chop_img)


def get_chop_image(image_a, image_b):
    # 打开图片并转换为灰度图
    image_a = image_a.convert("L")  # A 转为灰度图
    image_b = image_b.convert("L")  # B 转为灰度图

    # 对 A 和 B 图像进行二值化处理
    threshold_a = 180  # A 图像的二值化阈值
    binary_a = np.array(image_a) < threshold_a  # A 的二值图，黑色部分为 True
    threshold_b = 128  # B 图像的二值化阈值
    binary_b = np.array(image_b) < threshold_b  # B 的二值图，黑色部分为 True

    # 生成遮罩 mask，只保留 A 中不属于 B 的部分
    mask = np.logical_and(binary_a, ~binary_b)  # 在 A 中有但不在 B 中的部分

    # 将 mask 应用于 A，背景设为白色
    result_array = np.where(mask, 0, 255)  # 黑色部分 (0) 保留，其他变为白色 (255)

    # 转换回 PIL 图像
    result_image = Image.fromarray(result_array.astype("uint8"))
    return result_image

def crop_non_white_area(image):
    """
    Crops a PIL Image to the non-white region, preserving full dimensions if an entire axis is filled.
    
    Args:
    image (PIL.Image.Image): Input PIL Image
    
    Returns:
    PIL.Image.Image: Cropped image
    
    Raises:
    ValueError: If the entire image is white
    """
    # Convert image to grayscale numpy array
    gray = image.convert('L')
    gray_array = np.array(gray)
    
    # Define white threshold (allowing slight variation)
    white_threshold = 240
    
    # Create a binary mask of non-white regions
    non_white_mask = gray_array < white_threshold
    
    # Check if the entire image is white
    if not non_white_mask.any():
        raise ValueError("The entire image is white. Cannot crop.")
    
    # Find non-white regions
    rows = non_white_mask.any(axis=1)
    cols = non_white_mask.any(axis=0)
    
    # Find row and column ranges
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    # Crop the image
    cropped = image.crop((col_min, row_min, col_max+1, row_max+1))
    
    return cropped
