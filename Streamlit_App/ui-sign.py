import streamlit as st
from PIL import Image
import cv2
import os
from SOURCE.yolo_files import detect
from SOURCE.gan_files import test
from SOURCE.vgg_finetuned_model import vgg_verify
from SOURCE.cae_files import sign_clean
from SOURCE.cae_files.cae_model import UNetCAE
from helper_fns import gan_utils
import shutil
import glob
import SessionState
import math
import numpy as np

MEDIA_ROOT = 'media/documents/v20241204/'
SIGNATURE_ROOT = 'media/AnchorSignatures/'
YOLO_RESULT = 'results/yolov5/'
YOLO_OP = 'crops/DLSignature/'
GAN_IPS = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
GAN_OP = 'results/gan/gan_signdata_kaggle/test_latest/images/'
GAN_OP_RESIZED = 'results/gan/gan_signdata_kaggle/test_latest/images/'


def to_binary_image(image):
    img_array = np.array(image)
    threshold = 190
    img_array = np.where(img_array < threshold, img_array, 255)  # 较黑的像素保留，其他设置为白色
    filtered_array = np.where(img_array > threshold, img_array, 0)  # 较黑的像素保留，其他设置为白色

    # 转换回 PIL 图像
    filtered_img = Image.fromarray(filtered_array.astype("uint8"))
    return filtered_img

def selected_cleaned_image_path(selection):
    ''' Returns the path of cleaned image corresponding to the document the user selected '''
    return GAN_OP + selection + '_fake.png'


def selected_cleaned_image_chop_path(selection):
    ''' Returns the path of cleaned image corresponding to the document the user selected '''
    return GAN_OP + selection + '_chop.png'

def copy_and_overwrite(from_path, to_path):
    '''
    Copy files from results/yolo_ops/ to results/gan/gan_signdata_kaggle/gan_ips
    CycleGAN model requires ip files to be present in results/gan/gan_signdata_kaggle/gan_ips
    '''
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

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

def resize_with_aspect_ratio(img, target_h=None, target_w=None):
    """
    Resize an image to a specified target height or width while maintaining aspect ratio.

    :param image_path: Path to the input image
    :param target_height: Target height (optional)
    :param target_width: Target width (optional)
    :return: Resized PIL Image object
    """
    # Open the image
    original_width, original_height = img.size

    # Ensure one of target_height or target_width is provided
    if target_h is None and target_w is None:
        raise ValueError("Either target_height or target_width must be specified.")

    # Calculate new dimensions while maintaining aspect ratio
    if target_h is not None:
        # Calculate width to maintain aspect ratio
        new_width = int(original_width * (target_h / original_height))
        new_height = target_h
    elif target_w is not None:
        # Calculate height to maintain aspect ratio
        new_height = int(original_height * (target_w / original_width))
        new_width = target_w

    # Resize the image
    resized_img = img.resize((new_width, new_height))
    return resized_img


def pad_to_square(img, output_path=None):
    """
    Pad an image to a square shape with white background.

    :param image_path: Path to the input image.
    :param output_path: Optional path to save the padded image.
    :return: Padded PIL Image object.
    """
    # 获取原始宽度和高度
    width, height = img.size

    # 计算正方形边长
    side_length = max(width, height)

    # 创建白色背景的正方形图像
    square_img = Image.new("RGB", (side_length, side_length), (255, 255, 255))

    # 计算粘贴位置，使原图居中
    paste_position = ((side_length - width) // 2, (side_length - height) // 2)

    # 将原图粘贴到白色背景上
    square_img.paste(img, paste_position)

    # 保存或返回结果
    if output_path:
        square_img.save(output_path)
    return square_img

def signature_verify(selection):
    ''' Performs signature verification and displays the anchor image alongside 
        the detections from all the documents and their corresponding cosine 
        similarity score.

        For the demo, all three phases (signature detection, cleaning and 
        verification) are performed on all the documents. The cleaned image of 
        all documents are compared with the anchor image (the signature in the 
        database corresponding to the document the user selected from the
        dropdown) to demonstrate that matching pairs give a higher cosine 
        similarity score and non-matching pairs produce a lower score.
        Ideally, the anchor image should be selected from the "Account Name" and
        "Signatory Name" returned by Vikas' module.

        For POC, the motive is to show that matching signature pairs have a 
        score close to 1 and non-matching signatures have a lower score (<0.7).
        The logic used is a simple matching with filename.

        So the index or id User Signatures are denoted by their file name.
        Eg: User1's anchor signature in media/UserSignatureSquare will be named 
        as 1.png.
        For the sake of simplicity, I have matched the name of document in
        similar logic.
        Eg: Document 1 (media/documents/1.png) contains the signature of the 
        user. 
        The the signature to be compared with is selected on the basis of 
        filename.
        
    '''
    anchor_sign_image_path = SIGNATURE_ROOT + 'AnchorSign.png'
    anchor_sign_img = to_binary_image(replace_bg(Image.open(anchor_sign_image_path)))
    
    anchor_chop_image_path = SIGNATURE_ROOT + 'AnchorChop.png'
    anchor_chop_img = to_binary_image(replace_bg(Image.open(anchor_chop_image_path)))
    
    sign_feature_set, chop_feature_set = vgg_verify.verify(
        anchor_sign_img, GAN_OP_RESIZED, selection, anchor_chop_img
    )

    columns = st.columns(3)
    columns[0].markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><h3>Reference</h3></div>", unsafe_allow_html=True)
    columns[1].markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><h3>To Verify</h3></div>", unsafe_allow_html=True)
    columns[2].markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><h3>Score</h3></div>", unsafe_allow_html=True)

    for sign_image, score in sign_feature_set:
        score = round(score.item()*100, 2)
        columns = st.columns(3)

        with columns[0]:
            col0_container = st.container()
            col0_container.image(pad_to_square(anchor_sign_img), use_column_width=True)

        with columns[1]:
            col1_container = st.container()
            resized_sign_img = pad_to_square(sign_image).resize((anchor_sign_img.size[1], anchor_sign_img.size[1]))
            col1_container.image(resized_sign_img, use_column_width=True)

        with columns[2]:
            col2_container = st.container()
            col2_container.markdown(f"<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><h3>{score}%</h3></div>", unsafe_allow_html=True)

    for chop_image, score in chop_feature_set:
        score = round(score.item()*100, 2)
        columns = st.columns(3)
        with columns[0]:
            col0_container = st.container()
            col0_container.image(pad_to_square(anchor_chop_img), use_column_width=True)

        with columns[1]:
            col1_container = st.container()
            size = anchor_chop_img.size[1]
            resized_chop_img = pad_to_square(chop_image).resize((size,size))
            col1_container.image(resized_chop_img, use_column_width=True)

        with columns[2]:
            col2_container = st.container()
            col2_container.markdown(f"<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><h3>{score}%</h3></div>", unsafe_allow_html=True)

def signature_cleaning(selection):
    ''' Performs signature cleaning and displays the cleaned signatures '''
    # copy files from results/yolo_ops/ to results/gan/gan_signdata_kaggle/gan_ips
    selected_img_path = MEDIA_ROOT+selection+'.jpg'
    cleaned_img, cleaned_img_chop = sign_clean.clean(selected_img_path)
    cleaned_img.save(selected_cleaned_image_path(selection))
    cleaned_img_chop.save(selected_cleaned_image_chop_path(selection))

    det_num = 1
    #cleaned images are selected and displayed
    st.subheader(f"{det_num} signature(s) cleaned")
    columns = [column for column in st.columns(det_num+1)]
    for i, col in enumerate(columns):
        if i < det_num:
            postfix = '-'+str(i+1) if i > 0 else ''            
            cleaned_image = selected_cleaned_image_path(selection+postfix)
            col.image(cleaned_image)
        else:
            col.image(selected_cleaned_image_chop_path(selection))


# def signature_detection(selection):
#     ''' Performs signature detection and returns the results folder. '''

#     # call YOLOv5 detection fn on all images in the document folder.
#     img_det_num = detect.detect(MEDIA_ROOT, selection)
#     det_num = img_det_num[selection]
#     # get the path where last detected results are stored.
#     latest_detection = max(glob.glob(os.path.join(YOLO_RESULT, '*/')), key=os.path.getmtime)
#     # resize and add top and bottom padding to detected sigantures. 
#     # gan model expects ips in that particular format.
#     gan_utils.resize_images(os.path.join(latest_detection, YOLO_OP))

#     # selects and display the detections of the document which the user selected.
#     st.subheader(f"{det_num} signature(s) detected")
#     columns = [column for column in st.columns(det_num)]
#     for i, col in enumerate(columns):
#         postfix = '-'+str(i+1) if i > 0 else ''            
#         selection_detection =latest_detection + YOLO_OP + selection + postfix + '.jpg' 
#         col.image(selection_detection)
#     return latest_detection + YOLO_OP, det_num# return the yolo op folder

    
def select_document():
    '''
        Selects the document from the dropdown menu and displays the image.
        Returns an integer represeting the id of the document selected.
    '''
    left, right = st.columns([1,1]) # Create two columns
    # dropdown box in left column
    selection = str(left.selectbox('Select document for verfying:',[1,2,3,4,5,6]))
    # select corresponding document image from media/documents
    selection_image = MEDIA_ROOT+selection+'.jpg'
    #display image in right column.
    right.image(selection_image, use_column_width='always')
#     detect_button = left.button('Detect Signature')
    return selection

def main():
    # 初始化 session_state，仅在首次运行时
    if "selection" not in st.session_state:
        st.session_state.selection = ''
        st.session_state.previous_selection = None  # 保存上一次的选择
        st.session_state.yolo_op = ''
        st.session_state.detect_button_clicked = False
        st.session_state.clean_button_clicked = False
        st.session_state.verify_button_clicked = False

    st.header('Deep Learning based Signature Verification')

    # 用户选择文档
    current_selection = select_document()

    # 如果 selection 发生变化，重置状态
    if current_selection != st.session_state.previous_selection:
        st.session_state.detect_button_clicked = False
        st.session_state.clean_button_clicked = False
        st.session_state.verify_button_clicked = False
        st.session_state.yolo_op = ''
        st.session_state.previous_selection = current_selection  # 更新为新的选择

    st.session_state.selection = current_selection  # 更新当前选择

    # Detect Signature 按钮渲染
    # 但是 Cheque 这个不需要 Detect
#     if st.button('Detect Signature'):
    st.session_state.detect_button_clicked = True

    # 执行 Detect Signature 逻辑
    if st.session_state.detect_button_clicked:
#         st.session_state.yolo_op = signature_detection(st.session_state.selection)
#         st.write("Detection Complete.")

        # Render Clean Signature btn 按钮逻辑
        if st.button('Clean Signature'):
            st.session_state.clean_button_clicked = True

    # 执行 Clean Signature 逻辑
    if st.session_state.clean_button_clicked:
        signature_cleaning(st.session_state.selection)
        
        # Verify Signature 按钮逻辑
        if st.button('Verify Signature'):
            st.session_state.verify_button_clicked = True

    # 执行 Verify Signature 逻辑
    if st.session_state.verify_button_clicked:
        signature_verify(st.session_state.selection)

main()
