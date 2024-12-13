import streamlit as st
from PIL import Image
import cv2
import os
from SOURCE.yolo_files import detect
from SOURCE.gan_files import test
from SOURCE.vgg_finetuned_model import vgg_verify
from helper_fns import gan_utils
import shutil
import glob
import SessionState

MEDIA_ROOT = 'media/documents/'
SIGNATURE_ROOT = 'media/UserSignaturesSquare/'
YOLO_RESULT = 'results/yolov5/'
YOLO_OP = 'crops/DLSignature/'
GAN_IPS = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
GAN_OP = 'results/gan/gan_signdata_kaggle/test_latest/images/'
GAN_OP_RESIZED = 'results/gan/gan_signdata_kaggle/test_latest/images/'



def select_cleaned_image(selection):
    ''' Returns the path of cleaned image corresponding to the document the user selected '''
    return GAN_OP + selection + '_fake.png'

def copy_and_overwrite(from_path, to_path):
    '''
    Copy files from results/yolo_ops/ to results/gan/gan_signdata_kaggle/gan_ips
    CycleGAN model requires ip files to be present in results/gan/gan_signdata_kaggle/gan_ips
    '''
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

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
    anchor_image = SIGNATURE_ROOT + selection + '.png'
    # verify the anchor signature with the detctions on all documents
    feature_set = vgg_verify.verify(anchor_image, GAN_OP_RESIZED)
    for image, score in feature_set:
        columns = [column for column in st.columns(3)]
        columns[0].image(anchor_image)
        columns[1].image(image)
        columns[2].write(score)

def signature_cleaning(selection, yolo_op):
    ''' Performs signature cleaning and displays the cleaned signatures '''
    # copy files from results/yolo_ops/ to results/gan/gan_signdata_kaggle/gan_ips
    copy_and_overwrite(yolo_op, GAN_IPS)
    test.clean() # performs cleaning

    #cleaned images are selected and displayed
    cleaned_image = select_cleaned_image(selection)
    st.image(cleaned_image)

def signature_detection(selection):
    ''' Performs signature detection and returns the results folder. '''

    # call YOLOv5 detection fn on all images in the document folder.
    detect.detect(MEDIA_ROOT)
    # get the path where last detected results are stored.
    latest_detection = max(glob.glob(os.path.join(YOLO_RESULT, '*/')), key=os.path.getmtime)
    # resize and add top and bottom padding to detected sigantures. 
    # gan model expects ips in that particular format.
    gan_utils.resize_images(os.path.join(latest_detection, YOLO_OP))

    # selects and display the detections of the document which the user selected.
    selection_detection =latest_detection + YOLO_OP + selection + '.jpg'
    st.image(selection_detection)
    return latest_detection + YOLO_OP # return the yolo op folder

def select_document():
    '''
        Selects the document from the dropdown menu and displays the image.
        Returns an integer represeting the id of the document selected.
    '''
    left, right = st.columns(2) # Create two columns
    # dropdown box in left column
    selection = str(left.selectbox('Select document to run inference', [1, 2]))
    # select corresponding document image from media/documents
    selection_image = MEDIA_ROOT+selection+'.png'
    #display image in right column.
    right.image(selection_image, use_column_width='always')
    return selection

def main():
    # 初始化 session_state，仅在首次运行时
    if "selection" not in st.session_state:
        st.session_state.selection = ''
        st.session_state.counter = 0
        st.session_state.yolo_op = ''
        st.session_state.detect_button = False
        st.session_state.clean_button = False
        st.session_state.verify_button = False

    st.write('Deep Learning based Signature Detection and Verification')
    st.write('Built by [Amal Joseph](https://www.linkedin.com/in/amaljoseph/)')
    st.write('[Github Repo](https://github.com/amaljoseph)')
    st.session_state.counter += 1
    print(st.session_state.counter)
    
    # 文档选择
    st.session_state.selection = select_document()

    # 按钮逻辑：逐步触发
    if st.button('Detect Signature'):
        st.session_state.detect_button = True

    if st.session_state.detect_button:
        print('detect'+str(st.session_state.counter))
        st.session_state.yolo_op = signature_detection(st.session_state.selection)
        st.write("Detection Complete.")

        if st.button('Clean Signature'):
            st.session_state.clean_button = True

    if st.session_state.clean_button:
        print('clean'+str(st.session_state.counter))
        signature_cleaning(st.session_state.selection, st.session_state.yolo_op)
        st.write("Cleaning Complete.")

        if st.button('Verify Signature'):
            st.session_state.verify_button = True

    if st.session_state.verify_button:
        print('verify'+str(st.session_state.counter))
        signature_verify(st.session_state.selection)
        st.write("Verification Complete.")
main()