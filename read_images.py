import numpy as np
import os
import cv2


def read_stereo_images(source_img_path, id_list):
    """

    :param source_img_path:
    :param id_list: a list constructed with the index of source picture in the file
    :return:
    """
    # opencv读入的图像是BGR的格式
    left_path = os.path.join(source_img_path, r'Left')
    right_path = os.path.join(source_img_path, r'Right')
    left_list = os.listdir(left_path)
    right_list = os.listdir(right_path)
    left_list.sort(key= lambda x:int(x.split(".")[0].split("_")[2]))
    right_list.sort(key= lambda x:int(x.split(".")[0].split("_")[2]))
    #print(left_list)
    

    if len(right_list) == len(left_list):
        # read an image randomly to getting the size of image?
        left_name = os.path.join(left_path, left_list[id_list[0]])
        left_img = read_an_image(left_name)
        batch = len(id_list)
        height = np.shape(left_img)[0]
        width = np.shape(left_img)[1]
        channel = np.shape(left_img)[2]
        left_img_list = np.zeros([batch, height, width, channel], dtype=np.float32)
        right_img_list = np.zeros([batch, height, width, channel], dtype=np.float32)
        #jm_img_list = np.zeros([batch, height, width, 1], dtype=np.float32)

        for i in range(0, len(id_list)):
            left_name = os.path.join(left_path, str(left_list[id_list[i]]))
            left_img_list[i, :, :, :] = read_an_image(left_name)
            right_name = os.path.join(right_path, str(right_list[id_list[i]]))
            right_img_list[i, :, :, :] = read_an_image(right_name)
            #jm_img_list[i, :, :, :] = read_an_jmimage(right_name)
    else:
        print('check your file')
    return left_img_list, right_img_list


def read_an_image(img_name):
    img = cv2.imread(img_name, 1).astype(np.float32)#[...,::-1]
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b,g,r = cv2.split(img)
    # RGB图像
    img = cv2.merge([r,g,b])
    
    img = cv2.medianBlur(img, 3)
    return img



def read_an_jmimage(img_name):
    img = cv2.imread(img_name, 1).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, im_fixed = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY_INV)
    return im_fixed
