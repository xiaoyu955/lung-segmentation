from skimage import io
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
import cv2

colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]


# def img2label(im):





# for i in range(1000):
#     print("num",i)
#     img = cv2.imread("F:\\unet2\\image\\bone3\\img_new\\" + "%03d" % (i) + ".png")
#     label = cv2.imread("C:\\Users\\MSE\\Desktop\\mask\\mask_new7\\" + "%03d" % (i) + ".png")
#     label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
#
#     label1 = np.zeros_like(label)
#     label1[:, :, 0] = label
#     label1[:, :, 1] = label
#     label1[:, :, 2] = label
#     # label1 = img2label(label)
#     # img = Image.fromarray(np.uint8(label))
#     cv2.imwrite("C:\\Users\\MSE\\Desktop\\mask\\mask_new8\\" + "%03d"%(i) + ".png",label1)



img_path='D:\\unet1\\bone\\train_new\\jian_pen\\mask_new\\'
save_img_path='D:\\unet1\\bone\\train_new\\jian_pen\\mask_new4\\'
# def mask2label(img_path,save_img_path):
def mask2label1(img_path):

    # for img_name in os.listdir(img_path):
    # image = Image.open(img_path)

    # print(len(image.split()))
    # print(img_path+img_name)
    # img = cv2.imread(img_path + img_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray =img_path

    # img2 = np.zeros_like(img_path)
    img1 = np.expand_dims(img_path, 2)
    # print("img1",img1.shape)

    # img2 = np.expand_dims(img1, axis=2)
    img2 = np.concatenate((img1, img1, img1), axis=-1)
    # print("img2",img2.shape)



    img2[:, :, 0] = gray
    img2[:, :, 1] = gray
    img2[:, :, 2] = gray

    # print("img",img2.size)
    width, height, c = img2.shape

    for x in range(width):
        for y in range(height):
            if img2[x, y, 0] == 0 and img2[x, y, 1] == 0 and img2[x, y, 2] == 0:
                img2[x, y, 0] = 0
                img2[x, y, 0] = 0
                img2[x, y, 0] = 0

    for x in range(width):
        for y in range(height):
            if img2[x, y, 0] == 1 and img2[x, y, 2] == 1 and img2[x, y, 2] == 1:
                img2[x, y, 0] = 255
                img2[x, y, 1] = 255
                img2[x, y, 2] = 255

    # for x in range(width):
    #     for y in range(height):
    #         if img2[x, y, 0] == 2 and img2[x, y, 1] == 2 and img2[x, y, 2] == 2:
    #             img2[x, y, 0] = 0
    #             img2[x, y, 1] = 128
    #             img2[x, y, 2] = 0

    return img2





            # cv2.imwrite(save_img_path + img_name, img2)
            # image=Image.open(save_img_path+img_name)


