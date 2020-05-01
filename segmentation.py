import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import cv2
import imutils
from scipy.spatial import distance

print("Input image read")
image_name = "nostra.jpg"
image_source = "E:\\ROB\\Desktop\\PycharmWorkspace\\segm\\imgs\\preprocessed\\prep"+image_name
image_seg = "E:\\ROB\\Desktop\\PycharmWorkspace\\segm\\imgs\\segmented\\seg"+image_name
thresholded_image = "thresholded.jpg"

image1 = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)


blur = cv2.GaussianBlur(image1,(157,157),0,0)

ret3,th3 = cv2.threshold(image1,255,0,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)


ker = np.ones((5,5),np.float32)/25
blackhat = cv2.morphologyEx(th3, cv2.MORPH_OPEN, ker)

cv2.imwrite(thresholded_image, th3)
#cv2.imshow('thresholded', th3)
#cv2.waitKey(0)
print("applying chan-vese to image...")

# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(th3, mu=0.7, lambda1=11, lambda2=11, tol=1e-3, max_iter=200,
              dt=0.6, init_level_set="checkerboard", extended_output=True)


plt.imsave(image_seg, cv[0], cmap='gray')
img = cv2.imread(image_seg)
#cv2.imshow('ff', img)
#cv2.waitKey(0)
height, width = img.shape[:2]
height1 = int(height/2)
width1 = int(width/2)

print(f"{height}, {width}")


if ((img[height1,width1][0] == 0 and img[height1,width1][1] == 0 and img[height1,width1][2] == 0) and (img[height1+20,width1+20][0] == 0 and img[height1+20,width1+20][1] == 0 and img[height1+20,width1+20][2] == 0) and (img[height1-20,width1-20][0] == 0 and img[height1-20,width1-20][1] == 0 and img[height1-20,width1-20][2] == 0)):
    plt.imsave(image_seg, cv[0], cmap='Greys_r')
















