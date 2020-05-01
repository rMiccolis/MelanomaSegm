import cv2
from binarize import *
import skimage
from scipy import ndimage
from skimage.measure import perimeter, label, regionprops
from skimage import color, filters
import math
from centroid import *

import numpy as np
from sklearn.preprocessing import binarize


def get_border_irregularity_ci(img, show = False):
    img = skimage.color.rgb2gray(img)
    img = binarize(img)

    #Find area and perimeter
    label_img = label(img)
    region = regionprops(label_img)

    img_area = max([props.area for props in region]) #Want the max because they could be many spaces
    img_perimeter = max([props.perimeter for props in region])

    #Calculate CI's formula
    return (img_perimeter**2) / (4.*math.pi*img_area)




img = cv2.imread('roi.jpg')
asi = get_border_irregularity_ci(img)

print(asi)