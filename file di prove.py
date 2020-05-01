import os

import cv2
import imutils
import scipy
from PIL import Image
from skimage.color import rgb2gray
from scipy.spatial import distance
import numpy as np
from scipy import ndimage
from itertools import cycle

from scipy.ndimage import binary_dilation, binary_erosion, \
                        gaussian_filter, gaussian_gradient_magnitude


image_source = "IMD409.bmp"
image_seg = "img_seg.jpg"

def find_biggest_contour(image):
   image = image.copy()
   contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

   contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
   biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

   mask = np.zeros(image.shape, np.uint8)
   cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
   return biggest_contour, mask



img = cv2.imread(image_seg, cv2.IMREAD_GRAYSCALE)
basewidth = img.shape[1] / 340.
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_NONE)

cv2.imshow("thresh", thresh)
#cv2.imshow("Image", image)


height, width = img.shape[:2]
print("measuring infos...")

im = cv2.imread(image_source)


#cnt = contours[0]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 3)


xcentro = int(width/2)
ycentro = int(height/2)

#cnt = contours[0]
print(f"totalLen={len(contours)}")
cx= xcentro
cy = ycentro
print(f"Initial center= {cx, cy}")
success = False
m, a = find_biggest_contour(img)
for cnt in (m):
    M = cv2.moments(cnt)

    if(M['m00'] != 0):
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        area = M['m00']

        if((distance.euclidean((cx, cy), (xcentro, ycentro)) < 30)):
            success = True
            print(f"area={area}")
            break




#reading normal image
img2 = cv2.imread(image_source, cv2.IMREAD_COLOR)

# Reading same image in another
# variable and converting to gray scale.
img = cv2.imread(image_seg, cv2.IMREAD_GRAYSCALE)

# Converting image to a binary image
# ( black and white only image).
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# Detecting contours in image.
contours, sizes = cv2.findContours(threshold, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

real_max, boh = find_biggest_contour(img)

approx = cv2.approxPolyDP(real_max, 0.009 * cv2.arcLength(cnt, True), True)

# draws boundary of contours.
cv2.drawContours(img2, [approx], 0, (0, 255, 0), 1)

# Used to flatted the array containing
# the co-ordinates of the vertices.
#n = approx.ravel()


# Showing the final image.
cv2.imshow('image2', img2)
cv2.imwrite('imageNotReduced.jpeg', img2)
cv2.waitKey(0)

#########################################################################################################################################


original_image = img2
image = original_image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred, 120, 255, 1)
cv2.imshow('canny', canny)




# Find contours in the image
cnts = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Obtain area for each contour
contour_sizes = [(cv2.contourArea(contour), contour) for contour in cnts]

# Find maximum contour and crop for ROI section
largest_contour = approx
x,y,w,h = cv2.boundingRect(largest_contour)
cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
ROI = original_image[y:y+h, x:x+w]


cv2.imshow("ROI", ROI)
basewidth = img.shape[1] / 340.
ROI2 = cv2.resize(ROI, (int(img.shape[0]/basewidth), int(img.shape[1]/basewidth)))
cv2.imwrite('roi.jpg', ROI2)
#cv2.imshow("canny", canny)
#cv2.imshow("detected", image)
cv2.waitKey(0)