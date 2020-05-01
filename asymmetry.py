'''import cv2
import numpy as np

img = cv2.imread('img_seg.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)


#cv2.imshow("thresh", thresh)
#cv2.imshow("Image", image)
cv2.waitKey(0)

height, width = img.shape[:2]
print("measuring infos...")


#divido l'immagine in due immagini, sinistra e destra
height, width = img.shape[:2]
print (img.shape)
# Let's get the starting pixel coordiantes (top left of cropped top)
start_row, start_col = int(0), int(0)
# Let's get the ending pixel coordinates (bottom right of cropped top)
end_row, end_col = int(height), int(width * .5)
cropped_left = img[start_row:end_row , start_col:end_col]
print (start_row, end_row)
print (start_col, end_col)


cv2.waitKey(0)
cv2.destroyAllWindows()

# Let's get the starting pixel coordinates (top left of cropped bottom)
start_row, start_col = int(0), int(width * .5)
# Let's get the ending pixel coordinates (bottom right of cropped bottom)
end_row, end_col = int(height), int(width)
cropped_right = img[start_row:end_row , start_col:end_col]
print (start_row, end_row)
print (start_col, end_col)

#cv2.imshow("Cropped Top", cropped_left)
#cv2.imshow("Cropped Bot", cropped_right)
cv2.waitKey(0)
cv2.destroyAllWindows()

#a partire dalle due immagini calcolo se sono simmetriche in base alla percentuale
#di pixel bianchi
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cropped_left = cv2.cvtColor(cropped_left, cv2.COLOR_BGR2GRAY)
cropped_right = cv2.cvtColor(cropped_right, cv2.COLOR_BGR2GRAY)

total_white_pixels = cv2.countNonZero(img)
left_white_pixels = cv2.countNonZero(cropped_left)
right_white_pixels = cv2.countNonZero(cropped_right)
print(total_white_pixels, left_white_pixels, right_white_pixels)

left_percent = left_white_pixels/total_white_pixels
right_percent = right_white_pixels/total_white_pixels
print(np.abs(left_percent - right_percent))

if ((np.abs(left_percent - right_percent)) > 0.2):
    asymmetry = 1
else:
    asymmetry = 0

print(asymmetry)
'''
import numpy as np
import skimage
from skimage import color, filters
import cv2



def checkOverlap(shape1, shape2):
    #Find the accuracy of symmetry
    all_pixels = 0.
    correct = 0.
    wrong = 0.

    for i in range(shape1.shape[0]):
        for j in range(shape1.shape[1]):

            curr_pixel1 = (shape1[i][j])
            curr_pixel2 = (shape2[i][j])

            if(curr_pixel1 or curr_pixel2):
                all_pixels += 1
                if(curr_pixel1 and curr_pixel2):
                    correct += 1
                else:
                    wrong += 1

    return correct, wrong, all_pixels

def get_asymmetry_index(img):
    imgcolor = img
    img = skimage.color.rgb2gray(img)
    x = []
    y = []

    #DOING FOR THE FIRST TIME TO GET LEFT AND TOP
    top = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    left = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    #Don't want to take the white parts
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 1:
                x.append(j)
                y.append(i)

    # Trying to find center, x-intercept and y-intercept
    centroid = (sum(x) / len(x), sum(y) / len(y))
    y_axis = centroid[1]
    x_axis = centroid[0]

    #Performing splitting for top/down images
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] < 0.95):
                if(i < y_axis):
                    top[i][j] = True

    #Performing splitting for left/right images
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] < 0.95):
                if (j < x_axis):
                    left[i][j] = True


    #DOING FOR FLIP UP/DOWN TO GET THE DOWN PART
    flipped_ud = np.flipud(img)

    bottom = np.zeros((flipped_ud.shape[0], flipped_ud.shape[1]), dtype=bool)

    #Performing splitting for top/down images
    for i in range(flipped_ud.shape[0]):
        for j in range(flipped_ud.shape[1]):
            if(flipped_ud[i][j] < 0.95):
                if(i < y_axis):
                    bottom[i][j] = True

    #DOING FOR FLIP UP/DOWN TO GET THE DOWN PART
    flipped_lr = np.fliplr(img)

    right = np.zeros((flipped_lr.shape[0], flipped_lr.shape[1]), dtype=bool)

    #Performing splitting for top/down images
    for i in range(flipped_lr.shape[0]):
        for j in range(flipped_lr.shape[1]):
            if(flipped_lr[i][j] < 0.95):
                if(j < x_axis):
                    right[i][j] = True


    correct_TB, wrong_TB, all_TB = checkOverlap(top, bottom)
    correct_LR, wrong_LR, all_LR = checkOverlap(left, right)

    return 1 - sum([correct_TB / all_TB, correct_TB / all_LR])/2.



img = cv2.imread('imageNotReduced.jpeg')
asi = get_asymmetry_index(img)

print(asi)