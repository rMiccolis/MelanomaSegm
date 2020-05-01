import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import os


image_name = "nostra.jpg"


project_root = os.path.dirname(os.path.dirname(__file__)) + "/MelanomaDetection/"
image_source = project_root+ "imgs/"+image_name
no_hair_img = project_root+ "imgs/preprocessed/prep"
save_name = no_hair_img+image_name


print("started...")

src = cv2.imread(image_source)
print(src.shape)
'''

h,w = src.shape[:2]
#se le dimensioni sono troppo grandi, ridimensiona
width = 768
height = 560
dim = (width, height)

# resize image
resized = cv2.resize(src, dim, interpolation=cv2.INTER_NEAREST)
src = resized
cv2.imwrite(image_source, src)
print(f"new shape:{src.shape}")
'''

ker = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(src,-1,ker)
dst1 = cv2.GaussianBlur(dst, (95,95),0,0)
#cv2.imwrite('immagine solo filtrata.jpg', dst1)

# Convert the original image to grayscale
grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )


# Kernel for the morphological filtering
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))

# Perform the blackHat filtering on the grayscale image to find the
# hair countours
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)



# intensify the hair countours in preparation for the inpainting
# algorithm
ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)




# inpaint the original image depending on the mask
dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA)


cv2.imwrite(save_name, dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


grayScale = cv2.cvtColor( dst, cv2.COLOR_RGB2GRAY )

kernel = np.ones((5,5),np.uint8)
#dilation = cv2.dilate(grayScale,kernel,iterations = 2)
closing = cv2.morphologyEx(grayScale, cv2.MORPH_OPEN, kernel)


#cv2.imshow(save_name, closing)
cv2.imwrite(save_name, closing, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
cv2.waitKey(0)







