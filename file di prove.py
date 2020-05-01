import os
import cv2

project_root = os.path.dirname(os.path.dirname(__file__)) + "/MelanomaDetection/imgs/IMD409.bmp"
print(project_root)
img = cv2.imread(project_root)

cv2.imshow('dfe', img)
cv2.waitKey(0)