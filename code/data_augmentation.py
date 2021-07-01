import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./cat.jpg")     # defalut: BRG instead of RGB channels
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
cv2.imshow("imag", img)
cv2.waitKey(0)
cv2.destroyAllWindows()