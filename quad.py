import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread("result5.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 20, 255, 0)

ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


'''
rect = cv2.minAreaRect(contours[2])
#box = cv2.cv.BoxPoints(rect)  # for OpenCV 2.x
box = cv2.boxPoints(rect)      # for OpenCV 3.x
box = np.int0(box)
cv2.drawContours(image, [box], 0, (255, 255, 0), 2)
'''

cnt_len = cv2.arcLength(contours[2], True)
cnt = cv2.approxPolyDP(contours[2], 0.02*cnt_len, True)
down_cnt = sorted(cnt, key=lambda point : point[0][1], reverse=True)[:2]
print(down_cnt)
if len(cnt) == 4:
    cv2.drawContours(image, [cnt], -1, (255, 255, 0), 3)
plt.imshow(image)
plt.show()
