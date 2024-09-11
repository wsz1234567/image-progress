import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread("result6.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (255, 0, 0), 2)


x, y, w, h = cv2.boundingRect(contours)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(image)
plt.show()
