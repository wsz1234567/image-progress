# 标出图像轮廓线

import cv2
import numpy as np

'''
        created on Tues jan 08:28:51 2018
        @author: ren_dong

        contour detection
        cv2.findContours()    寻找轮廓
        cv2.drawContours()    绘制轮廓
'''
# 加载图像img
img = cv2.imread('result4.png')


'''
灰度化处理,注意必须调用cv2.cvtColor(),
如果直接使用cv2.imread('1.jpg',0),会提示图像深度不对,不符合cv2.CV_8U
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 调用cv2.threshold()进行简单阈值化,由灰度图像得到二值化图像
#  输入图像必须为单通道8位或32位浮点型
ret, thresh = cv2.threshold(gray, 20, 255, 0)


# 调用cv2.findContours()寻找轮廓,返回修改后的图像,轮廓以及他们的层次
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('image', image)

mask1= cv2.drawContours(img, contours,-1,(128,128,255),10)
cv2.imshow('mask1', mask1)
'''
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 如果逼近后的轮廓有四个顶点，则认为是四边形
    print(approx)
        # 提取四边形的角点坐标
    corner1 = tuple(approx[0][0])
    corner2 = tuple(approx[1][0])
    corner3 = tuple(approx[2][0])
    corner4 = tuple(approx[3][0])
    print(corner4)
    # 在图像上标记角点
    cv2.circle(img, corner1, 20, (0, 0, 255), -1)
    cv2.circle(img, corner2, 20, (0, 0, 255), -1)
    cv2.circle(img, corner3, 20, (0, 0, 255), -1)
    cv2.circle(img, corner4, 20, (0, 0, 255), -1)

# 显示标记了角点的图像
cv2.imshow("Image with Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

cv2.imwrite("contour2.jpg",mask1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''print('contours[0]:', contours[0])
print('len(contours):', len(contours))
print('hierarchy.shape:', hierarchy.shape)
print('hierarchy:', hierarchy)

# 调用cv2.drawContours()在原图上绘制轮廓
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('contours', img)

cv2.waitKey()
cv2.destroyAllWindows()
'''