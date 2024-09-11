import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from pylsd.lsd import lsd
import time

def calculate_intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    # 计算直线1的笛卡尔坐标参数
    x1 = rho1 * np.cos(theta1)
    y1 = rho1 * np.sin(theta1)
    x2 = x1 + 1000 * (-np.sin(theta1))
    y2 = y1 + 1000 * (np.cos(theta1))

    # 计算直线2的笛卡尔坐标参数
    x3 = rho2 * np.cos(theta2)
    y3 = rho2 * np.sin(theta2)
    x4 = x3 + 1000 * (-np.sin(theta2))
    y4 = y3 + 1000 * (np.cos(theta2))

    # 计算直线1和直线2的交点坐标
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(det) < 1e-6:
        return None  # 两直线平行或重合，不存在交点

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return x, y

def Houghfindlines(line_data, theta_rho, num):
    # 初始化K-Means模型，指定要分为二类
    kmeans = KMeans(n_clusters=num)
    # 当遇到小目标的跑道采用聚类4个值，只取其中两个category进行挑出进行标注
    # 进行聚类
    kmeans.fit(theta_rho)

    # 获取每个值所属的类别标签
    labels = kmeans.labels_

    # 创建四个空列表，用于存放每个类别的值
    category1 = []
    category2 = []


    # 将值根据类别标签分配到四个列表中
    for i, label in enumerate(labels):
        if label == 0:
            category1.append(theta_rho[i][0])
        elif label == 1:
            category2.append(theta_rho[i][0])


    category=[category1[0], category2[0]]

    return_lines = []
    for line in line_data:
        rho1, theta1 = line[0]
        if theta1 in category:
            category.remove(theta1)
            return_lines.append(line)

    return return_lines
    # 打印每个类别的值
def pointfind(point_data, x_data, num):
    # 初始化K-Means模型，指定要分为二类
    kmeans = KMeans(n_clusters=num)
    # 进行聚类
    kmeans.fit(x_data)
    # 获取每个值所属的类别标签
    labels = kmeans.labels_
    # 创建四个空列表，用于存放每个类别的值
    category1 = []
    category2 = []
    category3 = []
    category4 = []
    # 将值根据类别标签分配到四个列表中
    for i, label in enumerate(labels):
        if label == 0:
            category1.append(x_data[i][0])
        elif label == 1:
            category2.append(x_data[i][0])
        elif label == 2:
            category3.append(x_data[i][0])
        elif label == 3:
            category4.append(x_data[i][0])

    category=[category1[0], category2[0], category3[0], category4[0]]

    return_points = []
    for point in point_data:
        x1, y1 = point
        if x1 in category:
            category.remove(x1)
            return_points.append(point)

    return return_points

def find_contour(contours):
    w_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        w_data.append(w)
    wmax = max(w_data)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w == wmax:
            fin_contour = contour
            break
    return fin_contour

# 最小包络四边形算法
def quadframe(contours_canvas,gray):
    ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(find_contour(contours))
    cv2.rectangle(contours_canvas, (x, y), (x+w, y+h), 1, 3)
    # cv2.rectangle(image, (x, y), (x + w, y + h), 255, 1)
    cv2.imshow('contours_canvas', contours_canvas)
    # cv2.imshow('canvas2', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('C:/Users/LENOVO/Desktop/result.png', contours_canvas)
    # 边界点拟合
    cnt_len = cv2.arcLength(contours[0], True)
    cnt = cv2.approxPolyDP(contours[0], 0.02 * cnt_len, True)
    down_cnt = sorted(cnt, key=lambda point: point[0][1], reverse=True)[:2]

    return down_cnt

start_time = time.time()
# 读取图像并转为灰度图像 初始化图像参数
image = cv2.imread('result10.png')
image1 = cv2.imread('zhuanli.png')
image2 = cv2.imread('result10.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lines_canvas = np.zeros(image.shape, dtype=np.uint8)
lines_canvas = cv2.cvtColor(lines_canvas, cv2.COLOR_BGR2GRAY)
contours_canvas = np.zeros(image.shape, dtype=np.uint8)
contours_canvas = cv2.cvtColor(contours_canvas, cv2.COLOR_BGR2GRAY)
down_cnt = quadframe(contours_canvas, gray)


# 边缘检测
edges = cv2.Canny(gray, 150, 200)
# cv2.imshow('edges',edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 霍夫变换检测直线
lines = cv2.HoughLines(edges, 2, np.pi/180, threshold=60)


theta_rho = []
for line in lines:
    rho1, theta1 = line[0]
    theta_rho.append(theta1)

reshape_theta = np.array(theta_rho).reshape(-1, 1)
standard_lines = Houghfindlines(lines, reshape_theta, 2)


# 计算直线方程并求解交点坐标

intersections = []
for i in range(len(standard_lines)):
    for j in range(i+1, len(standard_lines)):
        intersection = calculate_intersection(standard_lines[i], standard_lines[j])
        if intersection == None:
            continue
        intersections.append(intersection)

v, w = intersection
v = int(v)
w = int(w)
# 画直线
for line in standard_lines:
    rho, theta = line[0]
    print(theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))

    # (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
    cv2.line(lines_canvas, (x1, y1), (x2, y2), 1, 1)
    cv2.line(image, (x1, y1), (x2, y2), 255, 3)

cv2.imshow('canvas2', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 求角平分线
# print(standard_lines[0])
'''
rho1, theta1 = standard_lines[0][0]
rho2, theta2 = standard_lines[1][0]

theta = (theta1+theta2) / 2
a = np.cos(theta)
b = np.sin(theta)

x1 = int(v + 2000 * (-b))
y1 = int(w + 2000 * (a))

# (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
cv2.line(image, (v, w), (x1, y1), 255, 3)


cv2.imshow('canvas2', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('C:/Users/LENOVO/Desktop/result.png', image)

'''

'''
for intersection in intersections:
    w, v = intersection
    cv2.circle(image, (int(w), int(v)), 5, (0, 255, 0), -1)

print(intersections)
'''

corners = []
corners_y, corners_x = np.where(contours_canvas + lines_canvas == 2)
corners += zip(corners_x, corners_y)
# corners = sorted(corners, key=lambda theta: math.tan(theta), reverse=True)[:2]
# corners = sorted(corners, key=lambda x: x[0], reverse=False)[:4]
# 这个是4个点

x_data = []
for corner in corners:
    x1, y1 = corner
    x_data.append(x1)

x_data = np.array(x_data).reshape(-1, 1)
corners = pointfind(corners, x_data, 4)

print(corners)

down_corners = sorted(corners, key=lambda y: y[1], reverse=True)[:2]
# 抽取最下面两个点

downx1, downy1 = down_corners[0]
downx2, downy2 = down_corners[1]

v1 = int((downx1 + downx2)/2.0)
w1 = int((downy2 + downy2)/2.0)

k = (w1 - w) / (v1 - v)
b = w1 - k * v1


x = 2000  # 从 -10 到 10
x1 = -2000
# 计算对应的 y 值
y = int(x * k + b)
y1 = int(x1 * k + b)


(x, y), (x1, y1) = sorted(((x, y), (x1, y1)), key=lambda pt: pt[0])
cv2.line(image, (x, y), (x1, y1),  255, 3)

'''
cv2.imshow('canvas2', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('C:/Users/LENOVO/Desktop/result.png', image)
'''

for corner in corners:
    x, y = corner
    cv2.circle(image, (int(x), int(y)), 15, (0, 255, 0), -1)
# 显示结果图像
cv2.imshow('Intersections', image)
cv2.imwrite('C:/Users/LENOVO/Desktop/result.png', image)
'''
for point_cnt in down_cnt:
    x, y = point_cnt
    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
'''

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
# cv2.imwrite('C:/Users/LENOVO/Desktop/result.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 程序作用：输入图片，通过霍夫变换得到交点坐标值，并在图片当中标出

