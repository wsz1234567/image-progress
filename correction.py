import cv2
import numpy as np

# 读取图像
image = cv2.imread('red.png')
image1 = cv2.imread('result9.png')

# 图像的高度和宽度
height, width = image1.shape[:2]
to_image = np.zeros((width, height, 3), dtype=np.uint8)


# 定义梯形的四个顶点
pts_original = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])

point = np.array([[100, 0]], dtype=np.float32)
point1 = np.array([[100, 200]], dtype=np.float32)
# 定义矩形的四个顶点（可以自定义）
pts_target = np.float32([(925, 192), (926, 192), (1169, 192), (0, 726)])
# pts_target = np.float32([(972, 162), (1006, 162), (699, 380), (1057, 400)])


# 获取透视变换矩阵
matrix = cv2.getPerspectiveTransform(pts_original, pts_target)

# 进行透视变换
result = cv2.warpPerspective(image, matrix, (width, height))  # 可以自定义输出图像的大小
transformed_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix)
transformed_point1 = cv2.perspectiveTransform(point1.reshape(-1, 1, 2), matrix)
# x = transformed_point[0][0]
# y = transformed_point[0][1]
# print(transformed_point[0][0])
x, y= transformed_point[0][0]
x1, y1 = transformed_point1[0][0]

point_0 = np.array([[x, y]], dtype=np.float32)
point_1 = np.array([[x1, y1]], dtype=np.float32)

inverse_perspective_matrix = np.linalg.inv(matrix)
transformed_points_0 = cv2.perspectiveTransform(point_0.reshape(-1, 1, 2), inverse_perspective_matrix)
transformed_points_1 = cv2.perspectiveTransform(point_1.reshape(-1, 1, 2), inverse_perspective_matrix)

transformed_x, transformed_y = transformed_points_0[0][0]
transformed_x1, transformed_y1 = transformed_points_1[0][0]

cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), -1)
cv2.circle(result, (int(x1), int(y1)), 10, (0, 255, 0), -1)

# 显示原始图像和透视变换后的图像

cv2.circle(image, (100, 0), 12, (0, 255, 0), -1)
cv2.circle(image, (100, 200), 12, (0, 255, 0), -1)



# cv2.imshow('Original Image', image)

cv2.imshow('Perspective Transformation', result)
cv2.imwrite('C:/Users/LENOVO/Desktop/result.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
