import cv2
import numpy as np

# 创建一个黑色背景的图像
image = np.zeros((200, 200, 3), dtype=np.uint8)

# 填充红色正方形
color = (42, 42, 165)  # BGR格式：红色
start_point = (0, 0)  # 正方形左上角坐标
end_point = (200, 200)  # 正方形右下角坐标
image = cv2.rectangle(image, start_point, end_point, color, -1)

# 显示图像
cv2.imshow("Red Square", image)
cv2.imwrite('red.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

