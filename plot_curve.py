import datetime
import matplotlib.pyplot as plt

import numpy as np

# 生成一些随机数据（示例中使用正弦函数）
x = np.linspace(0, 2 * np.pi, 100)  # 创建一个包含100个点的x值数组
y = np.sin(x)  # 使用正弦函数生成相应的y值

# 创建图表
plt.figure(figsize=(8, 6))  # 设置图表的大小

# 绘制折线图
plt.plot(x, y, label='Sine Wave', color='blue', linewidth=2)

# 添加标题和标签
plt.title('Sine Wave')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

# 添加图例
plt.legend()

# 显示图表
plt.show()

