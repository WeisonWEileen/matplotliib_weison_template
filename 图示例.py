#!/usr/bin/env python
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
import numpy as np

# 示例数据（替换为您的实际数据）
time = np.arange(0, 300, 10)  # 实验时间间隔
num_subjects = 5
num_trials = 5

# 生成随机示例数据（每个被试五次测量）
grip_force_data = np.random.normal(20, 1, (num_subjects, num_trials, len(time)))

# 生成理想的肘关节角度数据（sin曲线在0-90之间变化）
ideal_angle_data = 35 * (1 + np.sin(2 * np.pi * time / 90))

# 添加随机误差以模拟实际数据
error_amplitude = 5  # 误差振幅
angle_data = ideal_angle_data + np.random.uniform(-error_amplitude, error_amplitude, size=ideal_angle_data.shape)

# 计算均值和标准差
mean_grip_force = np.mean(grip_force_data, axis=(0, 1))
angle_data_mean = np.mean(angle_data, axis=0)
std_grip_force = np.std(grip_force_data, axis=(0, 1))
std_angle_data = np.std(angle_data, axis=0)

# 创建带有双坐标轴的曲线图
fig, ax1 = plt.subplots(figsize=(6, 4))

# 左侧坐标轴 - 握力均值曲线和方差带
ax1.set_xlabel('Time (s)', family='Times New Roman', color="black", size=16)
ax1.set_ylabel('Force (N)', family='Times New Roman', color="black", size=16)  # 蓝色
ax1.plot(time, mean_grip_force, label='Mean Grip Force', color='darkorange')
ax1.fill_between(time, mean_grip_force - std_grip_force, mean_grip_force + std_grip_force, alpha=0.2, color='darkorange')
ax1.tick_params(axis='y', labelcolor='darkorange')
ax1.grid(True)

# 调整左侧坐标轴的坐标刻度
ax1.set_yticks(np.arange(0, 40, 10))
ax1.set_ylim(0, 40)

# 右侧坐标轴 - 肘关节角度均值曲线和方差带
ax2 = ax1.twinx()  # 使用twinx创建右侧坐标轴
ax2.set_ylabel('Elbow Angle (degrees)', family='Times New Roman',  size=16, color='black')  # 橙色
ax2.plot(time, angle_data, label='Mean Elbow Angle', color='lightgreen')
ax2.fill_between(time, angle_data - std_angle_data, angle_data + std_angle_data, alpha=0.2, color='lightgreen')
ax2.tick_params(axis='y', labelcolor='limegreen')

# 调整右侧坐标轴的坐标刻度
ax2.set_yticks(np.arange(-30, 100, 30))
ax2.set_ylim(-30, 100)

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper right',prop={'family': 'Times New Roman', 'size': 12})

# 标题
plt.title('Multitasks')

output_path = 'Multitasks'
plt.savefig(output_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.0)


plt.show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# 示例数据（替换为您的实际数据）
time = np.arange(0, 300, 10)  # 实验时间间隔
num_subjects = 5
num_trials = 5

# 生成随机示例数据（每个被试五次测量）
grip_force_data = np.random.normal(20, 1, (num_subjects, num_trials, len(time)))

# 生成理想的肘关节角度数据（分段函数）
angle_data = np.zeros(len(time))
segment_duration = len(time) // 3  # Split the time into three segments

# First segment: Linear increase to 50
angle_data[:segment_duration] = np.linspace(0, 50, segment_duration)

# Second segment: Linear decrease from 50 to 100
angle_data[segment_duration:2*segment_duration] = np.linspace(50, 100, segment_duration)

# Third segment: Constant at 100
angle_data[2*segment_duration:] = 100

# 添加随机误差以模拟实际数据
error_amplitude =2  # 误差振幅
angle_data += np.random.uniform(-error_amplitude, error_amplitude, size=len(time))

# 计算均值和标准差
mean_grip_force = np.mean(grip_force_data, axis=(0, 1))
std_grip_force = np.std(grip_force_data, axis=(0, 1))
std_angle_data = np.std(angle_data)

# 创建带有双坐标轴的曲线图
fig, ax1 = plt.subplots(figsize=(6, 4))

# 左侧坐标轴 - 握力均值曲线和方差带
ax1.set_xlabel('Time (s)', family='Times New Roman', color="black", size=16)
ax1.set_ylabel('Force (N)', family='Times New Roman', color="black", size=16)  # 蓝色
ax1.plot(time, mean_grip_force, label='Mean Grip Force', color='darkorange')
ax1.fill_between(time, mean_grip_force - std_grip_force, mean_grip_force + std_grip_force, alpha=0.2, color='darkorange')
ax1.tick_params(axis='y', labelcolor='darkorange')
ax1.grid(True)

# 调整左侧坐标轴的坐标刻度
ax1.set_yticks(np.arange(0, 40, 10))
ax1.set_ylim(0, 40)

# 右侧坐标轴 - 肘关节角度均值曲线和方差带
ax2 = ax1.twinx()  # 使用twinx创建右侧坐标轴
ax2.set_ylabel('Shoulder Angle (degrees)', family='Times New Roman', size=16, color='black')  # 橙色
ax2.plot(time, angle_data, label='Mean Elbow Angle', color='lightgreen')
ax2.fill_between(time, angle_data - std_angle_data, angle_data + std_angle_data, alpha=0.2, color='lightgreen')
ax2.tick_params(axis='y', labelcolor='limegreen')

# 调整右侧坐标轴的坐标刻度
ax2.set_yticks(np.arange(0, 150, 25))
ax2.set_ylim(0, 150)

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper right', prop={'family': 'Times New Roman', 'size': 12})

# 标题
plt.title('Multitasks')

output_path = 'Multitasks_shoulder'
plt.savefig(output_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.0)

plt.show()


# In[ ]:




