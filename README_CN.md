简体中文 | [English](README.md)

# Rope Skipping Counter

## 简介

Rope Skipping Counter 是一个基于 OpenCV 和 MediaPipe 的视频处理应用，用于实时计数跳绳次数。通过检测用户的姿态变化，本应用可以准确识别跳绳动作并进行计数。

![demo](demo.gif)

## 功能

- 实时检测人体姿态，并提取髋关节和肩膀的位置。
- 计算并可视化中心Y轴，监测身体上下波动。
- 基于波动幅度计数跳绳次数。
- 实时绘制跳绳计数和中心点位置。

## 技术栈

- Python
- OpenCV
- MediaPipe
- Matplotlib
- NumPy

## 使用说明

### 环境准备

确保你已经安装了以下库：

```bash
pip install opencv-python mediapipe matplotlib numpy
```

### 代码运行

1. 将待处理视频文件命名为 `demo.mp4`，并放在代码文件夹内。
2. 运行代码：

   ```bash
   python rope_skipping_counter.py
   ```

3. 视频处理完成后，输出结果会保存为 `demo_output.mp4`。

### 参数设置

可以根据需要调整以下参数：

- `buffer_time`: 缓冲区时间，默认为 50。
- `dy_ratio`: 移动幅度阈值，默认为 0.3。
- `up_ratio`: 上升阈值，默认为 0.55。
- `down_ratio`: 下降阈值，默认为 0.35。
- `flag_low` 和 `flag_high`: 控制翻转标志的阈值。

### 结果展示

在运行过程中，程序会在视频帧上实时显示跳绳计数和中心点位置。完成后，可以在输出视频中查看跳绳过程的可视化效果。

## 感谢

* [pushup_counter](https://github.com/hacklavya/pushup_counter)