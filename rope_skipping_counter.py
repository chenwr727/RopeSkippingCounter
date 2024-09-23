import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np


class BufferList:
    def __init__(self, buffer_time, default_value=0):
        self.buffer = [default_value] * buffer_time

    def push(self, value):
        self.buffer.pop(0)
        self.buffer.append(value)

    def max(self):
        return max(self.buffer)

    def min(self):
        return min(filter(lambda x: x is not None, self.buffer), default=0)

    def smooth_update(self, old_value, new_value, alpha=0.5):
        return alpha * new_value + (1 - alpha) * old_value


def extract_landmarks(results, landmarks_indices, image_width, image_height):
    """提取指定关节的坐标"""
    return [
        (lm.x * image_width, lm.y * image_height)
        for i, lm in enumerate(results.pose_landmarks.landmark)
        if i in landmarks_indices
    ]


def calculate_center_y(hip_points, shoulder_points):
    """计算中心Y轴及肩-臀垂直距离"""
    cy_hip = int(np.mean([point[1] for point in hip_points]))
    cy_shoulder = int(np.mean([point[1] for point in shoulder_points]))
    return cy_hip, cy_hip - cy_shoulder


def update_counters(cy, cy_shoulder_hip, cy_max, cy_min, flip_flag, thresholds):
    """更新波动计数逻辑"""
    dy = cy_max - cy_min
    if dy > thresholds["dy_ratio"] * cy_shoulder_hip:
        if (
            cy > cy_max - thresholds["up_ratio"] * dy
            and flip_flag == thresholds["flag_low"]
        ):
            flip_flag = thresholds["flag_high"]
        elif (
            cy < cy_min + thresholds["down_ratio"] * dy
            and flip_flag == thresholds["flag_high"]
        ):
            flip_flag = thresholds["flag_low"]
    return flip_flag


def draw_visualizations(image, cx, cy, count, image_width, image_height):
    """绘制中心点、计数等信息"""
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(
        image,
        "centroid",
        (cx - 25, cy - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    cv2.putText(
        image,
        f"count = {count}",
        (int(image_width * 0.5), int(image_height * 0.2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )


def plot_buffers(buffers, ax):
    """实时绘制缓冲区数据"""
    ax.clear()
    for label, buf in buffers.items():
        ax.plot(buf.buffer, label=label)
    ax.legend(loc="upper right")
    plt.pause(0.1)


# 主代码
file_name = "demo.mp4"

mp_pose = mp.solutions.pose
hip_landmarks = [23, 24]
shoulder_landmarks = [11, 12]

# 阈值设置
thresholds = {
    "buffer_time": 50,  # 缓冲区时间
    "dy_ratio": 0.3,  # 移动幅度阈值
    "up_ratio": 0.55,  # 上升阈值
    "down_ratio": 0.35,  # 下降阈值
    "flag_low": 150,  # 翻转标志低点
    "flag_high": 250,  # 翻转标志高点
}

buffers = {
    "center_y": BufferList(thresholds["buffer_time"]),
    "center_y_up": BufferList(thresholds["buffer_time"]),
    "center_y_down": BufferList(thresholds["buffer_time"]),
    "center_y_flip": BufferList(thresholds["buffer_time"]),
    "center_y_pref_flip": BufferList(thresholds["buffer_time"]),
}

cy_max, cy_min = 100, 100
flip_flag = thresholds["flag_high"]
count = 0

cap = cv2.VideoCapture(file_name)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(
    file_name.replace(".mp4", "_output.mp4"),
    fourcc,
    fps,
    (int(cap.get(3)), int(cap.get(4))),
)

plt.ion()
fig, ax = plt.subplots()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_height, image_width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            hip_points = extract_landmarks(
                results, hip_landmarks, image_width, image_height
            )
            shoulder_points = extract_landmarks(
                results, shoulder_landmarks, image_width, image_height
            )
            cx = int(np.mean([point[0] for point in hip_points]))
            cy, cy_shoulder_hip = calculate_center_y(hip_points, shoulder_points)
        else:
            cx, cy, cy_shoulder_hip = 0, 0, 0

        buffers["center_y"].push(cy)
        cy_max = buffers["center_y"].smooth_update(cy_max, buffers["center_y"].max())
        buffers["center_y_up"].push(cy_max)
        cy_min = buffers["center_y"].smooth_update(cy_min, buffers["center_y"].min())
        buffers["center_y_down"].push(cy_min)

        prev_flip_flag = flip_flag
        flip_flag = update_counters(
            cy, cy_shoulder_hip, cy_max, cy_min, flip_flag, thresholds
        )
        buffers["center_y_flip"].push(flip_flag)
        buffers["center_y_pref_flip"].push(prev_flip_flag)

        if prev_flip_flag < flip_flag:
            count += 1

        draw_visualizations(image, cx, cy, count, image_width, image_height)
        plot_buffers(buffers, ax)

        cv2.imshow("MediaPipe Pose", image)
        out.write(image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
