English | [简体中文](README_CN.md)

# Rope Skipping Counter

## Introduction

Rope Skipping Counter is a video processing application based on OpenCV and MediaPipe, designed to count rope skipping repetitions in real-time. By detecting changes in the user's posture, this application accurately recognizes skipping actions and keeps count.

![demo](demo.gif)

## Features

- Real-time detection of human pose and extraction of hip and shoulder positions.
- Calculation and visualization of the Y-axis center to monitor vertical body movement.
- Counting rope skipping repetitions based on movement amplitude.
- Real-time drawing of skipping counts and center point positions.

## Tech Stack

- Python
- OpenCV
- MediaPipe
- Matplotlib
- NumPy

## Usage Instructions

### Environment Setup

Make sure you have the following libraries installed:

```bash
pip install opencv-python mediapipe matplotlib numpy
```

### Running the Code

1. Rename the video file to be processed as `demo.mp4` and place it in the code folder.
2. Run the code:

   ```bash
   python rope_skipping_counter.py
   ```

3. Once processing is complete, the output will be saved as `demo_output.mp4`.

### Parameter Settings

You can adjust the following parameters as needed:

- `buffer_time`: Buffer duration, default is 50.
- `dy_ratio`: Movement amplitude threshold, default is 0.3.
- `up_ratio`: Rising threshold, default is 0.55.
- `down_ratio`: Falling threshold, default is 0.35.
- `flag_low` and `flag_high`: Thresholds to control the flip flag.

### Result Display

During the run, the program will display the skipping count and center point position in the video frames. After completion, you can view the visualized skipping process in the output video.

## Thanks

* [pushup_counter](https://github.com/hacklavya/pushup_counter)