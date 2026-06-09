# AI-Based Abnormal Behavior Detection from Surveillance Cameras

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Object%20Detection-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20API-black.svg)

## Overview

This project is an AI-powered surveillance system designed to automatically detect abnormal human behaviors from security camera footage.

The system combines object detection, human tracking, pose estimation, and action recognition to identify suspicious activities such as:

* Fighting
* Falling
* Running
* Abnormal movements
* Other safety-related incidents

The goal is to reduce manual monitoring effort and provide real-time alerts for security and safety applications.

---

## Key Features

* Real-time video processing
* Human detection using YOLOv11
* Multi-object tracking
* Human pose extraction (Skeleton-based representation)
* Action recognition using SlowFast architecture
* Automatic abnormal behavior detection
* Alert generation and logging
* Web-based monitoring dashboard using Flask

---

## System Pipeline

```text
Video Stream
      │
      ▼
YOLOv11 Person Detection
      │
      ▼
Multi-Object Tracking
      │
      ▼
Pose Estimation
      │
      ▼
Skeleton Sequence Generation
      │
      ▼
SlowFast Action Recognition
      │
      ▼
Behavior Classification
      │
      ▼
Alert Generation & Visualization
```

---

## Why YOLO + Pose + SlowFast?

### YOLOv11

Responsible for detecting people in each frame.

### Multi-Object Tracking

Maintains unique IDs for detected individuals across video frames.

### Pose Estimation

Extracts human skeleton keypoints and removes unnecessary background information.

### SlowFast

Uses two pathways:

* Slow Pathway → captures spatial information
* Fast Pathway → captures motion dynamics

This combination improves abnormal behavior recognition performance in surveillance environments.

---

## Technology Stack

### AI & Deep Learning

* PyTorch
* YOLOv11
* SlowFast
* Pose Estimation
* NumPy

### Computer Vision

* OpenCV

### Backend

* Flask
* REST API

### Data Processing

* Pandas
* NumPy

---

## Project Structure

```text
AI-Based-Abnormal-Behavior-Detection/

├── app.py
├── requirements.txt
├── README.md

├── models/
│   ├── yolo/
│   ├── slowfast/
│   └── pose/

├── datasets/
│   ├── train/
│   ├── val/
│   └── test/

├── static/
├── templates/

├── uploads/
├── outputs/

├── utils/
│   ├── preprocessing.py
│   ├── tracking.py
│   ├── pose.py
│   └── inference.py

└── logs/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/LenoxVaugh/AI-Based-Abnormal-Behavior-Detection-from-Surveillance-Cameras.git

cd AI-Based-Abnormal-Behavior-Detection-from-Surveillance-Cameras
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / MacOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start Flask server:

```bash
python app.py
```

Open browser:

```text
http://localhost:5000
```

Upload a surveillance video and monitor abnormal behavior detection results through the web interface.

---

## Scalability Considerations

Current implementation is designed as a Proof of Concept (POC).

For production deployment, the following improvements are recommended:

* Redis / RabbitMQ message queue
* Dedicated AI inference workers
* PostgreSQL for alert metadata
* WebSocket for real-time status updates
* Docker containerization
* TensorRT optimization
* NVIDIA Triton Inference Server
* Multi-camera distributed processing

---

## Future Improvements

* Real-time RTSP camera support
* WebRTC live streaming
* Multi-GPU inference
* Distributed processing pipeline
* Alert notification via Email / Telegram
* Mobile dashboard

---

## Results

The system successfully demonstrates:

* Human detection
* Human tracking
* Pose-based behavior understanding
* Abnormal behavior classification
* Automated surveillance monitoring

---

## Author

**Bùi Vĩnh Lộc**

Artificial Intelligence Engineer

GitHub: https://github.com/LenoxVaugh
