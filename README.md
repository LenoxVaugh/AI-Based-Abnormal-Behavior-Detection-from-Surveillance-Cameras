#  AI-Based Abnormal Behavior Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)

##  Introduction
This project focuses on detecting abnormal human behaviors from surveillance camera footage using Artificial Intelligence. The system analyzes video streams to identify suspicious or unusual activities automatically, reducing the need for constant manual monitoring.

The model processes video frames, extracts **spatial-temporal features**, and classifies behaviors into **Normal** or **Abnormal** categories.

##  Objectives
* **Detect** abnormal behaviors (fighting, falling, etc.) in real-time.
* **Apply** state-of-the-art deep learning techniques for video analysis.
* **Build** an automated surveillance monitoring system.
* **Minimize** human error and manual monitoring effort.

---

##  System Architecture

| Stage | Process | Description |
| :--- | :--- | :--- |
| **1** | **Video Input** | Raw footage from surveillance cameras. |
| **2** | **Frame Extraction** | Breaking video into sequences of images. |
| **3** | **Preprocessing** | Resizing, normalization, and noise reduction. |
| **4** | **Feature Extraction** | Capturing spatial (CNN) and temporal (LSTM) features. |
| **5** | **Classification** | Deep Learning model determines the behavior type. |
| **6** | **Output** | Visual alerts and prediction results. |

---

##  Dataset
The dataset contains surveillance videos categorized into normal and abnormal activities.

### Abnormal Examples:
* 👊 **Fighting** or physical altercations.
* 🏃 **Running** in restricted areas.
* 📉 **Falling** (elderly care or workplace safety).
* 🕵️ **Suspicious movements** (loitering).

### Directory Structure:
```text
dataset/
├── train/
│   ├── normal/
│   └── abnormal/
└── test/
    ├── normal/
    └── abnormal/
```
### Technologies Used
* Language: Python
* Computer Vision: OpenCV
* Deep Learning: TensorFlow / PyTorch
* Data Science: NumPy, Matplotlib

### Installation & Setup
1. Clone the repository
Bash
git clone [https://github.com/LenoxVaughn/abnormal-behavior-detection.git](https://github.com/LenoxVaughn/abnormal-behavior-detection.git)
cd abnormal-behavior-detection
2. Create virtual environment
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
Bash
pip install -r requirements.txt
### Project Structure
```Plaintext
abnormal-behavior-detection/
├── dataset/            # Video data
├── models/             # Saved model weights
├── src/                # Source code
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```
### Usage
Training:
Bash
python src/train.py
Inference:
Bash
python src/predict.py --video input_video.mp4
### Author
AI / Computer Vision Project

GitHub: @LenoxVaughn
