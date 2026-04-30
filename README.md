# Low-Light Image Enhancement with Lane Detection

This project presents a hardware-aware real-time low-light image enhancement framework integrated with lane detection for night driving scenarios.

The system is designed to improve visibility in low-light conditions and enhance downstream perception tasks such as lane detection, while maintaining computational efficiency suitable for embedded deployment.

---

## 🚀 Features

- Adaptive low-light image enhancement (HSV-domain)
- Illumination estimation and brightness correction
- Wavelet-based detail preservation
- Lane detection using Canny + Hough Transform
- Real-time capable (~9–10 FPS on embedded hardware)
- Deployment on PYNQ-Z2 platform

---

## 📁 Repository Structure
repo/
│
├── code/
│ ├── enhancement_only.py
│ ├── lane_detection_only.py
│ ├── pc_pipeline.py
│ └── pynq_pipeline.py
│
├── dataset/
│ ├── enhancement_only/
│ └── lane_detection_only/
│
├── results/
│ ├── enhancement_only/
│ ├── lane_detection_only/
│ └── lane_detection_only/bad/
│
├── demo/
│ └── lane_detection_test_video/
│
└── README.md


---

## 🧠 Code Description

### 1. Enhancement Only
- File: `enhancement_only.py`
- Performs low-light image enhancement on dataset images

### 2. Lane Detection Only
- File: `lane_detection_only.py`
- Applies lane detection pipeline without enhancement

### 3. PC Pipeline
- File: `pc_pipeline.py`
- Real-time pipeline for laptop/webcam
- Enhancement + lane detection combined

### 4. PYNQ Pipeline
- File: `pynq_pipeline.py`
- Optimized version for embedded execution on PYNQ-Z2

---

## 📊 Dataset

The dataset consists of:
- Low-light road images for enhancement testing
- Separate lane detection dataset

These datasets are used for:
- Evaluation
- Visualization
- Report results

---

## 📈 Results

The results demonstrate:
- Improved brightness and contrast in low-light conditions
- Better edge visibility
- More stable and accurate lane detection

Some challenging cases are included in: results/lane_detection_only/bad/

---

## 🎥 Demo

A video demonstration of the system on real road scenarios is included in: lane_detection_test_video/


---

⚠️ Important Notes
Input and output paths in the code may need to be adjusted based on your system
The code is designed for flexibility, so directories should be modified as required
The repository includes experimental outputs and intermediate results for completeness
📦 Full Project Resources

The complete BTP work, including:

All experimental models (both successful and failed)
Full datasets
Additional outputs and intermediate results

is available at:

👉 https://drive.google.com/drive/folders/1rgADDbfVgeXslO2C90uwnUoJUyVfa9ZD?usp=sharing

🎯 Highlights
Real-time performance (~9–10 FPS)
Improved lane detection under low-light conditions
Lightweight design suitable for embedded systems
Integration of enhancement with downstream vision tasks

👨‍💻 Author
Praneel Vipul Vania, Aditya Sable
Dhirubhai Ambani University
