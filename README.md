# AI-Powered Campus Inspection Prototype

An AI-powered campus inspection prototype integrating ESP32-CAM, YOLO and OpenCV for real-time visual monitoring of college campus surroundings.

The project focuses on detecting selected waste categories, potholes and manholes through live video analysis, with the broader objective of identifying campus cleanliness issues, infrastructure damage, cracks and other potential vulnerabilities.

## Project Overview

The prototype combines computer vision and embedded hardware to explore automated campus monitoring. It uses an ESP32-CAM for visual data acquisition and YOLO-based object detection for identifying selected categories of interest.

The current implementation is a functional prototype with limited detection classes and scope for further improvement.

## Key Features

- Real-time visual monitoring
- YOLO-based object detection
- Selected waste category detection
- Pothole and manhole detection
- OpenCV-based image and video processing
- ESP32-CAM integration

## Technologies Used

- Python
- YOLOv8 / YOLO11
- OpenCV
- Ultralytics
- ESP32-CAM

## Project Structure

```text
Campus-Inspection-Prototype/
│
├── main.py
├── road_hazard_detector.py
├── run_detector.bat
├── best.pt
├── yolov8n.pt
│
├── weights/
│   ├── taco_yolo11s.pt
│   └── waste_yolov8n.pt
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ranj1t-Ga1kwad/Campus-Inspection-Prototype.git
```

### 2. Navigate to the project directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

## Hardware Components

- ESP32-CAM
- Supporting power supply and connectivity components

## Project Status

Prototype completed with selected object detection capabilities. Further improvements in detection accuracy, classification and campus infrastructure monitoring are possible.

## Future Scope

- Improved waste classification
- Crack and infrastructure damage detection
- GPS-based issue localization
- Automated campus inspection reports
- Expanded detection categories

## Disclaimer

This project is an experimental prototype developed for educational and demonstration purposes. It is not intended to replace professional infrastructure inspection systems.
