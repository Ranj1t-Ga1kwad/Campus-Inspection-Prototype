
# Model Weights

This folder contains the pre-trained YOLO model weight files used in the Campus Inspection Prototype.

These model files are required for running the project's object detection functionality.

## Model Files

- `taco_yolo11s.pt` – YOLO model weights for waste detection.
- `waste_yolov8n.pt` – YOLO model weights for waste detection.

## Usage

The model files are loaded by the Python scripts during detection. Ensure that the required model files are present in this folder and that their file paths are correctly configured in the source code.

**Note:** These are model weight files, not training datasets or source code.
