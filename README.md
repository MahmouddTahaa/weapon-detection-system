# (Weapon Detection System Documentation)[https://huggingface.co/spaces/mtaha48/Weapon-Detection]

## Overview
The Weapon Detection System is a real-time application that uses the YOLO v3 model to detect weapons in video streams from webcams or uploaded videos. It features a modular design with separate model and app components, built using Python, OpenCV, and Gradio.

## Features
- Real-time weapon detection via webcam.
- Processing of prerecorded videos with weapon detection and output generation.
- User-friendly interface via Gradio for easy interaction.
- Debug logging for detection analysis.

## System Architecture
- **weapon_detection_model.py**: Contains the YOLO v3 model logic, including frame processing and weapon detection.
- **app.py**: Manages the Gradio interface, webcam streaming, and video processing, importing detection functionality from the model file.

## Dependencies
- Python 3.x
- `gradio`
- `opencv-python`
- `numpy`

Install via:
```bash
pip install gradio opencv-python numpy
```

## Setup
1. Place `app.py` and `weapon_detection_model.py` in your project directory.
2. Ensure a `weapon_detection` subdirectory contains:
   - `yolov3_training_2000.weights`
   - `yolov3_testing.cfg`
3. Run the application:
   ```bash
   python app.py
   ```

## Usage
- **Webcam Detection**: Click "Start Webcam" to begin real-time detection; click "Stop Webcam" to end.
- **Video Upload**: Upload a video file and click "Process Video" to generate a processed video with detections.

## Training Dataset
The YOLO v3 model was trained on the COCO8 dataset (8 images from COCO train 2017), likely fine-tuned for the "Weapon" class.

## Practical Applications
- Security in public spaces (airports, schools).
- Safety at events (concerts, sports games).
- Integration with surveillance networks.
- Real-time support for law enforcement.

## Troubleshooting
- **Webcam Issues**: Ensure no other app uses the camera; test with `cv2.VideoCapture(0)`.
- **Misclassifications**: Check console logs for detection details; consider retraining or fine-tuning.
- **Performance**: Optimize with GPU support or reduce resolution if slow.

## Contact
For issues or contributions, contact me.
