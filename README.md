# ADAS YOLOv8 - Autonomous Driving Assistant Prototype

Prototype Python project for an ADAS-style system using YOLOv8, OpenCV and OpenVINO.

## Features

- Object detection with YOLOv8
- Collision risk estimation
- Simulated emergency braking
- Lane detection
- Lane departure warning
- Real-time HUD overlay
- Intel Iris Xe / OpenVINO support

## Warning

This is an educational prototype.  
It must not be connected to a real vehicle control system.

## Run

```bash
python -m adas.main --config configs/live_showcase_iris_xe.yaml --source data/dashcam1.mp4 --lite-ui