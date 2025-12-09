# Hand Mouse Control

Hand Mouse Control is an interactive computer vision project that allows users to control the mouse cursor using natural hand movements captured through a standard webcam.  
The system replaces traditional mouse input with gesture recognition, offering an intuitive and accessible way to interact with a computer.

This project demonstrates practical experience in computer vision, real-time processing, gesture detection, and human–computer interaction.

---

## About the Project

The application uses MediaPipe’s hand-tracking model to identify and monitor 21 key points on each hand.  
Based on these landmarks, the program interprets gestures and converts them into standard mouse actions:

- The right hand is used for cursor movement.
- The left hand is used for clicking and scrolling.

The goal of this project is to explore alternative input methods, improve accessibility, and showcase real-time processing capabilities in Python.

---

## Features

### Right Hand – Mouse Control
- Smooth and responsive cursor tracking based on the index fingertip.
- Dynamic acceleration depending on movement speed.
- Position smoothing to ensure stable tracking.

### Left Hand – Gesture Input
- Thumb–index gesture triggers left click (press and release).
- A "gun" gesture triggers right click, with a cooldown to prevent accidental repeats.
- Vertical wrist motion is used for scrolling.

### Technical Highlights
- Real-time landmark detection using MediaPipe.
- Optimized frame processing with OpenCV.
- Pixel-perfect mapping of normalized coordinates to screen resolution.
- Modular code structure for easy extension and experimentation.

---

## Requirements

```bash
pip install opencv-python mediapipe pyautogui
How to Run
bash
Copy code
python hand_mouse.py
Press Q at any time to close the application.

How It Works
The webcam feed is processed frame by frame.

MediaPipe detects hand landmarks and returns their normalized coordinates.

The program interprets these coordinates:

Right-hand movements are mapped to screen space.

Left-hand gestures are evaluated using geometric distances between landmarks.

PyAutoGUI simulates mouse actions based on recognized gestures.

Project Structure
text
Copy code
hand-mouse-control/
├── hand_mouse.py     # Main application script
├── README.md         # Documentation
└── requirements.txt  # (optional) Dependency list
Code Entry Point
python
Copy code
if __name__ == "__main__":
    handMouse()
Potential Extensions
Custom gesture recognition system

Calibration interface for gesture sensitivity

Multi-user gesture profiles

Integration with accessibility tools

License
This project is open-source and free to modify.
