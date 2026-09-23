# Hand Gesture Mouse Control

Control the Windows mouse with a webcam and hand gestures. The application uses
the current MediaPipe Tasks Vision API to detect up to two hands in real time,
then sends mouse commands through PyAutoGUI.

## Features

- Real-time hand landmark overlay for up to two hands.
- Right hand: move the cursor with the index-finger tip.
- Left hand: touch the thumb and index finger together to hold the left mouse
  button; release them to release the button. This supports click and drag.
- Left hand: make the `gun` gesture (thumb away from middle finger, index and
  middle fingers close together) for a right-click.
- Left hand: move the wrist vertically for scrolling.
- Mirrored preview window; press `Q` to close it.

## Requirements

- Windows with a working webcam.
- Python 3.12 or the Python version used by the included virtual environment.
- The dependencies in `requirements.txt`.
- `src/hand_landmarker.task`, the MediaPipe hand-landmarker model. It is already
  included in this project.

Face detection is not used by the current application, so
`face_detector.tflite` is not required.

## Installation

From the project root in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this once for the
current terminal session and activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run

```powershell
python .\src\HandFaceBASE.py
```

The script searches for `hand_landmarker.task` in `src` first, then in the
project root.

## Gesture reference

| Hand | Gesture | Result |
| --- | --- | --- |
| Right | Move index finger | Moves the cursor |
| Left | Thumb tip close to index tip | Left click / drag while held |
| Left | Gun gesture | Right click |
| Left | Fast wrist movement up or down | Scroll |

## Safety

PyAutoGUI's fail-safe is enabled. Moving the cursor to any screen corner stops
mouse control safely. The script also releases a held left mouse button when it
exits.

## Project structure

```text
Hand-Gesture-Mouse-Control/
|-- requirements.txt
`-- src/
    |-- HandFaceBASE.py       # Application entry point
    `-- hand_landmarker.task  # MediaPipe Tasks model
```
