"""Control the mouse with hand gestures using MediaPipe Tasks 1.x.

Right hand: move the cursor with the index finger.
Left hand: pinch thumb and index for left-click/drag, make the gun gesture for
right-click, and move the wrist vertically to scroll.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp
import pyautogui


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (SCRIPT_DIR / "hand_landmarker.task", SCRIPT_DIR.parent / "hand_landmarker.task")
HAND_MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.is_file()), MODEL_CANDIDATES[0])
pyautogui.FAILSAFE = True

# The Tasks API exposes landmarks but not the legacy ``mp.solutions`` drawer.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def require_model() -> None:
    """Raise a clear error when a required Tasks model is unavailable."""
    if not HAND_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing MediaPipe model file: {HAND_MODEL_PATH.name}. "
            "Download it before running this script."
        )


def draw_hand(frame, landmarks) -> None:
    """Draw the 21 normalized hand landmarks and their connections."""
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)


def distance(first, second) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


@dataclass
class MouseState:
    previous_right_x: float | None = None
    previous_right_y: float | None = None
    previous_left_wrist_y: float | None = None
    previous_left_wrist_time: float | None = None
    last_right_click_time: float = 0.0
    last_scroll_time: float = 0.0
    left_button_down: bool = False


def move_cursor(landmarks, state: MouseState, screen_width: int, screen_height: int, now: float) -> None:
    index_tip = landmarks[8]
    if state.previous_right_x is None:
        state.previous_right_x = index_tip.x
        state.previous_right_y = index_tip.y
        return

    smoothed_x = state.previous_right_x * 0.7 + index_tip.x * 0.3
    smoothed_y = state.previous_right_y * 0.7 + index_tip.y * 0.3
    cursor_x = max(0, min(int(smoothed_x * (screen_width - 1)), screen_width - 1))
    cursor_y = max(0, min(int(smoothed_y * (screen_height - 1)), screen_height - 1))
    pyautogui.moveTo(cursor_x, cursor_y, duration=0)

    state.previous_right_x = index_tip.x
    state.previous_right_y = index_tip.y


def handle_left_hand(landmarks, state: MouseState, now: float) -> None:
    thumb, index, middle, wrist = landmarks[4], landmarks[8], landmarks[12], landmarks[0]
    pinch_gesture = distance(thumb, index) < 0.05
    gun_gesture = distance(thumb, middle) > 0.1 and distance(index, middle) < 0.07

    if gun_gesture and now - state.last_right_click_time >= 0.33:
        pyautogui.rightClick()
        state.last_right_click_time = now

    if pinch_gesture and not state.left_button_down:
        pyautogui.mouseDown()
        state.left_button_down = True
    elif not pinch_gesture and state.left_button_down:
        pyautogui.mouseUp()
        state.left_button_down = False

    if state.previous_left_wrist_y is not None:
        elapsed = max(now - state.previous_left_wrist_time, 0.0001)
        wrist_speed_y = (wrist.y - state.previous_left_wrist_y) / elapsed
        if now - state.last_scroll_time >= 0.05:
            if wrist_speed_y > 1:
                pyautogui.scroll(-80)
                state.last_scroll_time = now
            elif wrist_speed_y < -1:
                pyautogui.scroll(80)
                state.last_scroll_time = now

    state.previous_left_wrist_y = wrist.y
    state.previous_left_wrist_time = now


def run_hand_mouse() -> None:
    require_model()

    vision = mp.tasks.vision
    base_options = mp.tasks.BaseOptions
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam 0.")

    state = MouseState()
    screen_width, screen_height = pyautogui.size()
    try:
        with vision.HandLandmarker.create_from_options(hand_options) as hands:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                hand_result = hands.detect_for_video(image, timestamp_ms)
                now = time.monotonic()

                for hand_index, (landmarks, handedness) in enumerate(
                    zip(hand_result.hand_landmarks, hand_result.handedness)
                ):
                    draw_hand(frame, landmarks)
                    label = handedness[0].category_name
                    if label == "Right":
                        move_cursor(landmarks, state, screen_width, screen_height, now)
                    elif label == "Left":
                        handle_left_hand(landmarks, state, now)

                    cv2.putText(
                        frame, label, (20, 50 + 30 * hand_index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                    )

                cv2.imshow("HAND MOUSE CONTROL", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except pyautogui.FailSafeException:
        print("Mouse control stopped by PyAutoGUI fail-safe (cursor in a screen corner).")
    finally:
        if state.left_button_down:
            pyautogui.mouseUp()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_mouse()
