import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_detection as mp_face
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

def run_hand_tracking():
    cam = cv2.VideoCapture(index = 0)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detector:
        
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
        
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            face_results = face_detector.process(frame_rgb)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            if face_results.detections:
                 for detection in face_results.detections:
                     mp_drawing.draw_detection(frame, detection)
            

            cv2.imshow('HANDS AND FACE DETECTION', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_tracking()
