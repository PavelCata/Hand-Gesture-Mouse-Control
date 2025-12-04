import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import pyautogui
import time

pyautogui.FAILSAFE = False

last_wrist_y = None
last_wrist_time = None
last_rightclick = 0

def handMouse():
    last_click_time = 0
    click_cooldown = 0.33
    
    cam = cv2.VideoCapture(index = 0)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width, height = pyautogui.size()
    
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                print("Ignore empty camera frame")
                continue

            frame = cv2.flip(frame,1)
            height_cam, width_cam, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
        
            if results.multi_hand_landmarks and results.multi_handedness:
                now = time.time()
                for i, hand in enumerate(results.multi_hand_landmarks):
                    lm = hand.landmark
                    label = results.multi_handedness[i].classification[0].label

                    if label == "Right":

                        def dist(a, b):
                            return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5
                        
                        cx = lm[8].x
                        cy = lm[8].y

                        if 'last_right_x' not in globals():
                            global last_right_x, last_right_y,last_right_time
                            last_right_x, last_right_y = lm[8].x, lm[8].y
                            last_right_time = time.time()

                        

                        dx = cx - last_right_x
                        dy = cy - last_right_y
                        dt = max(now - last_right_time, 0.0001)

                        
                        speed = ((dx*dx + dy*dy)**0.5) / dt

                        base_dpi = 1.2
                        acceleration = min(speed * 1.5, 3.5)
                        sensitivity = base_dpi + acceleration

                        smoothed_x = (last_right_x*0.7 + cx*0.3)
                        smoothed_y = (last_right_y*0.7 + cy*0.3)

                        screen_x = int(smoothed_x * width * sensitivity)
                        screen_y = int(smoothed_y * height * sensitivity)

                        screen_x = max(0, min(screen_x, width))
                        screen_y = max(0, min(screen_y, height))

                        pyautogui.moveTo(screen_x, screen_y, duration=0)

                        last_right_x = cx
                        last_right_y = cy
                        last_right_time = now
                    

                    if label == "Left":

                        def dist(a, b):
                            return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5

                        thumb = lm[4]
                        index = lm[8]
                        middle = lm[12]
                        wrist = lm[0]

                        d = dist(thumb, index)

                        ok_gesture = d < 0.05
                        gun_gesture = dist(thumb, middle) > 0.1 and dist(index, middle) < 0.07
                        
                        global last_rightclick 
                        

                        if gun_gesture:
                            if time.time() - last_rightclick > rightclick_cooldown:
                                pyautogui.rightClick()
                                last_rightclick = time.time()
                        else:
                            if ok_gesture:
                                pyautogui.mouseDown()
                            else:
                                pyautogui.mouseUp()


                        global last_wrist_y, last_wrist_time

                        if last_wrist_y is not None:
                            dy = wrist.y - last_wrist_y
                            dt = max(now - last_wrist_time, 0.0001)
                            speed_y = dy / dt

                            if speed_y > 1:
                                pyautogui.scroll(-80)
                                time.sleep(0.05)
                            
                            if speed_y < -1:
                                pyautogui.scroll(80)
                                time.sleep(0.05)
                        
                        last_wrist_y = wrist.y
                        last_wrist_time = now

                        
                        #Hand visualization for debugging
                        # cx1, cy1 = int(thumb.x * width_cam), int(thumb.y * height_cam)
                        # cx2, cy2 = int(index.x * width_cam), int(index.y * height_cam)
                        # cv2.circle(frame, (cx1, cy1), 12, (0,255,0), -1)
                        # cv2.circle(frame, (cx2, cy2), 12, (0,255,0), -1)
                        # cv2.putText(frame, f"OK={ok_gesture}", (20,300),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    #which hand
                    # cv2.putText(frame, label, (20, 50 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    

            cv2.imshow('HAND MOUSE CONTROL', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    handMouse()
