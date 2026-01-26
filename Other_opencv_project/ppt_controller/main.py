import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import pyautogui
import numpy as np
import time
import os

# --- PREFERENCES ---
# Set to True to see the visual debugging
DEBUG_VIEW = True

# --- CONFIGURATION ---
CAM_WIDTH, CAM_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING = 5
FRAME_REDUCTION = 100
MODEL_PATH = "../../hand_landmarker.task"

# --- INITIALIZATION ---
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        exit()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6, # Matches Hill Climb
    min_hand_presence_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO
)
detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# State Variables
master_active = False # Start in SLEEPING mode
last_toggle_time = 0
toggle_cooldown = 1.0 # 1 second between toggles

prev_x, prev_y = 0, 0
prev_palm_y = 0
swipe_start_x = 0
swipe_threshold = 80
swipe_cooldown = 0.5
last_swipe_time = 0
last_scroll_time = 0
scroll_cooldown = 0.15 # Slightly faster (approx 7 scrolls per second)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 # Match Hill Climb optimization

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16), (13,17), (17,18), (18,19), (19,20), (0,17)]

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, coords[start], coords[end], (255, 255, 255), 1)

def get_finger_states(landmarks, hand_label):
    fingers = []
    # Thumb (Handedness aware, mirrored logic)
    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else: # Left hand
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    
    # Other 4 fingers
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return fingers

print("AI PPT Controller Started (Master Switch Mode)!")
print("Gesture 🤙 (Pinky + Thumb): Toggle ACTIVE/SLEEPING")
print("ACTIVE Mode Gestures:")
print(" - Index Finger Up: Move Pointer / Swipe Pages")
print(" - Hand Open/Fist: Scroll Up/Down")
print("Press 'q' to Quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Use accurate timing from OpenCV
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    results = detector.detect_for_video(mp_image, timestamp)

    status = "SLEEPING" if not master_active else "ACTIVE"
    if results.hand_landmarks:
        for i, hand_lms in enumerate(results.hand_landmarks):
            hand_label = results.handedness[i][0].category_name
            draw_landmarks(frame, hand_lms)
            
            fingers = get_finger_states(hand_lms, hand_label)
            total_fingers = fingers.count(1)

            # Master Toggle Detection: Pinky (4) and Thumb (0) up, others down
            # fingers list: [thumb, index, middle, ring, pinky]
            if fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
                current_time = time.time()
                if current_time - last_toggle_time > toggle_cooldown:
                    master_active = not master_active
                    last_toggle_time = current_time
                    print(f"SWITCH: System is now {'ACTIVE' if master_active else 'SLEEPING'}")

            # Only process if active
            if master_active:
                h, w, _ = frame.shape
                # Pointer tip coords (landmark 8 is Index Tip)
                ix, iy = int(hand_lms[8].x * w), int(hand_lms[8].y * h)

                # 1. Pointer Mode (Index up, Middle down - ignore thumb for pointer)
                if fingers[1] == 1 and fingers[2] == 0:
                    status = "POINTER"
                    # Map coordinates (using index tip 8)
                    x3 = np.interp(ix, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, SCREEN_WIDTH))
                    y3 = np.interp(iy, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, SCREEN_HEIGHT))
                    
                    curr_x = prev_x + (x3 - prev_x) / SMOOTHING
                    curr_y = prev_y + (y3 - prev_y) / SMOOTHING
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                    cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)

                    # Swipe Logic
                    if swipe_start_x == 0:
                        swipe_start_x = ix
                    
                    current_time = time.time()
                    if current_time - last_swipe_time > swipe_cooldown:
                        diff_x = ix - swipe_start_x
                        if abs(diff_x) > swipe_threshold:
                            if diff_x > 0:
                                pyautogui.press('right')
                                status = "SWIPE NEXT"
                            else:
                                pyautogui.press('left')
                                status = "SWIPE PREV"
                            last_swipe_time = current_time
                            swipe_start_x = 0
                else:
                    swipe_start_x = 0

                # 2. Scroll Mode (Based on Open/Close state)
                if total_fingers >= 4:
                    status = "SCROLL UP (HAND OPEN)"
                    current_time = time.time()
                    if current_time - last_scroll_time > scroll_cooldown:
                        pyautogui.scroll(25) # Faster jump
                        last_scroll_time = current_time
                elif total_fingers <= 1 and fingers[1] == 0:
                    status = "SCROLL DOWN (FIST)"
                    current_time = time.time()
                    if current_time - last_scroll_time > scroll_cooldown:
                        pyautogui.scroll(-25) # Faster jump
                        last_scroll_time = current_time

    if DEBUG_VIEW:
        color = (0, 255, 0) if master_active else (0, 0, 255)
        # Draw a solid bar at the top for clear visual status
        cv2.rectangle(frame, (0,0), (640, 40), color, -1)
        cv2.putText(frame, f"MODE: {status}", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("AI PPT Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
