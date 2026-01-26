import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult
# No separate VisionRunningMode, use vision.RunningMode
import pyautogui
import numpy as np
import os
import urllib.request

# pyautogui optimizations for games
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Model URL and Path
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model file...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# Initialize Mediapipe Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6, # Lowered slightly for speed
    min_hand_presence_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
# Use 640x480 for much faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def count_fingers(landmarks, hand_label):
    fingers = []
    # Thumb Detection (Handedness aware)
    # MediaPipe reports the actual hand (Right/Left)
    # In a mirrored image (cv2.flip 1):
    # - Right hand thumb is on the LEFT of the palm (x_tip < x_joint)
    # - Left hand thumb is on the RIGHT of the palm (x_tip > x_joint)
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else: # Left hand
        if landmarks[4].x > landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other 4 fingers (Same for both hands)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)

# Hand skeleton connections
HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16), (13,17), (17,18), (18,19), (19,20), (0,17)]

def draw_landmarks(frame, hand_landmarks, handedness):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        x_min, y_min = min(x_min, cx), min(y_min, cy)
        x_max, y_max = max(x_max, cx), max(y_max, cy)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (cx, cy), 2, (255, 255, 255), cv2.FILLED)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, coords[start], coords[end], (255, 255, 255), 1)
    padding = 15
    cv2.rectangle(frame, (x_min - padding, y_min - padding), (x_max + padding, y_max + padding), (255, 0, 255), 2)
    hand_type = handedness[0].category_name
    cv2.putText(frame, hand_type, (x_min - padding, y_min - padding - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    return hand_type

# State tracker for smooth control
current_state = "IDLE" # GAS, BRAKE, IDLE

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detection_result = detector.detect_for_video(mp_image, timestamp)

    new_state = "IDLE"
    if detection_result.hand_landmarks:
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[i]
            hand_type = draw_landmarks(frame, hand_landmarks, handedness)
            
            # Count fingers using handedness info
            totalFingers = count_fingers(hand_landmarks, hand_type)
            
            cv2.putText(frame, f'Fingers: {totalFingers}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Logic for Hill Climb Racing
            # Trigger GAS on 4 or 5 fingers for better reliability
            if totalFingers >= 4:
                new_state = "GAS"
            elif totalFingers == 0:
                new_state = "BRAKE"

    # Minimal key command logic
    if new_state != current_state:
        if new_state == "GAS":
            pyautogui.keyUp('left')
            pyautogui.keyDown('right')
        elif new_state == "BRAKE":
            pyautogui.keyUp('right')
            pyautogui.keyDown('left')
        elif new_state == "IDLE":
            pyautogui.keyUp('right')
            pyautogui.keyUp('left')
        current_state = new_state

    # HUD Visuals
    color = (0, 255, 0) if current_state == "GAS" else (0, 0, 255) if current_state == "BRAKE" else (200, 200, 200)
    cv2.putText(frame, f"STATUS: {current_state}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    cv2.imshow('Hill Climb Pro (Fast Mode)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

pyautogui.keyUp('right')
pyautogui.keyUp('left')
detector.close()
cap.release()
cv2.destroyAllWindows()



