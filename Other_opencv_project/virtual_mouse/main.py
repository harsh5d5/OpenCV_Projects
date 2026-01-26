import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import os

# --- CONFIGURATION ---
CAM_WIDTH, CAM_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING = 5  # Smoothing factor
FRAME_REDUCTION = 100  # Frame reduction for better control
MODEL_PATH = "hand_landmarker.task"

# --- INITIALIZATION ---
# Ensure model is present (it should be in the parent dir as per README)
# If not, we'll look for it or the user should have it.
if not os.path.exists(MODEL_PATH):
    # Try looking in parent directory
    parent_model = os.path.join("..", "..", MODEL_PATH)
    if os.path.exists(parent_model):
        MODEL_PATH = parent_model
    else:
        print(f"Error: {MODEL_PATH} not found. Please ensure it exists.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Disable PyAutoGUI fail-safe to prevent crashes
pyautogui.FAILSAFE = False

print("AI Virtual Mouse Started (Tasks API)!")
print("Index Finger: Move Cursor")
print("Index + Thumb Pinch: Left Click")
print("Middle + Thumb Pinch: Right Click")
print("Press 'q' to Quit")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) # Flip for mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)

    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            # Draw landmarks (Manual drawing since we don't have mp_draw)
            h, w, c = img.shape
            landmarks = []
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if len(landmarks) >= 21:
                # Tips of fingers
                x1, y1 = landmarks[8][0], landmarks[8][1]   # Index tip
                x2, y2 = landmarks[12][0], landmarks[12][1]  # Middle tip
                x0, y0 = landmarks[4][0], landmarks[4][1]   # Thumb tip

                # 1. Check which fingers are up
                # Simplified finger up detection (y-coordinate relative)
                index_up = landmarks[8][1] < landmarks[6][1]
                middle_up = landmarks[12][1] < landmarks[10][1]
                
                # 2. Movement Mode (Index finger is up)
                if index_up and not middle_up:
                    # Map coordinates
                    x3 = np.interp(x1, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, SCREEN_WIDTH))
                    y3 = np.interp(y1, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, SCREEN_HEIGHT))

                    # Smoothing
                    curr_x = prev_x + (x3 - prev_x) / SMOOTHING
                    curr_y = prev_y + (y3 - prev_y) / SMOOTHING

                    pyautogui.moveTo(curr_x, curr_y)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

                # 3. Left Click (Index and Thumb tips close)
                dist_left = np.hypot(landmarks[8][0] - landmarks[4][0], landmarks[8][1] - landmarks[4][1])
                if dist_left < 30:
                    cv2.circle(img, (landmarks[8][0], landmarks[8][1]), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()

                # 4. Right Click (Middle and Thumb tips close)
                dist_right = np.hypot(landmarks[12][0] - landmarks[4][0], landmarks[12][1] - landmarks[4][1])
                if dist_right < 30:
                    cv2.circle(img, (landmarks[12][0], landmarks[12][1]), 15, (0, 0, 255), cv2.FILLED)
                    pyautogui.rightClick()

                # 5. Scroll (Index and Middle fingers both up and close)
                if index_up and middle_up:
                    dist_scroll = np.hypot(x1 - x2, y1 - y2)
                    if dist_scroll < 40:
                        movement = (y1 - prev_y) * 2
                        pyautogui.scroll(-int(movement))

    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()

