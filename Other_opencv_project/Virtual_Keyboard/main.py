import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from time import sleep
import cvzone
import numpy as np
import time
from pynput.keyboard import Controller
import os

# --- CONFIGURATION ---
MODEL_PATH = "c:/Users/Welcome/Downloads/Open_cv/Game/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.getcwd(), "hand_landmarker.task")

# Initialize the webcam
# We try for 1280x720 but fallback to whatever the camera provides
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize pynput
keyboard = Controller()

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# Define keys and layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""
last_click_time = 0
buttonList = []

class Button():
    def __init__(self, pos, text, size=[80, 80]):
        self.pos = pos
        self.size = size
        self.text = text

def createButtons(w, h):
    global buttonList
    buttonList = []
    # Dynamic sizing based on width
    # 10 keys per row, reduce from w/13 to w/14 for more margin
    btnSize = int(w / 14)
    gap = int(btnSize * 0.15)
    startX = (w - (10 * btnSize + 9 * gap)) // 2
    startY = h // 8

    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            x = startX + j * (btnSize + gap)
            y = startY + i * (btnSize + gap)
            buttonList.append(Button([x, y], key, [btnSize, btnSize]))

    # Functional keys at the bottom
    funcY = startY + 3 * (btnSize + gap)
    buttonList.append(Button([startX, funcY], "BACK", [btnSize * 2 + gap, btnSize]))
    buttonList.append(Button([startX + 2 * (btnSize + gap), funcY], "SPACE", [btnSize * 5 + 4 * gap, btnSize]))
    buttonList.append(Button([startX + 7 * (btnSize + gap) + gap, funcY], "CLR", [btnSize * 2 + gap, btnSize]))

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        bw, bh = button.size
        # Scaled drawing
        cvzone.cornerRect(imgNew, (x, y, bw, bh), 15, rt=0, colorC=(0, 255, 0))
        cv2.rectangle(imgNew, button.pos, (x + bw, y + bh), (255, 0, 255), cv2.FILLED)
        
        scale = bw / 40 # Dynamically scale font
        text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_PLAIN, scale, 2)[0]
        text_x = x + (bw - text_size[0]) // 2
        text_y = y + (bh + text_size[1]) // 2
        cv2.putText(imgNew, button.text, (text_x, text_y),
                    cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), 2)

    out = img.copy()
    alpha = 0.4
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

# Initial button creation - will be updated if frame size changes
buttons_initialized = False

while cap.isOpened():
    success, img = cap.read()
    if not success: break
        
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    if not buttons_initialized:
        createButtons(w, h)
        buttons_initialized = True
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)
    
    img = drawAll(img, buttonList)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            point8 = [int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)]
            point12 = [int(hand_landmarks[12].x * w), int(hand_landmarks[12].y * h)]
            
            # Simple Landmark Drawing
            for lm in [hand_landmarks[8], hand_landmarks[12], hand_landmarks[4]]:
                cv2.circle(img, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 0), cv2.FILLED)

            for button in buttonList:
                x, y = button.pos
                bw, bh = button.size

                if x < point8[0] < x + bw and y < point8[1] < y + bh:
                    cv2.rectangle(img, (x-3, y-3), (x + bw+3, y + bh+3), (150, 0, 150), cv2.FILLED)
                    scale = bw / 40
                    text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_PLAIN, scale, 2)[0]
                    cv2.putText(img, button.text, (x + (bw - text_size[0]) // 2, y + (bh + text_size[1]) // 2),
                                cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), 2)
                    
                    dist = np.linalg.norm(np.array(point8) - np.array(point12))
                    
                    if dist < 35 and (time.time() - last_click_time) > 0.4:
                        cv2.rectangle(img, button.pos, (x + bw, y + bh), (0, 255, 0), cv2.FILLED)
                        last_click_time = time.time()
                        
                        if button.text == "SPACE":
                            keyboard.press(" ")
                            finalText += " "
                        elif button.text == "BACK":
                            from pynput.keyboard import Key
                            keyboard.press(Key.backspace)
                            finalText = finalText[:-1]
                        elif button.text == "CLR":
                            finalText = ""
                        else:
                            keyboard.press(button.text)
                            finalText += button.text

    # Dynamic UI boxes
    box_w, box_h = int(w * 0.8), int(h * 0.12)
    box_x = (w - box_w) // 2
    box_y = int(h * 0.75)
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 0, 100), cv2.FILLED)
    cv2.putText(img, finalText, (box_x + 15, box_y + int(box_h * 0.7)), cv2.FONT_HERSHEY_PLAIN, box_h/30, (255, 255, 255), 3)
    
    brand_w, brand_h = int(w * 0.25), int(h * 0.08)
    cv2.rectangle(img, (box_x, box_y + box_h + 10), (box_x + brand_w, box_y + box_h + 10 + brand_h), (255, 0, 255), cv2.FILLED)
    cv2.putText(img, "CVZONE", (box_x + 10, box_y + box_h + 10 + int(brand_h * 0.75)), cv2.FONT_HERSHEY_PLAIN, brand_h/15, (255, 255, 255), 3)

    cv2.imshow("AIG-Perfect Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



try:
    detector.close()
except:
    pass
cap.release()
cv2.destroyAllWindows()
