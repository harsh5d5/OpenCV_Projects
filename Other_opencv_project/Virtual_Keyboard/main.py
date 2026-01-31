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

# --- INITIALIZATION ---
MODEL_PATH = "c:/Users/Welcome/Downloads/Open_cv/Game/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    # Try relative path just in case
    MODEL_PATH = os.path.join(os.getcwd(), "hand_landmarker.task")

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize pynput keyboard controller
keyboard = Controller()

# Setup MediaPipe Hand Landmarker (Tasks API)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# Define keys and layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0, colorC=(0, 255, 0))
        cv2.rectangle(imgNew, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        
        text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_PLAIN, 3, 3)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(imgNew, button.text, (text_x, text_y),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

# Populate the button list
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

buttonList.append(Button([50, 350], "BACK", [230, 85]))
buttonList.append(Button([300, 350], "SPACE", [450, 85]))
buttonList.append(Button([770, 350], "CLR", [230, 85]))

while cap.isOpened():
    success, img = cap.read()
    if not success: break
        
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Run hand detection
    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)
    
    # Draw the keyboard UI
    img = drawAll(img, buttonList)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            h, w, c = img.shape
            # Create a landmark list similar to what cvzone gave [x, y, z]
            lmList = []
            for lm in hand_landmarks:
                lmList.append([int(lm.x * w), int(lm.y * h), lm.z])
            
            # Point 8 is index finger tip
            point8 = lmList[8][:2]
            
            # Draw landmarks for visual feedback
            for lm in lmList:
                cv2.circle(img, (lm[0], lm[1]), 5, (0, 255, 0), cv2.FILLED)

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < point8[0] < x + w and y < point8[1] < y + h:
                    cv2.rectangle(img, (x-5, y-5), (x + w+5, y + h+5), (150, 0, 150), cv2.FILLED)
                    text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_PLAIN, 3, 3)[0]
                    cv2.putText(img, button.text, (x + (w - text_size[0]) // 2, y + (h + text_size[1]) // 2),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                    
                    # Distance between index (8) and middle finger (12) tips for "Click"
                    p1 = np.array(lmList[8][:2])
                    p2 = np.array(lmList[12][:2])
                    dist = np.linalg.norm(p1 - p2)
                    
                    if dist < 40: # Threshold for click
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + (w - text_size[0]) // 2, y + (h + text_size[1]) // 2),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                        
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
                        
                        sleep(0.3) # Avoid double typing

    # UI for text and Branding
    cv2.rectangle(img, (50, 480), (1035, 580), (100, 0, 100), cv2.FILLED)
    cv2.putText(img, finalText, (70, 545), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    cv2.rectangle(img, (50, 600), (400, 680), (255, 0, 255), cv2.FILLED)
    cv2.putText(img, "CVZONE", (75, 660), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("AIG-Perfect Virtual Keyboard (3.13 Fix)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
