import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import math
import time

# Model Setup
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize Mediapipe
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Constants
PURPLE = (255, 0, 255)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

class DragBox():
    def __init__(self, posCenter):
        self.posCenter = posCenter
        self.size = [200, 200]
        self.isGrabbed = False

    def update(self, cursor, is_pinching):
        cx, cy = self.posCenter
        w, h = self.size
        # Start Grab
        if not self.isGrabbed and is_pinching:
            if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
                self.isGrabbed = True
        # Follow/Drop
        if self.isGrabbed:
            self.posCenter = cursor
            if not is_pinching:
                self.isGrabbed = False
                return True
        return False

    def draw(self, frame):
        cx, cy = self.posCenter
        w, h = self.size
        cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), PURPLE, cv2.FILLED)
        cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (255, 255, 255), 2)

class VirtualButton():
    def __init__(self, pos):
        self.pos = pos
        self.size = (120, 120)

    def is_touched(self, cursor):
        x, y = self.pos
        w, h = self.size
        return x < cursor[0] < x + w and y < cursor[1] < y + h

    def draw(self, frame, touched=False):
        x, y = self.pos
        w, h = self.size
        color = (0, 255, 0) if touched else PURPLE
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.putText(frame, "+", (x + 35, y + 85), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 4)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def process_hand(landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    # Pinch detection (Index + Middle)
    is_pinching = get_distance(points[8], points[12]) < 50
    return is_pinching, points[8], points[12], points

def check_collision(boxes):
    if len(boxes) < 2: return
    for i, b1 in enumerate(boxes):
        for j, b2 in enumerate(boxes):
            if i != j and get_distance(b1.posCenter, b2.posCenter) < 140:
                boxes.pop(i)
                return

def draw_visuals(frame, points, w, h):
    # Hand Bounding Box
    x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = max(0, min(x_coords)-20), min(w, max(x_coords)+20)
    y_min, y_max = max(0, min(y_coords)-20), min(h, max(y_coords)+20)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # Skeleton & Landmarks
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, points[s], points[e], (255, 255, 255), 1)
    for p in points:
        cv2.circle(frame, p, 5, (255, 0, 255), cv2.FILLED)

# Init State
boxes = [DragBox([400, 400])]
spawn_btn = VirtualButton((1100, 50))
last_add_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    drop, touched = False, False
    if res.hand_landmarks:
        for lm in res.hand_landmarks:
            pinch, i_tip, m_tip, pts = process_hand(lm, w, h)
            draw_visuals(frame, pts, w, h)
            cursor = [(i_tip[0] + m_tip[0]) // 2, (i_tip[1] + m_tip[1]) // 2]
            
            if spawn_btn.is_touched(cursor):
                touched = True
                if time.time() - last_add_time > 1.0:
                    boxes.append(DragBox([w // 2, h // 2]))
                    last_add_time = time.time()
            
            for b in boxes:
                if b.update(cursor, pinch): drop = True
                if b.isGrabbed: break

    if drop: check_collision(boxes)
    spawn_btn.draw(frame, touched)
    for b in boxes: b.draw(frame)
    cv2.imshow('Hand Box', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

detector.close()
cap.release()
cv2.destroyAllWindows()
