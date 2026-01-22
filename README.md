# Hand-Controlled Hill Climb Racing 🏎️✋

This mini-project allows you to control the **Hill Climb Racing** game using hand gestures via your webcam. It uses **OpenCV** for computer vision, **MediaPipe** for hand tracking, and **PyAutoGUI** to simulate keyboard presses.

## ✨ Features
- **Python 3.13 Compatible**: Uses the modern MediaPipe Tasks API.
- **High Performance**: Optimized for fast FPS and low latency.
- **Handedness Aware**: Works perfectly with both Left and Right hands.
- **Visual HUD**: Real-time display of hand skeleton, bounding box, and game status (GAS/BRAKE).

## 🛠️ Required Versions & Libraries
This project requires specific versions for full compatibility:

- 🐍 **Python**: `v3.13.9`
- 🤖 **MediaPipe**: `v0.10.31`
- 📷 **OpenCV**: `v4.10.x`
- ⌨️ **PyAutoGUI**: `v0.9.54`
- 🔢 **NumPy**: `v2.x`

## 🚀 Installation

Open your terminal or Anaconda Prompt and run:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

> [!NOTE]
> The script will automatically download the required AI model file (`hand_landmarker.task`) the first time you run it.

## 🎮 How to Control
Once the script is running, open **Hill Climb Racing** and use these gestures in front of your webcam:

| Gesture | Hand State | Action | In-Game Command |
| :--- | :--- | :--- | :--- |
| **Open Hand** | 4 or 5 Fingers Up | **GAS** | Right Arrow Key |
| **Fist** | 0 Fingers Up | **BRAKE / REVERSE** | Left Arrow Key |
| **Others** | 1 to 3 Fingers Up | **IDLE** | Release all keys |

## 🏃 How to Run
1. Start the Hill Climb Racing game.
2. Run the Python script:
   ```bash
   python hill_climbe_racing.py
   ```
3. Position your hand in the camera view.
4. Press **'q'** on the camera window to stop the script.

---
Developed as a Computer Vision Mini-Project. Enjoy racing with your hands! 🏁
