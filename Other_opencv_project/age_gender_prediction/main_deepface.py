import cv2
from deepface import DeepFace
import os

def predict_advanced_age_gender():
    # Attempt to open the webcam
    video = cv2.VideoCapture(0)
    
    print("Initializing DeepFace (this may take a few seconds on first run)...")
    print("Press 'q' to quit.")

    while True:
        has_frame, frame = video.read()
        if not has_frame:
            break

        try:
            # DeepFace.analyze will detect faces and predict age, gender, and emotion
            # enforce_detection=False prevents crashing if no face is detected
            results = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False, silent=True)

            for res in results:
                # Get the region of the face
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                
                # Get predicted values
                age = int(res['age'])
                gender = res['dominant_gender']
                
                # Draw box and labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{gender}, Age: {age}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow("DeepFace Advanced Prediction (Real Age)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_advanced_age_gender()
