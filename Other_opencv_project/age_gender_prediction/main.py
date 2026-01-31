import cv2
import math
import os
from collections import deque, Counter

def highlight_face(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame_opencv_dnn, face_boxes

def predict_age_gender():
    model_path = "models"
    
    # Check if models exist
    required_files = [
        "face_deploy.prototxt", "face_net.caffemodel",
        "age_deploy.prototxt", "age_net.caffemodel",
        "gender_deploy.prototxt", "gender_net.caffemodel"
    ]
    
    if not all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
        print("Models not found. Please run download_models.py first.")
        return

    face_proto = os.path.join(model_path, "face_deploy.prototxt")
    face_model = os.path.join(model_path, "face_net.caffemodel")
    age_proto = os.path.join(model_path, "age_deploy.prototxt")
    age_model = os.path.join(model_path, "age_net.caffemodel")
    gender_proto = os.path.join(model_path, "gender_deploy.prototxt")
    gender_model = os.path.join(model_path, "gender_net.caffemodel")

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']

    face_net = cv2.dnn.readNet(face_model, face_proto)
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)

    video = cv2.VideoCapture(0)
    padding = 40  # Increased padding for better context
    
    # History for smoothing
    age_history = deque(maxlen=15)
    gender_history = deque(maxlen=15)

    print("Press 'q' to quit.")

    while True:
        has_frame, frame = video.read()
        if not has_frame:
            break

        result_img, face_boxes = highlight_face(face_net, frame)

        for face_box in face_boxes:
            face = frame[max(0, face_box[1] - padding): min(face_box[3] + padding, frame.shape[0] - 1),
                   max(0, face_box[0] - padding): min(face_box[2] + padding, frame.shape[1] - 1)]

            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender_history.append(gender_list[gender_preds[0].argmax()])
            
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_history.append(age_list[age_preds[0].argmax()])

            # Get most common prediction in recent history for stabilization
            gender = Counter(gender_history).most_common(1)[0][0]
            age = Counter(age_history).most_common(1)[0][0]

            label = f"{gender}, {age}"
            cv2.putText(result_img, label, (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("Stabilized Age & Gender Prediction", result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_age_gender()
