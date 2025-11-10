import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
#read
img = cv2.imread('female.jpg')
img = cv2.resize(img, (720, 640))

#define model
face_pbtxt = 'opencv_face_detector.pbtxt'
face_pb = 'opencv_face_detector_uint8.pb'
age_prototxt = 'age_deploy.prototxt'
age_model = 'age_net.caffemodel'
gender_prototxt = 'gender_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
MODEL_MEAN_VALUES = [104,117,123]
#Load model
face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gender = cv2.dnn.readNet(gender_model, gender_prototxt)

#Setup label
age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male','Female']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = models.resnet18(weights=None)
num_ftrs = emotion_model.fc.in_features
emotion_model.fc = nn.Linear(num_ftrs, 7)  # 7 emotions
emotion_model.load_state_dict(torch.load("resnet18_fer2013.pth", map_location=device))
emotion_model.to(device)
emotion_model.eval()
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

emotion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
#copy Image
img_cp = img.copy()
#get image dimensions and blob
def detect_faces(frame):
    img_h = frame.shape[0]
    img_w = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    face.setInput(blob)
    detected_faces = face.forward()
    face_bounds = []

    for i in range(detected_faces.shape[2]):
        confidence = detected_faces[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detected_faces[0, 0, i, 3] * img_w)
            y1 = int(detected_faces[0, 0, i, 4] * img_h)
            x2 = int(detected_faces[0, 0, i, 5] * img_w)
            y2 = int(detected_faces[0, 0, i, 6] * img_h)
            face_bounds.append([x1, y1, x2, y2])
    return face_bounds


def predict_all(frame, face_bounds):
    for face_bound in face_bounds:
        try:
            x1, y1, x2, y2 = face_bound
            face_crop = frame[
                max(0, y1 - 15): min(y2 + 15, frame.shape[0] - 1),
                max(0, x1 - 15): min(x2 + 15, frame.shape[1] - 1)
            ]

            # ------------------- Age & Gender -------------------
            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Gender
            gender.setInput(blob)
            gender_prediction = gender.forward()
            gender_label = gender_classifications[gender_prediction[0].argmax()]

            # Age
            age.setInput(blob)
            age_prediction = age.forward()
            age_label = age_classifications[age_prediction[0].argmax()]

            # ------------------- Emotion -------------------
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_tensor = emotion_transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = emotion_model(face_tensor)
                emotion_label = emotion_labels[output.argmax(1).item()]

            # ------------------- Draw -------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{gender_label},{age_label},{emotion_label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            print(e)
            continue


choice = input("Chọn nguồn: 1-Webcam, 2-Load ảnh: ")
if choice == '1':
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        faces = detect_faces(frame)
        predict_all(frame, faces)
        cv2.imshow("Age & Gender Detection", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    camera.release()

elif choice == '2':
    path = input("Nhập đường dẫn ảnh: ")
    img = cv2.imread(path)
    if img is not None:
        faces = detect_faces(img)
        predict_all(img, faces)
        cv2.imshow("Age & Gender Detection", img)
        cv2.waitKey(0)
    else:
        print("Không đọc được ảnh!")
cv2.destroyAllWindows()
