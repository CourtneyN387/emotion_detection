import cv2
import numpy as np
import time
from keras.models import model_from_json
import matplotlib.pyplot as plt
from collections import Counter

# Function to load the emotion detection model
def load_emotion_model(model_json_path, model_weights_path):
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights(model_weights_path)
    print("Loaded model from disk")
    return emotion_model

# Function to perform emotion detection
def detect_emotions(cap, emotion_model, emotion_dict):
    emotion_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            current_time = time.time()
            emotion_data.append((current_time, emotion_dict[maxindex]))

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion_data

def line_graph(emotion_data):
    timestamps = [x[0] for x in emotion_data]
    emotions = [x[1] for x in emotion_data]

    plt.plot(timestamps, emotions)
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.title('Emotions Over Time')
    plt.show()

def bar_graph(emotion_data):
    emotions = [emotion for _, emotion in emotion_data]

    # Count the frequency of each emotion
    emotion_counts = Counter(emotions)

    # Creating the bar graph
    plt.bar(emotion_counts.keys(), emotion_counts.values())

    # Adding titles and labels
    plt.title('Emotion Distribution')
    plt.xlabel('Emotions')
    plt.ylabel('Frequency')

    # Display the plot
    plt.show()


# Main script
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_model = load_emotion_model('model/emotion_model.json', 'model/emotion_model.h5')

# For webcam input
cap = cv2.VideoCapture(0)

# Call the function
emotion_data = detect_emotions(cap, emotion_model, emotion_dict)

line_graph(emotion_data)
bar_graph(emotion_data)
