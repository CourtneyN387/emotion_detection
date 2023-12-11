import cv2
import numpy as np
import time 
from keras.models import model_from_json

# Allows us to map emotions with label
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
# applying all the learnings to the new model 
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
# this will let us put in video from laptop camera 
cap = cv2.VideoCapture(0)

# pass here your video path -- this one is putting in a sample video 
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

emotion_data = []

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    # resize all image 
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    # first of all we have to detect the face 
    # here make use of haarcascades to do that 
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting image to grayscale

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    # provide position of each face in video stream
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        # cropping the full screen image, and store it into the roi_gray_frame 
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        # get a list of each percentage associated with each emotion 
        # return the max percentage and max emotion
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

# import matplotlib.pyplot as plt

# # Assuming emotion_data is a list of tuples (timestamp, emotion_label)
# timestamps = [x[0] for x in emotion_data]
# emotions = [x[1] for x in emotion_data]

# plt.plot(timestamps, emotions)
# plt.xlabel('Time')
# plt.ylabel('Emotion')
# plt.title('Emotions Over Time')
# plt.show()

import matplotlib.pyplot as plt
from collections import Counter

# Assuming emotion_data is a list of tuples (timestamp, emotion_label)
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
