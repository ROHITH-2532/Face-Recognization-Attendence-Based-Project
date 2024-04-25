from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import pickle
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)
video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier('D:/Rohith/data/haarcascade_frontalface_default.xml')

with open('D:/Rohith/data/names.pkl','rb') as f:
    LABELS = pickle.load(f)

with open('D:/Rohith/data/faces_data.pkl','rb') as f:
    FACES = pickle.load(f)   

num_samples = FACES.shape[0]
num_features = FACES.shape[1] * FACES.shape[2] * FACES.shape[3]  # height * width * channels

# Reshape the FACES array into a 2D array where each row is a flattened image
FACES_flattened = FACES.reshape(num_samples, num_features)

# Assume LABELS is already in the correct shape (num_samples,)
knn = KNeighborsClassifier()
knn.fit(FACES_flattened, LABELS)
imgBackground = cv2.imread('background.png')
COL_NAMES= ['NAME', 'TIME']
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50,50))
        resized_img = resized_img.flatten().reshape(1, -1)  # Flatten and reshape for prediction
        if len(faces) > 0:
            output = knn.predict(resized_img)
            label_text = str(output[0])  # Assuming labels are direct names, change accordingly
            cv2.putText(frame, label_text, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1) 
            ts=time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
            exist = os.path.isfile("D:/Rohith/data/Attendence" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        attendence=[str(output[0]), str(timestamp)]
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow('frame', imgBackground)
    k = cv2.waitKey(1)
    if k==ord('o'):
        speak("Attendence taken...")
        time.sleep(2)
        if  exist:
            with open("D:/Rohith/data/Attendence" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendence)
            csvfile.close()    
        else:
            with open("D:/Rohith/data/Attendence" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendence)
            csvfile.close()    
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
