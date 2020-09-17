#RECOGNIZING THE FACE

import pandas as pd
import numpy as np
import cv2
import pickle




face_cascade = cv2.CascadeClassifier('C:\\Program Files\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels ={}

with open('label.pickle','rb') as f:
   og_labels = pickle.load(f)
   labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)


while True:
    ret ,frame =  cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame1',gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (120,150,255)
            stroke = 2
            cv2.putText(frame,name,(x,y), font, 1 ,color, stroke,cv2.LINE_AA)
        
        #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    
    
cap.release() 


cv2.destroyAllWindows