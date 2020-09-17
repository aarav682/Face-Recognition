#TRAINING THE FACE DATASET

import os
from PIL import Image
import numpy as np
import pickle
import cv2

#print(dir(cv2.face))
bas_dir = os.path.dirname(os.path.abspath('C:\\Users\\aarav\\Desktop\\CETPA\\faces_train'))

x_train = []
y_labels = []
current_id = 0

label_ids={}

face_cascade = cv2.CascadeClassifier('C:\\Program Files\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
for root , dirs,files in os.walk(bas_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(' ','-').lower()
            #print(label,path)
            if not label in label_ids:
           
                label_ids[label] = current_id 
                current_id+=1
            id_= label_ids[label]
           # print(label_ids)
            #x_train.append(path)
            #y_label.append(label)
            pil_image = Image.open(path).convert('L')
            image_array = np.array(pil_image,'uint8')
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
                #print(y_labels)

               
with open('label.pickle','wb') as f:
    pickle.dump(label_ids,f)
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')


               
               
               