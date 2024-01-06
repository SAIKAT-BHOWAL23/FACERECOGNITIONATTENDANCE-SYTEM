from sklearn.neighbors import KNeighborsClassifier


import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime


video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('DATA/haarcascade_frontalface_default.xml')

print("PRESS 'p' TO GIVE ATTENDANCE")

with open('DATA/names.pkl','rb') as f:
        LABELS=pickle.load(f)
        
with open('DATA/faces_data.pkl', 'rb') as f:
        FACES=pickle.load(f)        


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME' , '   ' ,  'DATE', '   ' , 'TIME']


while True:
    ret, frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.5 , 5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,51)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist=os.path.isfile("records_of_attendence/Attendance_" + date + ".csv")
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendence=[str(output[0]),str(date), str(timestamp)]

    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)

    if k==ord('p'):
          if exist:
                with open("records_of_attendence/Attendance_" + date + ".csv", "+a") as csvfile:
                      writer=csv.writer(csvfile)
                     # writer.writerow(COL_NAMES)
                      writer.writerow(attendence)
                csvfile.close()
                print("YOUR ATTENDANCE IS TAKEN")     
          else:
                with open("records_of_attendence/Attendance_" + date + ".csv", "+a") as csvfile:
                      writer=csv.writer(csvfile)
                      writer.writerow(COL_NAMES)
                      writer.writerow(attendence)
                csvfile.close()
                print("YOUR ATTENDANCE IS TAKEN")
               
    
     
   
   
   
   
   
    if k==ord('s'):
        break
video.release()
cv2.destroyAllWindows()    


