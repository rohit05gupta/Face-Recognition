import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

i=0
name=input('enter your id')
while True:
    ret, im =cam.read();
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face."+str(name) +'.'+ str(i) + ".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        cv2.waitKey(100);
    cv2.imshow('im',im);
    cv2.waitKey(1);    
    if (i>20):
    	break
cam.release()
cv2.destroyAllWindows()

