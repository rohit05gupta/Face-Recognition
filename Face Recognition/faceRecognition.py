import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer/trainningData.yml")
id=0
# font=cv2.InitFont(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 4
fontcolor = (0, 255, 255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        nbr_predicted, conf = rec.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        if(nbr_predicted==2):
             nbr_predicted='Shahrukh'
        elif(nbr_predicted==1):
             nbr_predicted='Rohit'
        cv2.putText(im,str(nbr_predicted), (x,y+h),fontface, 2,(220,255,255)); #Draw the text
    cv2.imshow('im',im);
    if(cv2.waitKey(1)==ord('q')):
    	break;
cam.release()
cv2.destroyAllWindows()    	
