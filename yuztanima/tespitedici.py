import cv2
import numpy
import os

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'C:\Users\Fatma Turan\PycharmProjects\yuztanima\training\trainer.yml')
cascadePath="face.xml"
faceCascade=cv2.CascadeClassifier(cascadePath)
path=r"C:\Users\Fatma Turan\PycharmProjects\yuztanima\yuzverileri"

cam=cv2.VideoCapture(0)
while True:
    ret,im=cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        tahminEdilenKisi,conf = recognizer.predict(gray[y:y + h, x:x + w])
        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
        #print(conf)
        if (tahminEdilenKisi == 2):
            tahminEdilenKisi = 'Sevda'
        elif (tahminEdilenKisi == 4):
            tahminEdilenKisi = 'Fatmagul'
        elif (tahminEdilenKisi == 6):
            tahminEdilenKisi = 'Elif'
        else:
            tahminEdilenKisi = "Bilinmeyen kişi"

        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        cv2.putText(im, str(tahminEdilenKisi), (x, y + h), fontFace, fontScale, fontColor)
        cv2.imshow('image', im)
        cv2.waitKey(10)
