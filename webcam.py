import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

# from imutils.video import VideoStream
# from imutils import face_utils
# import datetime
# import imutils
# import dlib
# import time


cascPath = "assets/haarcascade_frontalface_default.xml"
eCascPath = "assets/haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eCascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Fix for OpenCV Window not closing
cv2.startWindowThread()

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # Paint Faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (237, 77, 38), 2)
        # Detect Eyes
        eyes = eyeCascade.detectMultiScale(gray[y:y+h, x:x+w])
        # Paint Eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(243, 243, 243), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
