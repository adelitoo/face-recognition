import cv2
import os

cv2.namedWindow("preview")
webcam = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if not webcam.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Cannot read frame")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=15, minSize=(40,40))

    for face in faces:
        print(face)
        x, y, w, h = face
        cv2.putText(frame, "face", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

    cv2.imshow("preview", frame)
    if cv2.waitKey(1) == ord("q"):
        break


cv2.destroyWindow("preview")
webcam.release()
