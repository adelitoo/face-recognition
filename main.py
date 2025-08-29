import cv2

cv2.namedWindow("preview")
webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("cv2/data/haarcascade_frontalface_default.xml")

if not webcam.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Cannot read frame")
        break

    cv2.imshow("preview", frame)
    # OpenCV needs a key to quit the window otherwise the winodow is frozen and never shows up
    if cv2.waitKey(1) == ord("q"):
        break


cv2.destroyWindow("preview")
webcam.release()
