import cv2

pipeline = (
    "libcamerasrc ! "
    "video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera failed")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame failed")
        break

    #cv2.imshow("test", frame)
    cv2.imwrite("testHEJHEJ.png", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()