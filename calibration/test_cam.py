import cv2

for i in range(15):
	cap = cv2.VideoCapture(i)
	ret, _ = cap.read()
	print("camera", i, ":", "ok" if ret else "FAIL")
	cap.release()
