import cv2
import numpy as np

def create_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([30, 30, 80])
    upper_yellow = np.array([80, 255, 255])
    mask = cv2.bitwise_not(cv2.inRange(hsv, lower_yellow, upper_yellow))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    result = cv2.bitwise_and(image, image, mask=mask)
    ratio = 1.0 / 3
    return result

cap = cv2.VideoCapture('driving.mov')

while cap.isOpened():
    ret, frame = cap.read()
    frame = create_mask(frame)
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
