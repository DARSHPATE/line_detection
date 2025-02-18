import cv2
import numpy as np

def create_mask(image):
    x_margin_top = 900
    x_margin_bottom = 225
    y_margin_top = 150
    y_margin_bottom = 500
    offset = 100
    vertices = np.array([[(x_margin_top, y_margin_top), (frame.shape[1] - x_margin_top + offset, y_margin_top), (frame.shape[1] - x_margin_bottom + offset, frame.shape[0] - y_margin_bottom), (x_margin_bottom, frame.shape[0] - y_margin_bottom)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    roi = cv2.bitwise_and(image, mask)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 30, 0])
    upper_yellow = np.array([30, 100, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)=9
    result = cv2.bitwise_and(roi, roi, mask=mask)

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
