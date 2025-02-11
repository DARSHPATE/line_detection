import cv2
import numpy as np

def create_mask(image):
    #do an roi
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([32, 25, 80])
    upper_yellow = np.array([80, 255, 255])
    mask = cv2.bitwise_not(cv2.inRange(hsv, lower_yellow, upper_yellow))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    result = cv2.bitwise_and(image, image, mask=mask)
    ratio = 1.0 / 3
    for i in range(2):
        if i == 0:
            side = result[:, :int(ratio * result.shape[1])]
        else:
            side = result[:, int((1 - ratio) * result.shape[1]):]
        for slice in side:
            zeroes = np.where(slice == (0, 0, 0))[0]
            if len(zeroes) > 0:
                if i == 0:
                    last_zero = zeroes[-1]
                    slice[:last_zero] = [0, 0, 0]
                else:
                    first_zero = zeroes[0]
                    slice[first_zero:] = [0, 0, 0]
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
