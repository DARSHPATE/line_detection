import cv2
import numpy as np


def create_mask(image):
    x_margin_top = 900
    x_margin_bottom = 225
    y_margin_top = 150
    y_margin_bottom = 500
    offset = 100
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vertical = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1))
    vertical = cv2.normalize(vertical, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # vertical = cv2.threshold(vertical, 10, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    vertical = cv2.erode(vertical, kernel, iterations=1)

    vertices = np.array([[(x_margin_top, y_margin_top), (frame.shape[1] - x_margin_top + offset, y_margin_top),
                          (frame.shape[1] - x_margin_bottom + offset, frame.shape[0] - y_margin_bottom),
                          (x_margin_bottom, frame.shape[0] - y_margin_bottom)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    roi = cv2.bitwise_and(image, mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 15, 100])
    upper_yellow = np.array([35, 100, 255])
    lower_white = np.array([0, 5, 0])
    upper_white = np.array([360, 15, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    white_covered = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(white_mask))

    yellow_edges = cv2.bitwise_and(vertical, vertical, mask=yellow_mask)
    white_edges = cv2.bitwise_and(vertical, vertical, mask=white_mask)

    edges = cv2.bitwise_or(yellow_edges, white_edges)

    kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=3)
    # mask = cv2.dilate(mask, kernel, iterations=3)

    point = ()

    return white_covered

def overlay_arrow(image, arrow, position=(50, 1700)):
    alpha_channel = arrow[:, :, 3] / 255
    x, y, _ = arrow.shape
    for color_index in range(0, 3):
        image[position[0]:position[0] + x, position[1]:position[1] + y, color_index] = (1 - alpha_channel) * image[position[0]:position[0] + x, position[1]:position[1] + y, color_index] + alpha_channel * arrow[:x, :y, color_index]
    return image

cap = cv2.VideoCapture('driving.mov')

arrow = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)
arrow = cv2.resize(arrow, (arrow.shape[1] // 8, arrow.shape[0] // 8), interpolation=cv2.INTER_AREA)

while cap.isOpened():
    ret, frame = cap.read()
    frame = overlay_arrow(frame, arrow)
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
