import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_binary(image):
    x_margin_top = 900
    x_margin_bottom = 225
    y_margin_top = 150
    y_margin_bottom = 500
    offset = 100
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vertical = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1))
    vertical = cv2.normalize(vertical, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    kernel = np.ones((3, 3), np.uint8)
    vertical = cv2.erode(vertical, kernel, iterations=1)

    vertices = np.array([[(x_margin_top, y_margin_top), (frame.shape[1] - x_margin_top + offset, y_margin_top),
                          (frame.shape[1] - x_margin_bottom + offset, frame.shape[0] - y_margin_bottom),
                          (x_margin_bottom, frame.shape[0] - y_margin_bottom)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, (255, 255, 255))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 15, 100])
    upper_yellow = np.array([35, 100, 255])
    lower_white = np.array([0, 0, 0])
    upper_white = np.array([360, 20, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    yellow_edges = cv2.bitwise_and(vertical, vertical, mask=yellow_mask)
    white_edges = cv2.bitwise_and(vertical, vertical, mask=white_mask)
    edges = cv2.bitwise_or(yellow_edges, white_edges)
    _, edges = cv2.threshold(edges, 2, 1, cv2.THRESH_BINARY)

    left_vertices = np.array([[(400, 150), (700, 120), (280, 360), (50, 330)]])
    right_vertices = np.array([[(600, 100), (810, 130), (1265, 370), (950, 370)]])
    mask = np.zeros_like(edges)
    mask = mask.astype('uint8')
    cv2.fillPoly(mask, left_vertices, 1)
    left_roi = cv2.bitwise_and(edges, edges, mask=mask)
    kernel = np.ones((3, 3), np.float32)
    left_roi = cv2.filter2D(left_roi, -1, kernel)
    _, left_roi = cv2.threshold(left_roi, 4, 1, cv2.THRESH_BINARY)
    mask = np.zeros_like(edges)
    mask = mask.astype('uint8')
    cv2.fillPoly(mask, right_vertices, 1)
    right_roi = cv2.bitwise_and(edges, edges, mask=mask)
    right_roi = cv2.filter2D(right_roi, -1, kernel)
    _, right_roi = cv2.threshold(right_roi, 4, 1, cv2.THRESH_BINARY)
    
    k = 2
    left_ones, right_ones = [], []
    for x, sliced in enumerate(left_roi):
        indices = np.where(sliced == 1)[0]
        if len(indices) > k:
            indices = indices[-k:]
        left_ones += [(x, y) for y in indices]
    left_x = np.array([p[0] for p in left_ones])
    left_y = np.array([p[1] for p in left_ones])
    left_fit = np.polyfit(left_x, left_y, 1)
    left_quadratic = np.poly1d(left_fit)
    x_range = np.linspace(200, 350, 1000)
    for x in x_range:
        cv2.circle(image, (int(left_quadratic(x)), int(x)), radius=0, color=(0, 255, 0), thickness=5)
    for x, sliced in enumerate(right_roi):
        indices = np.where(sliced == 1)[0]
        if len(indices) > k:
            indices = indices[:k]
        right_ones += [(x, y) for y in indices]
    right_x = np.array([p[0] for p in right_ones])
    right_y = np.array([p[1] for p in right_ones])
    right_fit = np.polyfit(right_x, right_y, 1)
    right_quadratic = np.poly1d(right_fit)
    x_range = np.linspace(200, 350, 1000)
    for x in x_range:
        cv2.circle(image, (int(right_quadratic(x)), int(x)), radius=0, color=(0, 255, 0), thickness=5)


    return image

def overlay_arrow(image, arrow, position=(30, 1100)):
    alpha_channel = arrow[:, :, 3] / 255
    x, y, _ = arrow.shape
    for color_index in range(0, 3):
        image[position[0]:position[0] + x, position[1]:position[1] + y, color_index] = (1 - alpha_channel) * image[position[0]:position[0] + x, position[1]:position[1] + y, color_index] + alpha_channel * arrow[:x, :y, color_index]
    return image

cap = cv2.VideoCapture('driving720.mov')

arrow = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)
arrow = cv2.resize(arrow, (arrow.shape[1] // 12, arrow.shape[0] // 12), interpolation=cv2.INTER_AREA)

while cap.isOpened():
    ret, frame = cap.read()
    frame = create_binary(frame)
    frame = overlay_arrow(frame, arrow)
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
