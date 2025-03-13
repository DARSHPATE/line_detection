import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
lane_detection
    input: current image frame
    finds the lane lines and computes the centerline
    output: overlayed frame
bluescale = bluescale(image)
edges = SobelFilter(bluescale)
blurred = blur(edges)
left_lane_roi, right_half_roi = mask(blurred)
left_line = linearInterpolate(left_lane_roi)
right_line = linearInterpolate(right_lane_roi)
center_line = angleBisector(left_line, right_line)
display(left_line, right_line, center_live)
'''

def lane_detection(image, turn_indicator, phase, in_turn):
    bluescale = image[:, :, 0]
    vertical = np.abs(cv2.Sobel(bluescale, cv2.CV_64F, 1, 0, ksize=1))
    vertical = cv2.normalize(vertical, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    kernel = np.ones((3, 3), np.uint8)
    vertical = cv2.erode(vertical, kernel)
    _, edges = cv2.threshold(vertical, 0, 1, cv2.THRESH_BINARY)
    median_blurred = cv2.medianBlur(edges, 3)
    kernel = np.ones((3, 3), np.uint8)
    local_sum = cv2.filter2D(median_blurred, -1, kernel)
    _, average_blurred = cv2.threshold(local_sum, 4, 1, cv2.THRESH_BINARY)

    left_vertices = np.array([[[(400, 150), (700, 120), (280, 360), (50, 330)], [(420, 130), (660, 100), (320, 310), (80, 310)], [(420, 130), (660, 100), (350, 300), (80, 300)]][phase]])
    right_vertices = np.array([[[(580, 100), (830, 130), (1215, 330), (930, 330)], [(615, 165), (790, 140), (1125, 315), (810, 315)], [(630, 120), (770, 120), (1090, 295), (790, 295)]][phase]])
    lanes = [left_vertices, right_vertices]
    k = 3
    rois = [None, None]
    for i in range(2):
        mask = np.zeros_like(bluescale)
        mask = mask.astype('uint8')
        cv2.fillPoly(mask, lanes[i], 1)
        roi = cv2.bitwise_and(average_blurred, average_blurred, mask=mask)
        rois[i] = cv2.bitwise_and(image, image, mask=mask)
        ones_set = []
        for x, sliced in enumerate(roi):
            indices = np.where(sliced == 1)[0]
            if len(indices) > k:
                y = indices[(2 * i - 1) * k]
                ones_set.append((x, y))
        x_values = np.array([p[0] for p in ones_set])
        y_values = np.array([p[1] for p in ones_set])
        linear_fit = np.polyfit(x_values, y_values, 1)
        line_approximation = np.poly1d(linear_fit)
        lanes[i] = line_approximation
        r2 = 1 - np.sum((y_values - line_approximation(x_values)) ** 2) / np.sum((y_values - np.mean(y_values)) ** 2)
        proportion = len(y_values) / np.abs(y_values[0] - y_values[-1])
        if r2 * proportion < 0.1:
            lanes[i] = None
    left_line, right_line = tuple(lanes)
    x_min, x_max = 170, 370
    if turn_indicator != 0:
        return image, phase, True
    elif in_turn:
        in_turn = False
        phase += 1
    if left_line is not None:
        cv2.line(image, (int(left_line(x_min)), x_min), (int(left_line(x_max)), x_max), (0, 255, 0), thickness=3)
    if right_line is not None:
        cv2.line(image, (int(right_line(x_min)), x_min), (int(right_line(x_max)), x_max), (0, 255, 0), thickness=3)
    if left_line is not None and right_line is not None:
        lc = left_line.c
        rc = right_line.c
        intersection = np.cross([lc[0], -1, lc[1]], [rc[0], -1, rc[1]])
        intersection = intersection / intersection[2]
        dir1 = np.array([1, lc[0]])
        dir2 = np.array([1, rc[0]])
        dir = dir1 / np.linalg.norm(dir1) + dir2 / np.linalg.norm(dir2)
        second_point = intersection + 100 * np.array([dir[0], dir[1], 0])
        center_line_coefficients = np.cross(intersection, second_point)
        center_line_coefficients = - np.array([center_line_coefficients[0], center_line_coefficients[2]]) / center_line_coefficients[1]
        center_line = np.poly1d(center_line_coefficients)
        cv2.line(image, (int(center_line(x_min)), x_min), (int(center_line(x_max)), x_max), (255, 0, 0), thickness=3)
    return image, phase, in_turn

'''
turn_prediction
    input: current image frame and last image frame
    determines what direction the car is turning
    output: indicator of current direction
current_image_top = slice(current_frame)
last_image_top = slice(last_frame)
left_difference = averageDifference(image frames staggered left)
right_difference = averageDifference(image frames staggered right)
turn_amount = right_difference - left_difference
if turn_amount > threshold:
    turn_indicator = right
else if turn_amount < -threshold:
    turn_indicator = left
else:
    turn_indicator = straight
return turn_indicator
'''

def turn_prediction(image, last_image, turn_index, last_occurrence, turn_status, top_height=40, turn_margin=6, change_threshold=13, decay=0.95, turn_threshold=30, smoothing_delay=4):
    if last_image is None:
        return 0, 0, 0, 0
    current_top = cv2.cvtColor(image[:top_height, :], cv2.COLOR_BGR2GRAY)
    last_top = cv2.cvtColor(last_image[:top_height, :], cv2.COLOR_BGR2GRAY)
    right_index = np.mean((current_top[:, turn_margin:] - last_top[:, :-turn_margin]) ** 2)
    left_index = np.mean((current_top[:, :-turn_margin] - last_top[:, turn_margin:]) ** 2)
    net_change = right_index - left_index
    if net_change > change_threshold:
        turn_index += net_change - change_threshold
    elif net_change < -change_threshold:
        turn_index += net_change + change_threshold
    else:
        turn_index = np.round(turn_index * decay, decimals=3)
    if np.abs(turn_index) > turn_threshold:
        last_occurrence = smoothing_delay
    else:
        if last_occurrence != 0:
            last_occurrence -= 1
    if (last_occurrence == 0 and turn_status in [1, 3]) or (last_occurrence != 0 and turn_status in [0, 2]):
        turn_status = (turn_status + 1) % 4
    turn_indicator = int(turn_status != 0) * np.sign(turn_index)
    return turn_indicator, turn_index, last_occurrence, turn_status

'''
overlay_arrow
    input: current image frame, arrow image, turn indicator
    overlays the arrow depending on what direction the car is turning
    output: overlayed frame
rotateImage(arrow, turn_indicator)
current_frame = current_frame + arrow
'''

def overlay_arrow(image, arrow, turn_indicator, position=(30, 1100)):
    arrow = np.rot90(arrow, k=-turn_indicator)
    alpha_channel = arrow[:, :, 3] / 255
    x, y, _ = arrow.shape
    for color_index in range(0, 3):
        image[position[0]:position[0] + x, position[1]:position[1] + y, color_index] = (1 - alpha_channel) * image[position[0]:position[0] + x, position[1]:position[1] + y, color_index] + alpha_channel * arrow[:x, :y, color_index]
    return image

cap = cv2.VideoCapture('driving720.mov')
arrow = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)
arrow = cv2.resize(arrow, (arrow.shape[1] // 12, arrow.shape[0] // 12), interpolation=cv2.INTER_AREA)

last_frame, turn_index, last_occurrence, turn_status = None, None, None, None
phase, in_turn = 0, False
while cap.isOpened():
    ret, frame = cap.read()
    turn_indicator, turn_index, last_occurrence, turn_status = turn_prediction(frame, last_frame, turn_index, last_occurrence, turn_status)
    last_frame = frame
    frame, phase, in_turn = lane_detection(frame, turn_indicator, phase, in_turn)
    frame = overlay_arrow(frame, arrow, turn_indicator)
    if ret == True:
        cv2.imshow('Overlay', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
