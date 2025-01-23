# Darsh Patel - circle detection overlay

import cv2
import numpy as np

# Perform image processing techniques like blurring, masking, and edge detection

def condition_image(image, gray_threshold=40, k=11, tlower=100, tupper=200):
    gaussian = cv2.GaussianBlur(image, (k, k), 0)
    hsv = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([360, gray_threshold, 100]))
    masked = cv2.bitwise_and(gaussian, gaussian, mask=mask)
    canny_transform = cv2.Canny(masked, tlower, tupper)
    return canny_transform

# Finds the circle in an image using the Hough Circle Transform

def find_circles(edge_detected, param1=30, param2=50):
    circles = cv2.HoughCircles(edge_detected, cv2.HOUGH_GRADIENT, dp=1, minDist=image.shape[0] + image.shape[1], param1=param1, param2=param2, minRadius=0, maxRadius=0)
    if circles is None:
        return circles
    else:
        return [[int(n) for n in circle[0]] for circle in circles]

# Marks the circle and its center on the image

def mark_circle(circles, image):
    if circles is not None:
        for circle in circles:
            cv2.circle(image, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            cv2.circle(image, (circle[0], circle[1]), 0, (0, 0, 255), 5)

# Executes image detection processes and then displays the result

image = cv2.imread('can_image.png')
mark_circle(find_circles(condition_image(image)), image)
cv2.imshow("Overlay", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
