# Darsh Patel - circle detection overlay

import cv2
import numpy as np

# Finds the circle in an image using the Hough Circle Transform

def find_circles(image, k=11, tlower=100, tupper=200, param1=30, param2=50):
    gaussian = cv2.GaussianBlur(image, (k, k), 0)
    canny_transform = cv2.Canny(gaussian, tlower, tupper)
    circles = cv2.HoughCircles(canny_transform, cv2.HOUGH_GRADIENT, dp=1, minDist=image.shape[0] + image.shape[1], param1=param1, param2=param2, minRadius=0, maxRadius=0)
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
mark_circle(find_circles(image), image)
cv2.imshow("Overlay", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
