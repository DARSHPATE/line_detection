import cv2
import numpy as np

def find_circles(image, k=11, tlower=50, tupper=150, param1=30, param2=50):
    gaussian = cv2.GaussianBlur(image, (k, k), 0)
    canny_transform = cv2.Canny(gaussian, tlower, tupper, L2gradient=True)
    circles = cv2.HoughCircles(canny_transform, cv2.HOUGH_GRADIENT, dp=1, minDist=image.shape[0] + image.shape[1], param1=param1, param2=param2, minRadius=0, maxRadius=0)
    if circles is None:
        return circles
    else:
        return [[int(n) for n in circle[0]] for circle in circles]

def mark_circle(circles, image):
    if circles is not None:
        for circle in circles:
            cv2.circle(image, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            cv2.circle(image, (circle[0], circle[1]), 0, (0, 0, 255), 5)

def separate_frames(video_capture):
    frames = []
    while True:
        ongoing, frame = video_capture.read()
        if not ongoing:
            break
        frames.append(frame)
    video_capture.release()
    return frames

def display_frames(frames):
    for frame in frames:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('Circle overlay', frame)
        if cv2.waitKey(25) & 0xFF == 27:
            break

cap = cv2.VideoCapture('IMG_2043.MOV')
frames = separate_frames(cap)
for frame in frames:
    mark_circle(find_circles(frame), frame)
display_frames(frames)
cap.release()
cv2.destroyAllWindows()
