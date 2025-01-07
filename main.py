# Darsh Patel - Edge Detection and Centerline Overlay

import cv2
import numpy as np

# merge edges that are similar enough to be considered the same

def merge_edges(edges, r_threshold = 100, theta_threshold = np.pi / 25, _formatted = False):
    if edges is None:
        return []
    if len(edges) > 900:
        return edges
    if not _formatted:
        edges = [(edge[0], edge[1], 1) for edge in edges]
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges[i + 1:]):
            if np.abs(edge1[0] - edge2[0]) < r_threshold and np.abs(edge1[1] - edge2[1]) < theta_threshold:
                edges.pop(i + j + 1)
                edges.pop(i)
                wc1 = edge1[2] / (edge1[2] + edge2[2])
                wc2 = edge2[2] / (edge1[2] + edge2[2])
                new_r = edge1[0] * wc1 + edge2[0] * wc2
                new_theta = edge1[1] * wc1 + edge2[1] * wc2
                new_weight = edge1[2] + edge2[2]
                edges.insert(0, (new_r, new_theta, new_weight))
                return merge_edges(edges, r_threshold, theta_threshold, True)
    return [[edge[0], edge[1]] for edge in edges]

# find the edges in the image using the Hough Line Transform

def find_edges(image, rho = 1, theta = np.pi / 180, threshold = 120):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_transform = cv2.Canny(gray, 100, 200)
    edges = cv2.HoughLines(canny_transform, rho = rho, theta = theta, threshold = threshold)
    if edges is not None:
        edges = [edge[0] for edge in edges]
    return edges

# determine pairs of edges that should be considered parallel

def find_parallel_pairs(edges, angle_threshold = np.pi / 4, _pairs = None):
    if _pairs is None:
        _pairs = []
    if edges is None:
        return ()
    edges = edges + []
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges[i + 1:]):
            diff = edge1[1] - edge2[1]
            if (np.array((abs(diff), abs(diff - np.pi), abs(diff + np.pi))) < angle_threshold).any():
                edges.pop(i + j + 1)
                edges.pop(i)
                _pairs.append((edge1, edge2))
                return find_parallel_pairs(edges, angle_threshold, _pairs)
    return _pairs

# display edges on the image

def mark_edges(edges, image):
    if edges is not None:
        for edge in edges:
            r, theta = edge
            v1 = (np.cos(theta), np.sin(theta))
            v2 = (-np.sin(theta), np.cos(theta))
            x1, y1 = (int(r * v1[0] + 10000 * v2[0]), int(r * v1[1] + 10000 * v2[1]))
            x2, y2 = (int(r * v1[0] - 10000 * v2[0]), int(r * v1[1] - 10000 * v2[1]))
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# find and mark centerlines on the image

def mark_centerlines(pairs, image):
    for pair in pairs:
        midpoints = [0, 0]
        for j, y in enumerate([0, 100]):
            points = [0, 0]
            for i in range(2):
                edge = pair[i]
                r, theta = edge
                t = (y - r * np.sin(theta)) / (np.cos(theta))
                points[i] = r * np.cos(theta) - t * np.sin(theta)
            midpoints[j] = ((points[0] + points[1]) / 2, y)
        x1, y1 = (int(101 * midpoints[0][0] - 100 * midpoints[1][0]), int(101 * midpoints[0][1] - 100 * midpoints[1][1]))
        x2, y2 = (int(-100 * midpoints[0][0] + 101 * midpoints[1][0]), int(-100 * midpoints[0][1] + 101 * midpoints[1][1]))
        try:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except:
            None

# live overlay

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    x_margin = 200
    y_margin = 150
    x1_roi = x_margin
    y1_roi = y_margin
    x2_roi = frame.shape[1] - x_margin
    y2_roi = frame.shape[0] - y_margin

    roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]                       # creates region of interest

    edges = merge_edges(find_edges(roi))                            # finds edges
    if edges is not None and len(edges) < 10:
        mark_edges(edges, roi)                                      # marks edges on image
    parallel_pairs = find_parallel_pairs(edges)
    if parallel_pairs is not None and len(parallel_pairs) < 3:
        mark_centerlines(find_parallel_pairs(edges), roi)           # marks centerlines on image

    frame[y1_roi:y2_roi, x1_roi:x2_roi] = roi
    cv2.rectangle(frame, (x_margin, y_margin), (frame.shape[1] - x_margin, frame.shape[0] - y_margin), (0, 0, 255), 2)

    cv2.imshow('Overlay', frame)                                    # display the overlay

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
