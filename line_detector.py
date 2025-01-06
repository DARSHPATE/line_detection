# Darsh Patel - Edge Detection and Centerline Overlay

import cv2
import numpy as np

# merge edges that are similar enough to be considered the same

def merge_edges(edges, r_threshold = 20, theta_threshold = np.pi / 60, formatted = False):
    if not edges:
        return edges
    if not formatted:
        if edges:
            edges = [(edge[0], edge[1], 1) for edge in edges]
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges[i + 1:]):
            if np.abs(edge1[0] - edge2[0]) < r_threshold and np.abs(edge1[1] - edge2[1]) < theta_threshold:
                edges.pop(j)
                edges.pop(i)
                wc1 = edge1[2] / (edge1[2] + edge2[2])
                wc2 = edge2[2] / (edge1[2] + edge2[2])
                new_r = edge1[0] * wc1 + edge2[0] * wc2
                new_theta = edge1[1] * wc1 + edge2[1] * wc2
                new_weight = edge1[2] + edge2[2]
                edges.append((new_r, new_theta, new_weight))
                return merge_edges(edges, r_threshold, theta_threshold, True)
    return [[edge[0], edge[1]] for edge in edges]

# find the edges in the image using the Hough Line Transform

def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_transform = cv2.Canny(gray, 100, 200)
    edges = cv2.HoughLines(canny_transform, rho = 1, theta = np.pi/180, threshold = 100)
    if edges:
        edges = [edge[0] for edge in edges]
    return edges

# determine pairs of edges that should be considered parallel

def find_parallel_pairs(edges, angle_threshold = np.pi / 6, pairs = []):
    if not edges:
        return ()
    edges = edges + []
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges[i + 1:]):
            diff = edge1[1] - edge2[1]
            if (np.array((abs(diff), abs(diff - np.pi), abs(diff + np.pi))) < angle_threshold).any():
                edges.pop(i + j + 1)
                edges.pop(i)
                pairs.append((edge1, edge2))
                return find_parallel_pairs(edges, angle_threshold, pairs)
    return pairs

# display edges on the image

def mark_edges(edges, image):
    if edges:
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
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# live overlay

feed = cv2.VideoCapture(0)

while True:
    ret, image = feed.read()

    edges = merge_edges(find_edges(image))                  # finds edges
    mark_edges(edges, image)                                # marks edges on image
    mark_centerlines(find_parallel_pairs(edges), image)     # marks centerlines on image

    # Display the result
    cv2.imshow('Overlay', image)                    # display the overlay

    # Press 'q' to quit
    if cv2.waitKey(0):
        break

feed.release()
cv2.destroyAllWindows()