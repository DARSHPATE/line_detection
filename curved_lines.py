# Darsh Patel - curved line detection

import cv2
import numpy as np

# Applies image processing functions to find a skeletonized version of the two parallel lines

def pre_processing(image, k=11, kernel=3, threshold=64):
    gaussian = cv2.GaussianBlur(image, (k, k), 0)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    black_and_white = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    inverted = cv2.bitwise_not(black_and_white)
    dilated = cv2.dilate(inverted, np.ones((kernel, kernel), np.uint8))
    skeleton = cv2.ximgproc.thinning(dilated)
    return skeleton

# Finds the lists of points for each line of the skeleton

def find_contours(skeleton):
    contours = []
    dir_lists = [[(1, 0), (-1, 0), (0, 1), (0, -1)], [(1, 1), (1, -1), (-1, 1), (-1, -1)]]
    while True:
        ones = np.where(skeleton != 0)
        if len(ones[0]) == 0:
            break
        y = ones[0][0]
        x = ones[1][0]
        neighbor_lists, neighbors = [[], []], []
        for k, dir_list in enumerate(dir_lists):
            for i, j in dir_list:
                if skeleton[y + i][x + j] != 0:
                    neighbor_lists[k].append((y + i, x + j))
        if len(neighbor_lists[0]) == 2:
            neighbors = neighbor_lists[0]
        elif len(neighbor_lists[0]) == 1:
            neighbors = neighbor_lists[0]
            if len(neighbor_lists[1]) == 2:
                if np.abs(neighbor_lists[1][0] - neighbors[0]) + np.abs(neighbor_lists[1][1] - neighbors[1]) == 1:
                    neighbors.append(neighbor_lists[1][0])
                else:
                    neighbors.append(neighbor_lists[1][1])
            else:
                neighbors += neighbor_lists[1]
        else:
            neighbors = neighbor_lists[1]
        new_contours = [None] * len(neighbors)
        for k, neighbor in enumerate(neighbors):
            new_contour = [(y, x), neighbors[k]]
            while True:
                current_y, current_x = new_contour[-1]
                current_neighbors = []
                for dir_list in dir_lists:
                    for i, j in dir_list:
                        if skeleton[current_y + i][current_x + j] != 0:
                            current_neighbors.append((current_y + i, current_x + j))
                flag = True
                for current_neighbor in current_neighbors:
                    if current_neighbor not in new_contour[-2:]:
                        new_contour.append(current_neighbor)
                        flag = False
                        break
                if flag:
                    break
            new_contours[k] = new_contour
        if len(new_contours) == 2:
            new_contour = new_contours[1][:0:-1] + new_contours[0]
        else:
            new_contour = new_contours[0]
        contours.append(new_contour)
        for y, x in new_contour:
            skeleton[y][x] = 0
    return contours

# Separates each parallel lines into equal segments by arc length

def separate_by_arc_length(contour, n, d):
    contour = contour[::d] + [contour[-1]]
    contour = [np.array(point) for point in contour]
    points = [None] * (n + 1)
    parameter, segment, fractional_part = 0, 0, 0
    current_point = contour[0]
    points[0] = current_point
    lengths = [np.linalg.norm(contour[i] - contour[i + 1]) for i in range(len(contour) - 1)]
    average = sum(lengths) / n
    for i in range(n):
        segment = int(np.floor(parameter))
        fractional_part = parameter - segment
        points[i] = contour[segment] + fractional_part * (contour[segment + 1] - contour[segment]) / lengths[segment]
        if i == n - 1:
            break
        deficit = average
        parameter += deficit / lengths[segment]
        if np.floor(parameter) > segment:
            deficit -= (1 - fractional_part) * lengths[segment]
            segment += 1
            while lengths[segment] < deficit:
                deficit -= lengths[segment]
                segment += 1
            parameter = segment + deficit / lengths[segment]
    points[-1] = contour[-1]
    return points

# Finds the centerline by averaging points from the two parallel lines

def find_midline(contours, n=1000, d=4):
    if len(contours) != 2:
        return []
    c1 = separate_by_arc_length(contours[0], n, d)
    c2 = separate_by_arc_length(contours[1], n, d)
    c1begin, c1end, c2begin, c2end = tuple(map(np.array, [c1[0], c1[-1], c2[0], c2[-1]]))
    if np.linalg.norm((c1end - c1begin) - (c2end - c2begin)) > np.linalg.norm((c1end - c1begin) + (c2end - c2begin)):
        c1.reverse()
    return [0.5 * (c1[i] + c2[i]) for i in range(n + 1)]

# Displays the edges and centerline

def display_overlay(image, contours, midline):
    for contour in contours:
        for i in range(len(contour) - 1):
            cv2.line(image, (contour[i][1], contour[i][0]), (contour[i + 1][1], contour[i + 1][0]), (255, 0, 0), 2)
    for i in range(len(midline) - 1):
        cv2.line(image, (int(midline[i][1]), int(midline[i][0])), (int(midline[i + 1][1]), int(midline[i + 1][0])), (0, 255, 0), 2)

image = cv2.imread('path.png')
skeleton = pre_processing(image)
contours = find_contours(skeleton)
midline = find_midline(contours)
display_overlay(image, contours, midline)
cv2.imshow('Overlay', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
