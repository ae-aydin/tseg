import cv2
import numpy as np

### Code is taken from https://github.com/ryouchinsa/Rectlabel-support/blob/master/general_json2yolo.py

def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, _ in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0

def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour

def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)

def mask_to_polygon(mask):
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = []
    contours_parent = []

    for i, contour in enumerate(contours):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    for contour in contours_parent_tmp:
        polygon = contour.squeeze().tolist()
        polygons.append(polygon)
        
    return polygons