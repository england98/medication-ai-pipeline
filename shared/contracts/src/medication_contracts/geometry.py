"""Geometry in normalized source-frame coordinates."""

import math


def center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def distance(a, b, width=1, height=1, scale=1):
    return math.hypot((a[0] - b[0]) * width, (a[1] - b[1]) * height) / scale


def point_box_distance(point, box, width, height, scale):
    closest = [max(box[0], min(point[0], box[2])), max(box[1], min(point[1], box[3]))]
    return distance(point, closest, width, height, scale)


def around(points, padding=0.015):
    return [max(0.0, min(p[0] for p in points) - padding),
            max(0.0, min(p[1] for p in points) - padding),
            min(1.0, max(p[0] for p in points) + padding),
            min(1.0, max(p[1] for p in points) + padding)]
