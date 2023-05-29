import cv2
import pandas as pd
import numpy as np


def get_corners(image: np.ndarray, detect_bounding_box: pd.DataFrame) -> np.ndarray:
    """get corners from image and detect bounding box

    Args:
        image (np.ndarray): raw image

        detect_bounding_box (pd.DataFrame): detected bounding boxes from yolo
    Returns:
        np.ndarray: corner points
    """
    corners = []

    for i, row in detect_bounding_box.iterrows():
        x1, y1, x2, y2, class_id, probability = row

        # bound x1, y1, x2, y2
        x1 = max(0, x1)
        x2 = min(image.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(image.shape[0], y2)

        # crop image and convert to grayscale
        im = image[int(y1) : int(y2), int(x1) : int(x2)]

        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # do edge detection
        edges = cv2.Canny(imgray, 50, 150, apertureSize=3)

        # turn the edge image into a binary image
        ret, thresh = cv2.threshold(edges, 127, 255, 0)

        # find the contour of the edge image
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # filter out the contour that has a small area
        contours = [contour for contour in contours if cv2.contourArea(contour) > 1]

        # find corners
        for i, contour in enumerate(contours):
            epsilon = 0.5 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx = approx.reshape(-1, 2) + np.array([x1, y1])
            corners.append(approx)

    return np.array(corners).astype(np.int8)
