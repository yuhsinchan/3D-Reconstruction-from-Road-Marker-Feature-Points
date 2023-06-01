import cv2
import pandas as pd
import numpy as np
import os
from typing import Union


def get_corners(
    seg_img=None, seq: str = "", frame: str = ""
) -> Union[np.ndarray, None]:
    """get corners from image and detect bounding box

    Args:
        seg_img (np.ndarray): segmented image
        seq (str): sequence name
        frame (str): frame name
    Returns:
        np.ndarray: corner points, shape: (n, 2)
    """

    if seg_img is None:
        frame_path = os.path.join(
            os.environ["root_path"], "ITRI_dataset", seq, "dataset", frame
        )
        seg_img = cv2.imread(os.path.join(frame_path, "seg.jpg"), cv2.IMREAD_GRAYSCALE)

    corners = None

    contours, hierarchy = cv2.findContours(
        seg_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, contour in enumerate(contours):
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if corners is None:
            corners = approx.reshape(-1, 2)
        else:
            corners = np.concatenate((corners, approx.reshape(-1, 2)), axis=0)

    return corners
