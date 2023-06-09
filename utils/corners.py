import cv2
import pandas as pd
import numpy as np
import os
from typing import Union
import scipy.sparse as sp


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
        if "seq" in seq:
            frame_path = os.path.join(
                os.environ["root_path"], "ITRI_dataset", seq, "dataset", frame
            )
            npz_path = os.path.join(frame_path, "seg.npz")
        else:
            frame_path = os.path.join(
                os.environ["root_path"], "ITRI_DLC", seq, "dataset", frame
            )
            npz_path = os.path.join(frame_path, "seg.npz")

        seg_img = sp.load_npz(npz_path).toarray()

    corners = None

    contours, hierarchy = cv2.findContours(
        seg_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, contour in enumerate(contours):
        # if contour is too small, ignore
        if cv2.contourArea(contour) < 300:
            continue

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if corners is None:
            corners = approx.reshape(-1, 2)
        else:
            corners = np.concatenate((corners, approx.reshape(-1, 2)), axis=0)

    return corners
