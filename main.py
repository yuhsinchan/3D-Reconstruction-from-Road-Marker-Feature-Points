import sys
import os

sys.path.append(os.environ["root_path"])

import argparse
import cv2
import numpy as np
import open3d as o3d
from utils.corners import get_corners
from utils.camera import Cameras
from rich.console import Console
from rich.progress import track

console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seq", type=str, help="seq1, seq2 or seq3", default="seq1"
    )

    args = parser.parse_args()

    dataset_path = os.path.join(
        os.environ["root_path"], "ITRI_dataset", args.seq, "dataset"
    )
    other_path = os.path.join(
        os.environ["root_path"], "ITRI_dataset", args.seq, "other_data"
    )

    console.log(f"dataset path: {dataset_path}")
    console.log(f"other path: {other_path}")

    # read camera info
    cameras = Cameras()

    frames = os.listdir(dataset_path)

    for f in track(frames):
        # read raw image
        raw_image = cv2.imread(os.path.join(dataset_path, f, "raw_image.jpg"))
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # get corners
        corners = get_corners(seq=args.seq, frame=f)
