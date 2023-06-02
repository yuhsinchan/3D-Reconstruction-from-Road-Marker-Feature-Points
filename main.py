import sys
import os

sys.path.append(os.environ["root_path"])

import csv
import argparse
import cv2
import numpy as np
import open3d as o3d
from utils.corners import get_corners
from utils.camera import Cameras
from utils.pincam import uv2xyz
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

    # write to csv
    for i, f in track(enumerate(frames)):
        if i == 3:
            break

        # read raw image
        raw_image = cv2.imread(os.path.join(dataset_path, f, "raw_image.jpg"))
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # get corners
        corners_uv = get_corners(seq=args.seq, frame=f)
        # transform uv to xyz
        corners_xyz = uv2xyz(corners_uv, os.path.join(dataset_path, f, "camera.csv"))
        # console.log(corners_xyz.shape)
        
        output_name = "output_" + f
        os.makedirs(output_name, exist_ok=True)
        # write corners_xyz to csv
        with open(os.path.join(output_name, "sub_map.csv"), "w") as file:
            writer = csv.writer(file)

            for point in corners_xyz:
                writer.writerow(point)
    
