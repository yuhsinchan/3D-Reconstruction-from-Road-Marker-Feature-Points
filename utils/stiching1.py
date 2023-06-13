import sys
import os

sys.path.append(os.environ["root_path"])

import cv2
import csv
import argparse
import numpy as np
from utils.corners import get_corners
from utils.camera import Cameras
from rich.console import Console
from rich.progress import track

console = Console()

# input 4 consecutive frames corner_xyz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seq", type=str, help="seq1, seq2 or seq3", default="seq1"
    )
    parser.add_argument("-t", "--test", type=str, help="test1 or test2", default=None)

    args = parser.parse_args()

    if args.test is not None:
        dataset_path = os.path.join(
            os.environ["root_path"], "ITRI_DLC", args.test, "dataset"
        )
        other_path = os.path.join(
            os.environ["root_path"], "ITRI_DLC", args.test, "other_data"
        )
    else:
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

    unsorted_frames = os.listdir(dataset_path)

    # sort the subfolders based on the capture time
    sorted_frames = sorted(
        unsorted_frames,
        key=lambda x: int(x.split("_")[0]) * 10**9
        + int(x.split("_")[1].rjust(9, "0")),
    )
    print(sorted_frames[-4:])
    # I have 1576 frames in seq1
    # read 4 frames at a time
    for num in track(range(0, len(sorted_frames) - 4)):
        frames = sorted_frames[num : num + 4]
        # console.log(f"frames: {frames}")
        camera_names = []
        corners_xyz = []

        # 'fl', 'f', 'b', 'fr'
        # read camera.csv to check the camera name sequence
        for frame in frames:
            with open(os.path.join(dataset_path, frame, "camera.csv"), "r") as file:
                reader = csv.reader(file)
                camera_name = next(reader)[0]
                camera_name = camera_name.split("_")[4]
                camera_names.append(camera_name)

            with open(os.path.join(dataset_path, frame, "test_map.csv"), "r") as file:
                reader = csv.reader(file)
                corner_xyz = np.array(list(reader), dtype=np.float32)
                # if corner_xyz.shape[0] == 0:
                #     console.log(frame)
                #     console.log(f"corner_xyz: {corner_xyz.shape}")
                corners_xyz.append(corner_xyz)

        with open(
            os.path.join(dataset_path, frames[0], "test_merge_map.csv"), "w"
        ) as file:
            writer = csv.writer(file)

            # calibrate the xyz of corners
            for i, corner_xyz in enumerate(corners_xyz):
                for point in corner_xyz:
                    writer.writerow(point)

        # console.log("done")

        # break
