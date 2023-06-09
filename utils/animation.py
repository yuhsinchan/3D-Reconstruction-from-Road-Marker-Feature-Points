import sys
import os
import cv2
import csv
import argparse
import open3d as o3d
import numpy as np
import time

from rich.console import Console
from rich.progress import track

print(os.environ)
root_path = os.environ["root_path"]
sys.path.append(root_path)

from utils.corners import get_corners
from utils.ICP import csv_reader, numpy2pcd
from utils.camera import Cameras

console = Console()
cameras = Cameras()
vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[0, 0, 0]])))

# tuning the camera position
points = np.random.rand(1000, 3)
points = points * 30
points = points - 15

# animation time
animation_time = 0.05


pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
# set all the points to black
pcd.paint_uniform_color([0, 0, 0])

vis.add_geometry(pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seq", type=str, help="seq1, seq2 or seq3", default="seq1")
    args = parser.parse_args()
    

    dataset_path = os.path.join(os.environ["root_path"], "ITRI_dataset", args.seq, "dataset")
    unsorted_frames = os.listdir(dataset_path)
    sorted_frames = sorted(unsorted_frames, key=lambda x: int(x.split("_")[0]) * 10**9 + int(x.split("_")[1].ljust(9, "0")))

    
    
    # I have 1576 frames in seq1
    for num in track(range(0, len(sorted_frames), 4)):
        # wait for 1 second
        time.sleep(animation_time)

        frames = sorted_frames[num : num + 4]
        camera_names = []
        corners_xyz = []
        points = []

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
                corners_xyz.append(corner_xyz)


        for i, corner_xyz in enumerate(corners_xyz):
            for point in corner_xyz:
                points.append(point)

        points = np.array(points)
        # if points is empty, then skip this frame
        if(points.size == 0):
            continue

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 0, 0])

        # Update the visualization
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

vis.destroy_window()





    