from utils.camera import Cameras
from rich.console import Console
import csv
import numpy as np

console = Console()
cameras = Cameras()

# input 

def uv2xyz(corners_uv, camera_name):
    # corners is an array of shape (n, 2)
    # camera is an instance of Camera

    if corners_uv is None:
        # console.log("corners_uv is empty")
        return np.array([])

    with open(camera_name, "r") as file:
            reader = csv.reader(file)
            camera_name = next(reader)[0]
            # console.log(f"camera name: {camera_name}")

    camera_name = camera_name.split("_")[4]
    # console.log(f"camera name: {camera_name}")

    projection_matrix = None

    if camera_name == "b":
        projection_matrix = cameras.back.projection_matrix
    elif camera_name == "f":
        projection_matrix = cameras.front.projection_matrix
    elif camera_name == "fl":
        projection_matrix = cameras.front_left.projection_matrix
    elif camera_name == "fr":
        projection_matrix = cameras.front_right.projection_matrix
    else:
        # console.log("camera name is not valid")
        return np.array([])
    
    corners_xyz = []
    # convert uv to xyz
    
    for uv in corners_uv:
        uv = np.append(uv, 1)
        xyz = np.linalg.pinv(projection_matrix) @ uv
        corners_xyz.append(xyz[:3])

    corners_xyz = np.array(corners_xyz)

    return corners_xyz
        

