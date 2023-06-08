from utils.camera import Cameras
from rich.console import Console
import csv
import numpy as np
import math

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
    extrinsic_matrix = None
    intrinsic_matrix = None

    if camera_name == "b":
        projection_matrix = cameras.back.projection_matrix
        extrinsic_matrix = cameras.back.extrinsic_matrix
        intrinsic_matrix = cameras.back.intrinsic_matrix
    elif camera_name == "f":
        projection_matrix = cameras.front.projection_matrix
        extrinsic_matrix = cameras.front.extrinsic_matrix
        intrinsic_matrix = cameras.front.intrinsic_matrix
    elif camera_name == "fl":
        projection_matrix = cameras.front_left.projection_matrix
        extrinsic_matrix = cameras.front_left.extrinsic_matrix
        intrinsic_matrix = cameras.front_left.intrinsic_matrix
    elif camera_name == "fr":
        projection_matrix = cameras.front_right.projection_matrix
        extrinsic_matrix = cameras.front_right.extrinsic_matrix
        intrinsic_matrix = cameras.front_right.intrinsic_matrix
    else:
        # console.log("camera name is not valid")
        return np.array([])
    
    corners_xyz = []
    # convert uv to xyz

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    f = math.sqrt(fx**2 + fy**2)
    
    for uv in corners_uv:

        # use the uv and projection matrix to calculate the ray
        uv = np.append(uv, 1)   
        line_cam = np.linalg.pinv(projection_matrix) @ uv
        direction_cam = line_cam / np.linalg.norm(line_cam)
        
        # transform the ground plane to the camera coordinate 
        # (z + 1.63 = 0)
        direction_world = extrinsic_matrix @ direction_cam
        camera_position_world = extrinsic_matrix @ np.array([0, 0, 0, 1])
        t = -1.63 / (direction_world[2]) # - camera_position_world[2])
        intersection_point_world = t * direction_world + camera_position_world
        corners_xyz.append(intersection_point_world[:3])

    corners_xyz = np.array(corners_xyz)

    return corners_xyz
        

