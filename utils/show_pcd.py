import os
import sys
import argparse
import open3d as o3d
import numpy as np

print(os.environ)
root_path = os.environ["root_path"]
sys.path.append(root_path)
from utils.camera import Cameras
from utils.ICP import csv_reader, numpy2pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show sub_map")
    parser.add_argument(
        "-m",
        "--map",
        type=str,
        default="sub_map",
        help="sub_map or test_map",
    )
    args = parser.parse_args()

    source = csv_reader(f"{root_path}/{args.map}")
    source_pcd = numpy2pcd(source)

    # create a new point cloud for the origin
    origin = np.array([[0, 0, 0]])
    origin_pcd = o3d.geometry.PointCloud()
    origin_pcd.points = o3d.utility.Vector3dVector(origin)
    origin_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # set the color to red

    # create a new point cloud for the scale
    scale = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    scale_pcd = o3d.geometry.PointCloud()
    scale_pcd.points = o3d.utility.Vector3dVector(scale)
    scale_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0], [0, 1, 0], [0, 1, 0]])

    # create a new LineSet geometry to connect the scale points and origin with lines
    lines = [[0, 1], [0, 2], [0, 3]]
    # red for x-axis, green for y-axis, blue for z-axis
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector()
    line_set.points.extend(origin_pcd.points)
    line_set.points.extend(scale_pcd.points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)


    # read camera info
    cameras = Cameras()
    matrix_b = cameras.back.extrinsic_matrix
    matrix_f = cameras.front.extrinsic_matrix
    matrix_fl = cameras.front_left.extrinsic_matrix
    matrix_fr = cameras.front_right.extrinsic_matrix

    # create camera point cloud
    camera_b = matrix_b @ np.array([[0, 0, 0, 1]]).T
    camera_b = camera_b[:3, :].T
    camera_b_pcd = o3d.geometry.PointCloud()
    camera_b_pcd.points = o3d.utility.Vector3dVector(camera_b)
    camera_b_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

    camera_f = matrix_f @ np.array([[0, 0, 0, 1]]).T
    camera_f = camera_f[:3, :].T
    camera_f_pcd = o3d.geometry.PointCloud()
    camera_f_pcd.points = o3d.utility.Vector3dVector(camera_f)
    camera_f_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

    camera_fl = matrix_fl @ np.array([[0, 0, 0, 1]]).T
    camera_fl = camera_fl[:3, :].T
    camera_fl_pcd = o3d.geometry.PointCloud()
    camera_fl_pcd.points = o3d.utility.Vector3dVector(camera_fl)
    camera_fl_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    
    camera_fr = matrix_fr @ np.array([[0, 0, 0, 1]]).T
    camera_fr = camera_fr[:3, :].T
    camera_fr_pcd = o3d.geometry.PointCloud()
    camera_fr_pcd.points = o3d.utility.Vector3dVector(camera_fr)
    camera_fr_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])



    # add the point clouds and LineSet to the visualization
    # o3d.visualization.draw_geometries([source_pcd, origin_pcd, line_set], point_show_normal=False)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    opt = vis.get_render_option()
    opt.point_show_normal = False

    vis.add_geometry(source_pcd)
    vis.add_geometry(origin_pcd)
    vis.add_geometry(line_set)

    vis.add_geometry(camera_b_pcd)
    vis.add_geometry(camera_f_pcd)
    vis.add_geometry(camera_fl_pcd)
    vis.add_geometry(camera_fr_pcd)
    
    vis.run()
    vis.destroy_window()