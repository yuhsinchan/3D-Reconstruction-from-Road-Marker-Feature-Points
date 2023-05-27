import os
import sys
import argparse
import open3d as o3d

print(os.environ)
root_path = os.environ["root_path"]
sys.path.append(root_path)
from utils.ICP import csv_reader, numpy2pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show sub_map")
    parser.add_argument(
        "-f",
        "--frame",
        type=str,
        default=f"{root_path}/ITRI_dataset/seq1/dataset/1681710717_532211005",
        help="sub_map path",
    )
    args = parser.parse_args()

    source = csv_reader(f"{args.frame}/sub_map.csv")
    source_pcd = numpy2pcd(source)

    o3d.visualization.draw_geometries([source_pcd], window_name="sub_map")
