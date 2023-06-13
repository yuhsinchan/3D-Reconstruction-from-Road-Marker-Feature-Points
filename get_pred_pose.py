import os
from utils.ICP import *
from rich.progress import track
import argparse

root_path = os.environ["root_path"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seq", type=str, help="seq1, seq2 or seq3", default=None
    )
    parser.add_argument("-t", "--test", type=str, help="test1, test2", default=None)
    parser.add_argument("--threshold", type=float, help="threshold", default=0.02)
    parser.add_argument("-it", "--iter", type=int, help="iteration", default=30)

    args = parser.parse_args()

    if args.seq is not None:
        path = os.path.join(root_path, "ITRI_dataset", args.seq)
        output_path = os.path.join(root_path, "ITRI_DLC", args.seq)
    elif args.test is not None:
        path = os.path.join(root_path, "ITRI_DLC", args.test)
        output_path = path
    else:
        raise ValueError("Please specify seq or test")

    with open(os.path.join(path, "localization_timestamp.txt"), "r") as f:
        localization_timestamps = f.readlines()

    poses = []

    correspondance = []

    for timestamp in track(localization_timestamps):
        timestamp = timestamp.strip()

        path_name = os.path.join(path, "dataset", timestamp)

        # Target point cloud
        target = csv_reader(os.path.join(path_name, "sub_map.csv"))
        target_pcd = numpy2pcd(target)

        source = csv_reader(os.path.join(path_name, "test_merge_map.csv"))

        if len(source.shape) == 0:
            source = np.array([[0, 0, 0]])
        if len(source.shape) == 1:
            source = source.reshape(-1, 3)

        source_pcd = numpy2pcd(source)

        # Initial pose
        init_pose = csv_reader(os.path.join(path_name, "initial_pose.csv"))

        # Implement ICP
        transformation, n_correspondance = ICP(
            source_pcd,
            target_pcd,
            threshold=args.threshold,
            init_pose=init_pose,
            iteration=args.iter,
        )
        pred_x = transformation[0, 3]
        pred_y = transformation[1, 3]

        poses.append(f"{pred_x} {pred_y}")
        correspondance.append(n_correspondance)

    with open(os.path.join(output_path, "pred_pose.txt"), "w") as f:
        f.write("\n".join(poses))

    # calculate the average and std correspondance
    correspondance = np.array(correspondance)
    print(f"Average correspondance: {np.mean(correspondance)}")
    print(f"Std correspondance: {np.std(correspondance)}")
