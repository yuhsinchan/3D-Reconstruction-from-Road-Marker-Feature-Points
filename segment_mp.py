import os
import cv2
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
from mpire import WorkerPool

from rich.console import Console

console = Console()

overlap_threshold = 0.5
in_box_threshold = 0.3
black_threshold = 0.1

dataset_path: str = ""
mask_path: str = ""
root_path: str = ""


def draw_box(image, box, color=(255, 0, 0), thickness=2):
    x1, y1, x2, y2, class_id, probability = box

    if probability < 0.2:
        return image

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return image


def draw_markers(anns, h, w, img_thresh):
    # draw a binary image
    img = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        m = ann["mask"]["segmentation"]

        area = m.sum()
        # calculate the portion of m in img_thresh
        img_thresh_area = np.logical_and(m, img_thresh).sum()

        if img_thresh_area / area > black_threshold:
            img[m] = 255

        if ann["per"] > in_box_threshold:
            img[m] = 255

    return img


def get_segment(f):
    if not os.path.exists(os.path.join(dataset_path, f, "raw_image.jpg")):
        return

    # load image
    img_path = os.path.join(dataset_path, f, "raw_image.jpg")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get camera info
    with open(os.path.join(dataset_path, f, "camera.csv"), "r") as fp:
        camera = fp.readlines()[0].strip()[1:]

    camera_mask = cv2.imread(
        os.path.join(
            root_path,
            "ITRI_dataset",
            "camera_info",
            f"{camera}_mask.png",
        ),
        cv2.IMREAD_GRAYSCALE,
    )

    # # downsample from 1440x928 to 640 x 480
    # img = cv2.resize(img, (640, 480))

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_thresh = cv2.threshold(img_gray, 127, 255, 0)[1]

    # get bounding box
    bounding_box_path = os.path.join(dataset_path, f, "detect_road_marker.csv")
    detect_road_marker = pd.read_csv(
        bounding_box_path,
        header=None,
        names=["x1", "y1", "x2", "y2", "class_id", "probability"],
    )

    # output bounding box image
    box_img = img.copy()
    for i, row in detect_road_marker.iterrows():
        box_img = draw_box(box_img, row.values)
    box_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(dataset_path, f, "box_image.jpg"), box_img)

    n = len(os.listdir(os.path.join(mask_path, f)))
    masks = []

    for i in range(n):
        npz_path = os.path.join(mask_path, f, f"{i}.npz")
        sparse_seg = sp.load_npz(npz_path)
        masks.append({"segmentation": sparse_seg.toarray().astype(bool)})
        masks[-1]["area"] = masks[-1]["segmentation"].sum()

    # print(len(masks))

    new_masks = []

    # get masks in bounding box
    for mask in masks:
        in_camera_mask = np.where(camera_mask == 255, mask["segmentation"], 0)
        # print(in_camera_mask.sum() / mask["segmentation"].sum())
        if in_camera_mask.sum() / mask["segmentation"].sum() > 0.1:
            continue

        for i, row in detect_road_marker.iterrows():
            x1, y1, x2, y2, class_id, probability = row

            if float(probability) < 0.3:
                continue

            points = np.where(mask["segmentation"] == 1)
            in_bounding_box = np.logical_and(
                np.logical_and(points[0] >= y1, points[0] <= y2),
                np.logical_and(points[1] >= x1, points[1] <= x2),
            )

            bounding_box_area = (x2 - x1) * (y2 - y1)

            per = in_bounding_box.sum() / mask["segmentation"].sum()
            if per > overlap_threshold:
                new_masks.append(
                    {
                        "mask": mask,
                        "per": in_bounding_box.sum() / bounding_box_area,
                    }
                )
                break

    # draw segmentation image
    bin_img = draw_markers(new_masks, img.shape[0], img.shape[1], img_thresh)

    # # upsample to 1440x928
    # bin_img = cv2.resize(bin_img, (1440, 928))
    cv2.imwrite(os.path.join(dataset_path, f, "seg.jpg"), bin_img)

    # stroe bin image into a npz file
    sp.save_npz(os.path.join(dataset_path, f, "seg.npz"), sp.csr_matrix(bin_img))


if __name__ == "__main__":
    root_path = os.environ["root_path"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seq", type=str, nargs="+", help="seq1, seq2 or seq3", default=None
    )
    parser.add_argument(
        "-t", "--test", type=str, nargs="+", help="test1 or test2", default=None
    )
    parser.add_argument("-m", "--mask", type=str, help="mask path", default="masks")
    args = parser.parse_args()

    dataset_paths = []
    mask_paths = []

    if args.test is not None:
        for test in args.test:
            dataset_paths.append(os.path.join(root_path, "ITRI_DLC", test, "dataset"))
            mask_paths.append(os.path.join(root_path, args.mask, test))
    if args.seq is not None:
        for seq in args.seq:
            dataset_paths.append(
                os.path.join(root_path, "ITRI_dataset", seq, "dataset")
            )
            mask_paths.append(os.path.join(root_path, args.mask, seq))

    for d, m in zip(dataset_paths, mask_paths):
        dataset_path = d
        mask_path = m

        os.makedirs(mask_path, exist_ok=True)
        frames = sorted(os.listdir(d))

        console.log(f"Processing {d}...")

        with WorkerPool(n_jobs=8) as pool:
            pool.map(get_segment, frames, progress_bar=True)
