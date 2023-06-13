import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seq", type=str, help="seq1, seq2, seq3, test1 or test2", default=None
    )

    args = parser.parse_args()
    seq = args.seq

    with open(f"./ITRI_DLC/{seq}/pred_pose.txt", "r") as fp:
        pred_pose = fp.readlines()

    track = []

    for line in pred_pose:
        i, j = line.strip().split()
        track.append([float(i), float(j)])

    x = [i[0] for i in track]
    y = [i[1] for i in track]
    c = [i for i in range(len(track))]

    x = np.array(x)
    y = np.array(y)

    std_x = np.std(x) * 0.7
    std_y = np.std(y) * 0.7

    def find_discontinue(x, y):
        std_x = np.std(x)
        std_y = np.std(y)

        discontinue = []
        for i in range(len(x) - 1):
            if abs(x[i + 1] - x[i]) > 3 * std_x or abs(y[i + 1] - y[i]) > 3 * std_y:
                discontinue.append(i + 1)
        return discontinue

    discon = find_discontinue(x, y)

    if len(discon) == 0:
        x_fine = gaussian_filter1d(x, sigma=std_x)
        y_fine = gaussian_filter1d(y, sigma=std_y)
    else:
        x_fine = gaussian_filter1d(x[: discon[0]], sigma=std_x)
        y_fine = gaussian_filter1d(y[: discon[0]], sigma=std_y)

        for i, point in enumerate(find_discontinue(x, y)[:-1]):
            x_fine = np.concatenate(
                (x_fine, gaussian_filter1d(x[discon[i] : discon[i + 1]], sigma=std_x))
            )
            y_fine = np.concatenate(
                (y_fine, gaussian_filter1d(y[discon[i] : discon[i + 1]], sigma=std_y))
            )

        x_fine = np.concatenate(
            (x_fine, gaussian_filter1d(x[discon[-1] :], sigma=std_x))
        )
        y_fine = np.concatenate(
            (y_fine, gaussian_filter1d(y[discon[-1] :], sigma=std_y))
        )

    if not os.path.exists(f"./solution/{seq}"):
        os.makedirs(f"./solution/{seq}")

    with open(f"./solution/{seq}/pred_pose.txt", "w") as fp:
        for i in range(len(x_fine)):
            fp.write(f"{x_fine[i]} {y_fine[i]}\n")
