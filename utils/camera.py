import yaml
import os
import cv2
import numpy as np
from rich.console import Console

console = Console()


class Camera:
    def __init__(self, camera) -> None:
        self.camera = camera

        config_path = os.path.join(
            os.environ["root_path"],
            "ITRI_dataset",
            "camera_info",
            "lucid_cameras_x00",
            f"{self.camera}_camera_info.yaml",
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.width = config["image_width"]
        self.height = config["image_height"]
        self.camera_name = config["camera_name"]
        self.intrinsic_matrix = np.array(config["camera_matrix"]["data"]).reshape(3, 3)
        self.distortion_model = config["distortion_model"]
        self.distortion_coefficients = np.array(
            config["distortion_coefficients"]["data"]
        )
        self.rectification_matrix = np.array(
            config["rectification_matrix"]["data"]
        ).reshape(3, 3)
        self.projection_matrix = np.array(config["projection_matrix"]["data"]).reshape(
            3, 4
        )

        self.mask = cv2.imread(
            os.path.join(
                os.environ["root_path"],
                "ITRI_dataset",
                "camera_info",
                "lucid_cameras_x00",
                f"{self.camera}_mask.png",
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        self.extrinsic_matrix = np.identity(4)

    def calculte_extrinsic_matrix(self, x, y, z, qx, qy, qz, qw):
        R = np.array(
            [
                [
                    1.0 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qw * qz),
                    2 * (qw * qy + qx * qz),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1.0 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1.0 - 2 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )

        T = np.array([x, y, z], dtype=np.float64).reshape(3, 1)

        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = T.reshape(3)

        self.extrinsic_matrix = transformation_matrix @ self.extrinsic_matrix


if __name__ == "__main__":
    camera = Camera("gige_100_fl_hdr")
    console.print("camera_matrix\n", camera.intrinsic_matrix)
    console.print("projection_matrix\n", camera.projection_matrix)
    console.print("distortion_coefficients\n", camera.distortion_coefficients)
    console.print("rectification_matrix\n", camera.rectification_matrix)
    console.print("width:", camera.width)
    console.print("height:", camera.height)
    console.print("camera_name:", camera.camera_name)
    console.print("distortion_model:", camera.distortion_model)
    console.print("mask", camera.mask.shape)
