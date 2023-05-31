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

    def calaulate_extrinsic_matrix(self, *args):
        for matrix in args:
            self.extrinsic_matrix = np.matmul(self.extrinsic_matrix, matrix)
    
    @staticmethod
    def get_extrinsic_matrix(x, y, z, qx, qy, qz, qw) -> np.ndarray:
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

        return transformation_matrix

class Cameras:
    def __init__(self):
        self.back = Camera("gige_100_b_hdr")
        self.front = Camera("gige_100_f_hdr")
        self.front_left = Camera("gige_100_fl_hdr")
        self.front_right = Camera("gige_100_fr_hdr")
        
        base_link_f = Camera.get_extrinsic_matrix(
            0.0, 0.0, 0.0, -0.5070558775462676, 0.47615311808704197, 
            -0.4812773544166568, 0.5334272708696808
        )
        f_fr = Camera.get_extrinsic_matrix(
            0.559084, 0.0287952, -0.0950537, -0.0806252, 0.607127, 
            0.0356452, 0.789699
        )
        f_fl = Camera.get_extrinsic_matrix(
            -0.564697, 0.0402756, -0.028059, -0.117199, -0.575476,
            -0.0686302, 0.806462
        )
        fl_b = Camera.get_extrinsic_matrix(
            0.06742502153707941, 1.723731468585929, 1.886103532139902,
            0.5070558775462676, -0.47615311808704197, 0.4812773544166568,
            0.5334272708696808
        )
        
        # update camera extrinsic matrix
        self.back.calaulate_extrinsic_matrix(base_link_f, f_fl, fl_b)
        self.front.calaulate_extrinsic_matrix(base_link_f)
        self.front_left.calaulate_extrinsic_matrix(base_link_f, f_fl)
        self.front_right.calaulate_extrinsic_matrix(base_link_f, f_fr)

if __name__ == "__main__":
    cameras = Cameras()
    console.print("back\n", cameras.back.extrinsic_matrix)
    console.print("front\n", cameras.front.extrinsic_matrix)
    console.print("front_left\n", cameras.front_left.extrinsic_matrix)
    console.print("front_right\n", cameras.front_right.extrinsic_matrix)
