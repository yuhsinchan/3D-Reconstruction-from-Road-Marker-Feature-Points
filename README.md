# magic_bye

## Get started
To get started with the project, you should follow these steps:

1. Create a new conda environment for the project using Python 3.8:
    ```bash
    conda create -n magic_bye python=3.8
    ```
2. Activate the new conda environment:
    ```bash
    conda activate magic_bye
    ```
3. Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

These steps will create a new conda environment for the project, activate the environment, and install the required Python packages. Once you have completed these steps, you should be ready to run the project code.
## Dataset

The dataset for this project is divided into two sets: a public set and a private set. 

### Public Set

The public set contains three video sequences:

- `seq1`
- `seq2`
- `seq3`

You can download the public set using the following link:

- [Download Public Set](https://140.112.48.121:25251/sharing/Lw8QTICUf)

### Private Set

The private set contains two video sequences:

- `test1`
- `test2`

You can download the private set using the following link:

- [Download Private Set](https://140.112.48.121:25251/sharing/PyViYwNsv)

### Path

To use the dataset, you should put the `ITRI_dataset` and `ITRI_DLC` folders in the root directory of your project. This will ensure that the code can find the necessary data files and dependencies.

## Usage

Before running any commands, make sure to source the `env.sh` file to set up your environment variables:

```bash
source env.sh
```

### Draw sub_map

Run the following command to visualize the point cloud map:

```bash
python utils/shwo_pcd.py
# or
python utils/show_pcd.py -f /path/to/frames/directory
```

This will open a window displaying the point cloud map of a sub-region of the environment. You can use the mouse to rotate and zoom in/out of the map.
