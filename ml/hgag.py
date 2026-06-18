import os

import numpy as np
import tensorflow as tf

SAMPLE_SIZE = 1_000

TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1


MODEL_FILE_NAME = "hgag_classifier"
MODEL_FILE_DIR = "../models/"
C_HEADER_DIR = "../models/"
PI = np.pi


def main():
    data = np.genfromtxt(
        "./datasets/HGAG-DATA/HGAG-DATA1/Clapping/Subject_1/.csv/accel_x_data.csv",
        delimiter=",",
    )

    print(data)
    print(data.shape)


if __name__ == "__main__":
    main()
