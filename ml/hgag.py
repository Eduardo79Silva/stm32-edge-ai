import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

SAMPLE_SIZE = 1_000

TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1

DATASET_DIR = "./datasets/HGAG-DATA/HGAG-DATA1"


MODEL_FILE_NAME = "hgag_classifier"
MODEL_FILE_DIR = "../models/"
C_HEADER_DIR = "../models/"
PI = np.pi

GESTURE_MAPPING = {"Clapping": 1, "Fist Making": 2, "Thumb Up": 3, "Wrist Flexion": 4}


def count_dirs_pathlib(path):
    return sum(1 for entry in Path(path).iterdir() if entry.is_dir())


def main():
    subjects_count = count_dirs_pathlib(f"{DATASET_DIR}/Clapping/")

    X = []
    y = []

    for gesture in GESTURE_MAPPING.keys():
        for subject in range(1, subjects_count + 1):
            subject_data = []
            for measure in ["accel", "gyro"]:
                for axis in ["x", "y", "z"]:
                    subject_data.append(
                        np.genfromtxt(
                            f"{DATASET_DIR}/{gesture}/Subject_{subject}/.csv/{measure}_{axis}_data.csv",
                            delimiter=",",
                        )
                    )

            subject_data = np.stack((subject_data), axis=1)
            subject_data = np.transpose(subject_data, axes=(0, 2, 1))
            X.append(subject_data)
            y += [GESTURE_MAPPING[gesture]] * len(subject_data)

    X = np.array(np.vstack(X))
    y = np.array(y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=42
    )

    print(f"X_train shape: {np.array(X_train).shape}")
    print(f"y_train shape: {np.array(y_train).shape}")
    print(f"X_test shape: {np.array(X_test).shape}")
    print(f"y_test shape: {np.array(y_test).shape}")
    print(f"X_val shape: {np.array(X_val).shape}")
    print(f"y_val shape: {np.array(y_val).shape}")


if __name__ == "__main__":
    main()
