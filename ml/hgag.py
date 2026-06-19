import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from pathlib import Path

SAMPLE_SIZE = 1_000

TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1

DATASET_DIR = "./datasets/HGAG-DATA/HGAG-DATA1"

MODEL_FILE_NAME = "hgag_classifier"
MODEL_FILE_DIR = "../models/"
MODEL_PATH = f"{MODEL_FILE_DIR}gesture_model.keras"
C_HEADER_DIR = "../models/"
PI = np.pi

GESTURE_MAPPING = {"Clapping": 0, "Fist Making": 1, "Thumb Up": 2, "Wrist Flexion": 3}


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
    print(y[0], y[2150], y[4300], y[6450])

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

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = Sequential()

        model.add(
            Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                input_shape=(250, 6),
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=7, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(4, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        model.summary()

        history = model.fit(
            X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val)
        )

        model.save(MODEL_PATH)

    _, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
