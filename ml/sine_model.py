import os

import numpy as np
import tensorflow as tf

SAMPLE_SIZE = 1_000

TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1


MODEL_FILE_NAME = "sine_model"
MODEL_FILE_DIR = "../models/"
C_HEADER_DIR = "../models/"
PI = np.pi


def main():
    RANDOM_SEED = 25
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    x = np.random.uniform(low=0, high=2 * PI, size=SAMPLE_SIZE)
    y = np.sin(x) + np.random.normal(0, 0.1, size=SAMPLE_SIZE)

    x_train, x_test, x_validate = np.split(
        x,
        [int(TRAIN_RATIO * SAMPLE_SIZE), int((TRAIN_RATIO + TEST_RATIO) * SAMPLE_SIZE)],
    )
    y_train, y_test, y_validate = np.split(
        y,
        [int(TRAIN_RATIO * SAMPLE_SIZE), int((TRAIN_RATIO + TEST_RATIO) * SAMPLE_SIZE)],
    )

    sine_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    sine_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    sine_model.fit(
        x_train,
        y_train,
        batch_size=100,
        epochs=2000,
        validation_data=(x_validate, y_validate),
    )

    if not os.path.isdir(MODEL_FILE_DIR):
        os.makedirs(MODEL_FILE_DIR)

    sine_model.save(f"{MODEL_FILE_DIR}sine_model.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(sine_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    sine_tflite_model = converter.convert()

    with open(f"{MODEL_FILE_DIR}{MODEL_FILE_NAME}.tflite", "wb") as file:
        file.write(sine_tflite_model)

    sine_tflite_model_split_line = np.array_split(
        [format(hex_value, "#04x") for hex_value in sine_tflite_model],
        len(sine_tflite_model) // 8,
    )

    if not os.path.isdir(C_HEADER_DIR):
        os.makedirs(C_HEADER_DIR)

    open(f"{C_HEADER_DIR}{MODEL_FILE_NAME.lower()}.h", "w").write(f"""
#ifndef {MODEL_FILE_NAME.upper()}_H
#define {MODEL_FILE_NAME.upper()}_H

    const unsigned int {MODEL_FILE_NAME.lower()}_len = {len(sine_tflite_model)};

    const unsigned char {MODEL_FILE_NAME.lower()}[{len(sine_tflite_model)}] = {{
        {",\n    ".join([", ".join(line) for line in sine_tflite_model_split_line])}
    }};

#endif // {MODEL_FILE_NAME.upper()}_H
    """)


if __name__ == "__main__":
    main()
