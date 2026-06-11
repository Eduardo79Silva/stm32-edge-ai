import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Dataset parameters
SAMPLE_SIZE = 1_000  # Total data points to generate

TRAIN_RATIO = 0.7  # 70% training data
TEST_RATIO = 0.2  # 20% testing data
VAL_RATIO = 0.1  # 10% validation data

# Model configuration
MODEL_FILE_NAME = "sine_model"
MODEL_FILE_DIR = "../models/"
C_HEADER_DIR = "../models/"  # Output directory for generated files
PI = np.pi  # Using numpy's more precise constant


def main():
    # Configuration for reproducibility
    RANDOM_SEED = 25
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)  # for reproduction
    tf.random.set_seed(RANDOM_SEED)

    x = np.random.uniform(low=0, high=2 * PI, size=SAMPLE_SIZE)
    y = np.sin(x) + np.random.normal(0, 0.1, size=SAMPLE_SIZE)
    print(x.shape, y.shape)

    # Split dataset into training, validation, and test sets
    x_train, x_test, x_validate = np.split(
        x,
        [int(TRAIN_RATIO * SAMPLE_SIZE), int((TRAIN_RATIO + TEST_RATIO) * SAMPLE_SIZE)],
    )
    y_train, y_test, y_validate = np.split(
        y,
        [int(TRAIN_RATIO * SAMPLE_SIZE), int((TRAIN_RATIO + TEST_RATIO) * SAMPLE_SIZE)],
    )

    # Visualize data splits
    plt.scatter(x_train, y_train, marker=".", label="Train")
    plt.scatter(x_test, y_test, marker=".", label="Test")
    plt.scatter(x_validate, y_validate, marker=".", label="Validate")
    plt.title("sin(x) vs x")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.show()

    sine_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    print(sine_model.summary())

    sine_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    history = sine_model.fit(
        x_train,
        y_train,
        batch_size=100,
        epochs=2000,
        validation_data=(x_validate, y_validate),
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training history
    axes[0].plot(history.history["loss"], label="training_loss")
    axes[0].plot(history.history["val_loss"], label="validation_loss")
    axes[0].set_title("Model loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (mse)")

    axes[1].plot(history.history["mae"], label="training_mae")
    axes[1].plot(history.history["val_mae"], label="validation_loss")
    axes[1].set_title("Model MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mae")

    plt.show()

    y_test_predict = sine_model.predict(x_test)
    y_test_true = np.sin(x_test)

    # Visual comparison
    plt.scatter(x_test, y_test_predict, marker=".", label="Predicted value")
    plt.scatter(x_test, y_test_true, marker=".", label="True value")
    plt.title("Prediction and True Values")
    plt.ylabel("sin(x)")
    plt.xlabel("x")
    plt.legend()
    plt.show()

    # Save Keras model
    sine_model.save(f"{MODEL_FILE_DIR}sine_model.keras")

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(sine_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    sine_tflite_model = converter.convert()

    # Create model path if it does not exist
    if not os.path.isdir(MODEL_FILE_DIR):
        os.makedirs(MODEL_FILE_DIR)

    # Save as .tflite file
    with open(f"{MODEL_FILE_DIR}{MODEL_FILE_NAME}.tflite", "wb") as file:
        file.write(sine_tflite_model)

    # Breaking the byte in several lines to fit better in the c header file
    sine_tflite_model_split_line = np.array_split(
        [format(hex_value, "#04x") for hex_value in sine_tflite_model],
        len(sine_tflite_model) // 8,
    )

    # Create model path if it does not exist
    if not os.path.isdir(C_HEADER_DIR):
        os.makedirs(C_HEADER_DIR)

    # Write TfLite model to a C header file
    open(f"{C_HEADER_DIR}{MODEL_FILE_NAME.lower()}.h", "w").write(f"""
#ifndef {MODEL_FILE_NAME.upper()}_H
#define {MODEL_FILE_NAME.upper()}_H

    const unsigned int {MODEL_FILE_NAME.lower()}_len = {len(sine_tflite_model)};

    const unsigned char {MODEL_FILE_NAME.lower()}[{len(sine_tflite_model)}] = {{
        {",\n    ".join([", ".join(line) for line in sine_tflite_model_split_line])}
    }};

#endif // {MODEL_FILE_NAME.upper()}_H
    """)

    interpreter = tf.lite.Interpreter(model_path=f"{MODEL_FILE_DIR}{MODEL_FILE_NAME}.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_test_tflite_predict = []
    for x_ in x_test:
        # Set input tensor
        interpreter.set_tensor(
            input_details[0]["index"], np.array([[x_]], dtype=np.float32)
        )
        # Run the inference
        interpreter.invoke()
        y_test_tflite_predict.append(interpreter.get_tensor(output_details[0]["index"]))

    # Compare performance
    y_test_tflite_predict = np.array(y_test_tflite_predict)
    y_test_predict = sine_model.predict(x_test)
    y_test_true = np.sin(x_test)
    mae_orginal_model = np.sum(np.abs(y_test_predict - y_test)) / y_test.shape[0]
    mae_converted_model = (
        np.sum(np.abs(y_test_tflite_predict - y_test)) / y_test.shape[0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(x_test, y_test_tflite_predict, marker=".", label="True value")
    axes[0].scatter(x_test, y_test_true, marker=".", label="Predicted value")
    axes[0].set_title("Prediction and True Values")
    axes[0].set_ylabel("sin(x)")
    axes[0].set_xlabel("x")
    axes[0].legend()

    axes[1].bar(
        ["orginal model", "converted model"], [mae_orginal_model, mae_converted_model]
    )
    axes[1].set_title("The effect of coversion to model accuracy")
    axes[1].set_ylabel("mae")
    axes[1].set_xlabel("model type")

    plt.show()


if __name__ == "__main__":
    main()
