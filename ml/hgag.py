import os

from scipy import signal
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from pathlib import Path

DATASET_DIR = "./datasets/HGAG-DATA/HGAG-DATA1"
GESTURE_MAPPING = {"Clapping": 0, "Fist Making": 1, "Thumb Up": 2, "Wrist Flexion": 3}

MODEL_FILE_NAME = "hgag_classifier"
MODEL_FILE_DIR = "../models/"
MODEL_PATH = f"{MODEL_FILE_DIR}gesture_model.keras"
C_HEADER_DIR = "../models/"

TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 32

FILTER_ORDER = 4
FILTER_LOW_HZ = 30
FILTER_HIGH_HZ = 90
SAMPLE_RATE_HZ = 200

SHOW_FILTER_PLOT = False


def count_dirs(path: str) -> int:
    """Count subdirectories in a given path."""
    return sum(1 for entry in Path(path).iterdir() if entry.is_dir())


def load_dataset(
    dataset_dir: str, gesture_mapping: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load HGAG-DATA1 dataset from disk.

    Each gesture folder contains per-subject CSV files for 6 axes
    (accel_x/y/z, gyro_x/y/z). Each CSV row is one repetition of 250 samples.

    Returns:
        X: shape (n_samples, 250, 6)
        y: shape (n_samples,) with integer class labels
    """
    subjects_count = count_dirs(f"{dataset_dir}/Clapping/")
    X, y = [], []

    for gesture, label in gesture_mapping.items():
        for subject in range(1, subjects_count + 1):
            subject_data = []
            for measure in ["accel", "gyro"]:
                for axis in ["x", "y", "z"]:
                    path = f"{dataset_dir}/{gesture}/Subject_{subject}/.csv/{measure}_{axis}_data.csv"
                    subject_data.append(np.genfromtxt(path, delimiter=","))

            subject_data = np.stack(subject_data, axis=1)
            subject_data = np.transpose(subject_data, axes=(0, 2, 1))
            X.append(subject_data)
            y += [label] * len(subject_data)

    return np.array(np.vstack(X)), np.array(y)


def split_dataset(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Split dataset into train, validation, and test sets (70/10/20).
    Uses stratified splitting to preserve class distribution.

    Note: random shuffle split — not subject-separated. Evaluation is optimistic.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape: tuple, n_classes: int) -> Sequential:
    """
    Build 1D CNN for gesture classification.

    Architecture: Conv1D(32) -> MaxPool -> Conv1D(64) -> MaxPool ->
                  GlobalAveragePooling -> Dense(n_classes, softmax)

    Args:
        input_shape: (timesteps, channels), e.g. (250, 6)
        n_classes: number of gesture classes
    """
    model = Sequential()
    model.add(
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=7, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Sequential:
    """Train model and save to disk."""
    model.summary()
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
    )
    model.save(MODEL_PATH)
    return model


def convert_to_tflite(model: Sequential, X_train: np.ndarray) -> bytes:
    """
    Convert Keras model to int8 quantized TFLite.

    Uses full integer quantization with a representative dataset sampled
    from X_train to calibrate activation ranges.
    Output type kept as int8; input type is int8.
    """

    def representative_dataset_gen():
        idx = np.random.choice(len(X_train), 1000, replace=False)
        for i in idx:
            yield [X_train[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def save_tflite_header(tflite_model: bytes, header_dir: str, model_name: str) -> None:
    """Serialize TFLite model as a C header file for firmware inclusion."""
    os.makedirs(header_dir, exist_ok=True)
    split_lines = np.array_split(
        [format(b, "#04x") for b in tflite_model],
        len(tflite_model) // 8,
    )
    with open(f"{header_dir}{model_name.lower()}.h", "w") as f:
        f.write(f"#ifndef {model_name.upper()}_H\n")
        f.write(f"#define {model_name.upper()}_H\n\n")
        f.write(
            f"    const unsigned int {model_name.lower()}_len = {len(tflite_model)};\n\n"
        )
        f.write(
            f"    const unsigned char {model_name.lower()}[{len(tflite_model)}] = {{\n"
        )
        f.write(",\n    ".join([", ".join(line) for line in split_lines]))
        f.write("\n    };\n\n")
        f.write(f"#endif // {model_name.upper()}_H\n")


def evaluate_quantized_model(
    tflite_model_path: str, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Evaluate quantized TFLite model accuracy on test set.

    Applies input quantization (scale + zero_point) before inference.
    Returns accuracy as a float.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    params = input_details[0]["quantization_parameters"]
    scale = params["scales"][0]
    zero_point = params["zero_points"][0]

    correct = 0
    for i in range(len(X_test)):
        sample = X_test[i : i + 1].astype(np.float32)
        sample_int8 = np.clip(np.round(sample / scale + zero_point), -128, 127).astype(
            np.int8
        )
        interpreter.set_tensor(input_details[0]["index"], sample_int8)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        if np.argmax(output) == y_test[i]:
            correct += 1

    return correct / len(X_test)


def compute_cmsis_filter_coeffs(
    order: int, low_hz: float, high_hz: float, fs: float
) -> np.ndarray:
    """
    Compute CMSIS-DSP biquad coefficients for a Butterworth bandpass filter.

    Converts scipy SOS format [b0,b1,b2,1,a1,a2] to CMSIS format [b0,b1,b2,-a1,-a2].
    Returns array of shape (order, 5).
    """
    sos = np.array(
        signal.butter(order, [low_hz, high_hz], "bandpass", fs=fs, output="sos")
    )
    cmsis = np.delete(sos, 3, axis=1)
    cmsis[:, 3] = -cmsis[:, 3]
    cmsis[:, 4] = -cmsis[:, 4]
    return cmsis


def print_cmsis_coeffs(cmsis_coeffs: np.ndarray) -> None:
    """Print CMSIS-DSP filter coefficients as a C array for firmware."""
    n_coeffs = cmsis_coeffs.shape[0] * cmsis_coeffs.shape[1]
    print(f"const float32_t filter_coeffs[{n_coeffs}] = {{")
    for row in cmsis_coeffs:
        print(f"    {row[0]}f, {row[1]}f, {row[2]}f, {row[3]}f, {row[4]}f,")
    print("};")


def plot_filter_response(
    sos: np.ndarray, fs: float, low_hz: float, high_hz: float
) -> None:
    """Plot frequency response of a filter given SOS coefficients."""
    w, h = signal.freqz_sos(sos, worN=1024, fs=fs)
    h_db = 20 * np.log10(np.abs(h))
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(w), h_db)
    plt.axvline(x=low_hz, color="r", linestyle="--", label=f"{low_hz}Hz")
    plt.axvline(x=high_hz, color="g", linestyle="--", label=f"{high_hz}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Butterworth Bandpass Filter {low_hz}-{high_hz}Hz")
    plt.legend()
    plt.grid(True)
    plt.ylim(-100, 5)
    plt.show()


def main():
    cache_path = f"{MODEL_FILE_DIR}dataset.npz"
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
    else:
        X, y = load_dataset(DATASET_DIR, GESTURE_MAPPING)
        np.savez(cache_path, X=X, y=y)

    print(f"Dataset: {X.shape[0]} samples, {len(GESTURE_MAPPING)} classes")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_model(input_shape=(250, 6), n_classes=len(GESTURE_MAPPING))
        model = train_model(model, X_train, y_train, X_val, y_val)

    _, test_accuracy = model.evaluate(X_test, y_test, verbose="0")
    print(f"Float model test accuracy: {test_accuracy:.4f}")

    tflite_model = convert_to_tflite(model, np.array(X_train))
    tflite_path = f"{MODEL_FILE_DIR}{MODEL_FILE_NAME}.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    save_tflite_header(tflite_model, C_HEADER_DIR, MODEL_FILE_NAME)

    quantized_accuracy = evaluate_quantized_model(tflite_path, np.array(X_test), y_test)
    print(f"Quantized model test accuracy: {quantized_accuracy:.4f}")

    cmsis_coeffs = compute_cmsis_filter_coeffs(
        FILTER_ORDER, FILTER_LOW_HZ, FILTER_HIGH_HZ, SAMPLE_RATE_HZ
    )
    print_cmsis_coeffs(cmsis_coeffs)

    if SHOW_FILTER_PLOT:
        sos = np.array(
            signal.butter(
                FILTER_ORDER,
                [FILTER_LOW_HZ, FILTER_HIGH_HZ],
                "bandpass",
                fs=SAMPLE_RATE_HZ,
                output="sos",
            )
        )
        plot_filter_response(sos, SAMPLE_RATE_HZ, FILTER_LOW_HZ, FILTER_HIGH_HZ)


if __name__ == "__main__":
    main()
