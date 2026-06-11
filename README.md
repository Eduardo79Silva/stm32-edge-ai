# stm32-edge-ai

TinyML inference on an STM32L476RG Nucleo-64 using TensorFlow Lite for Microcontrollers. This repository documents a hands-on exploration of the embedded ML workflow — from training a model in Python to deploying quantized inference on bare metal.

The initial project is a sine wave approximator: a small dense network trained in Keras, converted to TFLite with INT8 quantization, exported as a C header, and run on the MCU. It serves as a controlled baseline for understanding the full pipeline before moving to real sensor data.

## Hardware

| Component | Details |
|-----------|---------|
| Board | STM32 Nucleo-64 |
| MCU | STM32L476RG (ARM Cortex-M4, 80 MHz) |
| Flash | 1 MB |
| SRAM | 128 KB |
| Interface | UART2 via USB (ST-LINK, 115200 baud) |

## Repository Structure

```
stm32-edge-ai/
├── firmware/               # STM32CubeIDE project (HAL-based)
│   └── Core/
│       └── Src/
│           └── main.c      # Entry point, GPIO/UART init, EXTI callback
├── ml/
│   └── main.py             # Model training, TFLite conversion, C header export
├── models/                 # Generated model artefacts (ignored by git by default)
│   ├── sine_model.keras
│   ├── sine_model.tflite
│   └── sine_model.h        # C header for embedding in firmware
└── README.md
```

## Projects

### 1. Sine Wave Approximator

A 3-layer dense network (`32 → 16 → 1`, ReLU activations) trained on noisy `sin(x)` samples over `[0, 2π]`. After training, the model is:

1. Converted to TFLite with default optimizations (dynamic range quantization)
2. Serialized as a C byte array in `sine_model.h`
3. Linked into the firmware and executed via the TFLite Micro interpreter

### 2. Manually controlled green LED

The onboard button (B1) triggers an EXTI interrupt that currently toggles LD2 and sends a UART message.

### 3. Embedded inference model for gesture classification (goal)

Deploy a model trained on host that interacts with sensors to detect specific gestures.

## Getting Started

### Prerequisites

- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) (tested on 1.15+)
- Python 3.10+ with `tensorflow`, `numpy`, `matplotlib`
- A serial terminal (e.g. `minicom`, `picocom`, PuTTY) at **115200 8N1**

### Training the Model

```bash
cd ml
pip install tensorflow numpy matplotlib
python main.py
```

This produces `models/sine_model.tflite` and `models/sine_model.h`.

### Flashing the Firmware

1. Open `firmware/` as an STM32CubeIDE project.
2. Copy `models/sine_model.h` into `firmware/Core/Inc/`.
3. Build and flash via the IDE (or `st-flash` if using the open-source toolchain).
4. Connect a serial terminal to the ST-LINK virtual COM port at 115200 baud.
5. Press the blue B1 button: the LED toggles and a message appears on UART.

## References

- [Getting Started with Embedded AI — TinyML on STM32 Nucleo-L476RG](https://medium.com/@switches0011/getting-started-with-embedded-ai-tinyml-stm32-nucleo-l476rg-with-tensorflow-lite-micro-6476deb6fe5f)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [STM32L476RG Datasheet](https://www.st.com/en/microcontrollers-microprocessors/stm32l476rg.html)
- [UM1724 — STM32 Nucleo-64 User Manual](https://www.st.com/resource/en/user_manual/um1724-stm32-nucleo64-boards-mb1136-stmicroelectronics.pdf)

## License

MIT
