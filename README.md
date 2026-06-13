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

## Projects

### 1. Manually controlled green LED (Done - moved on)

The onboard button (B1) triggers an EXTI interrupt that currently toggles LD2 and sends a UART message.

### 2. Sine Wave Approximator (Currently done)

A 3-layer dense network (`32 → 16 → 1`, ReLU activations) trained on noisy `sin(x)` samples over `[0, 2π]`. After training, the model is:

1. Converted to TFLite with default optimizations (dynamic range quantization)
2. Serialized as a C byte array in `sine_model.h`
3. Linked into the firmware and executed via the TFLite Micro interpreter

<div align="center">
  <p align="center">
    <a href="https://github.com/Eduardo79Silva/stm32-edge-ai">
      <img src="public/sine_model.gif" alt="SineModel" width="30%" height="15%">
    </a>
    <br />
    <br />
    Neural Network infering sine wave loop bare-metal in a Cortex-M4
  </p>

</div>

<br />


### 3. Embedded inference model for gesture classification (Next phase)

Deploy a model trained on host that interacts with sensors to detect specific gestures.

## Getting Started

### Prerequisites

- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) (tested on 1.15+)
- Python 3.10+ with `tensorflow`, `numpy`, `matplotlib`
- A serial terminal (e.g. `minicom`, `picocom`, PuTTY) at **115200 8N1**

## Usage

### 1. Train and convert the model

```bash
cd ml
python3 main.py
```

This generates `models/sine_model.tflite` and `models/sine_model.h`.

### 2. Build the firmware

```bash
cd firmware
cmake --preset Debug
cmake --build --preset Debug -j$(nproc)
```

### 3. Flash the firmware

```bash
openocd -f interface/stlink.cfg -f target/stm32l4x.cfg \
  -c "program /path/to/repository/firmware/build/Debug/stm32-blinky.elf verify reset exit"
```

### 4. Monitor UART output

Find your serial port:

```bash
ls /dev/tty* | grep -E "ACM|USB"
```

Connect via minicom at 115200 baud:

```bash
minicom -D /dev/ttyACM0 -b 115200
```

You should see x and predicted sin(x) values streaming:

```
x: 0.0000, y: 0.0401
x: 0.0500, y: 0.0703
```

## References

- [Getting Started with Embedded AI — TinyML on STM32 Nucleo-L476RG](https://medium.com/@switches0011/getting-started-with-embedded-ai-tinyml-stm32-nucleo-l476rg-with-tensorflow-lite-micro-6476deb6fe5f)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [STM32L476RG Datasheet](https://www.st.com/en/microcontrollers-microprocessors/stm32l476rg.html)
- [UM1724 — STM32 Nucleo-64 User Manual](https://www.st.com/resource/en/user_manual/um1724-stm32-nucleo64-boards-mb1136-stmicroelectronics.pdf)

## License

MIT
