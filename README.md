# TinyML MNIST Digit Classification on STM32F4

A complete TinyML implementation that deploys a quantized neural network for MNIST digit classification on an STM32F446RE microcontroller. This project demonstrates the full pipeline from model training to edge deployment with optimized inference performance.

## Project Overview

This project implements a lightweight CNN that classifies 28x28 pixel handwritten digits (0-7) on a resource-constrained microcontroller, achieving **99.43% accuracy** with only **105.73 KiB** of model weights.

## Performance Metrics

### Model Specifications
| Metric | Value |
|--------|-------|
| **Model Type** | Quantized CNN (INT8) |
| **Input Size** | 28×28×1 (784 bytes) |
| **Output Classes** | 10 digits (0-9) |
| **Parameters** | 28,979 |
| **MAC Operations** | 271,563 |

### Memory Usage
| Component | Size | Description |
|-----------|------|-------------|
| **Model Weights (ROM)** | 108,268 B (105.73 KiB) | Quantized parameters stored in Flash |
| **Activations (RAM)** | 6,808 B (6.65 KiB) | Intermediate layer outputs |
| **Input Buffer** | 784 B | Single image input |
| **Output Buffer** | 7 B | 7-class logits |
| **Total RAM** | 7,599 B (7.42 KiB) | Runtime memory footprint |

### Latency Performance
| Platform | Inference Time | Notes |
|----------|----------------|-------|
| **STM32F446RE @ 180MHz** | *~9 ms* | Measured with HAL_GetTick() |
| **PC (CPU)** | 0.041 ms | ONNX Runtime reference |
| **PC (Non-Quantized)** | 0.028 ms | Float32 baseline |


### Model Accuracy
| Model Version | Accuracy | Latency |
|---------------|----------|---------|
| **Float32 Original** | 99.46% | 0.028 ms |
| **INT8 Quantized** | 99.43% | 0.041 ms |
| **Accuracy Loss** | -0.03% | +46% latency* |

Thq Quantized Model was 67.7% smaller than the Float32 original after static quantization in onnx. 

*PC latency comparison

## 🏗️ Project Structure

```
TinyML/
├── training/                    # Model development pipeline
│   ├── train.py                # Neural network training
│   ├── quantize.py             # INT8 quantization 
│   ├── export_onnx.py          # Model format conversion
│   └── compare.py              # Performance benchmarking
├── artifacts/                   # Generated model files
│   ├── tiny_mnist_best.pt      # PyTorch checkpoint
│   ├── tiny_mnist_best.onnx    # Float32 ONNX model
│   ├── tiny_mnist_best_quantized.onnx  # INT8 quantized model
│   └── model_comparison.json   # Performance metrics
├── stm/                        # STM32 embedded implementation
│   ├── Core/Src/main.c         # Main inference loop
│   ├── Core/Src/input_preproc.c  # Image preprocessing
│   ├── Core/Src/mnist_samples.c  # Test dataset
│   └── X-CUBE-AI/              # ST's AI framework integration
│       ├── App/network.c       # Generated neural network
│       ├── App/network_data.c  # Model weights and parameters
│       └── App/*.h             # Network interface headers
├── sample10/                   # Test images for validation
```

## 🚀 Getting Started

### Prerequisites
- STM32CubeIDE or compatible toolchain
- STM32F446RE Nucleo board (or similar STM32F4)
- Python 3.8+ with PyTorch, ONNX Runtime
- STM32CubeMX with X-CUBE-AI expansion pack

### Build and Deploy
1. **Train the model** (optional - pre-trained artifacts included):
   ```bash
   cd training/
   python train.py
   python quantize.py
   python export_onnx.py
   ```

2. **Generate STM32 code**:
   - Open STM32CubeMX
   - Enable X-CUBE-AI, import `artifacts/tiny_mnist_best_quantized.onnx`
   - Generate code for STM32F4 target

3. **Build and flash**:
   - Import project into STM32CubeIDE
   - Build and flash to target board
   - Monitor UART output at 115200 baud

### Expected Output
```
Sample 0 (label=3): pred=2, latency=Xms, out=[162, 143, 255, 193, 98, 62, 134]
Sample 1 (label=8): pred=2, latency=Xms, out=[185, 214, 238, 223, 101, 68, 96]
Sample 2 (label=6): pred=4, latency=Xms, out=[145, 166, 113, 118, 255, 130, 127]
...
```
