/*
 * Simple preprocessing utilities for MNIST images
 */
#ifndef INPUT_PREPROC_H
#define INPUT_PREPROC_H

#include <stdint.h>

/*
 * Normalize 28x28 U8 grayscale image to float using MNIST mean/std.
 * dst must have 784 floats available.
 */
void preprocess_u8_to_f32_norm(const uint8_t* src, float* dst);

/*
 * Normalize to float (mean/std), then quantize to U8 using
 * the network input quantization (scale, zero-point).
 * dst must have 784 bytes available.
 */
void preprocess_u8_to_u8_quant_norm(const uint8_t* src, uint8_t* dst);

#endif /* INPUT_PREPROC_H */


