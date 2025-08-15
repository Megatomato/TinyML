#ifndef MNIST_SAMPLES_H
#define MNIST_SAMPLES_H

#include <stdint.h>

#define MNIST_NUM_SAMPLES 10

extern const uint8_t mnist_images[MNIST_NUM_SAMPLES][28 * 28];
extern const int mnist_labels[MNIST_NUM_SAMPLES];

#endif /* MNIST_SAMPLES_H */


