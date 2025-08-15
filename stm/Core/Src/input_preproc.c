#include <stdint.h>
#include <math.h>
#include "input_preproc.h"

/* These are the mean/std used during MNIST training */
static const float kMnistMean = 0.1307f;
static const float kMnistStd  = 0.3081f;

/* These are taken from the generated network input quantization
 * metadata in `stm/X-CUBE-AI/App/network.c` (input_output_array_intq).
 * If you regenerate the network, update these accordingly.
 */
static const float kNetInScale = 0.012728233821690083f; /* scale */
static const int   kNetInZp    = 33;                     /* zero-point */

void preprocess_u8_to_f32_norm(const uint8_t* src, float* dst)
{
  for (int i = 0; i < 28 * 28; ++i) {
    const float x = (float)src[i] / 255.0f;
    dst[i] = (x - kMnistMean) / kMnistStd;
  }
}

void preprocess_u8_to_u8_quant_norm(const uint8_t* src, uint8_t* dst)
{
  for (int i = 0; i < 28 * 28; ++i) {
    const float x = (float)src[i] / 255.0f;
    const float norm = (x - kMnistMean) / kMnistStd;
    int q = (int)lrintf((norm / kNetInScale) + (float)kNetInZp);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    dst[i] = (uint8_t)q;
  }
}


