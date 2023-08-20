static const uniform int WeightScaleBits = 6;

export void clipped_relu_ispc(const uniform int input[], uniform uint8 output[], uniform int size) {
  foreach (j = 0 ... size) {
    output[j] = max(0, min(127, input[j] >> 6));
  }
}

export void sqr_clipped_relu_ispc(const uniform int input[], uniform uint8 output[], uniform int size) {
  foreach (j = 0 ... size) {
    int64 sqr = input[j] * input[j];
    // output[j] = max(0, min(127, (sqr >> (2 * 6)) / 128));
    output[j] = max(0, min(127, sqr >> 19));
  }
}

export void prefetch_ispc(const void *uniform ptr) {
  prefetch_l1(ptr);
}

export void affine_transform_ispc(
    const uniform uint8 input[], const uniform int8 weights[], const uniform int biases[],
    uniform int output[], uniform int InputDimensions, uniform int PaddedInputDimensions,
    uniform int OutputDimensions) {
  #define TILE 16
  assume(OutputDimensions % TILE == 0);
  uniform int sum[TILE];
  uniform uint8 ij;
  for (uniform int i = 0; i < OutputDimensions; i += TILE) {
    foreach (j = 0 ... TILE) sum[j] = 0;
    for (uniform int j = 0; j < InputDimensions; j++) {
      ij = input[j];
      foreach (k = 0 ... TILE) {
        sum[k] += weights[j * OutputDimensions + i + k] * (int)ij;
      }
    }
    foreach (j = 0 ... TILE)
      output[i + j] = biases[i + j] + sum[j];
  }
}

export void my_affine_transform_ispc(
    const uniform uint8 input[], const uniform int8 weights[], const uniform int biases[],
    uniform int output[], uniform int InputDimensions, uniform int PaddedInputDimensions,
    uniform int OutputDimensions) {

    for (uniform int i = 0; i < OutputDimensions; ++i) {
      uniform int offset = i * PaddedInputDimensions;
      int sum = 0;
      foreach(j = 0 ... InputDimensions) {
        sum += weights[offset + j] * input[j];
      }
      output[i] = biases[i] + reduce_add(sum);
    }
}

export uniform int affine_transform1_ispc(const uniform int8 weights[], const uniform uint8 input[],
                                     uniform int InputDimensions) {
  assume(InputDimensions % programCount == 0);

  int sum = 0;
  foreach (j = 0 ... InputDimensions) {
    sum += weights[j] * (int)input[j];
  }
  return reduce_add(sum);
}

// Specializations for common sizes.
export void affine_transform_1024_1024_16_ispc(const uniform uint8 input[], const uniform int8 weights[],
                                          const uniform int biases[], uniform int output[]) {
  affine_transform_ispc(input, weights, biases, output, 1024, 1024, 16);
}

export void affine_transform_30_32_32_ispc(const uniform uint8 input[], const uniform int8 weights[],
                                      const uniform int biases[], uniform int output[]) {
  affine_transform_ispc(input, weights, biases, output, 30, 32, 32);
}

