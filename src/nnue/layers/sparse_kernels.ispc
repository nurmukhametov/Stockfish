export void sparse_affine_transform_ispc(
    const uniform uint8 input[], const uniform int8 weights[], const uniform int biases[],
    uniform int output[], uniform int InputDimensions, uniform int PaddedInputDimensions,
    uniform int OutputDimensions) {

    uniform int nnz[2048];
    uniform int k = 0;
    for (uniform int j = 0; j < InputDimensions; j += programCount) {
        if (any(input[j + programIndex]))
            nnz[k++] = j;
    }
    uniform int count = k;

    for (uniform int i = 0; i < OutputDimensions; ++i) {
      uniform int offset = i * PaddedInputDimensions;
      int sum = 0;
      for (k = 0; k < count; k++) {
          int j = nnz[k] + programIndex;
          sum += weights[offset + j] * input[j];
      }
      output[i] = biases[i] + reduce_add(sum);
    }
}

// export void my_affine_transform_ispc(
//     const uniform uint8 input[], const uniform int8 weights[], const uniform int biases[],
//     uniform int output[], uniform int InputDimensions, uniform int PaddedInputDimensions,
//     uniform int OutputDimensions) {
// 
//     for (uniform int i = 0; i < OutputDimensions; ++i) {
//       uniform int offset = i * PaddedInputDimensions;
//       int sum = 0;
//       foreach(j = 0 ... InputDimensions) {
//         sum += weights[offset + j] * input[j];
//       }
//       output[i] = biases[i] + reduce_add(sum);
//     }
// }


export uniform int sparse_affine_transform1_ispc(const uniform int8 weights[], const uniform uint8 input[],
                                     uniform int InputDimensions) {
  assume(InputDimensions % programCount == 0);

  int sum = 0;
  foreach (j = 0 ... InputDimensions) {
      sum += weights[j] * (int)input[j];
  }
  return reduce_add(sum);
}

// Specializations for common sizes.
export void sparse_affine_transform_1024_1024_16_ispc(const uniform uint8 input[], const uniform int8 weights[],
                                          const uniform int biases[], uniform int output[]) {
  sparse_affine_transform_ispc(input, weights, biases, output, 1024, 1024, 16);
}

export void sparse_affine_transform_30_32_32_ispc(const uniform uint8 input[], const uniform int8 weights[],
                                      const uniform int biases[], uniform int output[]) {
  sparse_affine_transform_ispc(input, weights, biases, output, 30, 32, 32);
}
