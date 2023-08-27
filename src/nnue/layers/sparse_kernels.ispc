const uniform int InputSize = 2048;
const uniform int OutputSize = 16;

typedef uniform uint16 LOOKUP_TABLE_TYPE_8[8];

export void ispc_nnz_5_ptr_32(const uniform int32 input[], uniform uint16 out[],
                  uniform uint * uniform count,
                  uniform LOOKUP_TABLE_TYPE_8 lookup_tblind[]) {
  uniform uint c = 0;
  uniform uint base = 0;
  for (uniform int i = 0; i < InputSize / 4; i += programCount) {
    uint32 x = input[i + programIndex];
    uniform uint nnz = packmask(x);
    for (uniform int j = 0; j < 1; j++) {
      uniform uint b = (nnz >> (j * 8)) & 0xFF;
      uniform uint16 *ptr = (uniform uint16*)&lookup_tblind[b];
      // print("b: %, base: %, ptr:%\n", b, base, ptr[0 + programIndex]);
      out[c + programIndex] = base + ptr[0 + programIndex];
      base += 8;
      c += popcnt((uniform int32)b);
    }
  }
  *count = c;
}

varying int32 foo(varying int32 a, varying int32 b) {
    varying int32 x = 0;
    // print("a: %, b: %\n", a, b);
    for (int i = 0; i < 4; i++) {
        varying int32 ai = (varying int8)(((varying unsigned int32)a >> (i * 8)) & 0xFFu);
        varying int32 bi = (varying int8)(((varying unsigned int32)b >> (i * 8)) & 0xFFu);
        // print("ai: %, bi: %\n", ai, bi);
        x += ai * bi;
    }
    // print("x: %\n", x);
    return x;
}

extern "C" void ispc_space_affine_ptr(const uniform int8 input[], uniform int32 output[],
                  const uniform int8 weights[], uniform int32 biases[],
                  uniform LOOKUP_TABLE_TYPE_8 lookup_tblind[]) {
  uniform uint count = 0;
  uniform uint16 nnz[InputSize];
  const uniform int32* uniform input32 = (const uniform int32* uniform)input;

  // print("lookup table:\n");
  // for (uniform int i = 0; i < 100; i++) {
  //     print("%, ", lookup_tblind[i]);
  // }
  // print("\n");

  // print("input32: \n");
  // for (uniform int i = 0; i < InputSize / 4; i++) {
  //     print("%, ", input32[i]);
  // }
  // print("\n");

  ispc_nnz_5_ptr_32(input32, nnz, &count, lookup_tblind);

  // print("count %\n", count);
  // for (uniform int i = 0; i < count; i++) {
  //     print("%, ", nnz[i]);
  // }
  // print("\n");

  uniform int32 acc[OutputSize];
  foreach (i = 0 ... OutputSize) {
      acc[i] = biases[i];
  }

  for(uniform uint j = 0; j < count; j++) {
      uniform uint16 i = nnz[j];
      uniform int32 *col = (uniform int32 *) &weights[i * OutputSize * 4];
      varying int32 in32 = input32[i];
      foreach(k = 0 ... OutputSize) {
          varying int32 x = foo(col[k], in32);
          // print("acc: %, col:%, input32:%, foo:%\n",
          //         acc[k], col[k], in32, x);
          // acc[k] += (int) col[k] * in32;
          acc[k] += x;
      }
  }

  foreach (i = 0 ... OutputSize) {
      output[i] = acc[i];
  }
}
