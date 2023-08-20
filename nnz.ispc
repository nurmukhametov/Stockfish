const uniform int InputSize = 2048;

// extern "C" void ispc_nnz_0_ptr(const uniform int8 input[], uniform uint16 out[], uniform uint * uniform count) {
//   uniform uint c = 0;
//   for (uniform int16 i = 0; i < InputSize; ++i)
//   {
//     if (input[i]) {
//       out[c++] = i;
//     }
//   }
//   *count = c;
// }
// 
// extern "C" void ispc_nnz_1_ptr(const uniform int8 input[], uniform uint16 out[], uniform uint * uniform count) {
//   uniform uint c = 0;
//   for (uniform int i = 0; i < InputSize; i += programCount) {
//     uint8 x = input[i + programIndex];
//     uniform uint nnz = packmask(x);
//     while (nnz) {
//       uniform int lsbIndex = count_trailing_zeros(nnz);
//       nnz &= nnz - 1;
//       out[c++] = i + lsbIndex;
//     }
//   }
//   *count = c;
// }

typedef uniform uint16 LOOKUP_TABLE_TYPE[16];

extern "C" void ispc_nnz_5_ptr(const uniform int8 input[], uniform uint16 out[],
                  uniform uint * uniform count,
                  uniform LOOKUP_TABLE_TYPE lookup_tblind[]) {
  uniform uint c = 0;
  uniform uint base = 0;
  for (uniform int i = 0; i < InputSize; i += programCount) {
    uint8 x = input[i + programIndex];
    uniform uint nnz = packmask(x);
    for (uniform int j = 0; j < 2; j++) {
      uniform uint b = (nnz >> (j * 16)) & 0xFFFF;
      uniform uint16 *ptr = (uniform uint16*)&lookup_tblind[b];
      out[c + programIndex] = base + ptr[0 + programIndex];
      base += 16;
      c += popcnt((uniform int32)b);
    }
  }
  *count = c;
}
