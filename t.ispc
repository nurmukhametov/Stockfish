static const int FC_0_OUTPUTS = 15;
#define FC_0_OUTPUTS_P1 16
#define FC_1_OUTPUTS 32

static const uniform int WeightScaleBits = 6;

void propagate1(const uniform int32 * uniform input, uniform uint8 * uniform output, uniform int inDims) {
  // for (uniform int i = 0; i < inDims; ++i) {
  //   uniform int32 shifted = input[i] >> WeightScaleBits;
  //   uniform uint8 min_i = (uniform uint8) min(127, shifted);
  //   output[i] = max((uniform uint8) 0, min_i);
  //   print("% ", i);
  // }
  // print("\n");
  foreach (i = 0 ... inDims) {
    int32 shifted = input[i] >> WeightScaleBits;
    uint8 min_i = min(127, shifted);
    uint8 max_i = max((varying uint8)0, min_i);
    output[i] = max_i;
    print("% ", max_i);
  }
  print("\n");
}

void clip_relu(const uniform int input[], uniform uint8 output[], uniform int size, uniform uint8 scale) {
  foreach (j = 0 ... size) {
    output[j] = max(0, min(127, input[j] >> scale));
  }
}

extern "C" uniform int main() {
  uniform int32 in1[FC_0_OUTPUTS_P1] = {13765, 6461, 12830, 13081, 14936, 3236, 5492, 12586, 4551, 9076, 7821, 10303, 5976, 8411, 15601, 15010};
  uniform uint8 out1[FC_0_OUTPUTS_P1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // uniform int32 in2[FC_1_OUTPUTS] = {1301, 1469, 290, 1243, 33, 497, 281, 1646, 320, 821, 265, 222, 2045, 446, 1050, 1718, 1254, 606, 1305, 1073, 1010, 1992, 599, 1579, 1078, 1576, 819, 1825, 580, 721, 1654, 1882};
  // uniform uint8 out2[FC_1_OUTPUTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // print("in1: ");
  // for (uniform int i = 0; i < FC_0_OUTPUTS_P1; i++) {
  //   print("% ", in1[i]);
  // }
  // print("\n");

  propagate1(in1, out1, FC_0_OUTPUTS_P1);

  print("out1: ");
  for (uniform int i = 0; i < FC_0_OUTPUTS_P1; i++) {
    print("% ", out1[i]);
  }
  print("\n");

  clip_relu(in1, out1, FC_0_OUTPUTS_P1, WeightScaleBits);

  print("out1: ");
  for (uniform int i = 0; i < FC_0_OUTPUTS_P1; i++) {
    print("% ", out1[i]);
  }
  print("\n");

  return 0;
}
