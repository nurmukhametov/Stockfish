#include <vector>
#include <immintrin.h>

#include <cstdint>
#include <array>
#include <cstring>

#include <iostream>
#include <random>
#include <chrono>

#include <inttypes.h>
#include "sparse_affine_ref.h"

static constexpr int InputSize = 2048;
static constexpr int OutputSize = 16;
static constexpr int NumInputSets = 64;
static constexpr float density = 0.1f;
using IndexType = std::int32_t;

inline int popcount(std::uint64_t b) {
  return __builtin_popcountll(b);
}
inline int popcount(std::uint32_t b) {
  return __builtin_popcountl(b);
}


static inline int lsb(std::uint64_t b) {
  return __builtin_ctzll(b);
}
static inline int lsb(std::uint32_t b) {
  return __builtin_ctzl(b);
}

struct alignas(64) Input{
  std::int8_t v[InputSize];
};

std::vector<Input> make_inputs(float density)
{
  std::mt19937_64 rng(1234);
  std::uniform_int_distribution<int> df(0, 127);
  std::bernoulli_distribution db(density);
  std::vector<Input> inputs;
  inputs.resize(NumInputSets);
  for (int i = 0; i < NumInputSets; ++i)
  {
    for (int j = 0; j < InputSize; ++j)
    {
      inputs[i].v[j] = db(rng) * df(rng);
    }
  }
  return inputs;
}

static std::vector<Input> inputs;

IndexType out[InputSize + 16];
unsigned count_out;

// void affine_0(const std::uint8_t* in, IndexType* out)
// {
//     for (auto i = 0; i < OutputSize; i++) {
//         auto offset = i * InputSize;
//         std::int32_t sum = ref_biases[i];
//         printf("offset: %i\n", offset);
//         for (auto j = 0; j < InputSize; j++) {
//             printf("ref_weights[%i] = %i, in[%i] = %i\n",  offset+j, ref_weights[offset+j], j, in[j]);
//             sum += ref_weights[offset + j] * in[j];
//         }
//         printf("biases[%i] = %i, sum = %i\n", i, ref_biases[i], sum);
//         out[i] = sum;
//     }
// }

inline std::uint16_t pop_lsb(std::uint64_t & b) {
  const std::uint16_t s = lsb(b);
  b &= b - 1;
  return s;
}

static inline const std::array<std::array<std::uint16_t, 8>, 256> lookup_indices = [](){
  std::array<std::array<std::uint16_t, 8>, 256> v{};
  for (unsigned i = 0; i < 256; ++i)
  {
    std::uint64_t j = i, k = 0;
    while(j)
      v[i][k++] = pop_lsb(j);
  }
  return v;
}();

static const std::array<std::array<std::uint16_t, 16>, 256*256> lookup_4_indices = [](){
    std::array<std::array<std::uint16_t, 16>, 256*256> v;
    for (int i = 0; i < 256*256; ++i)
    {
        int j = i;
        int k = 0;
        while(j)
        {
            const IndexType lsbIndex = lsb(std::uint32_t(j));
            j &= j - 1;
            v[i][k] = lsbIndex;
            ++k;
        }
    }
    return v;
}();

extern "C" void ispc_nnz_5_ptr(const std::int8_t* in, IndexType* out, unsigned* count_out, const void *lookup_tblind);
inline void ispc_nnz_5(const std::int8_t* in, IndexType* out, unsigned& count_out) {
    unsigned count = 0;
    ispc_nnz_5_ptr(in, out, &count, &lookup_4_indices);
    count_out = count;
}

extern "C" void ispc_space_affine_ptr(const std::int8_t* in, IndexType* out,
    const std::int8_t *weights, const std::int32_t *biases,
    const void *lookup_tblind);
void ispc_sparse_affine(const std::uint8_t* in, IndexType* out) {
    ispc_space_affine_ptr((const std::int8_t*)in, out, ref_weights, ref_biases, &lookup_indices);
}

// template <typename FuncT>
// void test(FuncT f, std::string name)
// {
//     IndexType baseline_indices[InputSize + 16];
//     IndexType indices[InputSize + 16];
//     unsigned baseline_num_indices = 0;
//     unsigned num_indices = 0;
//     for (int i = 0; i < NumInputSets; ++i)
//     {
//         affine_0(inputs[0].v, baseline_indices, baseline_num_indices);
// 
//         f(inputs[0].v, indices, num_indices);
//         if (num_indices != baseline_num_indices)
//         {
//             std::cout << name << " invalid num indices.\n";
//             return;
//         }
//         else
//         {
//             for (int j = 0; j < baseline_num_indices; ++j)
//             {
//                 if (baseline_indices[j] != indices[j])
//                 {
//                     std::cout << name << " invalid index at " << j << '\n';
//                     return;
//                 }
//             }
//         }
//     }
// 
//     std::cout << name << " correct.\n";
// }
// 
// template <typename FuncT>
// void benchmark(FuncT f, std::string name)
// {
//     constexpr int num_iters = 10000;
//     for (int i = 0; i < NumInputSets; ++i)
//         f(inputs[i].v, out, count_out);
// 
//     auto t0 = std::chrono::high_resolution_clock::now();
// 
//     for (int i = 0; i < num_iters; ++i)
//     {
//         for (int j = 0; j < NumInputSets; ++j)
//         {
//             f(inputs[j].v, out, count_out);
//         }
//     }
// 
//     auto t1 = std::chrono::high_resolution_clock::now();
// 
//     double diff = (t1 - t0).count() / 1e9;
// 
//     std::cout << name << "; Time: " << diff << "s\n";
// }

int main()
{
    float densities[] = { 0.0f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.5f, 1.0f };
    // float densities[] = { 0.01f };

    std::int32_t output[16] = { };
    ispc_sparse_affine(ref_in, output);
    for (auto i = 0; i < OutputSize; i++) {
        // printf("result %" PRIi32 " != %" PRIi32 " at index %d\n", output[i], ref_out[i], i);
        if (ref_out[i] != output[i]) {
            printf("Incorrect result %" PRIi32 " != %" PRIi32 " at index %d\n", output[i], ref_out[i], i);
            break;
        }
    }
    std::cout << " correct.\n";

    //for (auto density : densities)
    //{
    //    inputs = make_inputs(density);

    //    // test(ispc_nnz_0, "ISPC version 0");
    //    // test(ispc_nnz_1, "ISPC version 1");
    //    // test(ispc_nnz_5, "ISPC version 5");

    //    // benchmark(ispc_nnz_0, "ISPC V. 0, Density: " + std::to_string(density));
    //    // benchmark(ispc_nnz_1, "ISPC V. 1, Density: " + std::to_string(density));
    //    // benchmark(ispc_nnz_5, "ISPC V. 5, Density: " + std::to_string(density));
    //}
}
