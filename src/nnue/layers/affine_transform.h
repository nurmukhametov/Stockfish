/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2023 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of layer AffineTransform of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include <iostream>
#include <algorithm>
#include <type_traits>
#include "../nnue_common.h"
#include "simd.h"

#if defined(USE_ISPC)
extern "C" void affine_transform_ispc(const uint8_t input[], const int8_t weights[], const int biases[],
    int output[], int InputDimensions,  int PaddedInputDimensions, int OutputDimensions);
extern "C" int affine_transform1_ispc(const int8_t weights[], const uint8_t input[], int InputDimensions);
extern "C" void affine_transform_1024_1024_16_ispc(const uint8_t input[], const int8_t weights[],
                                          const int biases[], int output[]);

extern "C" void affine_transform_30_32_32_ispc(const uint8_t input[], const int8_t weights[],
                                      const int biases[], int output[]);
extern "C" void sparse_affine_transform_ispc(const uint8_t input[], const int8_t weights[], const int biases[],
    int output[], int InputDimensions,  int PaddedInputDimensions, int OutputDimensions);
extern "C" int sparse_affine_transform1_ispc(const int8_t weights[], const uint8_t input[], int InputDimensions);
extern "C" void sparse_affine_transform_1024_1024_16_ispc(const uint8_t input[], const int8_t weights[],
                                          const int biases[], int output[]);

extern "C" void sparse_affine_transform_30_32_32_ispc(const uint8_t input[], const int8_t weights[],
                                      const int biases[], int output[]);
#endif

/*
  This file contains the definition for a fully connected layer (aka affine transform).

    - expected use-case is for when PaddedInputDimensions == 32 and InputDimensions <= 32.
      - that's why AVX512 is hard to implement
    - expected use-case is small layers
    - inputs are processed in chunks of 4, weights are respectively transposed
    - accumulation happens directly to int32s
*/

namespace Stockfish::Eval::NNUE::Layers {

// Fallback implementation for older/other architectures.
// Requires the input to be padded to at least 16 values.
#if !defined(USE_SSSE3)
  template <IndexType InputDimensions, IndexType PaddedInputDimensions, IndexType OutputDimensions>
  static void affine_transform_non_ssse3(std::int32_t* output, const std::int8_t* weights, const std::int32_t* biases, const std::uint8_t* input)
  {
# if defined(USE_SSE2)
    // At least a multiple of 16, with SSE2.
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const __m128i Zeros = _mm_setzero_si128();
    const auto inputVector = reinterpret_cast<const __m128i*>(input);

# elif defined(USE_MMX)
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 8;
    const __m64 Zeros = _mm_setzero_si64();
    const auto inputVector = reinterpret_cast<const __m64*>(input);

# elif defined(USE_NEON_DOTPROD)
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const auto inputVector = reinterpret_cast<const int8x16_t*>(input);

# elif defined(USE_NEON)
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const auto inputVector = reinterpret_cast<const int8x8_t*>(input);
# endif

    for (IndexType i = 0; i < OutputDimensions; ++i) {
      const IndexType offset = i * PaddedInputDimensions;

# if defined(USE_SSE2)
      __m128i sumLo = _mm_cvtsi32_si128(biases[i]);
      __m128i sumHi = Zeros;
      const auto row = reinterpret_cast<const __m128i*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        __m128i row_j = _mm_load_si128(&row[j]);
        __m128i input_j = _mm_load_si128(&inputVector[j]);
        __m128i extendedRowLo = _mm_srai_epi16(_mm_unpacklo_epi8(row_j, row_j), 8);
        __m128i extendedRowHi = _mm_srai_epi16(_mm_unpackhi_epi8(row_j, row_j), 8);
        __m128i extendedInputLo = _mm_unpacklo_epi8(input_j, Zeros);
        __m128i extendedInputHi = _mm_unpackhi_epi8(input_j, Zeros);
        __m128i productLo = _mm_madd_epi16(extendedRowLo, extendedInputLo);
        __m128i productHi = _mm_madd_epi16(extendedRowHi, extendedInputHi);
        sumLo = _mm_add_epi32(sumLo, productLo);
        sumHi = _mm_add_epi32(sumHi, productHi);
      }
      __m128i sum = _mm_add_epi32(sumLo, sumHi);
      __m128i sumHigh_64 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
      sum = _mm_add_epi32(sum, sumHigh_64);
      __m128i sum_second_32 = _mm_shufflelo_epi16(sum, _MM_SHUFFLE(1, 0, 3, 2));
      sum = _mm_add_epi32(sum, sum_second_32);
      output[i] = _mm_cvtsi128_si32(sum);

# elif defined(USE_MMX)
      __m64 sumLo = _mm_cvtsi32_si64(biases[i]);
      __m64 sumHi = Zeros;
      const auto row = reinterpret_cast<const __m64*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        __m64 row_j = row[j];
        __m64 input_j = inputVector[j];
        __m64 extendedRowLo = _mm_srai_pi16(_mm_unpacklo_pi8(row_j, row_j), 8);
        __m64 extendedRowHi = _mm_srai_pi16(_mm_unpackhi_pi8(row_j, row_j), 8);
        __m64 extendedInputLo = _mm_unpacklo_pi8(input_j, Zeros);
        __m64 extendedInputHi = _mm_unpackhi_pi8(input_j, Zeros);
        __m64 productLo = _mm_madd_pi16(extendedRowLo, extendedInputLo);
        __m64 productHi = _mm_madd_pi16(extendedRowHi, extendedInputHi);
        sumLo = _mm_add_pi32(sumLo, productLo);
        sumHi = _mm_add_pi32(sumHi, productHi);
      }
      __m64 sum = _mm_add_pi32(sumLo, sumHi);
      sum = _mm_add_pi32(sum, _mm_unpackhi_pi32(sum, sum));
      output[i] = _mm_cvtsi64_si32(sum);

# elif defined(USE_NEON_DOTPROD)
      int32x4_t sum = {biases[i]};
      const auto row = reinterpret_cast<const int8x16_t*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        sum = vdotq_s32(sum, inputVector[j], row[j]);
      }
      output[i] = vaddvq_s32(sum);

# elif defined(USE_NEON)
      int32x4_t sum = {biases[i]};
      const auto row = reinterpret_cast<const int8x8_t*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        int16x8_t product = vmull_s8(inputVector[j * 2], row[j * 2]);
        product = vmlal_s8(product, inputVector[j * 2 + 1], row[j * 2 + 1]);
        sum = vpadalq_s16(sum, product);
      }
      output[i] = sum[0] + sum[1] + sum[2] + sum[3];

# else
      std::int32_t sum = biases[i];
      printf("offset: %i\n", offset);
      for (IndexType j = 0; j < InputDimensions; ++j) {
        printf("weights[%i] = %i, in[%i] = %i\n", offset+j, weights[offset+j], j, input[j]);
        sum += weights[offset + j] * input[j];
      }
      printf("biases[%i] = %i, sum = %i\n", i, biases[i], sum);
      output[i] = sum;
# endif
    }

# if defined(USE_MMX)
    _mm_empty();
# endif
  }
#endif

  template <IndexType InDims, IndexType OutDims>
  class AffineTransform {
   public:
    // Input/output type
    using InputType = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
      std::uint32_t hashValue = 0xCC03DAE4u;
      hashValue += OutputDimensions;
      hashValue ^= prevHash >> 1;
      hashValue ^= prevHash << 31;
      return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i)
    {
      return
        (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4 +
        i / PaddedInputDimensions * 4 +
        i % 4;
    }

    static constexpr IndexType get_weight_index(IndexType i)
    {
#if defined (USE_SSSE3)
      return get_weight_index_scrambled(i);
#else
      return i;
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      read_little_endian<BiasType>(stream, biases, OutputDimensions);
      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

      return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
      write_little_endian<BiasType>(stream, biases, OutputDimensions);

      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

      return !stream.fail();
    }
    // Forward propagation
    void propagate(
        const InputType* input, OutputType* output) const {

#if defined(USE_ISPC)
      static_assert(OutputDimensions % 16 == 0 || OutputDimensions == 1);

      if constexpr (OutputDimensions == 16 && InputDimensions == 1024 && PaddedInputDimensions == 1024) {
        affine_transform_1024_1024_16_ispc(input, weights, biases, output);
      }
      else if constexpr (OutputDimensions == 32 && InputDimensions == 30 && PaddedInputDimensions == 32) {
        affine_transform_30_32_32_ispc(input, weights, biases, output);
      }
      else if constexpr (OutputDimensions > 1) {
        affine_transform_ispc(input, weights, biases, output, InputDimensions,
                               PaddedInputDimensions, OutputDimensions);
      }
      else {
        output[0] = biases[0] + affine_transform1_ispc(weights, input, InputDimensions);
      }
#else
#if defined (USE_AVX512)
      using vec_t = __m512i;
      #define vec_setzero _mm512_setzero_si512
      #define vec_set_32 _mm512_set1_epi32
      #define vec_add_dpbusd_32 Simd::m512_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m512_add_dpbusd_epi32x2
      #define vec_hadd Simd::m512_hadd
#elif defined (USE_AVX2)
      using vec_t = __m256i;
      #define vec_setzero _mm256_setzero_si256
      #define vec_set_32 _mm256_set1_epi32
      #define vec_add_dpbusd_32 Simd::m256_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m256_add_dpbusd_epi32x2
      #define vec_hadd Simd::m256_hadd
#elif defined (USE_SSSE3)
      using vec_t = __m128i;
      #define vec_setzero _mm_setzero_si128
      #define vec_set_32 _mm_set1_epi32
      #define vec_add_dpbusd_32 Simd::m128_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m128_add_dpbusd_epi32x2
      #define vec_hadd Simd::m128_hadd
#endif

#if defined (USE_SSSE3)
      const auto inputVector = reinterpret_cast<const vec_t*>(input);

      static constexpr IndexType OutputSimdWidth = sizeof(vec_t) / sizeof(OutputType);

      static_assert(OutputDimensions % OutputSimdWidth == 0 || OutputDimensions == 1);

      if constexpr (OutputDimensions % OutputSimdWidth == 0)
      {
        constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
        constexpr IndexType NumRegs = OutputDimensions / OutputSimdWidth;

        const auto input32 = reinterpret_cast<const std::int32_t*>(input);
        const vec_t* biasvec = reinterpret_cast<const vec_t*>(biases);
        vec_t acc[NumRegs];
        for (IndexType k = 0; k < NumRegs; ++k)
          acc[k] = biasvec[k];

        for (IndexType i = 0; i < NumChunks; i += 2)
        {
          const vec_t in0 = vec_set_32(input32[i + 0]);
          const vec_t in1 = vec_set_32(input32[i + 1]);
          const auto col0 = reinterpret_cast<const vec_t*>(&weights[(i + 0) * OutputDimensions * 4]);
          const auto col1 = reinterpret_cast<const vec_t*>(&weights[(i + 1) * OutputDimensions * 4]);
          for (IndexType k = 0; k < NumRegs; ++k)
            vec_add_dpbusd_32x2(acc[k], in0, col0[k], in1, col1[k]);
        }

        vec_t* outptr = reinterpret_cast<vec_t*>(output);
        for (IndexType k = 0; k < NumRegs; ++k)
          outptr[k] = acc[k];
      }
      else if constexpr (OutputDimensions == 1)
      {
        constexpr IndexType NumChunks = PaddedInputDimensions / SimdWidth;
        vec_t sum0 = vec_setzero();
        const auto row0 = reinterpret_cast<const vec_t*>(&weights[0]);

        for (int j = 0; j < (int)NumChunks; ++j)
        {
          const vec_t in = inputVector[j];
          vec_add_dpbusd_32(sum0, in, row0[j]);
        }
        output[0] = vec_hadd(sum0, biases[0]);
      }

# undef vec_setzero
# undef vec_set_32
# undef vec_add_dpbusd_32
# undef vec_add_dpbusd_32x2
# undef vec_hadd
#else
      // Use old implementation for the other architectures.
      affine_transform_non_ssse3<
        InputDimensions,
        PaddedInputDimensions,
        OutputDimensions>(output, weights, biases, input);
#endif
#endif
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
