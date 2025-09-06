/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "../primitive_ops.h"
#include "nncase/ntt/tensor_traits.h"

namespace nncase::ntt {
namespace ukernels {
template <ScalarOrVector T, size_t NumHeads, size_t HalfDim, bool Arch>
struct u_rope {
  public:
    template <Dimension TInputDimStrides, Dimension TOutputDimStrides>
    constexpr void
    operator()(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
               const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
               const TInputDimStrides &input_dim_strides,
               const TOutputDimStrides &output_dim_strides) noexcept {
        const T *NTT_RESTRICT input_dp = input;
        T *NTT_RESTRICT output_dp = output;
        for (size_t i = 0; i < HalfDim; i++) {
            const auto cos_0 = cos[i];
            const auto sin_0 = sin[i];
            const auto cos_1 = cos[HalfDim + i];
            const auto sin_1 = sin[HalfDim + i];

            for (size_t h = 0; h < NumHeads; h++) {
                const auto input_0 = input_dp[h];
                const auto input_1 = input_dp[h + input_dim_strides * HalfDim];

                // 1st half
                output_dp[h] = ntt::mul_sub(input_0, cos_0, input_1 * sin_0);

                // 2nd half
                output_dp[h + output_dim_strides * HalfDim] =
                    ntt::mul_add(input_1, cos_1, input_0 * sin_1);
            }

            input_dp += input_dim_strides;
            output_dp += output_dim_strides;
        }
    }
};
} // namespace ukernels

template <size_t NumHeads, size_t HalfDim, ScalarOrVector T,
          Dimension TInputDimStrides, Dimension TOutputDimStrides>
constexpr void u_rope(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
                      const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
                      const TInputDimStrides &input_dim_strides,
                      const TOutputDimStrides &output_dim_strides) noexcept {
    ukernels::u_rope<T, NumHeads, HalfDim, true> impl;
    impl(input, cos, sin, output, input_dim_strides, output_dim_strides);
}
} // namespace nncase::ntt
