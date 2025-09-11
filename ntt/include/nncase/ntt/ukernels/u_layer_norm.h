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
template <bool UseMean, class TEIn, class TEScale, class TEBias, class TEOut,
          FixedDimensions VectorizedAxes, Dimensions PadedNums,
          FixedDimension TAxis, bool Arch>
struct u_layer_norm {
  public:
    constexpr void operator()(const TEIn *input, const TEScale *scale,
                              const TEBias *bias, TEOut *output,
                              const float &epsilon, const VectorizedAxes &,
                              const PadedNums &, const TAxis &,
                              const size_t &inner_size,
                              const float &norm_factor) noexcept {
        using TElemScalar = element_or_scalar_t<TEIn>;
        using TAccElem = decltype(ntt::cast_elem<float>(std::declval<TEIn>()));
        auto mean = (TElemScalar)0;
        auto extended_sum = (TAccElem)0;
        if constexpr (UseMean) {
            auto extended_mean = (TAccElem)0;
            for (size_t i = 0; i < inner_size; i++)
                extended_mean += input[i];
            extended_mean *= norm_factor;
            auto extended_mean_s = reduce_sum(extended_mean);

            for (auto i = 0; i < inner_size; i++) {
                const auto val = ntt::square(ntt::cast_elem<float>(input[i]) -
                                             extended_mean_s);
                extended_sum += val;
            }
            mean = (TElemScalar)extended_mean_s;
        } else {
            for (auto i = 0; i < inner_size; i++) {
                const auto input_val = input[i];
                extended_sum = ntt::mul_add(input_val, input_val, extended_sum);
            }
        }

        const auto extended_sum_s = reduce_sum(extended_sum) * norm_factor;
        auto extended_add = extended_sum_s + epsilon;
        auto rsqrt = ntt::cast_elem<TElemScalar>(ntt::rsqrt(extended_add));

        if constexpr (UseMean) {
            for (auto i = 0; i < inner_size; i++) {
                auto val = (input[i] - mean) * rsqrt;
                output[i] = ntt::mul_add(val, scale[i], bias[i]);
            }
        } else {
            for (auto i = 0; i < inner_size; i++) {
                auto val = input[i] * rsqrt;
                output[i] = val * scale[i]; // RMSNorm doesn't need bias
            }
        }
    }
};
} // namespace ukernels

template <bool UseMean, class TEIn, class TEScale, class TEBias, class TEOut,
          FixedDimensions VectorizedAxes, Dimensions PadedNums,
          FixedDimension TAxis>
constexpr void
u_layer_norm(const TEIn *input, const TEScale *scale, const TEBias *bias,
             TEOut *output, const float &epsilon,
             const VectorizedAxes &vectorizedAxes, const PadedNums &padedNums,
             const TAxis &axis, const size_t &inner_size,
             const float &norm_factor) noexcept {
    ukernels::u_layer_norm<UseMean, TEIn, TEScale, TEBias, TEOut,
                           VectorizedAxes, PadedNums, TAxis, true>
        impl;
    impl(input, scale, bias, output, epsilon, vectorizedAxes, padedNums, axis,
         inner_size, norm_factor);
}
} // namespace nncase::ntt
