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
#include "../apply.h"
#include "../primitive_ops.h"
#include "../ukernels.h"
#include "../utility.h"
#include "reduce.h"

namespace nncase::ntt {

namespace vectorized_layer_norm_detail {

template <bool UseMean, Tensor TIn, Tensor TScale, Tensor TBias, typename TOut,
          FixedDimensions VectorizedAxes, Dimensions PadedNums,
          FixedDimension TAxis>
void within_axis_vectorize_impl(const TIn &input, const TScale &scale,
                                const TBias &bias, TOut &output,
                                const float &epsilon, const VectorizedAxes &,
                                const PadedNums &padedNums, const TAxis &axis) {

    using TElem = typename TIn::element_type;
    using TScaleElem = typename TScale::element_type;
    using TAccElem = decltype(ntt::cast_elem<float>(std::declval<TElem>()));

    constexpr auto rank = TIn::rank();

    auto input_shape = input.shape();
    auto input_strides = input.strides();

    constexpr auto axis_value = positive_index(TAxis::value, TIn::rank());

    const auto inner_size = input_shape[axis_value];

    constexpr VectorizedAxes vectorized_axes_temp;
    constexpr bool UseVectorReduce = vectorized_axes_temp.rank() == 1 &&
                                     vectorized_axes_temp[0] >= axis_value;

    using TElemScalar = element_or_scalar_t<TElem>;
    auto finner_size = (float)inner_size;
    if constexpr (UseVectorReduce) {
        finner_size *= TElem::size();
    }
    const auto norm_factor = 1.f / finner_size;

    const TScaleElem *NTT_RESTRICT scale_p = scale.elements().data();
    const TScaleElem *NTT_RESTRICT bias_p = bias.elements().data();

    auto stride_in = input_strides[axis_value];
    auto stride_out = output.strides()[axis_value];
    auto apply_shape = generate_shape<rank>([&](auto i) {
        if constexpr (i == axis_value)
            return 1_dim;
        else
            return input.shape()[i];
    });

    if (0 /*UseVectorReduce*/) {
        auto addr_scale = scale.elements().data();
        auto addr_bias = bias.elements().data();
        ntt::apply(apply_shape, [&](auto index) {
            auto addr_input = &input(index);
            auto addr_output = &output(index);
            ntt::u_layer_norm<UseMean>(
                addr_input, addr_scale, addr_bias, addr_output, epsilon,
                vectorized_axes_temp, padedNums, axis, inner_size, norm_factor,
                stride_in, stride_out);
        });

    } else {
        ntt::apply(apply_shape, [&](auto index) {
            const TElem *NTT_RESTRICT input_p = &input(index);
            TElem *NTT_RESTRICT output_p = &output(index);
            if constexpr (UseVectorReduce) {
                auto mean = (TElemScalar)0;
                auto extended_sum = (TAccElem)0;
                if constexpr (UseMean) {
                    auto extended_mean = (TAccElem)0;
                    for (size_t i = 0; i < inner_size; i++) {
                        extended_mean += input_p[i * stride_in];
                    }
                    extended_mean *= norm_factor;
                    auto extended_mean_s = reduce_sum(extended_mean);

                    for (auto i = 0; i < inner_size; i++) {
                        const auto val = ntt::square(
                            ntt::cast_elem<float>(input_p[i * stride_in]) -
                            extended_mean_s);
                        extended_sum += val;
                    }
                    mean = (TElemScalar)extended_mean_s;
                } else {
                    for (auto i = 0; i < inner_size; i++) {
                        const auto input_val = input_p[i * stride_in];
                        extended_sum =
                            ntt::mul_add(input_val, input_val, extended_sum);
                    }
                }

                const auto extended_sum_s =
                    reduce_sum(extended_sum) * norm_factor;
                auto extended_add = extended_sum_s + epsilon;
                auto rsqrt =
                    ntt::cast_elem<TElemScalar>(ntt::rsqrt(extended_add));

                if constexpr (UseMean) {
                    for (auto i = 0; i < inner_size; i++) {
                        auto val = (input_p[i * stride_in] - mean) * rsqrt;
                        output_p[i * stride_out] =
                            ntt::mul_add(val, scale_p[i], bias_p[i]);
                    }
                } else {
                    for (auto i = 0; i < inner_size; i++) {
                        auto val = input_p[i * stride_in] * rsqrt;
                        output_p[i * stride_out] =
                            val * scale_p[i]; // RMSNorm doesn't need bias
                    }
                }
            } else {
                auto mean = (TElem)0;
                auto extended_sum = (TAccElem)0;
                if constexpr (UseMean) {
                    auto extended_mean = (TAccElem)0;
                    for (size_t i = 0; i < inner_size; i++) {
                        extended_mean += input_p[i * stride_in];
                    }
                    extended_mean *= norm_factor;

                    for (auto i = 0; i < inner_size; i++) {
                        const auto val = ntt::square(
                            ntt::cast_elem<float>(input_p[i * stride_in]) -
                            extended_mean);
                        extended_sum += val;
                    }
                    mean = ntt::cast_elem<TElemScalar>(extended_mean);
                } else {
                    for (auto i = 0; i < inner_size; i++) {
                        const auto val = ntt::square(
                            ntt::cast_elem<float>(input_p[i * stride_in]));
                        extended_sum += val;
                    }
                }

                extended_sum *= norm_factor;
                auto extended_add = extended_sum + epsilon;
                auto rsqrt =
                    ntt::cast_elem<TElemScalar>(ntt::rsqrt(extended_add));

                if constexpr (UseMean) {
                    for (auto i = 0; i < inner_size; i++) {
                        auto val = (input_p[i * stride_in] - mean) * rsqrt;
                        output_p[i * stride_out] =
                            ntt::mul_add(val, scale_p[i], bias_p[i]);
                    }
                } else {
                    for (auto i = 0; i < inner_size; i++) {
                        auto val = input_p[i * stride_in] * rsqrt;
                        output_p[i * stride_out] = val * scale_p[i];
                    }
                }
            }
        });
    }
}

} // namespace vectorized_layer_norm_detail

template <bool UseMean = true, Tensor TIn, Tensor TScale, Tensor TBias,
          typename TOut, FixedDimension TAxis,
          FixedDimensions VectorizedAxes = shape_t<>,
          Dimensions PadedNums = shape_t<>>
void vectorized_layer_norm(const TIn &input, const TScale &scale,
                           const TBias &bias, TOut &&output,
                           const float &epsilon, const TAxis &axis = -1_dim,
                           const VectorizedAxes &vectorizedAxes = {},
                           const PadedNums &padedNums = {}) {
    static_assert(VectorizedAxes::rank() < 2,
                  "currently not support 2d packing.");

    vectorized_layer_norm_detail::within_axis_vectorize_impl<
        UseMean, TIn, TScale, TBias, TOut, VectorizedAxes, PadedNums, TAxis>(
        input, scale, bias, output, epsilon, vectorizedAxes, padedNums, axis);
}
} // namespace nncase::ntt
