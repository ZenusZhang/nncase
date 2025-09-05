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

namespace nncase::ntt {

template <Tensor TInput, Tensor TCos, Tensor TSin, class TOut>
void __attribute__((noinline)) rope(const TInput &input, const TCos &cos, const TSin &sin,
          TOut &&output) {
    constexpr auto head_axis = 1_dim;
    const auto half_dim = input.shape().back() / 2_dim;
    const auto num_heads = input.shape()[head_axis];

    using TElem = typename TInput::element_type;
    const TElem *NTT_RESTRICT input_p = input.elements().data();
    const TElem *NTT_RESTRICT cos_p = cos.elements().data();
    const TElem *NTT_RESTRICT sin_p = sin.elements().data();
    TElem *NTT_RESTRICT output_p = output.elements().data();

    // [seq, 1, dim]
    ntt::apply(
        cos.shape(),
        [&](auto, auto inout_offset, auto sincos_offset) {
            for (size_t i = 0; i < half_dim; i++) {
                const auto cos_0 = cos_p[sincos_offset + i];
                const auto sin_0 = sin_p[sincos_offset + i];
                const auto cos_1 = cos_p[sincos_offset + half_dim + i];
                const auto sin_1 = sin_p[sincos_offset + half_dim + i];

                auto input_hp = input_p;
                auto output_hp = output_p;
                for (size_t h = 0; h < num_heads; h++) {

                    const auto input_0 = input_hp[inout_offset + i];
                    const auto input_1 = input_hp[inout_offset + half_dim + i];

                    // 1st half
                    output_hp[inout_offset + i] =
                        ntt::mul_sub(input_0, cos_0, input_1 * sin_0);

                    // 2nd half
                    output_hp[inout_offset + i + half_dim] =
                        ntt::mul_add(input_1, cos_1, input_0 * sin_1);

                    input_hp += input.strides()[head_axis];
                    output_hp += output.strides()[head_axis];
                }
            }
        },
        input.strides(), sin.strides());
}
} // namespace nncase::ntt
