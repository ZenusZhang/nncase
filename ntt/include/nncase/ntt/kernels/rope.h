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
void rope(const TInput &input, const TCos &cos, const TSin &sin,
          TOut &&output) {
    const auto domain = input.shape().template slice<0, TInput::rank() - 1>();
    const auto in_strides =
        input.strides().template slice<0, TInput::rank() - 1>();
    const auto cos_strides = broadcast_strides<1>(cos.strides())
                                 .template slice<0, TInput::rank() - 1>();
    const auto half_dim = input.shape().back() / 2_dim;

    using TElem = typename TInput::element_type;
    const TElem *NTT_RESTRICT input_p = input.elements().data();
    const TElem *NTT_RESTRICT cos_p = cos.elements().data();
    const TElem *NTT_RESTRICT sin_p = sin.elements().data();
    TElem *NTT_RESTRICT output_p = output.elements().data();

    ntt::apply(
        domain,
        [&](auto, auto in_offset, auto cos_offset) {
            for (size_t i = 0; i < half_dim; i++) {
                const auto input_0 = input_p[in_offset + i];
                const auto input_1 = input_p[in_offset + i + half_dim];

                // 1st half
                output_p[in_offset + i] =
                    ntt::mul_add(input_0, cos_p[cos_offset + i],
                                 -input_1 * sin_p[cos_offset + i]);

                // 2nd half
                output_p[in_offset + i + half_dim] =
                    ntt::mul_add(input_1, cos_p[cos_offset + i + half_dim],
                                 input_0 * sin_p[cos_offset + i + half_dim]);
            }
        },
        in_strides, cos_strides);
}
} // namespace nncase::ntt
