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
#include "../ukernels/u_rope.h"

namespace nncase::ntt {

template <Tensor TInput, Tensor TCos, Tensor TSin, class TOut>
void rope(const TInput &input, const TCos &cos, const TSin &sin,
          TOut &&output) {
    constexpr auto head_axis = 2_dim;
    constexpr auto dim_axis = 1_dim;
    const auto half_dim = input.shape()[dim_axis] / 2_dim;
    const auto num_heads = input.shape()[head_axis];
    const auto domain = cos.shape().template slice<0, 1>();
    const auto in_strides = input.strides().template slice<0, 1>();
    const auto cos_strides = cos.strides().template slice<0, 1>();

    using TElem = typename TInput::element_type;
    const TElem *NTT_RESTRICT input_p = input.elements().data();
    const TElem *NTT_RESTRICT cos_p = cos.elements().data();
    const TElem *NTT_RESTRICT sin_p = sin.elements().data();
    TElem *NTT_RESTRICT output_p = output.elements().data();

    // [seq]
    ntt::apply(
        domain,
        [&](auto, auto inout_offset, auto sincos_offset) {
            ntt::u_rope<num_heads, half_dim>(
                input_p + inout_offset, cos_p + sincos_offset,
                sin_p + sincos_offset, output_p + inout_offset,
                input.strides()[dim_axis], output.strides()[dim_axis]);
        },
        in_strides, cos_strides);
}
} // namespace nncase::ntt
