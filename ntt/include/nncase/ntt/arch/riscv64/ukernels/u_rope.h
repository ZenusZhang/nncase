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
#include "../../../primitive_ops.h"
#include "../../../ukernels/u_rope.h"
#include "../arch_types.h"
#include "nncase/ntt/shape.h"
#include <riscv_vector.h>

namespace nncase::ntt::ukernels {
template <size_t NumHeads, size_t HalfDim>
struct u_rope<vector<half, NTT_VLEN / 16>, NumHeads, HalfDim, true> {
  public:
    using T = vector<half, NTT_VLEN / 16>;

    template <Dimension TSeqLen, Strides TInputStrides, Strides TCosStrides,
              Strides TSinStrides, Strides TOutputStrides>
    constexpr void
    operator()(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
               const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
               const TSeqLen &seq_len, const TInputStrides &input_strides,
               const TCosStrides &cos_strides, const TSinStrides &sin_strides,
               const TOutputStrides &output_strides) noexcept {
        using rope_layout = ukernels::rope_layout;

        constexpr auto unroll = 4_dim;
        ntt::apply_tiled(
            ntt::make_shape(fixed_dim_v<HalfDim>, seq_len),
            ntt::make_shape(1_dim, unroll),
            [&](auto index, auto in_offset, auto cos_offset, auto sin_offset,
                auto out_offset) {
                const auto seq_tile = ntt::min(unroll, seq_len - index[1_dim]);
                const auto vl = __riscv_vsetvl_e16m4(seq_tile * T::size());

                const vfloat16m4_t cos_0 = __riscv_vle16_v_f16m4(
                    (const _Float16 *)(cos + cos_offset), vl);
                const vfloat16m4_t sin_0 = __riscv_vle16_v_f16m4(
                    (const _Float16 *)(sin + sin_offset), vl);
                const vfloat16m4_t cos_1 = __riscv_vle16_v_f16m4(
                    (const _Float16
                         *)(cos + cos_offset +
                            HalfDim *
                                cos_strides[rope_layout::sincos_dim_axis]),
                    vl);
                const vfloat16m4_t sin_1 = __riscv_vle16_v_f16m4(
                    (const _Float16
                         *)(sin + sin_offset +
                            HalfDim *
                                sin_strides[rope_layout::sincos_dim_axis]),
                    vl);

#pragma unroll 1
                for (size_t h = 0; h < NumHeads; h++) {
                    const vfloat16m4_t input_0 = __riscv_vle16_v_f16m4(
                        (const _Float16
                             *)(input + in_offset +
                                h * input_strides[rope_layout::head_axis]),
                        vl);
                    const vfloat16m4_t input_1 = __riscv_vle16_v_f16m4(
                        (const _Float16
                             *)(input + in_offset +
                                h * input_strides[rope_layout::head_axis] +
                                HalfDim * input_strides[rope_layout::dim_axis]),
                        vl);

                    // 2nd half
                    vfloat16m4_t output_1 =
                        __riscv_vfmul_vv_f16m4(input_0, sin_1, vl);
                    output_1 =
                        __riscv_vfmacc_vv_f16m4(output_1, input_1, cos_1, vl);
                    __riscv_vse16_v_f16m4(
                        (_Float16
                             *)(output + out_offset +
                                h * output_strides[rope_layout::head_axis] +
                                HalfDim *
                                    output_strides[rope_layout::dim_axis]),
                        output_1, vl);

                    // 1st half
                    vfloat16m4_t output_0 =
                        __riscv_vfmul_vv_f16m4(input_1, sin_0, vl);
                    output_0 =
                        __riscv_vfmsac_vv_f16m4(output_0, input_0, cos_0, vl);
                    __riscv_vse16_v_f16m4(
                        (_Float16
                             *)(output + out_offset +
                                h * output_strides[rope_layout::head_axis]),
                        output_0, vl);
                }
            },
            input_strides.template slice<1>(), cos_strides, sin_strides,
            output_strides.template slice<1>());

        // constexpr auto vl = NTT_VLEN / 16;
        // constexpr auto sincos_stride_bytes = vl * unroll * sizeof(half);
        // for (size_t dim = 0; dim < HalfDim; dim++) {
        //     auto remain_seq_len = seq_len * vl;

        //     const T *NTT_RESTRICT cos_0p =
        //         cos + dim * cos_strides[rope_layout::sincos_dim_axis];
        //     const T *NTT_RESTRICT sin_0p =
        //         sin + dim * sin_strides[rope_layout::sincos_dim_axis];
        //     const T *NTT_RESTRICT cos_1p =
        //         cos_0p + HalfDim * cos_strides[rope_layout::sincos_dim_axis];
        //     const T *NTT_RESTRICT sin_1p =
        //         sin_0p + HalfDim * sin_strides[rope_layout::sincos_dim_axis];
        //     const T *NTT_RESTRICT input_0p = input;
        //     const T *NTT_RESTRICT input_1p =
        //         input_0p + HalfDim * input_strides[rope_layout::dim_axis];
        //     T *NTT_RESTRICT output_0p = output;
        //     T *NTT_RESTRICT output_1p =
        //         output_0p + HalfDim * output_strides[rope_layout::dim_axis];

        //     while (remain_seq_len) {
        //         const auto vl = __riscv_vsetvl_e16m4(remain_seq_len);
        //         remain_seq_len -= vl;

        //         // v0: cos_0
        //         // v4: sin_0
        //         // v8: cos_1
        //         // v12: sin_1
        //         // v16: input_4_0
        //         // v20: input_4_1
        //         // v24: tmp_0
        //         // v28: tmp_1
        //         asm volatile(
        //             "vle16.v v0, (%[cos_0p])\n"
        //             "addi %[cos_0p], %[cos_0p], %[sincos_stride_bytes]\n"
        //             "vle16.v v4, (%[sin_0p])\n"
        //             "addi %[sin_0p], %[sin_0p], %[sincos_stride_bytes]\n"
        //             "vle16.v v8, (%[cos_1p])\n"
        //             "addi %[cos_1p], %[cos_1p], %[sincos_stride_bytes]\n"
        //             "vle16.v v12, (%[sin_1p])\n"
        //             "addi %[sin_1p], %[sin_1p], %[sincos_stride_bytes]\n"
        //             : [cos_0p] "+r"(cos_0p), [sin_0p] "+r"(sin_0p),
        //               [cos_1p] "+r"(cos_1p), [sin_1p] "+r"(sin_1p)
        //             : [sincos_stride_bytes] "i"((size_t)sincos_stride_bytes)
        //             : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        //               "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        //               "memory");

        //         const T *NTT_RESTRICT input_0hp = input_0p;
        //         const T *NTT_RESTRICT input_1hp = input_1p;
        //         T *NTT_RESTRICT output_0hp = output_0p;
        //         T *NTT_RESTRICT output_1hp = output_1p;
        //         for (size_t h = 0; h < NumHeads; h++) {
        //             asm volatile(
        //                 "vle16.v v16, (%[input_4_0])\n"
        //                 "vle16.v v20, (%[input_4_1])\n"
        //                 // 1st half: output_dp[h] = ntt::mul_sub(input_0,
        //                 cos_0,
        //                 // input_1 * sin_0)
        //                 "vfmul.vv v24, v20, v4\n"  // tmp = input_1 * sin_0
        //                 "vfmsac.vv v24, v16, v0\n" // tmp = input_0 * cos_0 -
        //                                            // tmp
        //                 "vse16.v v24, (%[output_4_0])\n"
        //                 // 2nd half: output_dp[h + output_dim_strides *
        //                 HalfDim]
        //                 // = ntt::mul_add(input_1, cos_1, input_0 * sin_1)
        //                 "vfmul.vv v24, v16, v12\n" // tmp = input_0 * sin_1
        //                 "vfmacc.vv v24, v20, v8\n" // tmp = input_1 * cos_1 +
        //                                            // tmp
        //                 "vse16.v v24, (%[output_4_1])\n"
        //                 : [input_0hp] "+r"(input_0hp)
        //                 : [h_strides] "r"(
        //                       (size_t)(input_strides[rope_layout::head_axis]
        //                       *
        //                                NTT_VLEN / 16 * unroll *
        //                                sizeof(half))),
        //                   [input_4_1] "r"(
        //                       input_1hp +
        //                       h * input_strides[rope_layout::head_axis]),
        //                   [output_4_0] "r"(
        //                       output_0hp +
        //                       h * output_strides[rope_layout::head_axis]),
        //                   [output_4_1] "r"(
        //                       output_1hp +
        //                       h * output_strides[rope_layout::head_axis])
        //                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        //                 "v8",
        //                   "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        //                   "v16", "v17", "v18", "v19", "v20", "v21", "v22",
        //                   "v23", "v24", "v25", "v26", "v27", "memory");
        //         }
        //     }
        // }
    }
};
} // namespace nncase::ntt::ukernels
