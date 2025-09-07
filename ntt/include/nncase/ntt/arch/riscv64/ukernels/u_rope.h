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
#include <riscv_vector.h>

namespace nncase::ntt::ukernels {
#if 1
template <size_t NumHeads, size_t HalfDim>
struct u_rope<vector<half, NTT_VLEN / 16>, NumHeads, HalfDim, true> {
  public:
    using T = vector<half, NTT_VLEN / 16>;

    template <Dimension TInputDimStrides, Dimension TOutputDimStrides>
    constexpr void
    operator()(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
               const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
               const TInputDimStrides &input_dim_strides,
               const TOutputDimStrides &output_dim_strides) noexcept {
        constexpr auto vl = T::shape()[0_dim];

        asm("vsetvli zero, %[vl], e16, m4, ta, ma\n" ::[vl] "r"(vl * 4));
        const T *NTT_RESTRICT input_dp = input;
        T *NTT_RESTRICT output_dp = output;
        for (size_t i = 0; i < HalfDim; i++) {
            const auto sin_0 = sin[i];
            const auto cos_1 = cos[HalfDim + i];
            const auto sin_1 = sin[HalfDim + i];

            // v0: cos_4_0
            // v4: sin_4_0
            // v8: cos_4_1
            // v12: sin_4_1
            // v16: input_4_0
            // v20: input_4_1
            // v24: tmp
            asm volatile("vl1re16.v v0, (%[cos_0])\n"
                         "vl1re16.v v4, (%[sin_0])\n"
                         "vl1re16.v v8, (%[cos_1])\n"
                         "vl1re16.v v12, (%[sin_1])\n"
                         "vmv1r.v v1, v0\n"
                         "vmv1r.v v2, v0\n"
                         "vmv1r.v v3, v0\n"
                         "vmv1r.v v5, v4\n"
                         "vmv1r.v v6, v4\n"
                         "vmv1r.v v7, v4\n"
                         "vmv1r.v v9, v8\n"
                         "vmv1r.v v10, v8\n"
                         "vmv1r.v v11, v8\n"
                         "vmv1r.v v13, v12\n"
                         "vmv1r.v v14, v12\n"
                         "vmv1r.v v15, v12\n" ::[cos_0] "r"(&cos[i]),
                         [sin_0] "r"(&sin[i]), [cos_1] "r"(&cos[HalfDim + i]),
                         [sin_1] "r"(&sin[HalfDim + i])
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                           "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                           "memory");

            size_t h = 0;
            for (; h < NumHeads; h += 4) {
                asm volatile(
                    "vl4re16.v v16, (%[input_4_0])\n"
                    "vl4re16.v v20, (%[input_4_1])\n"
                    // 1st half: output_dp[h] = ntt::mul_sub(input_0, cos_0,
                    // input_1 * sin_0)
                    "vfmul.vv v24, v20, v4\n"  // tmp = input_1 * sin_0
                    "vfmsac.vv v24, v16, v0\n" // tmp = input_0 * cos_0 - tmp
                    "vs4r.v v24, (%[output_4_0])\n"
                    // 2nd half: output_dp[h + output_dim_strides * HalfDim] =
                    // ntt::mul_add(input_1, cos_1, input_0 * sin_1)
                    "vfmul.vv v24, v16, v12\n" // tmp = input_0 * sin_1
                    "vfmacc.vv v24, v20, v8\n" // tmp = input_1 * cos_1 + tmp
                    "vs4r.v v24, (%[output_4_1])\n" ::[input_4_0] "r"(
                        &input_dp[h]),
                    [input_4_1] "r"(&input_dp[h + input_dim_strides * HalfDim]),
                    [output_4_0] "r"(&output_dp[h]),
                    [output_4_1] "r"(
                        &output_dp[h + output_dim_strides * HalfDim])
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                      "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                      "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                      "v25", "v26", "v27", "memory");
            }

            input_dp += input_dim_strides;
            output_dp += output_dim_strides;
        }
    }
};
#endif
} // namespace nncase::ntt::ukernels
