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
template <size_t NumHeads, size_t HalfDim>
    requires(NumHeads % 4 == 0)
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

        asm("vsetvli zero, %[vl], e16, m1, ta, ma\n" ::[vl] "r"(vl));
        const T *NTT_RESTRICT cos_0p = cos;
        const T *NTT_RESTRICT sin_0p = sin;
        const T *NTT_RESTRICT cos_1p = cos + HalfDim;
        const T *NTT_RESTRICT sin_1p = sin + HalfDim;
        for (size_t i = 0; i < HalfDim; i++) {
            const T *NTT_RESTRICT input_0dp = input + i * input_dim_strides;
            const T *NTT_RESTRICT input_1dp =
                input + input_dim_strides * (HalfDim + i);
            T *NTT_RESTRICT output_0dp = output + i * output_dim_strides;
            T *NTT_RESTRICT output_1dp =
                output + output_dim_strides * (HalfDim + i);

            // v0: cos_0
            // v1: sin_0
            // v2: cos_1
            // v3: sin_1
            // v4: input_4_0
            // v8: input_4_1
            // v12: tmp_0
            // v16: tmp_1
            asm volatile("vl1re16.v v0, (%[cos_0p])\n"
                         "addi %[cos_0p], %[cos_0p], %[vl] * 2\n"
                         "vl1re16.v v1, (%[sin_0p])\n"
                         "addi %[sin_0p], %[sin_0p], %[vl] * 2\n"
                         "vl1re16.v v2, (%[cos_1p])\n"
                         "addi %[cos_1p], %[cos_1p], %[vl] * 2\n"
                         "vl1re16.v v3, (%[sin_1p])\n"
                         "addi %[sin_1p], %[sin_1p], %[vl] * 2\n"
                         : [cos_0p] "+r"(cos_0p), [sin_0p] "+r"(sin_0p),
                           [cos_1p] "+r"(cos_1p), [sin_1p] "+r"(sin_1p)
                         : [vl] "i"((size_t)vl)
                         : "v0", "v1", "v2", "v3", "memory");

            for (size_t h = 0; h < NumHeads; h += 4) {
                asm volatile(
                    "vl4re16.v v4, (%[input_0dp])\n"
                    "addi %[input_0dp], %[input_0dp], %[vl] * 2 * 4\n"
                    "vl4re16.v v8, (%[input_1dp])\n"
                    "addi %[input_1dp], %[input_1dp], %[vl] * 2 * 4\n"
                    // 2nd half: output_dp[h + output_dim_strides * HalfDim] =
                    // ntt::mul_add(input_1, cos_1, input_0 * sin_1)
                    "vfmul.vv v16, v4, v3\n"  // tmp_1[0] = input_0[0] * sin_1
                    "vfmul.vv v17, v5, v3\n"  // tmp_1[1] = input_0[1] * sin_1
                    "vfmul.vv v18, v6, v3\n"  // tmp_1[2] = input_0[2] * sin_1
                    "vfmul.vv v19, v7, v3\n"  // tmp_1[3] = input_0[3] * sin_1
                    "vfmacc.vv v16, v8, v2\n" // tmp_1[0] = input_1[0] * cos_1 +
                                              // tmp_1[0]
                    "vfmacc.vv v17, v9, v2\n" // tmp_1[1] = input_1[1] * cos_1 +
                                              // tmp_1[1]
                    "vfmacc.vv v18, v10, v2\n" // tmp_1[2] = input_1[2] * cos_1
                                               // + tmp_1[2]
                    "vfmacc.vv v19, v11, v2\n" // tmp_1[3] = input_1[3] * cos_1
                                               // + tmp_1[3]
                    "vs4r.v v16, (%[output_1dp])\n"
                    "addi %[output_1dp], %[output_1dp], %[vl] * 2 * 4\n"
                    // 1st half: output_dp[h] = ntt::mul_sub(input_0, cos_0,
                    // input_1 * sin_0)
                    "vfmul.vv v12, v8, v1\n"  // tmp_0[0] = input_1[0] * sin_0
                    "vfmul.vv v13, v9, v1\n"  // tmp_0[1] = input_1[1] * sin_0
                    "vfmul.vv v14, v10, v1\n" // tmp_0[2] = input_1[2] * sin_0
                    "vfmul.vv v15, v11, v1\n" // tmp_0[3] = input_1[3] * sin_0
                    "vfmsac.vv v12, v4, v0\n" // tmp_0[0] = input_0[0] * cos_0 -
                                              // tmp_0[0]
                    "vfmsac.vv v13, v5, v0\n" // tmp_0[1] = input_0[1] * cos_0 -
                                              // tmp_0[1]
                    "vfmsac.vv v14, v6, v0\n" // tmp_0[2] = input_0[2] * cos_0 -
                                              // tmp_0[2]
                    "vfmsac.vv v15, v7, v0\n" // tmp_0[3] = input_0[3] * cos_0 -
                                              // tmp_0[3]
                    "vs4r.v v12, (%[output_0dp])\n"
                    "addi %[output_0dp], %[output_0dp], %[vl] * 2 * 4\n"
                    : [input_0dp] "+r"(input_0dp), [input_1dp] "+r"(input_1dp),
                      [output_0dp] "+r"(output_0dp),
                      [output_1dp] "+r"(output_1dp)
                    : [vl] "i"((size_t)vl)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                      "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                      "v17", "v18", "v19", "memory");
            }
        }
    }
};
} // namespace nncase::ntt::ukernels
