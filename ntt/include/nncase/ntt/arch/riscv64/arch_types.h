/* Copyright 2019-2024 Canaan Inc.
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
#include "../../../bfloat16.h"
#include "../../../float8.h"
#include "../../../half.h"
#include "../../native_vector.h"
#ifdef __riscv_vector
#include <riscv_vector.h>

#ifndef __riscv_v_fixed_vlen
#error "-mrvv-vector-bits=zvl must be specified in toolchain compiler option."
#endif

#ifndef NTT_VLEN
#define NTT_VLEN __riscv_v_fixed_vlen
#endif

#ifndef NTT_VL_
#define NTT_VL(sew, op, lmul) ((NTT_VLEN) / (sew)op(lmul))
#endif

#define NTT_BEGIN_DEFINE_RVV_NATIVE_VECTOR2D_DEFAULT(                                               \
    element_type_, native_element_type, native_element_type_slim, lmul, lanes)                      \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(                                                                 \
        element_type_, fixed_v##native_element_type##m##lmul##_t, lmul, lanes)                      \
                                                                                                    \
    template <Dimensions TIndex>                                                                    \
    requires(TIndex::rank() == 1) static ntt::vector<element_type_, lanes>                          \
    get_element(const buffer_type &array,                                                           \
                [[maybe_unused]] const TIndex &index) noexcept {                                    \
        static_assert(FixedDimensions<TIndex>);                                                     \
        constexpr size_t index_value = TIndex{}[dim_zero];                                          \
        return __riscv_vget_v_##native_element_type_slim##m##lmul##_##native_element_type_slim##m1( \
            array, index_value);                                                                    \
    }                                                                                               \
    template <Dimensions TIndex>                                                                    \
    requires(TIndex::rank() == 2) static element_type_ get_element(                                 \
        const buffer_type &array, const TIndex &index) noexcept {                                   \
        return array[(size_t)ntt::linear_offset(index,                                              \
                                                fixed_shape_v<lmul, lanes>)];                       \
    }                                                                                               \
                                                                                                    \
    template <Dimensions TIndex>                                                                    \
    requires(TIndex::rank() == 1) static void set_element(                                          \
        buffer_type &array, [[maybe_unused]] const TIndex &index,                                   \
        ntt::vector<element_type_, lanes> value) noexcept {                                         \
        static_assert(FixedDimensions<TIndex>);                                                     \
        constexpr size_t index_value = TIndex{}[dim_zero];                                          \
        array =                                                                                     \
            __riscv_vset_v_##native_element_type_slim##m1_##native_element_type_slim##m##lmul(      \
                array, index_value, value);                                                         \
    }                                                                                               \
                                                                                                    \
    template <Dimensions TIndex>                                                                    \
    requires(TIndex::rank() == 2) static void set_element(                                          \
        buffer_type &array, const TIndex &index,                                                    \
        element_type_ value) noexcept {                                                             \
        array[(size_t)ntt::linear_offset(index, fixed_shape_v<lmul, lanes>)] =                      \
            value;                                                                                  \
    }

#if defined(__riscv_vector) &&                                                 \
    (defined(__riscv_zvfbfmin) || defined(__riscv_zvfbf))
#define REGISTER_BFLOAT16_TYPE_WITH_LMUL_LT1()                                 \
    typedef vbfloat16mf2_t fixed_vbfloat16mf2_t                                \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vbfloat16mf4_t fixed_vbfloat16mf4_t                                \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));

#define REGISTER_BFLOAT16_TYPE_WITH_LMUL_GE1(lmul)                             \
    typedef vbfloat16m##lmul##_t fixed_vbfloat16m##lmul##_t                    \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));

#else
#define REGISTER_BFLOAT16_TYPE_WITH_LMUL_LT1()
#define REGISTER_BFLOAT16_TYPE_WITH_LMUL_GE1(lmul)
#endif

#if defined(NNCASE_XPU_MODULE) && defined(SYS_MODE)
#define REGISTER_F8E4M3_TYPE_WITH_LMUL_LT1()                                   \
    typedef vfloat8e4m3mf2_t fixed_vfloat8e4m3mf2_t                            \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vfloat8e4m3mf4_t fixed_vfloat8e4m3mf4_t                            \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));

#define REGISTER_F8E4M3_TYPE_WITH_LMUL_GE1(lmul)                               \
    typedef vfloat8e4m3m##lmul##_t fixed_vfloat8e4m3m##lmul##_t                \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));

#else
#define REGISTER_F8E4M3_TYPE_WITH_LMUL_LT1()
#define REGISTER_F8E4M3_TYPE_WITH_LMUL_GE1(lmul)
#endif

// rvv fixed type
#define REGISTER_RVV_FIXED_TYPE_WITH_LMUL_LT1                                  \
    typedef vint8mf2_t fixed_vint8mf2_t                                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vint8mf4_t fixed_vint8mf4_t                                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vint8mf8_t fixed_vint8mf8_t                                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 8)));                  \
    typedef vuint8mf2_t fixed_vuint8mf2_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vuint8mf4_t fixed_vuint8mf4_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vuint8mf8_t fixed_vuint8mf8_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 8)));                  \
    typedef vint16mf2_t fixed_vint16mf2_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vint16mf4_t fixed_vint16mf4_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vuint16mf2_t fixed_vuint16mf2_t                                    \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vuint16mf4_t fixed_vuint16mf4_t                                    \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vint32mf2_t fixed_vint32mf2_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vuint32mf2_t fixed_vuint32mf2_t                                    \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vfloat16mf2_t fixed_vfloat16mf2_t                                  \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vfloat16mf4_t fixed_vfloat16mf4_t                                  \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    REGISTER_BFLOAT16_TYPE_WITH_LMUL_LT1()                                     \
    typedef vfloat32mf2_t fixed_vfloat32mf2_t                                  \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    REGISTER_F8E4M3_TYPE_WITH_LMUL_LT1()

#define REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(lmul)                            \
    typedef vint8m##lmul##_t fixed_vint8m##lmul##_t                            \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint8m##lmul##_t fixed_vuint8m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vint16m##lmul##_t fixed_vint16m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint16m##lmul##_t fixed_vuint16m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vint32m##lmul##_t fixed_vint32m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint32m##lmul##_t fixed_vuint32m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vint64m##lmul##_t fixed_vint64m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint64m##lmul##_t fixed_vuint64m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vfloat16m##lmul##_t fixed_vfloat16m##lmul##_t                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    REGISTER_BFLOAT16_TYPE_WITH_LMUL_GE1(lmul)                                 \
    typedef vfloat32m##lmul##_t fixed_vfloat32m##lmul##_t                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vfloat64m##lmul##_t fixed_vfloat64m##lmul##_t                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    REGISTER_F8E4M3_TYPE_WITH_LMUL_GE1(lmul)

REGISTER_RVV_FIXED_TYPE_WITH_LMUL_LT1
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(1)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(2)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(4)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(8)

#if defined(__riscv_vector) &&                                                 \
    (defined(__riscv_zvfbfmin) || defined(__riscv_zvfbf))
#define NTT_DEFINE_BFLOAT16_VECTORS_LT()                                       \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        bfloat16, fixed_vbfloat16mf2_t, NTT_VLEN / 8 / sizeof(bfloat16) / 2)   \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        bfloat16, fixed_vbfloat16mf4_t, NTT_VLEN / 8 / sizeof(bfloat16) / 4)   \
    NTT_END_DEFINE_NATIVE_VECTOR()

#define NTT_DEFINE_BFLOAT16_VECTORS_GE(lmul)                                   \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        bfloat16, fixed_vbfloat16m##lmul##_t,                                  \
        NTT_VLEN / 8 / sizeof(bfloat16) * lmul)                                \
    NTT_END_DEFINE_NATIVE_VECTOR()

#else
#define NTT_DEFINE_BFLOAT16_VECTORS_LT()
#define NTT_DEFINE_BFLOAT16_VECTORS_GE(lmul)
#endif

#if defined(NNCASE_XPU_MODULE) && defined(SYS_MODE)
#define NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_F2()                         \
    NTT_DEFINE_NATIVE_VECTOR_DEFAULT_BITCAST(                                  \
        float_e4m3_t, fixed_vfloat8e4m3mf2_t, fixed_vfloat8e4m3mf2_t,          \
        signed char, NTT_VLEN / 8 / sizeof(float_e4m3_t) / 2)                  \
    NTT_END_DEFINE_NATIVE_VECTOR()

#define NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_F4()                         \
    NTT_DEFINE_NATIVE_VECTOR_DEFAULT_BITCAST(                                  \
        float_e4m3_t, fixed_vfloat8e4m3mf4_t, fixed_vfloat8e4m3mf4_t,          \
        signed char, NTT_VLEN / 8 / sizeof(float_e4m3_t) / 4)                  \
    NTT_END_DEFINE_NATIVE_VECTOR()

#define NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_GE1(lmul)                    \
    NTT_DEFINE_NATIVE_VECTOR_DEFAULT_BITCAST(                                  \
        float_e4m3_t, fixed_vfloat8e4m3m##lmul##_t,                            \
        fixed_vfloat8e4m3m##lmul##_t, signed char,                             \
        NTT_VLEN / 8 / sizeof(float_e4m3_t) * lmul)                            \
    NTT_END_DEFINE_NATIVE_VECTOR()
#else
#define NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_F2()
#define NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_F4()
#define NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_GE1(lmul)
#endif

// rvv native vector
#define NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_LT1                                 \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int8_t, fixed_vint8mf2_t,           \
                                           NTT_VLEN / 8 / sizeof(int8_t) / 2)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int8_t, fixed_vint8mf4_t,           \
                                           NTT_VLEN / 8 / sizeof(int8_t) / 4)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int8_t, fixed_vint8mf8_t,           \
                                           NTT_VLEN / 8 / sizeof(int8_t) / 8)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(uint8_t) / 2) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8mf4_t,         \
                                           NTT_VLEN / 8 / sizeof(uint8_t) / 4) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8mf8_t,         \
                                           NTT_VLEN / 8 / sizeof(uint8_t) / 8) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, fixed_vint16mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(int16_t) / 2) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, fixed_vint16mf4_t,         \
                                           NTT_VLEN / 8 / sizeof(int16_t) / 4) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        uint16_t, fixed_vuint16mf2_t, NTT_VLEN / 8 / sizeof(uint16_t) / 2)     \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        uint16_t, fixed_vuint16mf4_t, NTT_VLEN / 8 / sizeof(uint16_t) / 4)     \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int32_t, fixed_vint32mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(int32_t) / 2) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        uint32_t, fixed_vuint32mf2_t, NTT_VLEN / 8 / sizeof(uint32_t) / 2)     \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(half, fixed_vfloat16mf2_t,          \
                                           NTT_VLEN / 8 / sizeof(half) / 2)    \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(half, fixed_vfloat16mf4_t,          \
                                           NTT_VLEN / 8 / sizeof(half) / 4)    \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_DEFINE_BFLOAT16_VECTORS_LT()                                           \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(float, fixed_vfloat32mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(float) / 2)   \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_F2()                             \
    NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_F4()

#define NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(lmul)                           \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        int8_t, fixed_vint8m##lmul##_t, NTT_VLEN / 8 / sizeof(int8_t) * lmul)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(uint8_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, fixed_vint16m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(int16_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint16_t, fixed_vuint16m##lmul##_t, \
                                           NTT_VLEN / 8 / sizeof(uint16_t) *   \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int32_t, fixed_vint32m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(int32_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint32_t, fixed_vuint32m##lmul##_t, \
                                           NTT_VLEN / 8 / sizeof(uint32_t) *   \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int64_t, fixed_vint64m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(int64_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint64_t, fixed_vuint64m##lmul##_t, \
                                           NTT_VLEN / 8 / sizeof(uint64_t) *   \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        float, fixed_vfloat32m##lmul##_t, NTT_VLEN / 8 / sizeof(float) * lmul) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(half, fixed_vfloat16m##lmul##_t,    \
                                           NTT_VLEN / 8 / sizeof(half) * lmul) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_DEFINE_BFLOAT16_VECTORS_GE(lmul)                                       \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(double, fixed_vfloat64m##lmul##_t,  \
                                           NTT_VLEN / 8 / sizeof(double) *     \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_DEFINE_F8E4M3_NATIVE_VECTOR_WITH_LMUL_GE1(lmul)

#define NTT_DEFINE_NATIVE_VECTOR2D_WITH_LMUL_GE1(lmul)                         \
    NTT_BEGIN_DEFINE_RVV_NATIVE_VECTOR2D_DEFAULT(float, float32, f32, lmul,    \
                                                 NTT_VLEN / 8 / sizeof(float)) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_RVV_NATIVE_VECTOR2D_DEFAULT(half, float16, f16, lmul,     \
                                                 NTT_VLEN / 8 / sizeof(half))  \
    NTT_END_DEFINE_NATIVE_VECTOR()

NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_LT1
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(1)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(2)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(4)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(8)

NTT_DEFINE_NATIVE_VECTOR2D_WITH_LMUL_GE1(2)

// mask vectors
#define NTT_DEFINE_NATIVE_MASK_VECTOR(bits)                                    \
    typedef vbool##bits##_t fixed_vbool##bits##_t                              \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / bits)));               \
                                                                               \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(bool, fixed_vbool##bits##_t,                \
                                   NTT_VLEN / bits)                            \
                                                                               \
    template <Dimensions TIndex>                                               \
    static bool get_element(const fixed_vbool##bits##_t &array,                \
                            const TIndex &index) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        const fixed_vint8m1_t i8_value =                                       \
            __riscv_vreinterpret_v_b##bits##_i8m1(array);                      \
        const auto offset = (size_t)index[dim_zero];                           \
        return (i8_value[offset / 8] & (1 << (offset % 8))) != 0;              \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(fixed_vbool##bits##_t &array, const TIndex &index, \
                            bool value) noexcept {                             \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        fixed_vint8m1_t i8_value =                                             \
            __riscv_vreinterpret_v_b##bits##_i8m1(array);                      \
        const auto offset = (size_t)index[dim_zero];                           \
        const auto mask = ~(1 << (offset % 8));                                \
        i8_value[offset / 8] =                                                 \
            (i8_value[offset / 8] & mask) | ((value ? 1 : 0) << (offset % 8)); \
        array = __riscv_vreinterpret_v_i8m1_b##bits(i8_value);                 \
    }                                                                          \
    NTT_END_DEFINE_NATIVE_VECTOR()

NTT_DEFINE_NATIVE_MASK_VECTOR(1)
NTT_DEFINE_NATIVE_MASK_VECTOR(2)
NTT_DEFINE_NATIVE_MASK_VECTOR(4)
NTT_DEFINE_NATIVE_MASK_VECTOR(8)
NTT_DEFINE_NATIVE_MASK_VECTOR(16)

#if !defined(__clang__) || __riscv_v_fixed_vlen >= 256
NTT_DEFINE_NATIVE_MASK_VECTOR(32)
#if !defined(__clang__) || __riscv_v_fixed_vlen >= 512
NTT_DEFINE_NATIVE_MASK_VECTOR(64)
#endif
#endif

#undef NTT_DEFINE_NATIVE_MASK_VECTOR
#endif
