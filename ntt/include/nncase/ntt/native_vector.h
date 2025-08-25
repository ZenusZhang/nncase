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
#include "detail/vector_storage.h"

#define NTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, ...)        \
    namespace nncase::ntt {                                                    \
    template <>                                                                \
    struct vector_storage_traits<element_type_, fixed_shape_t<__VA_ARGS__>> {  \
        using buffer_type = native_type;                                       \
        using element_type = element_type_;

#define NTT_END_DEFINE_NATIVE_VECTOR()                                         \
    }                                                                          \
    ;                                                                          \
    }

#define NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(element_type_, native_type,     \
                                               ...)                            \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, __VA_ARGS__)    \
                                                                               \
    template <Dimensions TIndex>                                               \
    static element_type_ get_element(const native_type &array,                 \
                                     const TIndex &index) noexcept {           \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        return array[(size_t)index[dim_zero]];                                 \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(native_type &array, const TIndex &index,           \
                            element_type_ value) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        array[(size_t)index[dim_zero]] = value;                                \
    }

#define NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(                           \
    element_type_, native_type, cast_type, ...)                                \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, __VA_ARGS__)    \
                                                                               \
    template <Dimensions TIndex>                                               \
    static element_type_ get_element(const native_type &array,                 \
                                     const TIndex &index) noexcept {           \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        return cast_type(array)[(size_t)index[dim_zero]];                      \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(native_type &array, const TIndex &index,           \
                            element_type_ value) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        auto casted_array = cast_type(array);                                  \
        casted_array[(size_t)index[dim_zero]] = value;                         \
        array = (native_type)casted_array;                                     \
    }

#define NTT_DEFINE_NATIVE_VECTOR_DEFAULT_BITCAST(                              \
    element_type_, native_type, cast_type, bitcast_type, lanes)                \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, lanes)          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static element_type_ get_element(const native_type &array,                 \
                                     const TIndex &index) noexcept {           \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        const auto casted_array = (cast_type)array;                            \
        return std::bit_cast<element_type_>(                                   \
            casted_array[(size_t)index[dim_zero]]);                            \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(native_type &array, const TIndex &index,           \
                            element_type_ value) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        auto casted_array = (cast_type)array;                                  \
        casted_array[(size_t)index[dim_zero]] =                                \
            std::bit_cast<bitcast_type>(value);                                \
        array = (native_type)casted_array;                                     \
    }
