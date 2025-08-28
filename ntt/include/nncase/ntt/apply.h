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
#include "dimension.h"
#include "shape.h"

namespace nncase::ntt {
namespace detail {
template <size_t Axis, class TIndex, class Shape, class Callable>
constexpr void apply_impl(const TIndex &index_prefix, const Shape &shape,
                          Callable &&callable) {
    for (dim_t i = 0; i < shape[fixed_dim_v<Axis>]; i++) {
        const auto index = index_prefix.append(i);
        if constexpr (Axis == Shape::rank() - 1) {
            callable(index);
        } else {
            apply_impl<Axis + 1>(index, shape,
                                 std::forward<Callable>(callable));
        }
    }
}

template <size_t Axis, class TIndex, class Shape, class Strides, class Callable>
constexpr void
apply_with_linear_offset_impl(const TIndex &index_prefix, dim_t linear_offset,
                              const Shape &shape, const Strides &strides,
                              Callable &&callable) {
    for (dim_t i = 0; i < shape[fixed_dim_v<Axis>]; i++) {
        const auto index = index_prefix.append(i);
        if constexpr (Axis == Shape::rank() - 1) {
            callable(index, linear_offset);
        } else {
            apply_with_linear_offset_impl<Axis + 1>(
                index, linear_offset, shape, strides,
                std::forward<Callable>(callable));
        }

        linear_offset += strides[fixed_dim_v<Axis>];
    }
}
} // namespace detail

template <class Shape, class Callable>
constexpr void apply(const Shape &shape, Callable &&callable) {
    if constexpr (Shape::rank()) {
        detail::apply_impl<0>(fixed_shape_v<>, shape,
                              std::forward<Callable>(callable));
    } else {
        callable(fixed_shape_v<>);
    }
}

template <Tensor TTensor, class Callable>
constexpr void apply_with_linear_offset(TTensor &tensor, Callable &&callable) {
    if constexpr (TTensor::rank()) {
        detail::apply_with_linear_offset_impl<0>(
            fixed_shape_v<>, 0, tensor.shape(), tensor.strides(),
            std::forward<Callable>(callable));
    } else {
        callable(fixed_shape_v<>, 0);
    }
}
} // namespace nncase::ntt
