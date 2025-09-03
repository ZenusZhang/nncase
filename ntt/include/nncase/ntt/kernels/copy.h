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
#include "unary.h"
#include <type_traits>

namespace nncase::ntt {
template <class TIn, class TOut>
void tensor_copy(const TIn &input, TOut &&output) noexcept {
    unary<ops::copy>(input, output);
}

namespace detail {
template <Tensor TTensor, bool Arch> struct tensor_zero_impl {
    constexpr void operator()(TTensor &tensor) noexcept {
        ntt::apply(tensor.shape(), [&](auto index) { tensor(index) = {}; });
    }
};
} // namespace detail

template <class TOut> void tensor_zero(TOut &&output) noexcept {
    detail::tensor_zero_impl<std::decay_t<TOut>, true>()(output);
}
} // namespace nncase::ntt
