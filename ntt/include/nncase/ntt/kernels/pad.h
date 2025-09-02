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
#include "../loop.h"
#include "../padding.h"
#include "../utility.h"

namespace nncase::ntt {
namespace pad_detail {
template <Tensor TIn, Tensor TOut, Paddings TPaddings, ScalarOrVector TElem>
void naive_apply(const TIn &input, TOut &output, const TPaddings &paddings,
                 const TElem &pad_value) {
    constexpr auto rank = TIn::rank();
    using element_type = typename TIn::element_type;
    ntt::apply(output.shape(), [&](auto out_index) {
        bool dopad = false;
        const auto in_index = generate_shape<rank>([&](auto i) {
            auto in_dim = out_index[i] - paddings[i].before;
            if (in_dim < 0 || in_dim >= input.shape()[i]) {
                dopad = true;
            }
            return in_dim;
        });
        auto addr_output =
            reinterpret_cast<element_type *>(&(output(out_index)));
        if (dopad) {
            ntt::u_unary(ntt::ops::copy<element_type>{}, &pad_value, 1,
                         addr_output, 1, 1);
        } else {
            auto addr_input =
                reinterpret_cast<const element_type *>(&(input(in_index)));
            ntt::u_unary(ntt::ops::copy<element_type>{}, addr_input, 1,
                         addr_output, 1, 1);
        }
    });
}

template <Tensor TIn, Tensor TOut, Paddings TPaddings, ScalarOrVector TElem>
void pad_impl(const TIn &input, TOut &output, const TPaddings &paddings,
              const TElem &pad_value) {
    constexpr auto rank = TIn::rank();
    if constexpr (FixedDimension<std::tuple_element_t<
                      rank - 1, typename TIn::shape_type>>) {
        using element_type = typename TIn::element_type;
        if (paddings[rank - 1][0] == 0 && paddings[rank - 1][1] == 0) {
            auto domains = output.shape().template slice<0, rank - 1>();
            auto input_strides = input.strides().template slice<0, rank - 1>();
            auto output_strides =
                output.strides().template slice<0, rank - 1>();
            constexpr auto last_dim =
                std::tuple_element_t<rank - 1, typename TIn::shape_type>{};

            ntt::apply(domains, [&](auto out_index) {
                bool dopad = false;
                auto addr_output = output.elements().data() +
                                   linear_offset(out_index, output_strides);

                const auto in_index = generate_shape<rank - 1>([&](auto i) {
                    auto in_dim = out_index[i] - paddings[i].before;
                    if (in_dim < 0 || in_dim >= input.shape()[i]) {
                        dopad = true;
                    }
                    return in_dim;
                });
                if (dopad) {
                    alignas(sizeof(element_type))
                        element_type pad_vals[last_dim]{};
                    std::fill(std::begin(pad_vals), std::end(pad_vals),
                              pad_value);
                    ntt::u_unary(ntt::ops::copy<element_type>{}, pad_vals, 1,
                                 addr_output, 1, last_dim);
                } else {
                    auto addr_input = input.elements().data() +
                                      linear_offset(in_index, input_strides);
                    ntt::u_unary(ntt::ops::copy<element_type>{}, addr_input, 1,
                                 addr_output, 1, last_dim);
                }
            });
        } else {
            naive_apply(input, output, paddings, pad_value);
        }
    } else {
        naive_apply(input, output, paddings, pad_value);
    }
}
} // namespace pad_detail

/**
 * @brief pad
 *
 * @param input input tensor.
 * @param output output tensor.
 * @param pad_value pad value.
 */
template <Tensor TIn, class TOut, Paddings TPaddings,
          ScalarOrVector TElem = typename TIn::element_type>
requires(bool(TIn::rank() == TPaddings::rank())) void pad(
    const TIn &input, TOut &&output, const TPaddings &paddings,
    const TElem &pad_value = {}) noexcept {
    pad_detail::pad_impl(input, output, paddings, pad_value);
}
} // namespace nncase::ntt
