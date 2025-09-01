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
#include "../post_ops.h"
#include "../tensor_ops.h"
#include "../ukernels.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut, FixedDimensions VectorizedAxes,
          template <class> class TPostOp>
class cast_impl {
    inline static constexpr size_t rank = TIn::rank();

    // FIXME: vector<bool> of x86 may fail.
    using InElemType = element_or_scalar_t<TIn>;
    using OutElemType = element_or_scalar_t<TOut>;
    static_assert((Vector<InElemType> && Vector<OutElemType>) ||
                      (Scalar<InElemType> && Scalar<OutElemType>),
                  "input & output must have the same type.");
    inline static constexpr auto in_ele_size =
        sizeof(std::conditional_t<Vector<InElemType>,
                                  element_or_scalar_t<InElemType>, size_t>);
    inline static constexpr auto out_ele_size =
        sizeof(std::conditional_t<Vector<OutElemType>,
                                  element_or_scalar_t<OutElemType>, size_t>);
    inline static constexpr float scale = (float)in_ele_size / out_ele_size;

    inline static constexpr auto in_offset_scale = scale > 1.0f ? (size_t)scale
                                                                : (size_t)1;
    inline static constexpr auto
        out_offset_scale = scale > 1.0f ? (size_t)1 : (size_t)(1.0f / scale);
    inline static constexpr auto axis =
        VectorizedAxes::rank() == 1 ? VectorizedAxes{}.at(0) : 0_dim;

  public:
    constexpr void operator()(const TIn &input, TOut &output,
                              const VectorizedAxes &) noexcept {

        constexpr auto vectorizedAxes = VectorizedAxes{};
        static_assert(vectorizedAxes.rank() == 0 || vectorizedAxes.rank() == 1,
                      "vectorizedAxes rank for cast must be 0 or 1");

        auto input_conti_dims = contiguous_dims(input.shape(), input.strides());
        auto output_conti_dims =
            contiguous_dims(output.shape(), output.strides());

        auto input_stride = vectorizedAxes.rank() == 1
                                ? input.strides()[vectorizedAxes.at(0)]
                                : 1;
        auto output_stride = vectorizedAxes.rank() == 1
                                 ? output.strides()[vectorizedAxes.at(0)]
                                 : 1;

        if (Vector<InElemType> && Vector<OutElemType> &&
            (input_conti_dims == TIn::rank()) &&
            (output_conti_dims == TOut::rank())) {

            constexpr auto rank = TIn::rank();
            auto input_domain_before_axis =
                input.shape().template slice<0, axis + 1>();
            auto input_domain_after_axis =
                input.shape().template slice<axis + 1, rank - axis - 1>();

            auto output_domain_before_axis =
                output.shape().template slice<0, axis + 1>();
            auto output_domain_after_axis =
                output.shape().template slice<axis + 1, rank - axis - 1>();
            dynamic_shape_t<rank> src_index;
            ntt::loop<rank>([&](auto &i) { src_index[i] = 0; });

            if constexpr (in_offset_scale > 1 && out_offset_scale == 1) {

                ntt::apply(output_domain_before_axis, [&](auto index) {
                    loop<axis + 1>([&](auto j) { src_index[j] = index[j]; });
                    auto in_index = src_index;
                    if constexpr (vectorizedAxes.rank() == 1)
                        in_index[fixed_dim_v<vectorizedAxes.at(0)>] *=
                            in_offset_scale;
                    auto in_ptr = &input(in_index);
                    auto out_ptr = &output(src_index);
                    auto len = output_domain_after_axis.length();
                    ntt::u_cast<in_offset_scale, out_offset_scale, TPostOp>(
                        in_ptr, input_stride, out_ptr, output_stride, len);
                });

            } else if constexpr (in_offset_scale == 1 && out_offset_scale > 1) {

                ntt::apply(input_domain_before_axis, [&](auto index) {
                    loop<axis + 1>([&](auto j) { src_index[j] = index[j]; });
                    auto out_index = src_index;
                    if constexpr (vectorizedAxes.rank() == 1)
                        out_index[fixed_dim_v<vectorizedAxes.at(0)>] *=
                            out_offset_scale;

                    auto in_ptr = &input(src_index);
                    auto out_ptr = &output(out_index);
                    auto len = input_domain_after_axis.length();
                    ntt::u_cast<in_offset_scale, out_offset_scale, TPostOp>(
                        in_ptr, input_stride, out_ptr, output_stride, len);
                });
            } else {
                auto in_ptr = input.elements().data();
                auto out_ptr = output.elements().data();
                auto len = input.shape().length();
                ntt::u_cast<in_offset_scale, out_offset_scale, TPostOp>(
                    in_ptr, 1, out_ptr, 1, len);
            }
        } else {
            if constexpr (in_offset_scale > 1 && out_offset_scale == 1) {
                ntt::apply(output.shape(), [&](auto index) {
                    auto in_index = index;
                    if constexpr (vectorizedAxes.rank() == 1)
                        in_index[fixed_dim_v<vectorizedAxes.at(0)>] *=
                            in_offset_scale;
                    prepend_lanes_t<InElemType, in_offset_scale> in_temp{};
                    ntt::loop<in_offset_scale>([&](auto i) {
                        in_temp(i) = input(in_index);
                        if constexpr (vectorizedAxes.rank() == 1) {
                            in_index[fixed_dim_v<vectorizedAxes.at(0)>] += 1;
                        }
                    });
                    output(index) =
                        ntt::cast_elem<element_or_scalar_t<OutElemType>>(
                            in_temp);
                    output(index) = TPostOp<OutElemType>()(output(index));
                });
            } else if constexpr (in_offset_scale == 1 && out_offset_scale > 1) {
                ntt::apply(input.shape(), [&](auto index) {
                    auto out_index = index;
                    if constexpr (vectorizedAxes.rank() == 1)
                        out_index[fixed_dim_v<vectorizedAxes.at(0)>] *=
                            out_offset_scale;

                    auto tmp_output =
                        ntt::cast_elem<element_or_scalar_t<OutElemType>>(
                            input(index));
                    ntt::loop<out_offset_scale>([&](auto s) {
                        output(out_index) = tmp_output(s);
                        output(out_index) =
                            TPostOp<OutElemType>()(output(out_index));
                        if constexpr (vectorizedAxes.rank() == 1)
                            out_index[fixed_dim_v<vectorizedAxes.at(0)>] += 1;
                    });
                });
            } else {
                ntt::apply(input.shape(), [&](auto index) {
                    output(index) =
                        ntt::cast_elem<element_or_scalar_t<OutElemType>>(
                            input(index));
                    output(index) = TPostOp<OutElemType>()(output(index));
                });
            }
        }
    }
};
} // namespace detail

template <template <class> class TPostOp = DefaultPostOp, Tensor TIn,
          Tensor TOut, FixedDimensions VectorizedAxes = shape_t<>>
void cast(const TIn &input, TOut &&output,
          const VectorizedAxes &vectorizedAxes = {}) noexcept {
    detail::cast_impl<TIn, std::decay_t<TOut>, VectorizedAxes, TPostOp> impl;
    impl(input, output, vectorizedAxes);
}
} // namespace nncase::ntt
