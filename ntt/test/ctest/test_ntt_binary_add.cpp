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
#include "test_ntt_binary.h"

//test case combination:
// 1. lhs/rhs
// 2. dynamic/fixed
// 3. lhs broadcast to rhs, rhs broadcast to lhs
// 3.1. 1 dim broadcast
// 3.2. 2 dims broadcast
// 4. scalar/vector/2d vector
// 5. tensor/ view

// TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_vector) {
//     // init
//     auto ntt_tensor_lhs =  make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1>);
//     NttTest::init_tensor(ntt_tensor_lhs, -10, 10);

//     auto ntt_tensor_rhs =  make_tensor<int>(ntt::fixed_shape_v<1, 3, 1, 16>);
//     NttTest::init_tensor(ntt_tensor_rhs, -10, 10);

//     // ntt
//     auto ntt_output1 = make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1, 3, 1, 16>);
//     ntt::binary<ntt::ops::add>(ntt_tensor_lhs, ntt_tensor_rhs, ntt_output1);

//     // // if mxn tensor-of-vector<v> op mxn tensor-of-scalar, 
//     // //broadcast the ntt mxn tensor-of-scalar to ort mxnxv tensor-of-scalar

//     // // ort
//     auto [ort_lhs, ort_rhs] = NttTest::convert_and_align_to_ort(ntt_tensor_lhs, ntt_tensor_rhs);
//     auto ort_output = ortki_Add(ort_lhs, ort_rhs);
//     // ortki_Add(ort_lhs, ort_rhs);
//     // // compare
//     auto ntt_output2 = make_tensor<ntt::vector<uint32_t, 8>>(ntt::fixed_shape_v<1, 3, 1, 16>);
//     NttTest::ort2ntt(ort_output, ntt_output2);
//     EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

// }



TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_1D_vector_rhs_2D_vector) {
    // init
    auto ntt_tensor_lhs =  make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_tensor_lhs, -10, 10);

    auto ntt_tensor_rhs =  make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_tensor_rhs, -10, 10);

    // ntt
    auto ntt_output1 = make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1>);
    ntt::binary<ntt::ops::ceil_div>(ntt_tensor_lhs, ntt_tensor_rhs, ntt_output1);

    // // if mxn tensor-of-vector<v> op mxn tensor-of-scalar, 
    // //broadcast the ntt mxn tensor-of-scalar to ort mxnxv tensor-of-scalar

    // // ort
    auto [ort_lhs, ort_rhs] = NttTest::convert_and_align_to_ort(ntt_tensor_lhs, ntt_tensor_rhs);
    auto ntt_neg1 = make_tensor<int>(ntt::fixed_shape_v<1>);
    ntt_neg1(0) = -1;
    auto ort_neg1 = NttTest::ntt2ort(ntt_neg1);

    auto ort_output = ortki_Div(ortki_Add(ortki_Add(ort_rhs,ort_neg1), ort_lhs), ort_rhs);
    // ortki_Add(ort_lhs, ort_rhs);
    // // compare
    auto ntt_output2 = make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

}


// //fixed fixed fixed group, for demonstrate the basic test macro
// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_normal,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_scalar,  
//                             (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_rhs_scalar,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_vector,  
//                             (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_rhs_vector,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_multidirectional,  
//                             (fixed_shape_v<1, 3, 1, 16>), (fixed_shape_v<3, 1, 16, 1>), (fixed_shape_v<3, 3, 16, 16>),
//                            int, add, Add) 

// //fixed dynamic dynamic group(with default shape)
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddint, fixed, dynamic,dynamic,  
//                            int, add, Add) 
// //dynamic fixed dynamic group
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddint, dynamic, fixed, dynamic,  
//                            int, add, Add) 
// //dynamic dynamic dynamic group
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddint, dynamic ,dynamic,dynamic,  
//                            int, add, Add) 
                           


// DEFINE_test_vector(add, Add)
// TEST(BinaryTestAddint, vector) {                                        
//     TEST_VECTOR(int)                                                    
//     TEST_VECTOR(int32_t)                                                  
//     TEST_VECTOR(int64_t)                                                  
// }                                                                          

int main(int argc, char *argv[]) {                                         
    ::testing::InitGoogleTest(&argc, argv);                                
    return RUN_ALL_TESTS();                                                
}

