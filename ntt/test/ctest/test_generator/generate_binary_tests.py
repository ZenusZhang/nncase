#test case combination:
# 1. lhs/rhs
# 2. dynamic/fixed
# 3. lhs broadcast to rhs, rhs broadcast to lhs
# 3.1. 1 dim broadcast
# 3.2. 2 dims broadcast
# 4. scalar/vector/2d vector
# 5. tensor/ view

import itertools
import os
from typing import List
from test_generator_base import *



class BinaryTestGenerator(BaseTestGenerator):
    def __init__(self):
        super().__init__()
        
        # ORT binary operations do not support these data types, need to cast to float32
        self.types_need_to_be_cast = [
            'bool',
            'uint8_t', 
            'uint16_t',
            'uint32_t',
            'uint64_t',
            'int8_t',
            'int16_t', 
            'bfloat16',
            'float_e4m3_t',
            'float_e5m2_t'
        ]
        self.op_str_map = {
            "add": f"auto ort_output = ortki_Add(ort_input_lhs, ort_input_rhs);",
            "sub": f"auto ort_output = ortki_Sub(ort_input_lhs, ort_input_rhs);",
            "mul": f"auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);",
            "div": f"auto ort_output = ortki_Div(ort_input_lhs, ort_input_rhs);",
            # "ceil_div": f"auto ort_output = ortki_CeilDiv(ort_input_lhs, ort_input_rhs);",
            # "floor_mod": f"auto ort_output = ortki_FloorMod(ort_input_lhs, ort_input_rhs);",
            # "mod": f"auto ort_output = ortki_Mod(ort_input_lhs, ort_input_rhs);",
            # "min": f"auto ort_output = ortki_Min(ort_input_lhs, ort_input_rhs);",
            # "max": f"auto ort_output = ortki_Max(ort_input_lhs, ort_input_rhs);",
            # "pow": f"auto ort_output = ortki_Pow(ort_input_lhs, ort_input_rhs);",
        }

    def is_div_operation(self) -> bool:
        """Check if the current operation is division, to disable zero generation."""
        result = (hasattr(self, 'ntt_op_str') and self.ntt_op_str in ["div", "mod"])
        return result

    def generate_test_name(self, datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape, 
        lhs_dims_spec, rhs_dims_spec, 
        lhs_vector_rank, rhs_vector_rank, 
        lhs_continuity, rhs_continuity):
        
        parts = []
        
        # 1. 数据类型
        parts.append(f"{datatype.name_suffix}")
        
        # 2. 左操作数信息
        lhs_shape_type = "dynamic" if lhs_is_dynamic_shape else "fixed"
        parts.append(f"lhs_{lhs_shape_type}")
        
        # 左操作数向量维度
        if lhs_vector_rank == 0:
            parts.append("scalar")
        else:
            parts.append(f"{lhs_vector_rank}D_vector")
        
        # 左操作数连续性 - contiguous改成view，non_contiguous改成raw_tensor
        if lhs_continuity.is_contiguous:
            parts.append("raw_tensor")
        else:
            op_str = "mul2" if lhs_continuity.big_tensor_op == "*2" else "add3" if lhs_continuity.big_tensor_op == "+3" else "add7"
            parts.append(f"view_{lhs_continuity.non_contiguous_dim}_{op_str}")
        
        # 3. 右操作数信息
        rhs_shape_type = "dynamic" if rhs_is_dynamic_shape else "fixed"
        parts.append(f"rhs_{rhs_shape_type}")
        
        # 右操作数向量维度
        if rhs_vector_rank == 0:
            parts.append("scalar")
        else:
            parts.append(f"{rhs_vector_rank}D_vector")
        
        # 右操作数连续性 - contiguous改成view，non_contiguous改成raw_tensor
        if rhs_continuity.is_contiguous:
            parts.append("raw_tensor")
        else:
            op_str = "mul2" if rhs_continuity.big_tensor_op == "*2" else "add3" if rhs_continuity.big_tensor_op == "+3" else "add7"
            parts.append(f"view_dim{rhs_continuity.non_contiguous_dim}_{op_str}")
        
        # 4. 广播信息 - 重新设计命名避免与元素类型的scalar/vector混淆
        # 检测广播类型，使用更清晰的命名
        if lhs_dims_spec == rhs_dims_spec:
            broadcast_info = "no_broadcast"
        elif lhs_dims_spec == [1]:
            broadcast_info = "lhs_singleton_broadcast"  # [1] 表示单元素广播
        elif rhs_dims_spec == [1]:
            broadcast_info = "rhs_singleton_broadcast"  # [1] 表示单元素广播
        elif len(lhs_dims_spec) == 1 and len(rhs_dims_spec) > 1:
            broadcast_info = "lhs_1d_broadcast"  # 左操作数是一维张量广播
        elif len(rhs_dims_spec) == 1 and len(lhs_dims_spec) > 1:
            broadcast_info = "rhs_1d_broadcast"  # 右操作数是一维张量广播
        else:
            broadcast_info = "multi_broadcast"  # 多维广播
            
        parts.append(broadcast_info)
        
        return "_".join(parts)

    def get_binary_output_shape(self, lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                lhs_shape, rhs_shape):
        output_is_dynamic_shape = lhs_is_dynamic_shape or rhs_is_dynamic_shape

        if len(lhs_shape) < len(rhs_shape):
            shorter_shape, longer_shape = lhs_shape, rhs_shape
        else:
            shorter_shape, longer_shape = rhs_shape, lhs_shape

        # Prepend 1s to the shorter shape to match the rank of the longer shape for broadcasting.
        rank_diff = len(longer_shape) - len(shorter_shape)
        padded_shorter_shape = [1] * rank_diff + shorter_shape
        
        # Check for broadcasting compatibility.
        for dim1, dim2 in zip(longer_shape, padded_shorter_shape):
            assert dim1 == dim2 or min(dim1, dim2) == 1, \
                f"Shapes {lhs_shape} and {rhs_shape} are not broadcast-compatible"
        
        # The output shape is the element-wise maximum of the two shapes.
        output_shape = [max(dim1, dim2) for dim1, dim2 in zip(longer_shape, padded_shorter_shape)]
        
        return output_is_dynamic_shape, output_shape


    def get_op_call_lines(self, ntt_op_str):
        """Generate NTT binary operation code"""
        return [
            "// Execute binary operation",
            f"ntt::binary<ntt::ops::{ntt_op_str}>(ntt_input_lhs, ntt_input_rhs, ntt_output);",
            ""
        ]

    def generate_ort_output(self, datatype, ntt_op_str):
        ort_type = self.ort_datatype_map.get(datatype.cpp_type, 'DataType_FLOAT')
        return [
            "// Execute binary operation",
            f"{self.op_str_map[ntt_op_str]}",
            ""
        ]

    def _prepare_contiguous_input(self, input_name, datatype, vector_rank, pack_param, 
                                  is_dynamic_shape, dims_spec, continuity):
        
        continuity_var_name = input_name
        element_type = self.get_element_cpp_type(datatype.cpp_type, vector_rank, pack_param)
        code = []
        
        if not continuity.is_contiguous:
            continuity_var_name = f"{input_name}_contiguous"
            copy_code, _ = self.generate_copy_to_contiguous_code(
                element_type,
                is_dynamic_shape,
                dims_spec,
                input_name,
                continuity_var_name
            )
            continuity_var_name = f"*{continuity_var_name}"
            code.extend(copy_code)
        
        return continuity_var_name, code

    def generate_ort_golden_output(self, datatype, 
                                    lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                    lhs_dims_spec, rhs_dims_spec,
                                    lhs_vector_rank, rhs_vector_rank,
                                    lhs_continuity, rhs_continuity,
                                    lhs_pack_param, rhs_pack_param,
                                    ntt_op_str, output_shape_expr):
        code = []
        
        # Check if datatype needs to be cast to float32
        need_cast = datatype.cpp_type in self.types_need_to_be_cast
            
        lhs_continuity_var_name, lhs_copy_code = self._prepare_contiguous_input(
            "ntt_input_lhs", datatype, lhs_vector_rank, lhs_pack_param,
            lhs_is_dynamic_shape, lhs_dims_spec, lhs_continuity
        )
        code.extend(lhs_copy_code)
        ort_input_lhs = lhs_continuity_var_name

        rhs_continuity_var_name, rhs_copy_code = self._prepare_contiguous_input(
            "ntt_input_rhs", datatype, rhs_vector_rank, rhs_pack_param,
            rhs_is_dynamic_shape, rhs_dims_spec, rhs_continuity
        )
        code.extend(rhs_copy_code)
        ort_input_rhs = rhs_continuity_var_name

        if need_cast:
            # Cast inputs to float32 before sending to ort
            code.append("// Cast inputs to float32 for ORT computation")
            
            # Lambda function to cast input to float32
            cast_to_float = lambda side, input_var, vector_rank, pack_param, is_dynamic, dims_spec: (
                code.append(f"auto ntt_{side}_float = ntt::make_tensor<{self.get_element_cpp_type('float', vector_rank, pack_param)}>({self.generate_shape_init(is_dynamic, dims_spec)});"),
                code.append(f"ntt::cast({input_var}, ntt_{side}_float);")
            )
            
            # Cast both inputs
            cast_to_float("lhs", ort_input_lhs, lhs_vector_rank, lhs_pack_param, lhs_is_dynamic_shape, lhs_dims_spec)
            cast_to_float("rhs", ort_input_rhs, rhs_vector_rank, rhs_pack_param, rhs_is_dynamic_shape, rhs_dims_spec)
            
            # Update variable references
            ort_input_lhs = "ntt_lhs_float"
            ort_input_rhs = "ntt_rhs_float"
            
            code.append("")

        code.extend([f"auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort({ort_input_lhs}, {ort_input_rhs});"])
        code.extend(self.generate_ort_output(datatype, ntt_op_str))

        return code

    def generate_ntt_output_to_test(self, datatype,
                                    lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                    lhs_dims_spec, rhs_dims_spec,
                                    lhs_vector_rank, rhs_vector_rank,
                                    lhs_continuity, rhs_continuity,
                                    lhs_pack_param, rhs_pack_param,
                                    ntt_op_str):
        indent = "    "
        code = []
        # generate ntt_input_lhs, ntt_input_rhs, ntt_output
        code.append(f"{indent}//---init ntt_input_lhs---")
        tensor_init_lhs_code = self.generate_tensor_init( datatype=datatype,
            shape_type=lhs_is_dynamic_shape, dim_spec=lhs_dims_spec,
            continuity=lhs_continuity, var_name="ntt_input_lhs",
            name_suffix="_lhs", vector_rank=lhs_vector_rank,
            P=lhs_pack_param)
        code.extend([f"{indent}{line}" for line in tensor_init_lhs_code])

        code.append(f"{indent}//---init ntt_input_rhs---")
        tensor_init_rhs_code = self.generate_tensor_init( datatype=datatype,
            shape_type=rhs_is_dynamic_shape, dim_spec=rhs_dims_spec,
            continuity=rhs_continuity, var_name="ntt_input_rhs",
            name_suffix="_rhs", vector_rank=rhs_vector_rank,
            P=rhs_pack_param)
        code.extend([f"{indent}{line}" for line in tensor_init_rhs_code])

        output_is_dynamic_shape, output_dims_spec = self.get_binary_output_shape(
            lhs_is_dynamic_shape, rhs_is_dynamic_shape,
            lhs_dims_spec, rhs_dims_spec)
        output_vector_rank = max(lhs_vector_rank, rhs_vector_rank)
        code.append(f"{indent}//---generate output tensor---")

        output_shape_expr = self.generate_shape_init(output_is_dynamic_shape, output_dims_spec)
        # For binary ops, output vector rank matches inputs. Assume lhs.
        output_pack_param = lhs_pack_param if lhs_pack_param else rhs_pack_param
        output_element_type = self.get_element_cpp_type(datatype.cpp_type, output_vector_rank, output_pack_param)

        output_op_call_lines = self.get_op_call_lines(ntt_op_str)
        ntt_output_and_op_code = self.generate_ntt_output_and_op_section(
            datatype=datatype,
            output_shape_expr=output_shape_expr,
            cast_mode=0,  # Placeholder for now
            ntt_op_call_lines=output_op_call_lines,
            output_var_name="ntt_output",
            output_element_type=output_element_type
        )
        code.extend([f"{indent}{line}" for line in ntt_output_and_op_code])
        return code, output_shape_expr, output_element_type




    # lhs_dynamic: bool, lhs is dynamic or fixed
    # rhs_dynamic: bool, rhs is dynamic or fixed
    # lhs_shape: list[int], lhs shape, [1, 77, 3]
    # rhs_shape: list[int], rhs shape, [1, 77, 3]
    # braodcast_ways: list[int], broadcast ways, 0: no_broadcast 1: lhs_to_rhs, 2: rhs_to_lhs, 
    # lhs_vector_ranks: list[int], lhs vector ranks, 0, 1, 2
    # rhs_vector_ranks: list[int], rhs vector ranks, 0, 1, 2, 3
    # lhs_tensor: list[int], lhs is tensor or view, 0: tensor, 1: view
    # rhs_tensor: list[int], rhs is tensor or view
    def generate_test_case(
            self,
            datatype,
            lhs_is_dynamic_shape: bool,
            rhs_is_dynamic_shape: bool,
            lhs_dims_spec: List[int],
            rhs_dims_spec: List[int],
            lhs_vector_rank: int,
            rhs_vector_rank: int,
            lhs_continuity: Continuity,
            rhs_continuity: Continuity,
            ntt_op_str):
        
        self.ntt_op_str = ntt_op_str  # Store operation type for is_div_operation check
        

        test_name = self.generate_test_name(datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape, 
            lhs_dims_spec, rhs_dims_spec, 
            lhs_vector_rank, rhs_vector_rank, 
            lhs_continuity, rhs_continuity)


        P = f"NTT_VLEN / (sizeof({datatype.cpp_type}) * 8)"
        code: List[str] = []
        lhs_pack_param = P if lhs_vector_rank > 0 else None
        rhs_pack_param = P if rhs_vector_rank > 0 else None

        # 1. Test header and constants
        code.extend(self.generate_function_name(f"BinaryTest{ntt_op_str}", datatype, test_name))
        code.extend(self.generate_min_max_constants(datatype))
        if lhs_vector_rank > 0 or rhs_vector_rank > 0:
            code.extend(self.generate_P_constants(P))

        # # Generate output to test in ntt format
        ntt_output_code, output_shape_expr, output_element_type = self.generate_ntt_output_to_test(datatype,
                            lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                            lhs_dims_spec, rhs_dims_spec,
                            lhs_vector_rank, rhs_vector_rank,
                            lhs_continuity, rhs_continuity,
                            lhs_pack_param, rhs_pack_param,
                            ntt_op_str)
        code.extend(ntt_output_code)


        # Generate golden output in ort format
        golden_output_code = self.generate_ort_golden_output(datatype,lhs_is_dynamic_shape, rhs_is_dynamic_shape,
            lhs_dims_spec, rhs_dims_spec,
            lhs_vector_rank, rhs_vector_rank,
            lhs_continuity, rhs_continuity,
            lhs_pack_param, rhs_pack_param,
            ntt_op_str, output_shape_expr)
        code.extend([f"    {line}" for line in golden_output_code])
        cast_mode = 2 if datatype.cpp_type in self.types_need_to_be_cast else 0
        # Compare outputs
        compare_code = self.generate_ort_back2ntt_and_compare_section(
            datatype,
            output_element_type,
            output_shape_expr,
            cast_mode=cast_mode,
            ntt_output_var_name="ntt_output",
            ort_output_var_name="ort_output")
        code.extend([f"    {line}" for line in compare_code])

        return "\n".join(code)

    def generate_all_tests_for_type(self, datatype, op_str):
        code = []
        
        # Define combinations for test cases
        is_dynamic_options = [False, True]
        is_view_options = [False, True]
        vector_rank_options = [0, 1, 2]  # 0: tensor, 1: 1d vector, etc. Keep it simple for now

        simple_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="+3"),
        ]

        dims_specs_options = [
                # No broadcast
                ([2, 3, 16, 16], [2, 3, 16, 16]),
                # Scalar broadcast
                ([1], [2, 3, 16, 16]),
                ([2, 3, 16, 16], [1]),
                # Vector broadcast
                ([16], [2, 3, 16, 16]),
                ([2, 3, 16, 16], [16]),
                # Multidirectional broadcast
                ([2, 1, 16, 1], [1, 3, 1, 16]),
            ]

        code.append(self.generate_header())

        param_combinations = itertools.product(
            is_dynamic_options,          # lhs_is_dynamic_shape 2
            is_dynamic_options,          # rhs_is_dynamic_shape 2
            dims_specs_options,   # (lhs_dims_spec, rhs_dims_spec) 6
            vector_rank_options,         # lhs_vector_rank 3
            vector_rank_options,         # rhs_vector_rank 3
            simple_continuities,         # lhs_continuity
            simple_continuities          # rhs_continuity
        )
        # 2*2*6*3*3*2*2*2*2/4 = 3456/4 = 864
        for lhs_is_dynamic, rhs_is_dynamic, (lhs_shape, rhs_shape), lhs_vec_rank, rhs_vec_rank, lhs_continuity, rhs_continuity in param_combinations:
            # Skip invalid combinations if any in the future
            # e.g. if lhs_shape == rhs_shape and ...
            if not lhs_continuity.is_contiguous and (lhs_shape == [1]):
                continue
            if rhs_shape == [1] and not rhs_continuity.is_contiguous:
                continue

            # set non_contiguous_dim for 1 dimension tensor
            if not lhs_continuity.is_contiguous and lhs_shape == [16]:
                lhs_continuity = lhs_continuity._replace(non_contiguous_dim=0)
            if not rhs_continuity.is_contiguous and rhs_shape == [16]:
                rhs_continuity = rhs_continuity._replace(non_contiguous_dim=0)

            test_code = self.generate_test_case(
                datatype,
                lhs_is_dynamic_shape=lhs_is_dynamic,
                rhs_is_dynamic_shape=rhs_is_dynamic,
                lhs_dims_spec=lhs_shape,
                rhs_dims_spec=rhs_shape,
                lhs_vector_rank=lhs_vec_rank,
                rhs_vector_rank=rhs_vec_rank,
                lhs_continuity=lhs_continuity,
                rhs_continuity=rhs_continuity,
                ntt_op_str=op_str
            )
            code.append(test_code)

        code.append(self.generate_footer())
        return "\n".join(code)

def generate_tests_for_op(op_str, generator):
    for datatype in ALL_DATATYPES:
        if datatype.cpp_type == "bool":
            continue
        test_code = generator.generate_all_tests_for_type(datatype, op_str)
        filename = f"test_ntt_binary_{datatype.name_suffix.lower()}_{op_str}_generated.cpp"
        output_filepath = os.path.join(generated_directory, filename)

        with open(output_filepath, "w") as f:
            f.write(test_code)
        
        print(f"Test file generated: {output_filepath}")
        generated_filenames.append(filename)
    

if __name__ == "__main__":
    generator = BinaryTestGenerator()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (ctest) and then the generated subdirectory
    ctest_directory = os.path.dirname(script_directory)
    generated_directory = os.path.join(ctest_directory, "generated")
    
    # Ensure generated directory exists
    os.makedirs(generated_directory, exist_ok=True)
    
    generated_filenames = []  # collect all generated file names

    # for datatype in ALL_DATATYPES:
    #     test_code = generator.generate_all_tests_for_type(datatype)
    #     filename = f"test_ntt_binary_{datatype.name_suffix.lower()}_generated.cpp"
    #     output_filepath = os.path.join(generated_directory, filename)

    #     with open(output_filepath, "w") as f:
    #         f.write(test_code)
        
    #     print(f"Test file generated: {output_filepath}")
    #     generated_filenames.append(filename)
    
    for op_str in generator.op_str_map.keys():
        generate_tests_for_op(op_str, generator)
    # Generate cmake list file in the generated directory
    generate_cmake_list(generated_directory, generated_filenames, "generated_binary_tests.cmake", "GENERATED_BINARY_TEST_SOURCES")