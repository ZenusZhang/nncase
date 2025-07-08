#!/usr/bin/env python3
"""
Generate test cases for NTT cast operations
Covering the following cases:
1. Input/Output type combinations: all 15 * 14 type pairs
2. Shape types: fixed/dynamic
3. Vector dimensions: scalar/1D/2D
4. Tensor continuity: contiguous/non-contiguous
5. Tensor dimensions: 3D/4D
"""

import itertools
from typing import List, Tuple
from test_generator_base import *
import os

class CastTestGenerator(BaseTestGenerator):
    def __init__(self):
        super().__init__()
        

        
    def generate_test_name(self, from_type, to_type, shape_type, vector_dim, continuity: Continuity, ndim):
        parts = []
        parts.append(f"from_{from_type.name_suffix}_to_{to_type.name_suffix}")
        parts.append(shape_type)
        
        if vector_dim == 0:
            parts.append("scalar")
        else:
            parts.append(f"{vector_dim}D_vector")
        
        if continuity.is_contiguous:
            parts.append("contiguous")
        else:
            op_str = "mul2" if continuity.big_tensor_op == "*2" else "add5"
            parts.append(f"non_contiguous_dim{continuity.non_contiguous_dim}_{op_str}")

        parts.append(f"{ndim}D")
        return "_".join(parts)

    def generate_ort_output(self, to_type):
        """Generate ORT reference implementation for cast operation"""
        ort_type = self.ort_datatype_map.get(to_type.cpp_type, 'DataType_FLOAT')
        return [
            "// ORT reference implementation",
            f"auto ort_output = ortki_Cast(ort_input, 1, {ort_type});",
            ""
        ]

    def generate_ntt_ops(self):
        """Generate NTT cast operation code"""
        return [
            "// Execute cast operation",
            "ntt::cast(ntt_input, ntt_output1);",
            ""
        ]


    def generate_ntt_output_to_test(self, from_type, to_type, shape_type, dim_names, continuity, vector_dim, P, pack_axes):
        """Generate the NTT output to be tested"""
        code = []
        
        # 1. NTT input creation
        code.extend(self.generate_ntt_input_section(
            datatype=from_type,
            shape_type=shape_type,
            dims_spec=dim_names,
            continuity=continuity,
            vector_rank=vector_dim,
            P=P,
            var_name="ntt_input"))

        # 2. NTT output tensor creation
        output_element_type = self.get_element_cpp_type(to_type.cpp_type, vector_dim, P)
        output_shape_expr = self.generate_shape_init(shape_type, dim_names)

        code.append(f"// Create output tensor")
        code.append(f"auto ntt_output1 = ntt::make_tensor<{output_element_type}>({output_shape_expr});")
        code.append("")

        # 3. NTT operation (cast)
        cast_call_code = self.generate_ntt_ops()

        op_code = self.generate_ntt_operation_section(cast_call_code)
        code.extend(op_code)

        return code, output_shape_expr, output_element_type

    def generate_ntt_cast_golden_output_fp8(self, from_type, to_type, shape_type, dim_names, continuity, P, vector_dim):
        code = []
        tensor_element_type = self.get_element_cpp_type(from_type.cpp_type, vector_dim, P)
        output_element_type = self.get_element_cpp_type(to_type.cpp_type, vector_dim, P)

        # 1. copy to contiguous tensor of scalar or vector
        if not continuity.is_contiguous:
            copy_code, continuous_input_var_name = self.generate_copy_to_contiguous_code(tensor_element_type, shape_type, dim_names)
            code.extend(copy_code)
        else:
            continuous_input_var_name = "ntt_input"

        unpack_axes = [len(dim_names)-1] if vector_dim == 1 else [len(dim_names)-2, len(dim_names)-1]
        # 2. unpack to scalar tensor
        if 'vector' in tensor_element_type:
            unpacked_dims = self.get_unpacked_dims(dim_names, unpack_axes)
            code.append(f"auto ntt_scalar_input = ntt::make_tensor<{from_type.cpp_type}>({self.generate_shape_init(shape_type, unpacked_dims)});")
            code.append(f"ntt::unpack({continuous_input_var_name}, ntt_scalar_input, {self.generate_pack_axes_str(unpack_axes)});")
        else:
            code.append(f"auto ntt_scalar_input = {continuous_input_var_name};")
        #3. generate golden output
        code.append(f"auto ntt_golden_scalar = ntt::make_tensor<{to_type.cpp_type}>(ntt_scalar_input.shape());")
        code.append(
            f"ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){{\n"
            f"      (ntt_golden_scalar)(index) = static_cast<{to_type.cpp_type}>(ntt_scalar_input(index));\n"
            f"    }});"
        )

        # 4. generate under test scalar output 
        if "vector" in tensor_element_type:
            code.append(f"auto ntt_golden_vector = ntt::make_tensor<{output_element_type}>({self.generate_shape_init(shape_type, dim_names)});")
            code.append(f"ntt::pack(ntt_golden_scalar, ntt_golden_vector, {self.generate_pack_axes_str(unpack_axes)});")
            code.append(f"auto& ntt_golden = ntt_golden_vector;")
        else:
            code.append(f"auto& ntt_golden = ntt_golden_scalar;")


        return code


    def generate_ort_golden_output(self, from_type, to_type, shape_type, dim_names, continuity, P, pack_axes, vector_dim, deal_fp8):
        """Generate golden output using ORT or lambda-based reference"""
        code = []
        is_fp8_cast = 'float_e' in from_type.cpp_type or 'float_e' in to_type.cpp_type

        if not is_fp8_cast:
            # Generate ORT input section
            code.extend(self.generate_ort_input_section(
                datatype=from_type,
                shape_type=shape_type,
                dims_spec=dim_names,
                continuity=continuity,
                deal_fp8=deal_fp8,
                P=P,
                vector_rank=vector_dim,
                ntt_input_var_name="ntt_input"))
            
            # Use ORT output
            ort_kernel_lines = self.generate_ort_output(to_type)
            code.extend(self.generate_ort_operation_section(ort_kernel_lines))
        else:
            # Use lambda-based reference
            code.extend(self.generate_ntt_cast_golden_output_fp8(from_type, to_type, shape_type, dim_names, continuity, P, vector_dim))
            
        return code
    


    def generate_test_case(self, from_type, to_type, shape_type, vector_dim, continuity, ndim):
        """Generate a single test case"""
        # 1. Initialize dimensions and other basic variables
        is_from_fp8 = 'float_e' in from_type.cpp_type
        is_to_fp8 = 'float_e' in to_type.cpp_type
        deal_fp8 = 1 if (is_from_fp8 or is_to_fp8) else 0
        is_fp8_cast = is_from_fp8 or is_to_fp8

        P = f"NTT_VLEN / (sizeof({from_type.cpp_type}) * 8)"
        if ndim == 3:
            dims, dim_names = [1, 77, 3], ['C', 'H', 'W']
        elif ndim == 4:
            dims, dim_names = [2, 8, 4, 4], ['N', 'C', 'H', 'W']
        else:
            dims, dim_names = [2, 8, 4, 4, 2], ['N', 'C', 'H', 'W', 'D']

        # Determine unpack axes based on vector dimension, maybe used in fp8 golden
        if vector_dim == 0:
            pack_axes = []
        elif vector_dim == 1:
            pack_axes = [-2]  # Pack along first axis
        else:  # vector_dim == 2
            pack_axes = [-2, -1]  # Pack along first two axes

        test_name = self.generate_test_name(from_type, to_type, shape_type, vector_dim, continuity, ndim)
        
        code: List[str] = []


        # 1. Test header and constants
        code.extend(self.generate_function_name("CastTest", from_type, test_name))
        P_would_be_used = True if vector_dim > 0 else False
        code.extend(self.generate_demension_constants(dim_names, dims, from_type, P if P_would_be_used else None))
        code.extend(self.generate_min_max_constants(from_type))

        # 2. Generate output to test in NTT format
        ntt_output_code, output_shape_expr, output_element_type = self.generate_ntt_output_to_test(
            from_type, to_type, shape_type, dim_names, continuity, vector_dim, P, pack_axes)
        code.extend([f"    {line}" for line in ntt_output_code])

        # 3. Generate golden output in ORT format, or in ntt format for fp8 cast
        golden_output_code = self.generate_ort_golden_output(
            from_type, to_type, shape_type, dim_names, continuity, P, pack_axes, vector_dim, deal_fp8)
    
        code.extend([f"    {line}" for line in golden_output_code])

        # 4. Compare outputs
        if is_fp8_cast:
            # Direct comparison for FP8 cast
            code.extend([
                "    // Compare results",
                "    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));",
                "}"
            ])
        else:
            # ORT-based comparison
            compare_code = self.generate_ort_back2ntt_and_compare_section(
                to_type,
                output_element_type,
                output_shape_expr,
                deal_fp8,
                ntt_output_var_name="ntt_output1",
                ort_output_var_name="ort_output")
            code.extend([f"    {line}" for line in compare_code])

        return "\n".join(code)

    def generate_all_tests_for_from_type(self, from_type):
        """Generate all test combinations for a given input datatype"""
        shape_types = ["fixed", "dynamic"]
        vector_dims = [0, 1, 2]  # scalar, 1D vector, 2D vector
        
        # Full continuity test combinations, mainly for 4D
        full_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="+7"),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="+7"),
        ]

        # Simplified continuity test combinations, for non-4D
        simple_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
        ]
        
        code = []
        
        # Generate file header
        code.append(self.generate_header())
        
        # Generate test cases for all target types (except the same type)
        for to_type in ALL_DATATYPES:
            if from_type.cpp_type == to_type.cpp_type:
                continue  # Skip same type cast
            
            # Generate test cases for different dimensions
            for ndim in [3, 4]:
                # Select continuity test strategy based on dimension
                current_continuities = full_continuities if ndim == 3 else simple_continuities

                for shape_type, vector_dim, continuity in itertools.product(shape_types, vector_dims, current_continuities):
                    # Skip unreasonable combinations
                    if vector_dim > ndim:  # Can't have more vector dimensions than tensor dimensions
                        continue
                    test_code = self.generate_test_case(from_type, to_type, shape_type, vector_dim, continuity, ndim)
                    code.append(test_code)
                    
        # Generate main function
        code.append(self.generate_footer())
        
        return "\n".join(code)


if __name__ == "__main__":
    generator = CastTestGenerator()
    script_directory = os.path.dirname(os.path.abspath(__file__))   
    generated_filenames = []  # collect all generated file names

    for from_type in ALL_DATATYPES:
        test_code = generator.generate_all_tests_for_from_type(from_type)
        filename = f"test_ntt_cast_from_{from_type.name_suffix.lower()}_generated.cpp"
        output_filepath = os.path.join(script_directory, filename)

        with open(output_filepath, "w") as f:
            f.write(test_code)
        
        print(f"Test file generated: {output_filepath}")
        generated_filenames.append(filename)
    
    generate_cmake_list(script_directory, generated_filenames, "generated_cast_tests.cmake", "GENERATED_CAST_TEST_SOURCES") 