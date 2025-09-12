#!/usr/bin/env python3
"""
RISC-V Vector to Scalar Conversion Tool

Converts RVV intrinsics in macro bodies to scalar C++ with debug hooks.
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ParsedOperation:
    intrinsic: str
    result: str
    operands: List[str]
    line_number: int
    original_line: str


class RVVConverter:
    def __init__(self, config_file: str = "rvv_conversion_config.json"):
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.operations: List[ParsedOperation] = []
        self.variables: Dict[str, str] = {}

    def is_scalar_constant(self, operand: str) -> bool:
        for pattern in self.config["operand_detection"]["scalar_constant_patterns"]:
            if re.match(pattern, operand.strip()):
                return True
        return False

    def parse_intrinsic_call(self, line: str) -> Optional[ParsedOperation]:
        clean_line = re.sub(r"/\*.*?\*/", "", line).strip()
        clean_line = clean_line.rstrip("\\").strip()
        if not clean_line or clean_line.startswith("//"):
            return None

        if any(skip in clean_line for skip in ["#define", "inline", "const", "size_t"]):
            return None

        def extract_paren_content(s: str, open_idx: int) -> Optional[str]:
            # s[open_idx] should be '('
            if open_idx < 0 or open_idx >= len(s) or s[open_idx] != '(':
                return None
            depth = 0
            for i in range(open_idx, len(s)):
                ch = s[i]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        return s[open_idx + 1:i]
            return None

        # First try to match an assignment form and capture up to the opening '('
        m_assign = re.search(r"(?:auto\s+|vfloat32m\w+_t\s+)?(\w+)\s*=\s*(__riscv_[\w#]+)\s*\(", clean_line)
        if m_assign:
            result = m_assign.group(1).strip()
            intrinsic = m_assign.group(2).strip()
            open_idx = m_assign.end() - 1  # points to '('
            extracted = extract_paren_content(clean_line, open_idx)
            if extracted is None:
                return None
            operands_str = extracted.strip()
        else:
            # Match a direct intrinsic call without assignment
            m_direct = re.search(r"(__riscv_[\w#]+)\s*\(", clean_line)
            if not m_direct:
                return None
            result = "temp_var"
            intrinsic = m_direct.group(1)
            open_idx = m_direct.end() - 1
            extracted = extract_paren_content(clean_line, open_idx)
            if extracted is None:
                return None
            operands_str = extracted.strip()

        # Remove RVV macro concatenations like '##lmul' or '##lmul##'
        intrinsic = re.sub(r"##\w+(?:##)?", "", intrinsic)
        operands = self.parse_operands(operands_str)
        return ParsedOperation(intrinsic=intrinsic, result=result, operands=operands, line_number=0, original_line=line)

    def parse_operands(self, operands_str: str) -> List[str]:
        ops: List[str] = []
        depth = 0
        cur = ""
        for ch in operands_str:
            if ch == "," and depth == 0:
                ops.append(cur.strip())
                cur = ""
            else:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                cur += ch
        if cur.strip():
            ops.append(cur.strip())
        return [op for op in ops if op.strip() != "vl"]

    def match_intrinsic_pattern(self, intrinsic: str, pattern: str) -> bool:
        regex_pattern = pattern.replace("*", r"\w*")
        return bool(re.match(f"^{regex_pattern}$", intrinsic))

    def get_intrinsic_mapping(self, intrinsic: str) -> Optional[Dict]:
        clean = re.sub(r"##\w+(?:##)?", "", intrinsic)
        if clean in self.config["intrinsic_mappings"]:
            return self.config["intrinsic_mappings"][clean]
        for pat, mapping in self.config["intrinsic_mappings"].items():
            if self.match_intrinsic_pattern(clean, pat):
                return mapping
        for pat, mapping in self.config["bit_operations"].items():
            if self.match_intrinsic_pattern(clean, pat):
                return mapping
        return None

    def convert_constant_to_double(self, constant: str) -> str:
        c = constant.strip()
        if c in self.config["constants"]["double_version"]:
            return self.config["constants"]["double_version"][c]
        if c.endswith("f") and not c.startswith("0x"):
            return c[:-1]
        return c

    def convert_operation_to_scalar(self, op: ParsedOperation, precision: str) -> str:
        mapping = self.get_intrinsic_mapping(op.intrinsic)
        # If this was a direct-return intrinsic (no LHS), avoid emitting
        # dangling code that references an undeclared temp variable.
        if op.result == "temp_var" and op.original_line.strip().startswith("return"):
            if not mapping:
                return f"// NOTE: return {op.intrinsic}(...) omitted (no mapping)"
            # Still omit actual code emission for returns; leave a breadcrumb.
            return f"// NOTE: return {op.intrinsic}(...) mapping omitted"
        if not mapping:
            return f"// WARNING: No mapping found for {op.intrinsic}"
        setup = ""
        if "setup" in mapping:
            setup = mapping["setup"].format(result=op.result) + "\n    "
        subs = {"result": op.result}
        for i, operand in enumerate(op.operands):
            key = f"operand{i+1}"
            if precision == "double" and self.is_scalar_constant(operand):
                subs[key] = self.convert_constant_to_double(operand)
            else:
                subs[key] = operand
        try:
            return setup + mapping["pattern"].format(**subs)
        except KeyError as e:
            missing = str(e).strip("'")
            return f"// WARNING: Mapping substitution failed for {op.intrinsic}: missing {missing}"

    def parse_macro_definition(self, content: str) -> List[ParsedOperation]:
        operations: List[ParsedOperation] = []
        content = content.replace("\\\n", " ")
        m = re.search(r"\{(.*?)\}", content, re.DOTALL)
        if not m:
            return operations
        body = m.group(1)
        statements: List[str] = []
        depth = 0
        cur = ""
        for ch in body:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ";" and depth == 0:
                if cur.strip():
                    statements.append(cur.strip())
                cur = ""
                continue
            cur += ch
        if cur.strip():
            statements.append(cur.strip())
        for i, stmt in enumerate(statements):
            op = self.parse_intrinsic_call(stmt)
            if op:
                op.line_number = i
                operations.append(op)
        return operations

    def generate_debug_struct(self) -> str:
        s = self.config["debug_struct"]
        # Open the struct and list fields; do not prematurely close it.
        code = f"struct {s['name']} {{\n"
        for f in s["fields"]:
            desc = f.get("description", "")
            code += f"    {f['type']} {f['name']}; // {desc}\n"
        code += "};\n\n"
        return code

    def generate_f32_function(self, ops: List[ParsedOperation]) -> str:
        code = "SinDebugValues sin_scalar_f32_debug(float v) {\n"
        code += "    SinDebugValues dbg = {};\n"
        code += "    dbg.input_v = v;\n\n"
        code += "    // Variable declarations\n"
        code += "    float r, n;\n"
        code += "    int32_t ki;\n"
        code += "    uint32_t sign, odd;\n"
        code += "    float r2, y;\n\n"
        code += "    // Converted operations\n"
        for op in ops:
            scalar_line = self.convert_operation_to_scalar(op, "float")
            code += f"    {scalar_line}; // {op.original_line.strip()}\n"
        code += "\n    return dbg;\n"
        code += "}\n\n"
        return code

    def generate_f64_function(self, ops: List[ParsedOperation]) -> str:
        code = "SinDebugValues sin_scalar_f64_debug(double v) {\n"
        code += "    SinDebugValues dbg = {};\n"
        code += "    dbg.input_v = v;\n\n"
        code += "    // Variable declarations\n"
        code += "    double r, n;\n"
        code += "    int32_t ki;\n"
        code += "    uint32_t sign, odd;\n"
        code += "    double r2, y;\n\n"
        code += "    // Converted operations\n"
        for op in ops:
            scalar_line = self.convert_operation_to_scalar(op, "double")
            code += f"    {scalar_line}; // {op.original_line.strip()}\n"
        code += "\n    return dbg;\n"
        code += "}\n\n"
        return code

    def generate_header(self, ops: List[ParsedOperation]) -> str:
        header = """#pragma once

#include <cmath>
#include <cstring>
#include <cstdint>

// Generated by convert_rvv_to_scalar.py
// Scalar versions of RISC-V vector sin implementation for precision analysis

"""
        header += self.generate_debug_struct()
        header += self.generate_f32_function(ops)
        header += self.generate_f64_function(ops)
        return header

    def convert_file(self, input_file: str, output_file: str):
        try:
            with open(input_file, "r") as f:
                content = f.read()
            ops = self.parse_macro_definition(content)
            header = self.generate_header(ops)
            with open(output_file, "w") as f:
                f.write(header)
            print(f"Generated {len(ops)} scalar operations in {output_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error processing file: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    import os
    parser = argparse.ArgumentParser(description="Convert RISC-V vector intrinsics to scalar C++ functions")
    parser.add_argument("-i", "--input", required=True, help="Input file containing RISC-V macro definition")
    parser.add_argument("-o", "--output", required=False, help="Output C++ header file (optional)")
    parser.add_argument("--config", default="rvv_conversion_config.json", help="Configuration file")
    args = parser.parse_args()

    # Derive a sensible default output if not provided
    out = args.output
    if not out:
        base = os.path.basename(args.input).lower()
        # Heuristic: name by known kernel type
        if "exp" in base:
            out_name = "exp_scalar_functions.h"
        else:
            # Default to sin when ambiguous (input_macro.txt case)
            out_name = "sin_scalar_functions.h"
        out = os.path.join(os.path.dirname(args.input) or ".", out_name)

    converter = RVVConverter(args.config)
    converter.convert_file(args.input, out)


if __name__ == "__main__":
    main()
