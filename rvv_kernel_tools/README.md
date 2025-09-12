# RVV Kernel Tools

A small toolkit to:

- Convert RISC-V Vector (RVV) intrinsics in macro-based kernels into scalar C++ for precision analysis.
- Run a reusable, configurable test framework that compares your scalarized kernel against a configurable golden implementation and prints step-by-step diagnostics when errors are significant.

This repo currently includes a scalarized float32 `sin` kernel with detailed debug fields and a framework to test it (and other kernels you generate similarly).

## Project Layout

- `convert_rvv_to_scalar.py` — Heuristic RVV → scalar code generator.
- `rvv_conversion_config.json` — Mapping rules for supported intrinsics, constants, and debug fields.
- `input_macro.txt` — Example RVV `sin` macro used by the generator.
- `sin_scalar_functions.h` — Pre-generated scalar debug implementation for `sin` (used by default).
- `kernel_test_framework.cpp` — Reusable test harness for scalar kernels.
- `kernel_test_config.json` — JSON config to choose golden function and input ranges.
- `Makefile` — Builds and runs the framework; provides a `regen` target to re-run the generator.
- `validate_sin_scalar.cpp` — Legacy validation program (kept for reference, not used by default).

## Requirements

- g++ with C++17 support
- make
- Python 3.8+

## Quick Start (sin)

1) Build the test framework:

```
cd rvv_kernel_tools
make
```

2) Run comprehensive tests (prints only cases exceeding the ULP threshold):

```
make test
```

3) Detailed analysis for a specific value:

```
make detailed VALUE=3.14159
```

4) Benchmark a curated set of values:

```
make benchmark
```

5) Regenerate the scalar header from the RVV macro (optional):

```
make regen
```

Note: `make` does not regenerate the header automatically; it uses the pre-generated `sin_scalar_functions.h`. Use `make regen` to re-run the generator.

## Test Framework

The executable is `validate_sin_scalar`. It loads `kernel_test_config.json` to select the golden function and drive input ranges. When the float32 implementation differs from the golden by more than `threshold_ulp`, the framework prints a step-by-step comparison using the debug fields exposed by your scalar functions.

### Run Modes

- Default: `make test` — sweeps a configured range and prints only failing cases.
- Detailed: `make detailed VALUE=x` — prints a full step-by-step comparison for a single input.
- Benchmark: `make benchmark` — compares a curated set of values prone to precision pitfalls.

### JSON Configuration (`kernel_test_config.json`)

- `golden`: one of `sinf`, `cosf`, `expf`, `tanf`. Controls the golden function used for comparison.
- `threshold_ulp`: integer ULP threshold for reporting.
- `range_start`, `range_end`, `range_step`: sweep parameters for `make test`.
- `include_specials`: include 0, ±π/2, ±π, ±3π/2, ±2π, near-π cases in the sweep.
- `include_large_multiples`: include larger multiples of π in the sweep.

Example:

```
{
  "golden": "sinf",
  "threshold_ulp": 2,
  "range_start": -12.566371,
  "range_end": 12.566371,
  "range_step": 0.01,
  "include_specials": true,
  "include_large_multiples": true
}
```

### Interpreting Output

For failing cases, the framework prints key intermediate values such as range-reduction results, transformed argument, polynomial evaluations, and final result before/after sign. This mirrors the fields in your `*_DebugValues` struct, enabling fast pinpointing of precision-loss sources.

## Switching to Other Kernels

The framework is kernel-agnostic. Point it at another generated header and debug functions via Makefile variables at build time:

- `KERNEL_HEADER` — header file to include (default: `sin_scalar_functions.h`).
- `KERNEL_F32` — float debug function (default: `sin_scalar_f32_debug`).
- `KERNEL_F64` — double debug function (default: `sin_scalar_f64_debug`).
- `KERNEL_STRUCT` — debug struct type (default: `SinDebugValues`).
- `KERNEL_SIN_FIELDS` — set to `0` to suppress sin-specific step printing.

Example (for an exp kernel you generated similarly):

```
make \
  KERNEL_HEADER=exp_scalar_functions.h \
  KERNEL_F32=exp_scalar_f32_debug \
  KERNEL_F64=exp_scalar_f64_debug \
  KERNEL_STRUCT=ExpDebugValues \
  KERNEL_SIN_FIELDS=0
```

Then set `"golden": "expf"` in `kernel_test_config.json`.

## RVV → Scalar Generator

`convert_rvv_to_scalar.py` parses a macro body and maps supported RVV intrinsics to scalar operations using `rvv_conversion_config.json`. It emits a header with:

- A debug struct (fields defined in the JSON).
- Two functions: a float32 version and a high-precision double reference.

Run it via:

```
make regen
```

Inputs: `input_macro.txt`, `rvv_conversion_config.json`
Output: `sin_scalar_functions.h`

### Current Limitations

- The parser is regex-based and best-effort; nested/complex intrinsics (e.g., reinterpret-calls inside operands) may not map automatically.
- Only a focused subset of intrinsics is covered (those used by the provided `sin` path). Extend `rvv_conversion_config.json` to add more.
- The generator does not currently emit all the handcrafted debug assignments present in the pre-generated header.

Recommendation: Use the pre-generated header for critical testing; regenerate iteratively as you expand mappings.

## Troubleshooting

- “header not found”: ensure `sin_scalar_functions.h` exists (run `make` or restore the header). `make clean` does not remove the header by default.
- “math symbols unresolved”: the framework uses `::sinf`, `::cosf`, `::expf`, `::tanf` for float. On some toolchains, you may need `-lm` when using `clang++` (g++ links libm by default).
- Regeneration warnings: if the generator prints `WARNING: No mapping found for ...`, add/extend patterns in `rvv_conversion_config.json`.

## License

This folder is part of the parent repository and follows its license.

