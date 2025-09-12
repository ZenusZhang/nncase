CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
PYTHON = python3

# Target executable
TARGET = validate_sin_scalar

# Framework source
FRAMEWORK_SRC = kernel_test_framework.cpp

# Kernel binding (override for other kernels like exp)
KERNEL_HEADER ?= sin_scalar_functions.h
KERNEL_F32 ?= sin_scalar_f32_debug
KERNEL_F64 ?= sin_scalar_f64_debug
KERNEL_STRUCT ?= SinDebugValues
KERNEL_SIN_FIELDS ?= 1

# Generated header file
GENERATED_HEADER = sin_scalar_functions.h

# Python script
CONVERTER_SCRIPT = convert_rvv_to_scalar.py

# Configuration file
CONFIG_FILE = rvv_conversion_config.json

# Input macro file
INPUT_MACRO = input_macro.txt

# Test framework config (for golden selection and ranges)
TEST_CONFIG = kernel_test_config.json

# Default target
all: $(TARGET)

# Generate the header file using the Python script (only when explicitly requested via `make regen`)
$(GENERATED_HEADER):
	@echo "Header is pre-generated. Run 'make regen' to regenerate from macros."

# Build the validation/framework program
$(TARGET): $(FRAMEWORK_SRC)
	@echo "Compiling kernel test framework..."
	$(CXX) $(CXXFLAGS) \
	  -DKERNEL_HEADER='"$(KERNEL_HEADER)"' \
	  -DKERNEL_F32_DEBUG_FUNC=$(KERNEL_F32) \
	  -DKERNEL_F64_DEBUG_FUNC=$(KERNEL_F64) \
	  -DKERNEL_DEBUG_STRUCT=$(KERNEL_STRUCT) \
	  -DKERNEL_HAS_SIN_FIELDS=$(KERNEL_SIN_FIELDS) \
	  -o $@ $(FRAMEWORK_SRC)

# Run comprehensive tests
test: $(TARGET)
	@echo "Running comprehensive sin validation tests..."
	./$(TARGET) --config $(TEST_CONFIG)

# Run detailed analysis for a specific value (e.g., make detailed VALUE=3.14159)
detailed: $(TARGET)
	@echo "Running detailed analysis for value: $(VALUE)"
	./$(TARGET) --config $(TEST_CONFIG) detailed $(VALUE)

# Run benchmark mode with specific test cases
benchmark: $(TARGET)
	@echo "Running benchmark comparison..."
	./$(TARGET) --config $(TEST_CONFIG) benchmark

# Test specific problematic values
test-problematic: $(TARGET)
	@echo "Testing problematic values..."
	@echo "=== Testing π ==="
	./$(TARGET) --config $(TEST_CONFIG) detailed 3.14159265359
	@echo "=== Testing π/2 ==="
	./$(TARGET) --config $(TEST_CONFIG) detailed 1.57079632679
	@echo "=== Testing large value (1000π) ==="
	./$(TARGET) --config $(TEST_CONFIG) detailed 3141.59265359

# Clean generated files
clean:
	rm -f $(TARGET)

# Force regeneration of header
regen:
	rm -f $(GENERATED_HEADER)
	@echo "Generating scalar functions from RISC-V macro..."
	$(PYTHON) $(CONVERTER_SCRIPT) -i $(INPUT_MACRO) -o $(GENERATED_HEADER)

# Show help
help:
	@echo "Available targets:"
	@echo "  all              - Build everything (default)"
	@echo "  test             - Run comprehensive validation tests (uses $(TEST_CONFIG))"  
	@echo "  detailed VALUE=x - Run detailed analysis for specific value x"
	@echo "  benchmark        - Run benchmark comparison with known values"
	@echo "  test-problematic - Test known problematic values"
	@echo "  clean            - Remove generated files"
	@echo "  regen            - Force regeneration of header file"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test                             # uses $(TEST_CONFIG)"
	@echo "  make detailed VALUE=3.14159           # uses $(TEST_CONFIG)"
	@echo "  make benchmark                        # uses $(TEST_CONFIG)"
	@echo "  make test-problematic"
	@echo ""
	@echo "Override kernel binding (example for exp):"
	@echo "  make KERNEL_HEADER=exp_scalar_functions.h KERNEL_F32=exp_scalar_f32_debug \\\n+	        KERNEL_F64=exp_scalar_f64_debug KERNEL_STRUCT=ExpDebugValues KERNEL_SIN_FIELDS=0"

# Mark phony targets
.PHONY: all test detailed benchmark test-problematic clean regen help

# Dependencies
kernel_test_framework.cpp: sin_scalar_functions.h
