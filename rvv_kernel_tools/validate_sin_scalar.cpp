#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include "sin_scalar_functions.h"

class SinValidator {
private:
    std::vector<float> test_values;
    
public:
    SinValidator() {
        // Generate comprehensive test cases
        generateTestValues();
    }
    
    void generateTestValues() {
        // Special values
        test_values = {0.0f, M_PI/6, M_PI/4, M_PI/3, M_PI/2, 
                      2*M_PI/3, 3*M_PI/4, 5*M_PI/6, M_PI, 
                      7*M_PI/6, 5*M_PI/4, 4*M_PI/3, 3*M_PI/2,
                      5*M_PI/3, 7*M_PI/4, 11*M_PI/6, 2*M_PI};
        
        // Negative values
        std::vector<float> negatives;
        for (float val : test_values) {
            if (val != 0.0f) {
                negatives.push_back(-val);
            }
        }
        test_values.insert(test_values.end(), negatives.begin(), negatives.end());
        
        // Large values to test range reduction
        std::vector<float> large_vals = {10*M_PI, 100*M_PI, 1000*M_PI, 
                                        -10*M_PI, -100*M_PI, -1000*M_PI};
        test_values.insert(test_values.end(), large_vals.begin(), large_vals.end());
        
        // Small values near zero
        std::vector<float> small_vals = {1e-6f, 1e-7f, 1e-8f, -1e-6f, -1e-7f, -1e-8f};
        test_values.insert(test_values.end(), small_vals.begin(), small_vals.end());
        
        // Random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
        for (int i = 0; i < 100; i++) {
            test_values.push_back(dis(gen));
        }
    }
    
    void compareResults(float input, const SinDebugValues& f32_result, 
                       const SinDebugValues& f64_result) {
        float reference = std::sin(input);
        
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Input: " << input << std::endl;
        std::cout << "Reference sin(): " << reference << std::endl;
        std::cout << "F32 result: " << f32_result.final_result << std::endl;
        std::cout << "F64 result: " << f64_result.final_result << std::endl;
        
        float f32_error = std::abs((float)f32_result.final_result - reference);
        double f64_error = std::abs(f64_result.final_result - reference);
        
        std::cout << "F32 error: " << f32_error << std::endl;
        std::cout << "F64 error: " << f64_error << std::endl;
        
        // Analyze intermediate values for precision loss
        analyzeIntermediateValues(f32_result, f64_result);
        std::cout << "---" << std::endl;
    }
    
    void analyzeIntermediateValues(const SinDebugValues& f32, 
                                  const SinDebugValues& f64) {
        // Compare range reduction precision
        double range_reduction_diff = std::abs(f32.r_reduced - f64.r_reduced);
        if (range_reduction_diff > 1e-6) {
            std::cout << "WARNING: Significant range reduction difference: " 
                     << range_reduction_diff << std::endl;
        }
        
        // Compare argument transformation (r')
        double r_prime_diff = std::abs(f32.r_prime - f64.r_prime);
        if (r_prime_diff > 1e-6) {
            std::cout << "WARNING: Significant r' transformation difference: " 
                     << r_prime_diff << std::endl;
        }
        
        // Compare polynomial evaluation
        double poly_diff = std::abs(f32.poly_result - f64.poly_result);
        if (poly_diff > 1e-6) {
            std::cout << "WARNING: Significant polynomial difference: " 
                     << poly_diff << std::endl;
        }
        
        // Check for catastrophic cancellation in range reduction
        if (std::abs(f32.r_reduced) < 1e-15 && std::abs(f32.input_v) > 1.0) {
            std::cout << "WARNING: Possible catastrophic cancellation in range reduction" << std::endl;
        }
    }
    
    void runAllTests() {
        std::cout << "Running comprehensive sin validation tests..." << std::endl;
        
        int failed_tests = 0;
        int catastrophic_failures = 0;
        double max_f32_error = 0.0;
        double max_f64_error = 0.0;
        float worst_input = 0.0f;
        
        for (float test_val : test_values) {
            SinDebugValues f32_result = sin_scalar_f32_debug(test_val);
            SinDebugValues f64_result = sin_scalar_f64_debug(test_val);
            
            float reference = std::sin(test_val);
            float f32_error = std::abs((float)f32_result.final_result - reference);
            double f64_error = std::abs(f64_result.final_result - reference);
            
            if (f32_error > max_f32_error) {
                max_f32_error = f32_error;
                worst_input = test_val;
            }
            if (f64_error > max_f64_error) {
                max_f64_error = f64_error;
            }
            
            // Check for catastrophic failures (> 0.1 absolute error)
            if (f32_error > 0.1) {
                catastrophic_failures++;
                std::cout << "CATASTROPHIC FAILURE for input " << test_val 
                         << ", error: " << f32_error << std::endl;
            } else if (f32_error > 1e-5) {  // Significant error
                failed_tests++;
            }
        }
        
        std::cout << std::endl << "=== TEST SUMMARY ===" << std::endl;
        std::cout << "Total tests: " << test_values.size() << std::endl;
        std::cout << "Tests with significant error (>1e-5): " << failed_tests << std::endl;
        std::cout << "Catastrophic failures (>0.1): " << catastrophic_failures << std::endl;
        std::cout << "Maximum F32 error: " << max_f32_error << " (input: " << worst_input << ")" << std::endl;
        std::cout << "Maximum F64 error: " << max_f64_error << std::endl;
        
        if (catastrophic_failures > 0) {
            std::cout << std::endl << "Analyzing worst case..." << std::endl;
            detailedAnalysis(worst_input);
        }
    }
    
    void detailedAnalysis(float specific_input) {
        std::cout << "=== DETAILED ANALYSIS FOR INPUT: " << specific_input << " ===" << std::endl;
        
        SinDebugValues f32 = sin_scalar_f32_debug(specific_input);
        SinDebugValues f64 = sin_scalar_f64_debug(specific_input);
        
        std::cout << "Reference std::sin(): " << std::sin(specific_input) << std::endl << std::endl;
        
        // Print all intermediate values
        std::cout << "F32 Debug Values:" << std::endl;
        printDebugValues(f32);
        
        std::cout << std::endl << "F64 Debug Values:" << std::endl;
        printDebugValues(f64);
        
        std::cout << std::endl;
        compareResults(specific_input, f32, f64);
        
        // Analyze step-by-step precision loss
        analyzeStepByStepPrecision(f32, f64);
    }
    
    void printDebugValues(const SinDebugValues& dbg) {
        std::cout << std::scientific << std::setprecision(15);
        std::cout << "  input_v: " << dbg.input_v << std::endl;
        std::cout << "  r_abs: " << dbg.r_abs << std::endl;
        std::cout << "  n_unrounded: " << dbg.n_unrounded << std::endl;
        std::cout << "  ki: " << dbg.ki << std::endl;
        std::cout << "  n_rounded: " << dbg.n_rounded << std::endl;
        std::cout << "  sign_bits: 0x" << std::hex << dbg.sign_bits << std::dec << std::endl;
        std::cout << "  odd_bits: 0x" << std::hex << dbg.odd_bits << std::dec << std::endl;
        std::cout << "  final_sign_bits: 0x" << std::hex << dbg.final_sign_bits << std::dec << std::endl;
        std::cout << "  r_reduced: " << dbg.r_reduced << std::endl;
        std::cout << "  r_prime: " << dbg.r_prime << std::endl;
        std::cout << "  r2: " << dbg.r2 << std::endl;
        std::cout << "  poly_result: " << dbg.poly_result << std::endl;
        std::cout << "  final_result_before_sign: " << dbg.final_result_before_sign << std::endl;
        std::cout << "  final_result: " << dbg.final_result << std::endl;
    }
    
    void analyzeStepByStepPrecision(const SinDebugValues& f32, const SinDebugValues& f64) {
        std::cout << "=== STEP-BY-STEP PRECISION ANALYSIS ===" << std::endl;
        
        // Range reduction analysis
        double rr_diff = std::abs(f32.r_reduced - f64.r_reduced);
        std::cout << "Range reduction difference: " << rr_diff;
        if (rr_diff > 1e-6) {
            std::cout << " *** SIGNIFICANT";
            if (rr_diff > 1e-3) {
                std::cout << " *** CRITICAL";
            }
        }
        std::cout << std::endl;
        
        // Argument transformation analysis
        double rt_diff = std::abs(f32.r_prime - f64.r_prime);
        std::cout << "R' transformation difference: " << rt_diff;
        if (rt_diff > 1e-6) {
            std::cout << " *** SIGNIFICANT";
        }
        std::cout << std::endl;
        
        // Polynomial evaluation analysis
        double poly_diff = std::abs(f32.poly_result - f64.poly_result);
        std::cout << "Polynomial evaluation difference: " << poly_diff;
        if (poly_diff > 1e-6) {
            std::cout << " *** SIGNIFICANT";
        }
        std::cout << std::endl;
        
        // Final result analysis
        double final_diff = std::abs(f32.final_result - f64.final_result);
        std::cout << "Final result difference: " << final_diff;
        if (final_diff > 1e-6) {
            std::cout << " *** SIGNIFICANT";
        }
        std::cout << std::endl;
        
        // Identify primary source of error
        if (rr_diff > rt_diff && rr_diff > poly_diff) {
            std::cout << "Primary error source: RANGE REDUCTION" << std::endl;
        } else if (rt_diff > poly_diff) {
            std::cout << "Primary error source: ARGUMENT TRANSFORMATION" << std::endl;
        } else {
            std::cout << "Primary error source: POLYNOMIAL EVALUATION" << std::endl;
        }
    }
    
    void benchmarkModeComparison() {
        std::cout << "=== BENCHMARK MODE: COMPARING SPECIFIC VALUES ===" << std::endl;
        
        // Test cases known to be problematic for float precision
        std::vector<float> benchmark_values = {
            M_PI,           // Should be exactly 0
            M_PI/2,         // Should be exactly 1  
            3*M_PI/2,       // Should be exactly -1
            2*M_PI,         // Should be exactly 0
            1000*M_PI,      // Large multiple of PI
            M_PI + 1e-6f,   // Near PI but not exactly
            0.1f,           // Simple decimal
            1.0f,           // Unit value
            10.0f,          // Moderate value
            100.0f          // Large value
        };
        
        for (float val : benchmark_values) {
            std::cout << std::endl;
            detailedAnalysis(val);
        }
    }
};

int main(int argc, char* argv[]) {
    SinValidator validator;
    
    if (argc > 1) {
        std::string mode = argv[1];
        
        if (mode == "detailed" && argc > 2) {
            // Detailed analysis for specific input
            float input = std::stof(argv[2]);
            validator.detailedAnalysis(input);
        } else if (mode == "benchmark") {
            // Benchmark mode
            validator.benchmarkModeComparison();
        } else {
            std::cout << "Usage:" << std::endl;
            std::cout << "  " << argv[0] << "                    # Run all tests" << std::endl;
            std::cout << "  " << argv[0] << " detailed <value>   # Detailed analysis for specific value" << std::endl;
            std::cout << "  " << argv[0] << " benchmark          # Compare specific benchmark values" << std::endl;
            return 1;
        }
    } else {
        // Run all tests
        validator.runAllTests();
    }
    
    return 0;
}