// Generic kernel test framework using generated *_scalar_functions.h
// - Selects golden function via simple JSON config
// - Reuses debugable f32/f64 functions from the generated header

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Default kernel binding for "sin". Override via -DKERNEL_* macros when building other kernels.
#ifndef KERNEL_HEADER
#define KERNEL_HEADER "sin_scalar_functions.h"
#endif
#ifndef KERNEL_F32_DEBUG_FUNC
#define KERNEL_F32_DEBUG_FUNC sin_scalar_f32_debug
#endif
#ifndef KERNEL_F64_DEBUG_FUNC
#define KERNEL_F64_DEBUG_FUNC sin_scalar_f64_debug
#endif
#ifndef KERNEL_DEBUG_STRUCT
#define KERNEL_DEBUG_STRUCT SinDebugValues
#endif

// Enable detailed step printing for the sin debug struct by default
#ifndef KERNEL_HAS_SIN_FIELDS
#define KERNEL_HAS_SIN_FIELDS 1
#endif

// Include the generated header for the selected kernel
#include KERNEL_HEADER

// Minimal configuration holder (parsed from JSON)
struct Config {
    std::string golden = "sinf";   // one of: sinf, expf, cosf, tanhf
    int threshold_ulp = 2;          // print if ULP error > threshold
    double range_start = -4.0 * M_PI;
    double range_end = 4.0 * M_PI;
    double range_step = 0.01;
    bool include_specials = true;
    bool include_large_multiples = true;
};

static inline std::string read_file_or_empty(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static inline std::string strip_ws(const std::string &s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) if (!isspace(static_cast<unsigned char>(c))) r.push_back(c);
    return r;
}

// Very small JSON parser for a limited schema; tolerant and optional
static void parse_config_json(const std::string &json_raw, Config &cfg) {
    if (json_raw.empty()) return;
    const std::string j = strip_ws(json_raw);
    auto find_string = [&](const std::string &key, std::string &out) {
        std::string pat = std::string("\"") + key + "\":";
        size_t pos = j.find(pat);
        if (pos == std::string::npos) return;
        pos += pat.size();
        if (pos >= j.size() || j[pos] != '"') return;
        size_t end = j.find('"', pos + 1);
        if (end == std::string::npos) return;
        out = j.substr(pos + 1, end - (pos + 1));
    };
    auto find_number = [&](const std::string &key, double &out) {
        std::string pat = std::string("\"") + key + "\":";
        size_t pos = j.find(pat);
        if (pos == std::string::npos) return;
        pos += pat.size();
        char *endp = nullptr;
        out = std::strtod(j.c_str() + pos, &endp);
    };
    auto find_int = [&](const std::string &key, int &out) {
        double tmp = static_cast<double>(out);
        find_number(key, tmp);
        out = static_cast<int>(tmp);
    };
    auto find_bool = [&](const std::string &key, bool &out) {
        std::string pat = std::string("\"") + key + "\":";
        size_t pos = j.find(pat);
        if (pos == std::string::npos) return;
        pos += pat.size();
        if (j.compare(pos, 4, "true") == 0) out = true;
        else if (j.compare(pos, 5, "false") == 0) out = false;
    };

    find_string("golden", cfg.golden);
    find_int("threshold_ulp", cfg.threshold_ulp);
    find_number("range_start", cfg.range_start);
    find_number("range_end", cfg.range_end);
    find_number("range_step", cfg.range_step);
    find_bool("include_specials", cfg.include_specials);
    find_bool("include_large_multiples", cfg.include_large_multiples);
}

// ULP error for float
static inline int32_t ulp_error_f32(float a, float b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<int32_t>::max();
    int32_t ia, ib;
    std::memcpy(&ia, &a, sizeof(int32_t));
    std::memcpy(&ib, &b, sizeof(int32_t));
    return std::abs(ia - ib);
}

// Map golden name -> function pointers
struct Golden {
    std::function<float(float)> f32;
    std::function<double(double)> f64;
};

static Golden make_golden(const std::string &name) {
    Golden g{};
    if (name == "sinf") {
        g.f32 = [](float x) { return ::sinf(x); };
        g.f64 = [](double x) { return std::sin(x); };
    } else if (name == "cosf") {
        g.f32 = [](float x) { return ::cosf(x); };
        g.f64 = [](double x) { return std::cos(x); };
    } else if (name == "expf") {
        g.f32 = [](float x) { return ::expf(x); };
        g.f64 = [](double x) { return std::exp(x); };
    } else if (name == "tanf") {
        g.f32 = [](float x) { return ::tanf(x); };
        g.f64 = [](double x) { return std::tan(x); };
    } else {
        // Default to sin
        g.f32 = [](float x) { return ::sinf(x); };
        g.f64 = [](double x) { return std::sin(x); };
    }
    return g;
}

// Print a detailed comparison for SinDebugValues (guarded for sin kernel)
static void print_detailed_if_sin(float input,
                                  const KERNEL_DEBUG_STRUCT &f32,
                                  const KERNEL_DEBUG_STRUCT &f64,
                                  float golden_f32,
                                  double golden_f64,
                                  int32_t ulp_err) {
#if KERNEL_HAS_SIN_FIELDS
    std::cout << std::string(103, '-') << std::endl;
    std::cout << "Input: " << std::scientific << std::setprecision(8) << input << std::endl;
    std::cout << "F32 Impl Result: " << f32.final_result << std::endl;
    std::cout << "Golden (f32): " << golden_f32 << std::endl;
    std::cout << "Final ULP Error (f32 vs golden): " << ulp_err << std::endl;
    std::cout << std::string(103, '-') << std::endl;

    auto pr = [&](const std::string &name, double v32, double v64) {
        float v32f = static_cast<float>(v32);
        float v64f = static_cast<float>(v64);
        int32_t ulp = ulp_error_f32(v32f, v64f);
        std::cout << std::left << std::setw(28) << name
                  << std::scientific << std::setprecision(8)
                  << std::setw(20) << v32f
                  << std::setw(20) << v64f
                  << std::setw(20) << std::abs(v64 - v32)
                  << std::setw(15) << ulp << std::endl;
    };

    std::cout << std::left << std::setw(28) << "Step"
              << std::setw(20) << "f32 (Impl)"
              << std::setw(20) << "f64->f32 (Ref.)"
              << std::setw(20) << "Abs Error (f64)"
              << std::setw(15) << "ULP Error" << std::endl;
    std::cout << std::string(103, '-') << std::endl;

    pr("Input Value", f32.input_v, f64.input_v);
    pr("r = abs(v)", f32.r_abs, f64.r_abs);
    pr("n = r * (1/pi)", f32.n_unrounded, f64.n_unrounded);
    std::cout << std::left << std::setw(28) << "ki = round(n)"
              << std::setw(20) << f32.ki
              << std::setw(20) << f64.ki
              << std::setw(20) << std::abs(f64.ki - f32.ki)
              << std::setw(15) << "N/A" << std::endl;
    pr("n_rounded = float(ki)", f32.n_rounded, f64.n_rounded);
    pr("r_reduced = r - n*pi", f32.r_reduced, f64.r_reduced);
    pr("r_prime = pi/2 - r_reduced", f32.r_prime, f64.r_prime);
    pr("r2 = r_prime^2", f32.r2, f64.r2);
    std::cout << "--- Polynomial Evaluation ---" << std::endl;
    pr("y (+c12)", f32.y_c12, f64.y_c12);
    pr("y (+c10)", f32.y_c10, f64.y_c10);
    pr("y (+c8)", f32.y_c8, f64.y_c8);
    pr("y (+c6)", f32.y_c6, f64.y_c6);
    pr("y (+c4)", f32.y_c4, f64.y_c4);
    pr("y (+c2)", f32.y_c2, f64.y_c2);
    pr("poly_result (+c0)", f32.poly_result, f64.poly_result);
    std::cout << "--- Finalization ---" << std::endl;
    pr("Result (before sign)", f32.final_result_before_sign, f64.final_result_before_sign);
    pr("Final Result", f32.final_result, f64.final_result);
    pr("Golden f32", f32.final_result, golden_f32);
    pr("Golden f64", f32.final_result, golden_f64);
    std::cout << std::endl;
#else
    (void)input; (void)f32; (void)f64; (void)golden_f32; (void)golden_f64; (void)ulp_err;
#endif
}

static void generate_default_inputs(const Config &cfg, std::vector<float> &inputs) {
    // Range sweep
    for (double v = cfg.range_start; v <= cfg.range_end; v += cfg.range_step) {
        inputs.push_back(static_cast<float>(v));
    }
    if (cfg.include_specials) {
        float pi = static_cast<float>(M_PI);
        inputs.push_back(0.0f);
        inputs.push_back(-0.0f);
        inputs.push_back(pi / 2.0f);
        inputs.push_back(pi);
        inputs.push_back(3.0f * pi / 2.0f);
        inputs.push_back(2.0f * pi);
        inputs.push_back(pi - 1e-6f);
        inputs.push_back(pi + 1e-6f);
    }
    if (cfg.include_large_multiples) {
        float pi = static_cast<float>(M_PI);
        for (float m = 8.0f * pi; m < 100.0f * pi; m += pi / 3.0f) inputs.push_back(m);
    }
}

static void run_compare(const Config &cfg, const Golden &golden, float input) {
    auto f32 = KERNEL_F32_DEBUG_FUNC(input);
    auto f64 = KERNEL_F64_DEBUG_FUNC(static_cast<double>(input));
    float g32 = golden.f32(input);
    double g64 = golden.f64(static_cast<double>(input));
    int32_t final_ulp = ulp_error_f32(static_cast<float>(f32.final_result), g32);
    if (final_ulp > cfg.threshold_ulp) {
        print_detailed_if_sin(input, f32, f64, g32, g64, final_ulp);
    }
}

int main(int argc, char *argv[]) {
    // Parse optional --config <path>
    std::string config_path;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--config" && i + 1 < args.size()) {
            config_path = args[i + 1];
        }
    }

    Config cfg;
    if (!config_path.empty()) {
        parse_config_json(read_file_or_empty(config_path), cfg);
    }
    Golden golden = make_golden(cfg.golden);

    if (!args.empty() && args[0] == std::string("detailed") && args.size() >= 2) {
        float v = std::stof(args[1]);
        run_compare(cfg, golden, v);
        return 0;
    }
    if (!args.empty() && args[0] == std::string("benchmark")) {
        std::vector<float> bench = {
            static_cast<float>(M_PI), static_cast<float>(M_PI/2), static_cast<float>(3*M_PI/2), static_cast<float>(2*M_PI),
            1000.0f * static_cast<float>(M_PI), static_cast<float>(M_PI) + 1e-6f, 0.1f, 1.0f, 10.0f, 100.0f
        };
        for (float v : bench) run_compare(cfg, golden, v);
        return 0;
    }

    std::cout << "Starting kernel precision analysis..." << std::endl;
    std::cout << "Only inputs with ULP error > " << cfg.threshold_ulp
              << " versus golden (" << cfg.golden << ") are printed." << std::endl;

    std::vector<float> inputs;
    generate_default_inputs(cfg, inputs);
    for (float v : inputs) run_compare(cfg, golden, v);

    std::cout << "Analysis complete." << std::endl;
    return 0;
}
