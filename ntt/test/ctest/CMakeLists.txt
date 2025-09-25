enable_testing()

# How to run a single test case:
# 1. Build only the specific test executable (using test_ntt_pack_generated_Uint32 as an example):
# cmake --build build/<BuildType> --target test_ntt_pack_generated_Uint32
# where <BuildType> is typically Debug, Release, etc.
# 2. Execute the corresponding test in the generated test directory (use the same <BuildType>):
# ctest -C <BuildType> --test-dir build/<BuildType>/ntt/test/ctest -R test_ntt_pack_generated_Uint32
# -C specifies the build configuration, and -R filters the test by regex.
find_package(ortki)
find_package(GTest REQUIRED)
find_package(Python3 REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

# add_definitions(-DDE_BUG)

# --- Generate test source files ---
# Define kernel names for automatic test generation. Add more kernels here.
# Limit coverage runs to handwritten tests only to keep CI runtime minimal
set(KERNEL_NAMES )
# set(KERNEL_NAMES binary pack unpack)
# set(KERNEL_NAMES  pack unpack)
# set(KERNEL_NAMES binary)
# set(KERNEL_NAMES unary)

# Set directories
set(TEST_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_generator)
set(GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)

# Create generated directory if it doesn't exist
file(MAKE_DIRECTORY ${GENERATED_DIR})

# Macro to run a generator script only when the output is missing or outdated
macro(run_generator_if_needed script_path output_file)
    if(NOT EXISTS ${output_file} OR ${script_path} IS_NEWER_THAN ${output_file})
        message(STATUS "Running ${script_path} to generate test files.")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${script_path}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE return_code
        )

        if(NOT return_code EQUAL 0)
            message(FATAL_ERROR "${script_path} failed to run. Error code: ${return_code}")
        endif()
    else()
        message(STATUS "Skipping script execution, ${output_file} is up-to-date.")
    endif()
endmacro()


# Automatically process all kernel generators
foreach(kernel ${KERNEL_NAMES})
    string(TOUPPER ${kernel} KERNEL_UPPER)

    set(GENERATOR_SCRIPT ${TEST_GENERATOR_DIR}/generate_${kernel}_tests.py)
    set(GENERATED_CMAKE ${GENERATED_DIR}/generated_${kernel}_tests.cmake)

    if(EXISTS ${GENERATOR_SCRIPT})
        run_generator_if_needed(${GENERATOR_SCRIPT} ${GENERATED_CMAKE})

        if(EXISTS ${GENERATED_CMAKE})
            include(${GENERATED_CMAKE})
            list(APPEND GENERATED_TEST_SOURCES ${GENERATED_${KERNEL_UPPER}_TEST_SOURCES})
        endif()
    else()
        message(WARNING "Generator script for ${kernel} not found: ${GENERATOR_SCRIPT}")
    endif()
endforeach()

macro(add_test_exec test_source_file)
    get_filename_component(tname ${test_source_file} NAME_WE)
    add_executable(${tname} ${test_source_file})
    target_compile_options(${tname} PRIVATE )
    target_link_libraries(${tname} PRIVATE GTest::gtest_main nncaseruntime ortki::ortki)
    add_test(NAME ${tname} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:${tname}> -P ${NNCASE_CMAKE_DIR}/run_test.cmake)

endmacro()

set(HANDWRITTEN_TESTS
    test_ntt_playground.cpp
)

# Combine handwritten and generated tests
list(APPEND TEST_NAMES ${HANDWRITTEN_TESTS})
# list(APPEND TEST_NAMES ${HANDWRITTEN_TESTS} )
# list(APPEND TEST_NAMES ${GENERATED_TEST_SOURCES} )


foreach(test_file ${TEST_NAMES})
    add_test_exec(${test_file})
endforeach()
