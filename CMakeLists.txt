cmake_minimum_required(VERSION 3.16)
project(dstt
    VERSION 2.0.0
    DESCRIPTION "Dynamic Semi-Trained Topology — deterministic adaptive multimedia generation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# ── Library ──────────────────────────────────────────────────────────
set(DSTT_SOURCES
    src/utils/math.cpp
    src/utils/random.cpp
    src/core/partition.cpp
    src/core/cfm.cpp
    src/core/afm.cpp
    src/core/softmax.cpp
    src/core/arm.cpp
    src/ea/chromosome.cpp
    src/ea/operators.cpp
    src/ea/fitness.cpp
    src/ea/population.cpp
    src/fdmp/embeddings.cpp
    src/fdmp/fdmp.cpp
    src/fdmp/tokenizer.cpp
    src/mge/branch_predictor.cpp
    src/mge/synthesiser.cpp
    src/mge/mge.cpp
    src/training/trainer.cpp
    src/model/dstt_model.cpp
)

add_library(dstt_lib ${DSTT_SOURCES})
target_include_directories(dstt_lib PUBLIC include)

# ── Demo executable ──────────────────────────────────────────────────
add_executable(dstt_demo examples/demo.cpp)
target_link_libraries(dstt_demo PRIVATE dstt_lib)

# ── Interactive CLI ──────────────────────────────────────────────────
add_executable(dstt_cli examples/dstt_cli.cpp)
target_link_libraries(dstt_cli PRIVATE dstt_lib)

# ── Tests ────────────────────────────────────────────────────────────
enable_testing()

function(dstt_add_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE dstt_lib)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

dstt_add_test(test_partition   tests/test_partition.cpp)
dstt_add_test(test_cfm_afm     tests/test_cfm_afm.cpp)
dstt_add_test(test_ea          tests/test_ea.cpp)
dstt_add_test(test_integration tests/test_integration.cpp)
dstt_add_test(test_training    tests/test_training.cpp)
