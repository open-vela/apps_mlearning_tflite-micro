# =============================================================================
# Benchmarking Configuration Variables
# =============================================================================

# Allow the root project to pass in the TensorFlow Lite Micro source directory
# and the pre-built tflite_micro library via cache variables. Fallback to
# reasonable defaults when they are not provided.
set(TFLM_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../../tflite-micro"
    CACHE PATH "Path to the TensorFlow Lite Micro checkout")
set(TFLM_LIBRARY
    tflite_micro
    CACHE STRING "TFLite Micro library target to link")
set(CMSIS_NN_LIBRARY
    cmsis-nn
    CACHE STRING "CMSIS-NN library target to link")
set(TFLM_INCLUDE_DIRS
    ""
    CACHE STRING "Additional include directories required for TFLM")

# Option to specify model path for embedded models
set(BENCHMARK_MODEL_PATH
    ""
    CACHE PATH "Path to .tflite model file for embedded model build")

# Option to specify custom arena size
set(BENCHMARK_TENSOR_ARENA_SIZE
    ""
    CACHE STRING "Custom tensor arena size for embedded models")

# Option to specify alternate memory attributes/size for compression
set(BENCHMARK_ALT_MEM_ATTR
    ""
    CACHE STRING "Alternate memory attribute for compression support")
set(BENCHMARK_ALT_MEM_SIZE
    ""
    CACHE STRING "Alternate memory size for compression support")

# Directory setup
set(BENCHMARKING_DIR ${CMAKE_CURRENT_LIST_DIR})
set(GENERATED_SRCS_DIR ${CMAKE_CURRENT_LIST_DIR}/genfiles/)

# Source file definitions
set(BENCHMARKING_SRCS ${BENCHMARKING_DIR}/generic_model_benchmark.cc
                      ${BENCHMARKING_DIR}/metrics.cc)
