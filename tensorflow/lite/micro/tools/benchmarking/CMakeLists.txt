# =============================================================================
# Benchmarking application CMake configuration
# =============================================================================

cmake_minimum_required(VERSION 3.23)
project(tflm_benchmarking CXX)

# =============================================================================
# Include Configuration and Functions
# =============================================================================
include(CMakeConfig.cmake)

# =============================================================================
# Model Data Generation
# =============================================================================

# Clean and create generated sources directory
file(REMOVE_RECURSE ${GENERATED_SRCS_DIR})
file(MAKE_DIRECTORY ${GENERATED_SRCS_DIR})

# Generate model data if model path is provided
if(CONFIG_BENCHMARK_MODEL_PATH)
  set(BENCHMARK_MODEL_PATH ${CONFIG_BENCHMARK_MODEL_PATH})
  # Convert relative path to absolute path for scripts
  get_filename_component(BENCHMARK_MODEL_PATH_ABS ${BENCHMARK_MODEL_PATH}
                         ABSOLUTE)

  # Get model name for file generation
  get_filename_component(BENCHMARK_MODEL_NAME ${BENCHMARK_MODEL_PATH} NAME_WE)

  # Set defines for embedded model
  set(EXTRA_DEFINES
      "GENERIC_BENCHMARK_USING_BUILTIN_MODEL"
      "GENERIC_BENCHMARK_MODEL_HEADER_PATH=\"${GENERATED_SRCS_DIR}${BENCHMARK_MODEL_NAME}_model_data.h\""
      "GENERIC_BENCHMARK_MODEL_NAME=${BENCHMARK_MODEL_NAME}")

  # Generate model data files in source directory
  set(GENERATED_MODEL_DATA_CC
      ${GENERATED_SRCS_DIR}${BENCHMARK_MODEL_NAME}_model_data.cc)
  set(GENERATED_MODEL_DATA_H
      ${GENERATED_SRCS_DIR}${BENCHMARK_MODEL_NAME}_model_data.h)

  # Custom command to generate model data
  add_custom_command(
    OUTPUT ${GENERATED_MODEL_DATA_CC} ${GENERATED_MODEL_DATA_H}
    COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../generate_cc_arrays.py
            ${GENERATED_SRCS_DIR} ${BENCHMARK_MODEL_PATH_ABS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${BENCHMARK_MODEL_PATH}
    COMMENT "Generating model data from ${BENCHMARK_MODEL_PATH}")

  # Create custom target for model data generation
  add_custom_target(
    tflm_benchmark_generate_model_data
    DEPENDS ${GENERATED_MODEL_DATA_CC} ${GENERATED_MODEL_DATA_H}
    COMMENT "Target for generating model data for tflm_benchmark")

  # Add generated source to build
  set(MODEL_DATA_SRCS ${GENERATED_MODEL_DATA_CC})
  set(HAS_MODEL_DATA TRUE)
else()
  message("No BENCHMARK_MODEL_PATH specified. "
          "The benchmarking tool will be built without an embedded model.")
  set(MODEL_DATA_SRCS "")
  set(EXTRA_DEFINES "")
  set(HAS_MODEL_DATA FALSE)
endif()

# =============================================================================
# Meta Data Generation
# =============================================================================

# Handle model file path - use existing absolute path or set to empty
if(CONFIG_BENCHMARK_MODEL_PATH)
  set(MODEL_FILE_PATH ${BENCHMARK_MODEL_PATH_ABS})
else()
  set(MODEL_FILE_PATH "")
endif()

set(METADATA_COMPILE_FLAGS ${COMMON_FLAGS})
set(GENERATED_SHOW_META_DATA_CC ${GENERATED_SRCS_DIR}show_meta_data.cc)
# Add custom command to generate show_meta_data.cc
add_custom_command(
  OUTPUT ${GENERATED_SHOW_META_DATA_CC}
  COMMAND
    ${CMAKE_COMMAND} -E env
    TEMPLATE_FILE="${CMAKE_CURRENT_LIST_DIR}/show_meta_data.cc.template"
    GENERATED_FILE="${GENERATED_SHOW_META_DATA_CC}"
    TENSORFLOW_ROOT="${TFLM_ROOT}/" MODEL_FILE="${MODEL_FILE_PATH}"
    CC="${CMAKE_C_COMPILER}" CXX="${CMAKE_CXX_COMPILER}"
    CC_FLAGS="${METADATA_COMPILE_FLAGS}" CXX_FLAGS="${METADATA_COMPILE_FLAGS}"
    ${CMAKE_CURRENT_LIST_DIR}/collect_meta_data.sh
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Generating simplified show_meta_data.cc")

# Create custom target for meta data generation
add_custom_target(
  tflm_benchmark_generate_meta_data
  DEPENDS ${GENERATED_SHOW_META_DATA_CC}
  COMMENT "Target for generating show_meta_data.cc for tflm_benchmark")

# Combine all generated sources
set(GENERATED_SRCS ${MODEL_DATA_SRCS} ${GENERATED_SHOW_META_DATA_CC})

# =============================================================================
# Executable Target Definition
# =============================================================================

nuttx_add_application(
  NAME
  tflm_benchmark
  STACKSIZE
  ${CONFIG_TFLITEMICRO_BENCHMARK_TOOL_STACKSIZE}
  PRIORITY
  ${CONFIG_TFLITEMICRO_BENCHMARK_TOOL_PRIORITY}
  MODULE
  ${CONFIG_TFLITEMICRO_BENCHMARK_TOOL}
  SRCS
  ${BENCHMARKING_SRCS}
  ${GENERATED_SRCS}
  DEPENDS
  ${TFLM_LIBRARY}
  ${CMSIS_NN_LIBRARY})

# The nuttx_add_application creates a target named 'apps_tflm_benchmark'
set(TFLM_BENCHMARK_TARGET apps_tflm_benchmark)

# Set dependencies for generated files
if(HAS_MODEL_DATA)
  add_dependencies(${TFLM_BENCHMARK_TARGET} tflm_benchmark_generate_model_data
                   tflm_benchmark_generate_meta_data)
else()
  add_dependencies(${TFLM_BENCHMARK_TARGET} tflm_benchmark_generate_meta_data)
endif()

# Configure include directories
target_include_directories(
  ${TFLM_BENCHMARK_TARGET}
  PRIVATE ${BENCHMARKING_DIR} ${TFLM_ROOT}
          ${CMAKE_BINARY_DIR}/benchmarking/genfiles ${TFLM_INCLUDE_DIRS})

target_compile_options(${TFLM_BENCHMARK_TARGET}
                       PRIVATE ${METADATA_COMPILE_FLAGS})

# Set compile definitions
target_compile_definitions(${TFLM_BENCHMARK_TARGET} PRIVATE ${EXTRA_DEFINES})

# =============================================================================
# Target Configuration
# =============================================================================

# Handle arena size configuration
if(CONFIG_BENCHMARK_TENSOR_ARENA_SIZE)
  target_compile_definitions(
    ${TFLM_BENCHMARK_TARGET}
    PRIVATE
      "GENERIC_BENCHMARK_TENSOR_ARENA_SIZE=${CONFIG_BENCHMARK_TENSOR_ARENA_SIZE}"
  )
endif()

# Handle compression memory configuration
if(BENCHMARK_ALT_MEM_ATTR AND BENCHMARK_ALT_MEM_SIZE)
  target_compile_definitions(
    ${TFLM_BENCHMARK_TARGET}
    PRIVATE "GENERIC_BENCHMARK_ALT_MEM_ATTR=${BENCHMARK_ALT_MEM_ATTR}"
            "GENERIC_BENCHMARK_ALT_MEM_SIZE=${BENCHMARK_ALT_MEM_SIZE}")
elseif(BENCHMARK_ALT_MEM_ATTR OR BENCHMARK_ALT_MEM_SIZE)
  message(WARNING "Both BENCHMARK_ALT_MEM_ATTR and BENCHMARK_ALT_MEM_SIZE "
                  "must be specified for compression support. Ignoring.")
endif()
