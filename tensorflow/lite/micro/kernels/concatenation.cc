/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/kernels/internal/reference/concatenation.h"

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

constexpr int kMaxInputNum = 64;  // Maximum number of input tensors
constexpr int kOutputTensor = 0;

struct OpData {
  ConcatenationParams params;
};

// Handles negative axis index, coerces to positive index value.
inline int CalculatePositiveAxis(int axis, const TfLiteTensor* output_tensor) {
  if (axis >= 0) {
    return axis;
  } else {
    return NumDimensions(output_tensor) + axis;
  }
}

// The following functions are helpers to get tensor data in the format that the
// reference op implementation expects. They provide the same functionality as
// class VectorOfTensors and class VectorOfQuantizedTensors in TFLite.

// Gets shapes from a list of tensors.
inline void GetAllInputTensorShapes(const TfLiteContext* context,
                                    const TfLiteNode* node,
                                    RuntimeShape all_shapes[kMaxInputNum]) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
    RuntimeShape shape = tflite::micro::GetTensorShape(t);
    all_shapes[i].ReplaceWith(shape.DimensionsCount(), shape.DimsData());
  }
}

// Get shape pointers from a list of shapes.
inline void GetShapesPointers(const RuntimeShape* shapes, size_t num,
                              const RuntimeShape* pointers[]) {
  for (size_t i = 0; i < num; ++i) {
    pointers[i] = &shapes[i];
  }
}

// Gets data pointers from a list of tensors.
template <typename T>
inline void GetAllInputTensorData(const TfLiteContext* context,
                                  const TfLiteNode* node,
                                  T* all_data[kMaxInputNum]) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
    all_data[i] = tflite::micro::GetTensorData<T>(t);
  }
}

template <typename data_type>
void EvalUnquantized(TfLiteContext* context, TfLiteNode* node) {
  // Collect the shapes and data pointer of input tensors
  RuntimeShape inputs_shape[kMaxInputNum];
  const RuntimeShape* inputs_shape_ptr[kMaxInputNum];
  const data_type* inputs_data[kMaxInputNum];
  GetAllInputTensorShapes(context, node, inputs_shape);
  GetShapesPointers(inputs_shape, node->inputs->size, inputs_shape_ptr);
  GetAllInputTensorData(context, node, inputs_data);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  reference_ops::Concatenation(data->params, inputs_shape_ptr, inputs_data,
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<data_type>(output));
}

void* ConcatenationInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus ConcatenationPrepare(TfLiteContext* context, TfLiteNode* node) {
  // This function only checks the types. Additional shape validations are
  // performed in the reference implementation called during Eval().
  const TfLiteConcatenationParams* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input_tensor = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input_tensor != nullptr);
  TfLiteType input_type = input_tensor->type;
  TfLiteTensor* output_tensor =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output_tensor != nullptr);
  TfLiteType output_type = output_tensor->type;

  micro_context->DeallocateTempTfLiteTensor(input_tensor);
  micro_context->DeallocateTempTfLiteTensor(output_tensor);

  // Check activation and input type
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteInt8 ||
                     input_type == kTfLiteInt16 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64 || input_type == kTfLiteBool);

  // Output type must match input type
  TF_LITE_ENSURE_EQ(context, output_type, input_type);

  // This implementation does not support large number of input tensors
  const int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs <= kMaxInputNum);

  // Shapes with dimensions >4 are not yet supported with static allocation.
  for (int i = 0; i < num_inputs; ++i) {
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, i);
    TF_LITE_ENSURE(context, input != nullptr);
    int num_dimensions = NumDimensions(input);

    if (num_dimensions > RuntimeShape::kMaxSmallSize) {
      MicroPrintf(
          "Op Concatenation does not currently support num dimensions > %d "
          "Tensor has %d dimensions.",
          RuntimeShape::kMaxSmallSize, num_dimensions);
      return kTfLiteError;
    }
    micro_context->DeallocateTempTfLiteTensor(input);
  }

  // Calculate OpData.
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  switch (output_type) {  // Already know in/outtypes are same.
    case kTfLiteBool:
    case kTfLiteFloat32:
    case kTfLiteInt16:
    case kTfLiteInt32:
    case kTfLiteInt64: {
      data->params.axis = CalculatePositiveAxis(params->axis, output);
      data->params.inputs_count = node->inputs->size;
      break;
    }
    case kTfLiteInt8: {
      data->params.axis = CalculatePositiveAxis(params->axis, output);
      data->params.inputs_count = node->inputs->size;

      float* input_scales =
          reinterpret_cast<float*>(context->AllocatePersistentBuffer(
              context, node->inputs->size * sizeof(float)));

      int32_t* input_zero_points =
          reinterpret_cast<int32_t*>(context->AllocatePersistentBuffer(
              context, node->inputs->size * sizeof(int32_t)));

      // Allocate persistent scale and zeropoint buffers.
      // Store input scale and zero point values in OpParams:
      for (int i = 0; i < node->inputs->size; ++i) {
        TfLiteTensor* t = micro_context->AllocateTempInputTensor(node, i);
        TF_LITE_ENSURE(context, t != nullptr);
        input_scales[i] = t->params.scale;
        input_zero_points[i] = t->params.zero_point;
        micro_context->DeallocateTempTfLiteTensor(t);
      }

      data->params.input_scale = input_scales;
      data->params.input_zeropoint = input_zero_points;
      data->params.output_zeropoint = output->params.zero_point;
      data->params.output_scale = output->params.scale;
      break;
    }
    default:
      MicroPrintf("Op Concatenation does not currently support Type '%s'.",
                  TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus ConcatenationEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* output_tensor =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output_tensor != nullptr);
  TfLiteType output_type = output_tensor->type;

  switch (output_type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      EvalUnquantized<float>(context, node);
      break;
    case kTfLiteInt32:
      EvalUnquantized<int32_t>(context, node);
      break;
    case kTfLiteInt8:
      EvalUnquantized<int8_t>(context, node);
      break;
    case kTfLiteInt64:
      EvalUnquantized<int64_t>(context, node);
      break;
    case kTfLiteInt16:
      EvalUnquantized<int16_t>(context, node);
      break;
    case kTfLiteBool:
      EvalUnquantized<bool>(context, node);
      break;

    default:
      MicroPrintf("Op Concatenation does not currently support Type '%s'.",
                  TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus ConcatenationCompile(TfLiteContext* context, TfLiteNode* node,
                                  TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/kernels/internal/reference/concatenation.h\""
          << std::endl
          << "#include <cstring>" << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      const TfLiteEvalTensor* output_tensor =
          tflite::micro::GetEvalOutput(context, node, kOutputTensor);
      TfLiteType output_type = output_tensor->type;

      // Only support int8 for now
      if (output_type != kTfLiteInt8) {
        ofs << "// Concatenation compile: only int8 is supported" << std::endl;
        return kTfLiteError;
      }

      TFLITE_DCHECK(node->user_data != nullptr);
      const OpData* data = static_cast<const OpData*>(node->user_data);

      const int num_inputs = node->inputs->size;

      ofs << "{ // concatenation int8" << std::endl;

      // Generate input data addresses
      // First, generate individual address variables for each input
      for (int i = 0; i < num_inputs; ++i) {
        const TfLiteEvalTensor* input =
            tflite::micro::GetEvalInput(context, node, i);
        std::string var_name = "input_data_" + std::to_string(i);
        tflite::micro::CompileAddress(
            ofs, var_name.c_str(),
            tflite::micro::GetTensorData<int8_t>(input));
      }

      // Then create the input_data array using these variables
      ofs << "static const int8_t* input_data[" << num_inputs << "] = {";
      for (int i = 0; i < num_inputs; ++i) {
        if (i > 0) ofs << ", ";
        ofs << "reinterpret_cast<const int8_t*>(input_data_" << i << ")";
      }
      ofs << "};" << std::endl;

      // Generate output data address
      tflite::micro::CompileAddress(
          ofs, "output_data",
          tflite::micro::GetTensorData<int8_t>(output_tensor));

      // Generate input shapes
      for (int i = 0; i < num_inputs; ++i) {
        const TfLiteEvalTensor* input =
            tflite::micro::GetEvalInput(context, node, i);
        const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
        ofs << "static const int32_t input_dims_" << i << "[] = {";
        for (int j = 0; j < input_shape.DimensionsCount(); ++j) {
          if (j > 0) ofs << ", ";
          ofs << input_shape.Dims(j);
        }
        ofs << "};" << std::endl;
      }

      // Generate RuntimeShape objects for inputs
      ofs << "tflite::RuntimeShape input_shapes[" << num_inputs << "];" << std::endl;
      for (int i = 0; i < num_inputs; ++i) {
        const TfLiteEvalTensor* input =
            tflite::micro::GetEvalInput(context, node, i);
        const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
        ofs << "input_shapes[" << i << "].ReplaceWith("
            << input_shape.DimensionsCount() << ", input_dims_" << i << ");"
            << std::endl;
      }

      // Generate shape pointer array
      ofs << "const tflite::RuntimeShape* input_shapes_ptr[" << num_inputs
          << "];" << std::endl;
      ofs << "for (int i = 0; i < " << num_inputs << "; ++i) {" << std::endl;
      ofs << "  input_shapes_ptr[i] = &input_shapes[i];" << std::endl;
      ofs << "}" << std::endl;

      // Generate output shape
      const RuntimeShape& output_shape =
          tflite::micro::GetTensorShape(output_tensor);
      ofs << "static const int32_t output_dims[] = {";
      for (int i = 0; i < output_shape.DimensionsCount(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << output_shape.Dims(i);
      }
      ofs << "};" << std::endl;

      // Generate ConcatenationParams
      ofs << "tflite::ConcatenationParams params;" << std::endl;
      ofs << "params.axis = " << static_cast<int>(data->params.axis) << ";" << std::endl;
      ofs << "params.inputs_count = " << data->params.inputs_count << ";"
          << std::endl;

      // Call reference concatenation
      ofs << "tflite::reference_ops::Concatenation(params," << std::endl;
      ofs << "    input_shapes_ptr," << std::endl;
      ofs << "    input_data," << std::endl;
      ofs << "    tflite::RuntimeShape(" << output_shape.DimensionsCount()
          << ", output_dims)," << std::endl;
      ofs << "    reinterpret_cast<int8_t*>(output_data));" << std::endl;

      ofs << "}" << std::endl;

    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif  // TFLITE_MODEL_COMPILER

TFLMRegistration Register_CONCATENATION() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(ConcatenationInit, ConcatenationPrepare,
                                  ConcatenationEval, ConcatenationCompile);
#else
  return tflite::micro::RegisterOp(ConcatenationInit, ConcatenationPrepare,
                                   ConcatenationEval);
#endif
}

}  // namespace tflite
