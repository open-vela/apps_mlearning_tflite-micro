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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AveragePoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
      AveragePoolingEvalQuantized<int8_t>(context, node, params, data, input,
                                          output);
      break;
    case kTfLiteInt16:
      AveragePoolingEvalQuantized<int16_t>(context, node, params, data, input,
                                           output);
      break;
    default:
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      MaxPoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
      MaxPoolingEvalQuantized<int8_t>(context, node, params, data, input,
                                      output);
      break;
    case kTfLiteInt16:
      MaxPoolingEvalQuantized<int16_t>(context, node, params, data, input,
                                       output);
      break;
    default:
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus MaxCompile(TfLiteContext* context, TfLiteNode* node,
                        TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h\""
          << std::endl
          << "#include \"tensorflow/lite/kernels/internal/types.h\""
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->builtin_data != nullptr);
      auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

      TFLITE_DCHECK(node->user_data != nullptr);
      const OpDataPooling* data =
          static_cast<const OpDataPooling*>(node->user_data);

      const TfLiteEvalTensor* input =
          micro::GetEvalInput(context, node, kPoolingInputTensor);
      TfLiteEvalTensor* output =
         micro::GetEvalOutput(context, node, kPoolingOutputTensor);

      ofs << "{ // maxpool" << std::endl;

      tflite::micro::CompileAddress(ofs, "input_data", input->data.data);
      tflite::micro::CompileAddress(ofs, "output_data", output->data.data);
      tflite::micro::CompileArray(ofs, "const int32_t", "input_dims_data",
                                  input->dims->data, input->dims->size);
      tflite::micro::CompileArray(ofs, "const int32_t", "output_dims_data",
                                  output->dims->data, output->dims->size);

      ofs << "tflite::PoolParams params;"
          << "params.stride_width = " << params->stride_width << ";"
          << "params.stride_height = " << params->stride_height << ";"
          << "params.filter_width = " << params->filter_width << ";"
          << "params.filter_height = " << params->filter_height << ";"
          << "params.padding_values.width = " << data->padding.width << ";"
          << "params.padding_values.height = " << data->padding.height << ";"
          << "params.quantized_activation_min = " << data->activation_min << ";"
          << "params.quantized_activation_max = " << data->activation_max << ";"
          << std::endl;

      switch (input->type) {
        case kTfLiteInt8:
          ofs << "tflite::reference_integer_ops::MaxPool(params,"
              << "tflite::RuntimeShape("<< input->dims->size << ",input_dims_data),"
              << "(int8_t*)input_data,"
              << "tflite::RuntimeShape("<< output->dims->size << ",output_dims_data),"
              << "(int8_t*)output_data);"
              << std::endl;
          break;

        default:
          ofs << "Type " << TfLiteTypeGetName(input->type)
              << " not currently supported." << std::endl;
          return kTfLiteError;
      }

      ofs << "}" << std::endl;
    } break;

    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}
#endif // TFLITE_MODEL_COMPILER

void* PoolInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataPooling));
}
}  // namespace

TFLMRegistration Register_AVERAGE_POOL_2D() {
  return tflite::micro::RegisterOp(PoolInit, PoolingPrepare, AverageEval);
}

TFLMRegistration Register_MAX_POOL_2D() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(PoolInit, PoolingPrepare, MaxEval, MaxCompile);
#else
  return tflite::micro::RegisterOp(PoolInit, PoolingPrepare, MaxEval);
#endif
}

}  // namespace tflite
