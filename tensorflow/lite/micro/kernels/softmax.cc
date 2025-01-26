/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void SoftmaxQuantized(const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                      const SoftmaxParams& op_data) {
  if (input->type == kTfLiteInt8) {
    if (output->type == kTfLiteInt16) {
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
    } else {
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
    }
  } else {
    tflite::reference_ops::SoftmaxInt16(
        op_data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
  }
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8:
    case kTfLiteInt16: {
      SoftmaxQuantized(input, output, op_data);
      return kTfLiteOk;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus SoftmaxCompile(TfLiteContext* context, TfLiteNode* node,
                            TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/kernels/internal/reference/softmax.h\""
          << std::endl
          << "#include \"tensorflow/lite/kernels/internal/types.h\""
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
      TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

      TFLITE_DCHECK(node->user_data != nullptr);
      SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);

      ofs << "{ // softmax" << std::endl;

      tflite::micro::CompileAddress(ofs, "input_data", input->data.data);
      tflite::micro::CompileAddress(ofs, "output_data", output->data.data);
      tflite::micro::CompileArray(ofs, "const int32_t", "input_dims_data",
                                  input->dims->data, input->dims->size);
      tflite::micro::CompileArray(ofs, "const int32_t", "output_dims_data",
                                  output->dims->data, output->dims->size);

      ofs << "tflite::SoftmaxParams params;"
          << "params.beta=" << op_data.beta << ";"
          << "params.input_multiplier=" << op_data.input_multiplier << ";"
          << "params.input_left_shift=" << op_data.input_left_shift << ";"
          << "params.reverse_scaling_divisor=" << op_data.reverse_scaling_divisor << ";"
          << "params.reverse_scaling_right_shift=" << op_data.reverse_scaling_right_shift << ";"
          << "params.diff_min=" << op_data.diff_min << ";"
          << "params.zero_point=" << op_data.zero_point << ";"
          << "params.scale=" << op_data.scale << ";"
          << std::endl;

      switch(input->type) {
        case kTfLiteInt8:
          ofs << "tflite::reference_ops::Softmax(params,"
              << "tflite::RuntimeShape("<< input->dims->size << ",input_dims_data),"
              << "(int8_t*)input_data,"
              << "tflite::RuntimeShape("<< output->dims->size << ",output_dims_data),"
              << "(int8_t*)output_data);"
              << std::endl;
          break;
        default:
          ofs << "Input type " << TfLiteTypeGetName(input->type)
              << " not supported." << std::endl;
          return kTfLiteError;
      }

      ofs << "}" << std::endl;
      break;
    }

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif
}  // namespace

TFLMRegistration Register_SOFTMAX() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(SoftmaxInit, SoftmaxPrepare, SoftmaxEval,
                                  SoftmaxCompile);
#else
  return tflite::micro::RegisterOp(SoftmaxInit, SoftmaxPrepare, SoftmaxEval);
#endif
}

}  // namespace tflite
