/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/reshape.h"

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

TfLiteStatus EvalReshapeReference(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kReshapeInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kReshapeOutputTensor);

  // TODO(b/162522304): storing input bytes in OpData increases some models
  // significantly, possibly due to alignment issues.
  size_t input_bytes;
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(input->type, &input_bytes));
  input_bytes *= ElementCount(*input->dims);

  // Do nothing for in-place reshape.
  if (input->data.raw != output->data.raw) {
    // Otherwise perform reshape with copy.
    memcpy(output->data.raw, input->data.raw, input_bytes);
  }
  return kTfLiteOk;
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus ComplileReshapeReference(TfLiteContext* context, TfLiteNode* node,
                                      TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/micro/kernels/reshape.h\"" << std::endl
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, kReshapeInputTensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kReshapeOutputTensor);

      size_t input_bytes;
      TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(input->type, &input_bytes));
      input_bytes *= ElementCount(*input->dims);

      ofs << "{ // reshape" << std::endl;

      if (input->data.raw != output->data.raw) {

        micro::CompileAddress(ofs, "input", input->data.raw);
        micro::CompileAddress(ofs, "output", output->data.raw);

        ofs << "memcpy(output, input,"
            << input_bytes
            << ");" << std::endl;
      }

      ofs << "}" << std::endl;
    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif

}  // namespace

TFLMRegistration Register_RESHAPE() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(nullptr, PrepareReshapeReference,
                                  EvalReshapeReference, ComplileReshapeReference);
#else
  return tflite::micro::RegisterOp(nullptr, PrepareReshapeReference,
                                   EvalReshapeReference, nullptr, nullptr);
#endif
}

}  // namespace tflite
