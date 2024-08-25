/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

namespace {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

void ExtractShape(const TfLiteEvalTensor* input, int32_t* output_data) {
  for (int i = 0; i < input->dims->size; ++i) {
    output_data[i] = input->dims->data[i];
  }
}

TfLiteStatus ShapePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus ShapeEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  if (output->type != kTfLiteInt32) {
    MicroPrintf("Output type %s (%d) not supported.",
                TfLiteTypeGetName(output->type), output->type);
    return kTfLiteError;
  } else {
    ExtractShape(input, tflite::micro::GetTensorData<int32_t>(output));
  }

  return kTfLiteOk;
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus ShapeCompile(TfLiteContext* context, TfLiteNode* node,
                          TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include <cstring>" << std::endl;
      break;

    case kTfLiteCompileStepEval: {

      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, kInputTensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kOutputTensor);

      if (output->type != kTfLiteInt32) {
        ofs << "#error Unsupported output type" << std::endl;
        return kTfLiteError;
      } else {
        ofs << "{ // shape" << std::endl;

        tflite::micro::CompileAddress(ofs, "output_data", output->data.data);

        ofs << "const int32_t input_dims_size = " << input->dims->size
            << ";" << std::endl;

        int32_t *input_dims_data = const_cast<int32_t*>(input->dims->data);

        tflite::micro::CompileArray(ofs, "int32_t", "input_dims",
                                    input_dims_data,
                                    input->dims->size);

        ofs << "memcpy(reinterpret_cast<int32_t*>(output_data),"
            << "&input_dims, input_dims_size * sizeof(int32_t));" << std::endl;

        ofs << "}" << std::endl;
      }
    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif

}  // namespace

TFLMRegistration Register_SHAPE() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(nullptr, ShapePrepare, ShapeEval,
                                  ShapeCompile);
#else
  return tflite::micro::RegisterOp(nullptr, ShapePrepare, ShapeEval);
#endif
}

}  // namespace tflite
