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
#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

TfLiteStatus StridedSliceEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const StridedSliceParams& op_params =
      *(static_cast<const StridedSliceParams*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kStridedSliceInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kStridedSliceOutputTensor);
  switch (output->type) {
    case kTfLiteFloat32:
      reference_ops::StridedSlice(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<float>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::StridedSlice(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<int8_t>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::StridedSlice(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
      break;
    case kTfLiteInt32:
      reference_ops::StridedSlice(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int32_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int32_t>(output));
      break;
    case kTfLiteBool:
      reference_ops::StridedSlice(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<bool>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<bool>(output));
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus StridedSliceCompile(TfLiteContext* context, TfLiteNode* node,
                                 TfLiteCompileStep step, std::ofstream& ofs) {

  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/kernels/internal/reference/"
          <<  "strided_slice.h\"" << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->user_data != nullptr);
      const StridedSliceParams& op_params =
          *(static_cast<const StridedSliceParams*>(node->user_data));

      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, kStridedSliceInputTensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kStridedSliceOutputTensor);

      ofs << "{ // transpose" << std::endl;

      tflite::micro::CompileAddress(
          ofs, "input", tflite::micro::GetTensorData<int8_t>(input));
      tflite::micro::CompileAddress(
          ofs, "output", tflite::micro::GetTensorData<int8_t>(output));

      ofs << "const tflite::StridedSliceParams& op_params = {"
          << ".start_indices_count="
          << std::to_string(op_params.start_indices_count)
          << ",.start_indices={";
      for (int i = 0; i < 5; ++i)
        ofs << op_params.start_indices[i] << ",";
      ofs << "},.stop_indices_count="
          << std::to_string(op_params.stop_indices_count)
          << ",.stop_indices={";
      for (int i = 0; i < 5; i++)
        ofs << op_params.stop_indices[i] << ",";
      ofs << "},.strides_count="
          << std::to_string(op_params.strides_count)
          << ",.strides={";
      for (int i = 0; i < 5; i++)
        ofs << op_params.strides[i] << ",";
      ofs << "},.begin_mask=" << op_params.begin_mask
          << ",.ellipsis_mask=" << op_params.ellipsis_mask
          << ",.end_mask=" << op_params.end_mask
          << ",.new_axis_mask=" << op_params.new_axis_mask
          << ",.shrink_axis_mask=" << op_params.shrink_axis_mask
          << ",.offset=" << op_params.offset
          << "};" <<std::endl;

      tflite::micro::CompileArray(ofs, "const int32_t", "input_dims_data",
                          input->dims->data, input->dims->size);
      tflite::micro::CompileArray(ofs, "const int32_t", "output_dims_data",
                          output->dims->data, output->dims->size);

      const char* type_str;
      switch (output->type) {
        case kTfLiteFloat32:
          type_str = "float*";
          break;
        case kTfLiteInt8:
          type_str = "int8_t*";
          break;
        case kTfLiteInt16:
          type_str = "int16_t*";
          break;
        case kTfLiteInt32:
          type_str = "int32_t*";
          break;
        case kTfLiteBool:
          type_str = "bool*";
          break;
        default:
          MicroPrintf("Type %s (%d) not supported.",
                      TfLiteTypeGetName(input->type), input->type);
          return kTfLiteError;
      }

      ofs << "tflite::reference_ops::StridedSlice(op_params, "
          << "tflite::RuntimeShape(" << input->dims->size
          << ", input_dims_data), "
          << "reinterpret_cast<" << type_str << ">(input), "
          << "tflite::RuntimeShape(" << output->dims->size
          << ", output_dims_data), "
          << "reinterpret_cast<" << type_str << ">(output));"
          << std::endl;

      ofs << "}" << std::endl;

    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif

}  // namespace

TFLMRegistration Register_STRIDED_SLICE() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(StridedSliceInit, StridedSlicePrepare,
                                   StridedSliceEval, StridedSliceCompile);
#else
  return tflite::micro::RegisterOp(StridedSliceInit, StridedSlicePrepare,
                                   StridedSliceEval);
#endif
}

}  // namespace tflite
