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

#include "tensorflow/lite/kernels/internal/reference/dequantize.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/dequantize.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

void* DequantizeInit(TfLiteContext* context, const char* buffer,
                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(DequantizeOpData));
}

TfLiteStatus DequantizeEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  DequantizeOpData* data = static_cast<DequantizeOpData*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  // Output type ensured to be kTfLiteFloat32 at the Prepare stage
  TFLITE_DCHECK(output->type == kTfLiteFloat32);

  switch (input->type) {
    case kTfLiteInt8: {
#if HIFI_VFPU && (defined(HIFI5) || defined(HIFI4) || defined(HIFI3))
      int err;
      const int8_t* input_data_ptr;
      float* output_data_ptr;
      const int flat_size =
          MatchingFlatSize(tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorShape(output));
      input_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
      output_data_ptr = tflite::micro::GetTensorData<float>(output);

      err = xa_nn_elm_dequantize_asym8s_f32(
          output_data_ptr, input_data_ptr, data->quantization_params.zero_point,
          data->quantization_params.scale, flat_size);
      TF_LITE_ENSURE(context, (err == 0));
#else   //  HIFI_VFPU && (defined(HIFI5) || defined(HIFI4) || defined(HIFI3))
      reference_ops::Dequantize(data->quantization_params,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<int8_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
#endif  //  HIFI_VFPU && (defined(HIFI5) || defined(HIFI4) || defined(HIFI3))
      break;
    }
    case kTfLiteInt16: {
#if HIFI_VFPU && (defined(HIFI5) || defined(HIFI4) || defined(HIFI3))
      int err;
      const int16_t* input_data_ptr;
      float* output_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);
      input_data_ptr = tflite::micro::GetTensorData<int16_t>(input);
      output_data_ptr = tflite::micro::GetTensorData<float>(output);
      err = xa_nn_elm_dequantize_asym16s_f32(
          output_data_ptr, input_data_ptr, data->quantization_params.zero_point,
          data->quantization_params.scale, flat_size);
      TF_LITE_ENSURE(context, (err == 0));
#else   // HIFI_VFPU && (defined(HIFI5) || defined(HIFI4) || defined(HIFI3))
      reference_ops::Dequantize(data->quantization_params,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<int16_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
#endif  // HIFI_VFPU && (defined(HIFI5) || defined(HIFI4) || defined(HIFI3))
      break;
    }
    case kTfLiteUInt8:
      reference_ops::Dequantize(data->quantization_params,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<uint8_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      break;
    default:
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus DequantizeCompile(TfLiteContext* context, TfLiteNode* node,
                               TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/micro/kernels/xtensa/xtensa.h\""
          << std::endl
          << "#include \"tensorflow/lite/micro/kernels/dequantize.h\""
          << std::endl
          << "#include \"tensorflow/lite/kernels/internal/reference/dequantize.h\""
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->user_data != nullptr);
      const DequantizeOpData* data =
          static_cast<const DequantizeOpData*>(node->user_data);

      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, 0);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, 0);

      // Only support quantized input types to Float32 output
      if (output->type != kTfLiteFloat32) {
        ofs << "// Output type " << TfLiteTypeGetName(output->type)
            << " not supported for Xtensa compilation" << std::endl;
        return kTfLiteError;
      }

      ofs << "{ // xtensa dequantize" << std::endl;

      // 1. Generate runtime data addresses
      tflite::micro::CompileAddress(ofs, "input_data", input->data.data);
      tflite::micro::CompileAddress(ofs, "output_data", output->data.data);

      // 2. Generate dequantization parameters
      const int32_t inp_zero_point = data->quantization_params.zero_point;
      const float inp_scale = data->quantization_params.scale;

      ofs << "const int32_t inp_zero_point = " << inp_zero_point << ";"
          << std::endl
          << "const float inp_scale = " << inp_scale << "f;" << std::endl;

      // 3. Generate flat size
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      ofs << "const int flat_size = " << flat_size << ";" << std::endl
          << std::endl;

      // 4. Generate dequantization code for int8 only (HIFI4 VFPU optimized)
      // Only support int8 to float32 dequantization for HIFI4
      if (input->type != kTfLiteInt8) {
        ofs << "// Unsupported input type: " << TfLiteTypeGetName(input->type)
            << " (only int8 supported)" << std::endl;
        return kTfLiteError;
      }

      // Generate direct call to HIFI4 VFPU nnlib function (no conditional compilation)
      ofs << "// Int8 to Float32 dequantization" << std::endl
          << "xa_nn_elm_dequantize_asym8s_f32(" << std::endl
          << "    reinterpret_cast<float*>(output_data)," << std::endl
          << "    reinterpret_cast<const int8_t*>(input_data)," << std::endl
          << "    inp_zero_point, inp_scale, flat_size);" << std::endl;

      ofs << "}" << std::endl;

    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif  // TFLITE_MODEL_COMPILER

TFLMRegistration Register_DEQUANTIZE() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(DequantizeInit, DequantizePrepare,
                                  DequantizeEval, DequantizeCompile);
#else
  return tflite::micro::RegisterOp(DequantizeInit, DequantizePrepare,
                                   DequantizeEval);
#endif
}

// Type-specific registration function for INT8 dequantization
TFLMRegistration Register_DEQUANTIZE_INT8() {
  return Register_DEQUANTIZE();
}

}  // namespace tflite
