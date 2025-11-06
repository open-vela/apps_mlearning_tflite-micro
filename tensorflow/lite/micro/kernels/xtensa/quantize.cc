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

#include "tensorflow/lite/kernels/internal/reference/quantize.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
TfLiteStatus EvalXtensa(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteUInt8: {
      switch (output->type) {
        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          reference_ops::Requantize(
              tflite::micro::GetTensorData<uint8_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              op_data->quantization_params.zero_point,
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }

        default:
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
      }
      break;
    }

    case kTfLiteInt8: {
      switch (output->type) {
        case kTfLiteUInt8: {
          int size = ElementCount(*input->dims);
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int8_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              op_data->quantization_params.zero_point,
              tflite::micro::GetTensorData<uint8_t>(output));
          break;
        }

        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const int8_t* input_data_ptr;
          int8_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
          output_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_requantize_asym8s_asym8s(
                  output_data_ptr, input_data_ptr, op_data->input_zero_point,
                  zero_point, op_data->requantize_output_shift,
                  op_data->requantize_output_multiplier, size),
              0);
          break;
        }

        case kTfLiteInt16: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int8_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              zero_point, tflite::micro::GetTensorData<int16_t>(output));
          break;
        }

        case kTfLiteInt32: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const int8_t* input_data_ptr;
          int32_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
          output_data_ptr = tflite::micro::GetTensorData<int32_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_requantize_asym8s_asym32s(
                  output_data_ptr, input_data_ptr, op_data->input_zero_point,
                  zero_point, op_data->requantize_output_shift,
                  op_data->requantize_output_multiplier, size),
              0);
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt16: {
      switch (output->type) {
        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          TF_LITE_ENSURE_EQ(context,
                            xa_nn_elm_requantize_asym16s_asym8s(
                                tflite::micro::GetTensorData<int8_t>(output),
                                tflite::micro::GetTensorData<int16_t>(input),
                                op_data->input_zero_point,
                                op_data->quantization_params.zero_point,
                                op_data->requantize_output_shift,
                                op_data->requantize_output_multiplier, size),
                            0);
          break;
        }

        case kTfLiteInt16: {
          int size = ElementCount(*input->dims);
          TF_LITE_ENSURE_EQ(context,
                            xa_nn_elm_requantize_asym16s_asym16s(
                                tflite::micro::GetTensorData<int16_t>(output),
                                tflite::micro::GetTensorData<int16_t>(input),
                                op_data->input_zero_point,
                                op_data->quantization_params.zero_point,
                                op_data->requantize_output_shift,
                                op_data->requantize_output_multiplier, size),
                            0);
          break;
        }

        case kTfLiteInt32: {
          int size = ElementCount(*input->dims);
          TF_LITE_ENSURE_EQ(context,
                            xa_nn_elm_requantize_asym16s_asym32s(
                                tflite::micro::GetTensorData<int32_t>(output),
                                tflite::micro::GetTensorData<int16_t>(input),
                                op_data->input_zero_point,
                                op_data->quantization_params.zero_point,
                                op_data->requantize_output_shift,
                                op_data->requantize_output_multiplier, size),
                            0);
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt32: {
      switch (output->type) {
        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int32_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              op_data->quantization_params.zero_point,
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }

        case kTfLiteInt16: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int32_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              zero_point, tflite::micro::GetTensorData<int16_t>(output));
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteFloat32: {
      switch (output->type) {
        case kTfLiteInt8: {
#if HIFI_VFPU
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const float* input_data_ptr;
          int8_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<float>(input);
          output_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_quantize_f32_asym8s(
                  output_data_ptr, input_data_ptr,
                  static_cast<float>(op_data->quantization_params.scale),
                  zero_point, size),
              0);
#else   // #if HIFI_VFPU
          reference_ops::AffineQuantize(
              op_data->quantization_params,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<float>(input),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
#endif  // #if HIFI_VFPU
          break;
        }

        case kTfLiteInt16: {
#if HIFI_VFPU
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const float* input_data_ptr;
          int16_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<float>(input);
          output_data_ptr = tflite::micro::GetTensorData<int16_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_quantize_f32_asym16s(
                  output_data_ptr, input_data_ptr,
                  static_cast<float>(op_data->quantization_params.scale),
                  zero_point, size),
              0);
#else   // #if HIFI_VFPU
          reference_ops::AffineQuantize(
              op_data->quantization_params,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<float>(input),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
#endif  // #if HIFI_VFPU
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    default: {
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);

  auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);
  op_data->quantization_params.zero_point = output->params.zero_point;
  op_data->quantization_params.scale =
      static_cast<double>(output->params.scale);

  op_data->input_zero_point = input->params.zero_point;

  double effective_scale = static_cast<double>(input->params.scale) /
                           static_cast<double>(output->params.scale);
  QuantizeMultiplier(effective_scale, &op_data->requantize_output_multiplier,
                     &op_data->requantize_output_shift);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  return EvalXtensa(context, node);
#else
  return EvalQuantizeReference(context, node);
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
}

}  // namespace

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus QuantizeCompile(TfLiteContext* context, TfLiteNode* node,
                             TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/micro/kernels/xtensa/xtensa.h\""
          << std::endl
          << "#include \"tensorflow/lite/micro/kernels/quantize.h\"" << std::endl
          << "#include \"tensorflow/lite/kernels/internal/reference/quantize.h\""
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->user_data != nullptr);
      auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);

      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, 0);
      TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

      const int size = ElementCount(*input->dims);

      ofs << "{ // xtensa quantize" << std::endl;

      // Generate runtime data addresses
      tflite::micro::CompileAddress(ofs, "input_data", input->data.data);
      tflite::micro::CompileAddress(ofs, "output_data", output->data.data);

      // Generate size and quantization parameters
      ofs << "const int size = " << size << ";" << std::endl
          << "const int32_t input_zero_point = " << op_data->input_zero_point
          << ";" << std::endl
          << "const int32_t output_zero_point = "
          << op_data->quantization_params.zero_point << ";" << std::endl
          << "const int32_t multiplier = "
          << op_data->requantize_output_multiplier << ";" << std::endl
          << "const int32_t shift = " << op_data->requantize_output_shift << ";"
          << std::endl
          << "const float scale = "
          << static_cast<float>(op_data->quantization_params.scale) << "f;"
          << std::endl;

      // Type dispatch based on input and output types
      if (input->type == kTfLiteUInt8 && output->type == kTfLiteInt8) {
        // uint8 to int8 using reference
        ofs << "// uint8 to int8 quantize using reference" << std::endl
            << "reference_ops::Requantize(" << std::endl
            << "    reinterpret_cast<const uint8_t*>(input_data), size,"
            << std::endl
            << "    multiplier, shift, input_zero_point, output_zero_point,"
            << std::endl
            << "    reinterpret_cast<int8_t*>(output_data));" << std::endl;
      } else if (input->type == kTfLiteInt8 && output->type == kTfLiteUInt8) {
        // int8 to uint8 using reference
        ofs << "// int8 to uint8 quantize using reference" << std::endl
            << "reference_ops::Requantize(" << std::endl
            << "    reinterpret_cast<const int8_t*>(input_data), size,"
            << std::endl
            << "    multiplier, shift, input_zero_point, output_zero_point,"
            << std::endl
            << "    reinterpret_cast<uint8_t*>(output_data));" << std::endl;
      } else if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
        // int8 to int8 using nnlib
        ofs << "// int8 to int8 requantize using nnlib" << std::endl
            << "xa_nn_elm_requantize_asym8s_asym8s(" << std::endl
            << "    reinterpret_cast<int8_t*>(output_data)," << std::endl
            << "    reinterpret_cast<const int8_t*>(input_data)," << std::endl
            << "    input_zero_point, output_zero_point," << std::endl
            << "    shift, multiplier, size);" << std::endl;
      } else if (input->type == kTfLiteInt8 && output->type == kTfLiteInt16) {
        // int8 to int16 using reference
        ofs << "// int8 to int16 quantize using reference" << std::endl
            << "reference_ops::Requantize(" << std::endl
            << "    reinterpret_cast<const int8_t*>(input_data), size,"
            << std::endl
            << "    multiplier, shift, input_zero_point, output_zero_point,"
            << std::endl
            << "    reinterpret_cast<int16_t*>(output_data));" << std::endl;
      } else if (input->type == kTfLiteInt8 && output->type == kTfLiteInt32) {
        // int8 to int32 using nnlib
        ofs << "// int8 to int32 requantize using nnlib" << std::endl
            << "xa_nn_elm_requantize_asym8s_asym32s(" << std::endl
            << "    reinterpret_cast<int32_t*>(output_data)," << std::endl
            << "    reinterpret_cast<const int8_t*>(input_data)," << std::endl
            << "    input_zero_point, output_zero_point," << std::endl
            << "    shift, multiplier, size);" << std::endl;
      } else if (input->type == kTfLiteInt16 && output->type == kTfLiteInt8) {
        // int16 to int8 using nnlib
        ofs << "// int16 to int8 requantize using nnlib" << std::endl
            << "xa_nn_elm_requantize_asym16s_asym8s(" << std::endl
            << "    reinterpret_cast<int8_t*>(output_data)," << std::endl
            << "    reinterpret_cast<const int16_t*>(input_data)," << std::endl
            << "    input_zero_point, output_zero_point," << std::endl
            << "    shift, multiplier, size);" << std::endl;
      } else if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
        // int16 to int16 using nnlib
        ofs << "// int16 to int16 requantize using nnlib" << std::endl
            << "xa_nn_elm_requantize_asym16s_asym16s(" << std::endl
            << "    reinterpret_cast<int16_t*>(output_data)," << std::endl
            << "    reinterpret_cast<const int16_t*>(input_data)," << std::endl
            << "    input_zero_point, output_zero_point," << std::endl
            << "    shift, multiplier, size);" << std::endl;
      } else if (input->type == kTfLiteInt16 && output->type == kTfLiteInt32) {
        // int16 to int32 using nnlib
        ofs << "// int16 to int32 requantize using nnlib" << std::endl
            << "xa_nn_elm_requantize_asym16s_asym32s(" << std::endl
            << "    reinterpret_cast<int32_t*>(output_data)," << std::endl
            << "    reinterpret_cast<const int16_t*>(input_data)," << std::endl
            << "    input_zero_point, output_zero_point," << std::endl
            << "    shift, multiplier, size);" << std::endl;
      } else if (input->type == kTfLiteInt32 && output->type == kTfLiteInt8) {
        // int32 to int8 using reference
        ofs << "// int32 to int8 quantize using reference" << std::endl
            << "reference_ops::Requantize(" << std::endl
            << "    reinterpret_cast<const int32_t*>(input_data), size,"
            << std::endl
            << "    multiplier, shift, input_zero_point, output_zero_point,"
            << std::endl
            << "    reinterpret_cast<int8_t*>(output_data));" << std::endl;
      } else if (input->type == kTfLiteInt32 && output->type == kTfLiteInt16) {
        // int32 to int16 using reference
        ofs << "// int32 to int16 quantize using reference" << std::endl
            << "reference_ops::Requantize(" << std::endl
            << "    reinterpret_cast<const int32_t*>(input_data), size,"
            << std::endl
            << "    multiplier, shift, input_zero_point, output_zero_point,"
            << std::endl
            << "    reinterpret_cast<int16_t*>(output_data));" << std::endl;
      } else if (input->type == kTfLiteFloat32 && output->type == kTfLiteInt8) {
        // float32 to int8 quantize using nnlib with VFPU (HIFI4 optimized, no conditional compilation)
        ofs << "// float32 to int8 quantize using nnlib with VFPU" << std::endl
            << "xa_nn_elm_quantize_f32_asym8s(" << std::endl
            << "    reinterpret_cast<int8_t*>(output_data)," << std::endl
            << "    reinterpret_cast<const float*>(input_data)," << std::endl
            << "    scale, output_zero_point, size);" << std::endl;
      } else {
        // Unsupported type combination
        ofs << "// Unsupported type combination: " << TfLiteTypeGetName(input->type)
            << " to " << TfLiteTypeGetName(output->type) << std::endl
            << "return kTfLiteError;" << std::endl;
      }

      ofs << "}" << std::endl;

    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif  // TFLITE_MODEL_COMPILER

TFLMRegistration Register_QUANTIZE() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(Init, Prepare, Eval, QuantizeCompile);
#else
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
#endif
}

// Type-specific registration function for FLOAT32 to INT8 quantization
TFLMRegistration Register_QUANTIZE_FLOAT32_INT8() {
  return Register_QUANTIZE();
}

}  // namespace tflite
