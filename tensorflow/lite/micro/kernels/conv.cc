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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus ConvCompile(TfLiteContext* context, TfLiteNode* node,
                         TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"tensorflow/lite/kernels/internal/reference/integer_ops/conv.h\""
          << std::endl
          << "#include \"tensorflow/lite/kernels/internal/types.h\""
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->builtin_data != nullptr);
      const auto& params =
          *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
      TFLITE_DCHECK(node->user_data != nullptr);
      const auto& data = *(static_cast<const OpDataConv*>(node->user_data));
      MicroContext* micro_context = GetMicroContext(context);

      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, kConvInputTensor);
      const TfLiteEvalTensor* filter =
          tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
      const TfLiteEvalTensor* bias =
          tflite::micro::GetEvalInput(context, node, kConvBiasTensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

      ofs << "{ // conv" << std::endl;

      switch (input->type) {
        case kTfLiteInt8: {
          switch (filter->type) {
            case kTfLiteInt8: {
              tflite::micro::CompileAddress(ofs, "input_data", input->data.data);
              tflite::micro::CompileAddress(ofs, "output_data", output->data.data);
              tflite::micro::CompileArray(ofs, "const int8_t", "filter_data",
                  tflite::micro::GetTensorData<int8_t>(filter),
                  ElementCount(*filter->dims));
              tflite::micro::CompileArray(ofs, "const int32_t", "bias_data",
                  tflite::micro::GetTensorData<int32_t>(bias),
                  ElementCount(*bias->dims));

              const int num_channels = output->dims->data[3];
              tflite::micro::CompileArray(ofs, "int32_t", "multiplier",
                  data.per_channel_output_multiplier, num_channels);
              tflite::micro::CompileArray(ofs, "int32_t", "shift",
                  data.per_channel_output_shift, num_channels);

              tflite::micro::CompileArray(ofs, "const int32_t", "input_dims_data",
                  input->dims->data, input->dims->size);
              tflite::micro::CompileArray(ofs, "const int32_t", "filter_dims_data",
                  filter->dims->data, filter->dims->size);
              tflite::micro::CompileArray(ofs, "const int32_t", "bias_dims_data",
                  bias->dims->data, bias->dims->size);
              tflite::micro::CompileArray(ofs, "const int32_t", "output_dims_data",
                  output->dims->data, output->dims->size);

              ofs << "tflite::ConvParams params;"
                  << "params.padding_type=(tflite::PaddingType)"
                  << (int)tflite::micro::RuntimePaddingType(params.padding)
                  << ";params.padding_values.width=" << data.padding.width
                  << ";params.padding_values.height=" << data.padding.height
                  << ";params.stride_height=" << params.stride_height
                  << ";params.stride_width=" << params.stride_width
                  << ";params.dilation_height_factor=" << params.dilation_height_factor
                  << ";params.dilation_width_factor=" << params.dilation_width_factor
                  << ";params.input_offset=" << -data.input_zero_point
                  << ";params.weights_offset=" << -data.filter_zero_point
                  << ";params.output_offset=" << data.output_zero_point
                  << ";params.output_multiplier=" << data.output_multiplier
                  << ";params.output_shift=" << data.output_shift
                  << ";params.quantized_activation_min=" << data.output_activation_min
                  << ";params.quantized_activation_max=" << data.output_activation_max
                  << ";" << std::endl;

              ofs << "tflite::reference_integer_ops::ConvPerChannel("
                     "params,multiplier,shift,"
                     "tflite::RuntimeShape(" << input->dims->size
                  << ",input_dims_data),(int8_t*)input_data,"
                     "tflite::RuntimeShape(" << filter->dims->size
                  << ",filter_dims_data),filter_data,"
                     "tflite::RuntimeShape(" << bias->dims->size
                  << ",bias_dims_data),bias_data,"
                     "tflite::RuntimeShape(" << output->dims->size
                  << ",output_dims_data),(int8_t*)output_data);"
                  << std::endl;
            } break;

            default:
              ofs << "filter type" << TfLiteTypeGetName(filter->type)
                  << "not currently supported." << std::endl;
              return kTfLiteError;
          }
        } break;

        default:
          ofs << "Input type" << TfLiteTypeGetName(input->type)
              << "not currently supported." << std::endl;
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

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      if (bias == nullptr || bias->type == kTfLiteInt32) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else if (bias->type == kTfLiteInt64) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<std::int64_t>(bias),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default:
          MicroPrintf("Weight type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(ConvInit, ConvPrepare, ConvEval, ConvCompile);
#else
  return tflite::micro::RegisterOp(ConvInit, ConvPrepare, ConvEval);
#endif
}

}  // namespace tflite
