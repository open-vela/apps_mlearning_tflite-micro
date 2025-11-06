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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFIMINI)
  return ConvReferenceEvalInt8(context, node);
#else
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  return ConvEvalHifiInt8(context, node, params, op_data, input, filter, bias,
                          output);
#elif defined(VISION_P6)
  return ConvEvalVision(context, node, params, op_data, input, filter, bias,
                        output);
#endif

#endif  // defined(HIFIMINI)
}

TfLiteStatus EvalInt16(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  return ConvEvalHifiInt16(context, node, params, op_data, input, filter, bias,
                           output);
#else
  return ConvReferenceEvalInt16(context, node);
#endif
}

}  // namespace

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus ConvCompileInt8(TfLiteContext* context, TfLiteNode* node,
                         TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      ofs << "#include \"include/nnlib/xa_nnlib_api.h\"" << std::endl
          << "#include \"include/nnlib/xa_nnlib_standards.h\"" << std::endl
          << "#include \"tensorflow/lite/kernels/internal/types.h\"" << std::endl;
#else
      ofs << "#include \"tensorflow/lite/kernels/internal/reference/integer_ops/conv.h\""
          << std::endl
          << "#include \"tensorflow/lite/kernels/internal/types.h\"" << std::endl;
#endif
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->builtin_data != nullptr);
      const auto& params =
          *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
      TFLITE_DCHECK(node->user_data != nullptr);
      const auto& data = *(static_cast<const XtensaConvOpData*>(node->user_data));

      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, kConvInputTensor);
      const TfLiteEvalTensor* filter =
          tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
      const TfLiteEvalTensor* bias =
          tflite::micro::GetEvalInput(context, node, kConvBiasTensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

      ofs << "{ // conv int8 hifi" << std::endl;

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      // Generate tensor data arrays
      tflite::micro::CompileArray(ofs, "const int8_t", "filter_data",
                                  tflite::micro::GetTensorData<int8_t>(filter),
                                  ElementCount(*filter->dims));
      tflite::micro::CompileArray(ofs, "const int32_t", "bias_data",
                                  tflite::micro::GetTensorData<int32_t>(bias),
                                  ElementCount(*bias->dims));
      tflite::micro::CompileAddress(ofs, "input_data", input->data.data);
      tflite::micro::CompileAddress(ofs, "output_data", output->data.data);

      // Generate quantization parameters
      const int num_channels = output->dims->data[3];
      tflite::micro::CompileArray(ofs, "int32_t", "multiplier",
                                  data.reference_op_data.per_channel_output_multiplier,
                                  num_channels);
      tflite::micro::CompileArray(ofs, "int32_t", "shift",
                                  data.reference_op_data.per_channel_output_shift,
                                  num_channels);

      // Generate dimension variables
      ofs << "const int batches = " << input->dims->data[0] << ";" << std::endl;
      ofs << "const int input_height = " << input->dims->data[1] << ";" << std::endl;
      ofs << "const int input_width = " << input->dims->data[2] << ";" << std::endl;
      ofs << "const int input_depth = " << input->dims->data[3] << ";" << std::endl;
      ofs << "const int filter_height = " << filter->dims->data[1] << ";" << std::endl;
      ofs << "const int filter_width = " << filter->dims->data[2] << ";" << std::endl;
      ofs << "const int output_height = " << output->dims->data[1] << ";" << std::endl;
      ofs << "const int output_width = " << output->dims->data[2] << ";" << std::endl;
      ofs << "const int output_depth = " << output->dims->data[3] << ";" << std::endl;

      // Generate convolution parameters
      ofs << "const int32_t input_offset = "
          << -data.reference_op_data.input_zero_point << ";" << std::endl;
      ofs << "const int32_t output_offset = "
          << data.reference_op_data.output_zero_point << ";" << std::endl;
      ofs << "const int stride_width = " << params.stride_width << ";" << std::endl;
      ofs << "const int stride_height = " << params.stride_height << ";" << std::endl;
      ofs << "const int pad_width = " << data.reference_op_data.padding.width
          << ";" << std::endl;
      ofs << "const int pad_height = " << data.reference_op_data.padding.height
          << ";" << std::endl;
      ofs << "const int32_t output_activation_min = "
          << data.reference_op_data.output_activation_min << ";" << std::endl;
      ofs << "const int32_t output_activation_max = "
          << data.reference_op_data.output_activation_max << ";" << std::endl;
      ofs << "const int output_data_format = 0;" << std::endl;
      ofs << "const int out_length = output_height * output_width * output_depth;"
          << std::endl;

      // Generate scratch buffer
      tflite::micro::CompileAddress(
          ofs, "p_scratch",
          context->GetScratchBuffer(context, data.scratch_tensor_index));

      // Generate convolution code based on filter size
      ofs << "if (filter_height == 1 && filter_width == 1) {" << std::endl;
      ofs << "  for (int batch = 0; batch < batches; ++batch) {" << std::endl;
      ofs << "    int8_t* p_out_temp = reinterpret_cast<int8_t*>(output_data) + "
          << "batch * out_length;" << std::endl;
      ofs << "    xa_nn_conv2d_pointwise_per_chan_sym8sxasym8s(" << std::endl;
      ofs << "        p_out_temp, const_cast<WORD8*>(filter_data)," << std::endl;
      ofs << "        const_cast<WORD8*>(reinterpret_cast<int8_t*>(input_data) + "
          << "batch * input_height * input_width * input_depth)," << std::endl;
      ofs << "        const_cast<WORD32*>(bias_data), input_height, input_width,"
          << std::endl;
      ofs << "        input_depth, output_depth, input_offset," << std::endl;
      ofs << "        multiplier, shift, output_offset, output_data_format);"
          << std::endl;
      ofs << "    xa_nn_vec_activation_min_max_8_8(" << std::endl;
      ofs << "        p_out_temp, p_out_temp, output_activation_min," << std::endl;
      ofs << "        output_activation_max, out_length);" << std::endl;
      ofs << "  }" << std::endl;
      ofs << "} else {" << std::endl;
      ofs << "  for (int batch = 0; batch < batches; ++batch) {" << std::endl;
      ofs << "    int8_t* p_out_temp = reinterpret_cast<int8_t*>(output_data) + "
          << "batch * out_length;" << std::endl;
      ofs << "    xa_nn_conv2d_std_per_chan_sym8sxasym8s(" << std::endl;
      ofs << "        p_out_temp," << std::endl;
      ofs << "        reinterpret_cast<int8_t*>(input_data) + "
          << "batch * input_height * input_width * input_depth," << std::endl;
      ofs << "        const_cast<int8_t*>(filter_data)," << std::endl;
      ofs << "        bias_data, input_height, input_width, input_depth," << std::endl;
      ofs << "        filter_height, filter_width, output_depth, stride_width,"
          << std::endl;
      ofs << "        stride_height, pad_width, pad_height, output_height,"
          << std::endl;
      ofs << "        output_width, input_offset, multiplier, shift," << std::endl;
      ofs << "        output_offset, output_data_format, p_scratch);" << std::endl;
      ofs << "    xa_nn_vec_activation_min_max_8_8(" << std::endl;
      ofs << "        p_out_temp, p_out_temp, output_activation_min," << std::endl;
      ofs << "        output_activation_max, out_length);" << std::endl;
      ofs << "  }" << std::endl;
      ofs << "}" << std::endl;

#else
      // Fallback to reference implementation for non-HIFI platforms
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
                                  data.reference_op_data.per_channel_output_multiplier,
                                  num_channels);
      tflite::micro::CompileArray(ofs, "int32_t", "shift",
                                  data.reference_op_data.per_channel_output_shift,
                                  num_channels);

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
          << ";params.padding_values.width=" << data.reference_op_data.padding.width
          << ";params.padding_values.height=" << data.reference_op_data.padding.height
          << ";params.stride_height=" << params.stride_height
          << ";params.stride_width=" << params.stride_width
          << ";params.dilation_height_factor=" << params.dilation_height_factor
          << ";params.dilation_width_factor=" << params.dilation_width_factor
          << ";params.input_offset=" << -data.reference_op_data.input_zero_point
          << ";params.weights_offset=" << -data.reference_op_data.filter_zero_point
          << ";params.output_offset=" << data.reference_op_data.output_zero_point
          << ";params.quantized_activation_min="
          << data.reference_op_data.output_activation_min
          << ";params.quantized_activation_max="
          << data.reference_op_data.output_activation_max
          << ";" << std::endl;

      ofs << "tflite::reference_integer_ops::ConvPerChannel("
             "params,multiplier,shift,"
             "tflite::RuntimeShape(" << input->dims->size
          << ",input_dims_data),reinterpret_cast<int8_t*>(input_data),"
             "tflite::RuntimeShape(" << filter->dims->size
          << ",filter_dims_data),filter_data,"
             "tflite::RuntimeShape(" << bias->dims->size
          << ",bias_dims_data),bias_data,"
             "tflite::RuntimeShape(" << output->dims->size
          << ",output_dims_data),reinterpret_cast<int8_t*>(output_data));"
          << std::endl;
#endif

      ofs << "}" << std::endl;
    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif  // TFLITE_MODEL_COMPILER

TFLMRegistration Register_CONV_2D_INT8() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(ConvInitXtensa, ConvPrepareXtensa, EvalInt8,
                                  ConvCompileInt8);
#else
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa, EvalInt8);
#endif
}

TFLMRegistration Register_CONV_2D_INT16() {
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa,
                                   EvalInt16);
}

}  // namespace tflite
