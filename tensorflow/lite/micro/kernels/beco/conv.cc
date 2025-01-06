/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TFLITE_MODEL_COMPILER
#include "beco_nnfunctions.h"
#endif

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/beco/utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

struct OpData {
  OpDataConv reference_op_data;
  int filter_buffer_idx;
  int bias_buffer_idx;
  int io_buffer_idx;
  int hw_buffer_idx;
  int padding;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kConvBiasTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  // Consistency check.
  if (input->type != kTfLiteInt8) {
    MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                input->type);
    return kTfLiteError;
  }
  TFLITE_DCHECK_EQ(input->dims->data[0], output->dims->data[0]);
  TFLITE_DCHECK_EQ(input->dims->data[3], filter->dims->data[3]);
  TFLITE_DCHECK_EQ(filter->dims->data[0], output->dims->data[3]);
  TFLITE_DCHECK_EQ(bias->dims->data[0], output->dims->data[3]);

  // Padding to mutiple of 8.
  data->padding = (8 - (output->dims->data[3] & 7)) & 7;

  // Prepare quant buffers.
  const int num_channels = output->dims->data[3] + data->padding;
  data->reference_op_data.per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->reference_op_data.per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // Initialize conv reference parameters.
  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input->dims->data[2], input->dims->data[1],
      filter->dims->data[2], filter->dims->data[1], output->dims->data[2],
      output->dims->data[1], input->type, &data->reference_op_data));

  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = -data->reference_op_data.input_zero_point;
  conv_params.output_offset = data->reference_op_data.output_zero_point;
  conv_params.stride.h = params.stride_height;
  conv_params.stride.w = params.stride_width;
  conv_params.padding.h = data->reference_op_data.padding.height;
  conv_params.padding.w = data->reference_op_data.padding.width;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;
  conv_params.activation.min = data->reference_op_data.output_activation_min;
  conv_params.activation.max = data->reference_op_data.output_activation_max;

  cmsis_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input->dims->data[3];

  cmsis_nn_dims filter_dims;
  filter_dims.n = filter->dims->data[0] + data->padding;
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3] + data->padding;

  // Request convert scratch buffers.
  int input_size = ElementCount(*input->dims);
  int output_size = output_dims.n * output_dims.h * output_dims.w * output_dims.c;
  int filter_size = filter_dims.n * filter_dims.h * filter_dims.w * filter_dims.c;
  int bias_size = (bias->dims->data[0] + data->padding) * sizeof(int32_t);
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      tflite::Max(input_size, output_size), &data->io_buffer_idx));
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      filter_size, &data->filter_buffer_idx));
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      bias_size, &data->bias_buffer_idx));

  // hardware buffer
  auto hw_buffer_size = beco_convolve_s8_get_buffer_size(
      &input_dims, &output_dims, &conv_params, &filter_dims);
  if (hw_buffer_size > 0) {
    TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      hw_buffer_size, &data->hw_buffer_idx));
  } else {
    data->hw_buffer_idx = -1;
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus Compile(TfLiteContext* context, TfLiteNode* node,
                     TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"beco_nnfunctions.h\"" << std::endl
          << "#include \"tensorflow/lite/micro/kernels/conv.h\"" << std::endl
          << "#include \"tensorflow/lite/micro/kernels/beco/utils.h\""
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->builtin_data != nullptr);
      const auto& params =
          *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
      TFLITE_DCHECK(node->user_data != nullptr);
      const OpData* data = static_cast<const OpData*>(node->user_data);
      MicroContext* micro_context = GetMicroContext(context);

      const TfLiteEvalTensor* input =
          tflite::micro::GetEvalInput(context, node, kConvInputTensor);
      const TfLiteEvalTensor* filter =
          tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
      const TfLiteEvalTensor* bias =
          tflite::micro::GetEvalInput(context, node, kConvBiasTensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

      ofs << "{ // beco conv int8" << std::endl;

      // Tensor data.
      tflite::micro::CompileArray(ofs, "const int8_t", "filter_data",
                                  tflite::micro::GetTensorData<int8_t>(filter),
                                  ElementCount(*filter->dims));
      tflite::micro::CompileArray(ofs, "const int32_t", "bias_data",
                                  tflite::micro::GetTensorData<int32_t>(bias),
                                  ElementCount(*bias->dims));
      tflite::micro::CompileAddress(ofs, "input_data", input->data.data);
      tflite::micro::CompileAddress(ofs, "output_data", output->data.data);

      // Conv parameters.
      ofs << "static const cmsis_nn_conv_params conv_params = {.input_offset="
          << -data->reference_op_data.input_zero_point
          << ", .output_offset=" << data->reference_op_data.output_zero_point
          << ", .stride={.w=" << params.stride_width
          << ", .h=" << params.stride_height
          << "}, .padding={.w=" << data->reference_op_data.padding.width
          << ", .h=" << data->reference_op_data.padding.height
          << "}, .dilation={.w=" << params.dilation_width_factor
          << ", .h=" << params.dilation_height_factor << "}, .activation={.min="
          << data->reference_op_data.output_activation_min
          << ", .max=" << data->reference_op_data.output_activation_max << "}};"
          << std::endl;

      // Quant parameters.
      const int num_channels = output->dims->data[3] + data->padding;
      tflite::micro::CompileArray(
          ofs, "int32_t", "multiplier",
          data->reference_op_data.per_channel_output_multiplier, num_channels);
      tflite::micro::CompileArray(
          ofs, "int32_t", "shift",
          data->reference_op_data.per_channel_output_shift, num_channels);

      ofs << "static const cmsis_nn_per_channel_quant_params quant_params = "
             "{.multiplier=multiplier, .shift=shift};"
          << std::endl;

      // Tensor demensions.
      ofs << "static const cmsis_nn_dims input_dims = {.n="
          << input->dims->data[0] << ", .h=" << input->dims->data[1]
          << ", .w=" << input->dims->data[2] << ", .c=" << input->dims->data[3]
          << "};" << std::endl;

      ofs << "static const cmsis_nn_dims filter_dims = {.n="
          << filter->dims->data[0] + data->padding
          << ", .h=" << filter->dims->data[1]
          << ", .w=" << filter->dims->data[2]
          << ", .c=" << filter->dims->data[3] << "};" << std::endl;

      ofs << "static const cmsis_nn_dims bias_dims = {.n=1, .h=1, .w=1, .c="
          << bias->dims->data[0] + data->padding << "};" << std::endl;

      ofs << "static const cmsis_nn_dims output_dims = {.n="
          << output->dims->data[0] << ", .h=" << output->dims->data[1]
          << ", .w=" << output->dims->data[2]
          << ", .c=" << output->dims->data[3] + data->padding << "};"
          << std::endl;

      // Computations.
      tflite::micro::CompileAddress(
          ofs, "io_buffer",
          micro_context->GetScratchBuffer(data->io_buffer_idx));
      tflite::micro::CompileAddress(
          ofs, "filter_buffer",
          micro_context->GetScratchBuffer(data->filter_buffer_idx));
      tflite::micro::CompileAddress(
          ofs, "bias_buffer",
          micro_context->GetScratchBuffer(data->bias_buffer_idx));

      ofs << "tflite::beco::hwc2chw<int8_t>(reinterpret_cast<int8_t*>("
             "io_buffer), reinterpret_cast<int8_t*>(input_data), "
          << input->dims->data[1] << ", " << input->dims->data[2] << ", "
          << input->dims->data[3] << ");" << std::endl;
      ofs << "std::memcpy(input_data, io_buffer, " << ElementCount(*input->dims)
          << ");" << std::endl;
      ofs << "tflite::beco::ohwi2ihwo<int8_t>(reinterpret_cast<int8_t*>("
             "filter_buffer), filter_data, "
          << data->padding << ", " << filter->dims->data[0] << ", "
          << filter->dims->data[1] << ", " << filter->dims->data[2] << ", "
          << filter->dims->data[3] << ");" << std::endl;
      int32_t input_offset = -data->reference_op_data.input_zero_point;
      if (input_offset != 0) {
        ofs << "tflite::beco::addOffset2Bias<int32_t,int8_t>(reinterpret_cast<"
               "int32_t*>(bias_buffer), bias_data, "
            << input_offset << ", reinterpret_cast<int8_t*>(filter_buffer), "
            << data->padding << ", " << filter->dims->data[0] << ", "
            << filter->dims->data[1] << ", " << filter->dims->data[2] << ", "
            << filter->dims->data[3] << ");" << std::endl;
      } else {
        ofs << "std::memcpy(bias_buffer, bias_data, "
            << bias->dims->data[0] * sizeof(int32_t) << ");" << std::endl;
      }

      ofs << "BECO_INIT();" << std::endl
          << "beco_convolve_s8(nullptr, &conv_params, &quant_params, "
             "&input_dims, reinterpret_cast<int8_t*>(input_data), "
             "&filter_dims, reinterpret_cast<int8_t*>(filter_buffer), "
             "&bias_dims, reinterpret_cast<int32_t*>(bias_buffer), "
             "&output_dims, reinterpret_cast<int8_t*>(io_buffer));"
          << std::endl
          << "BECO_EXIT(0);" << std::endl;

      ofs << "tflite::beco::chw2hwc<int8_t>(reinterpret_cast<int8_t*>(output_"
             "data), reinterpret_cast<int8_t*>(io_buffer), "
          << output->dims->data[1] << ", " << output->dims->data[2] << ", "
          << output->dims->data[3] << ");" << std::endl;

      ofs << "}" << std::endl;

    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#else
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);
  MicroContext* micro_context = GetMicroContext(context);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  // Initialize parameters.
  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = -data->reference_op_data.input_zero_point;
  conv_params.output_offset = data->reference_op_data.output_zero_point;
  conv_params.stride.h = params.stride_height;
  conv_params.stride.w = params.stride_width;
  conv_params.padding.h = data->reference_op_data.padding.height;
  conv_params.padding.w = data->reference_op_data.padding.width;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;
  conv_params.activation.min = data->reference_op_data.output_activation_min;
  conv_params.activation.max = data->reference_op_data.output_activation_max;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(
      data->reference_op_data.per_channel_output_multiplier);
  quant_params.shift =
      const_cast<int32_t*>(data->reference_op_data.per_channel_output_shift);

  cmsis_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input->dims->data[3];

  cmsis_nn_dims filter_dims;
  filter_dims.n = filter->dims->data[0] + data->padding;
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = bias->dims->data[0] + data->padding;

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3] + data->padding;

  cmsis_nn_context ctx;
  ctx.size = 0; // NOTE: not used in BECO
  if (data->hw_buffer_idx > 0)
    ctx.buf = micro_context->GetScratchBuffer(data->hw_buffer_idx);
  else
    ctx.buf = nullptr;

  // Convert tensors.
  auto io_buffer = reinterpret_cast<int8_t*>(
      micro_context->GetScratchBuffer(data->io_buffer_idx));
  auto filter_buffer = reinterpret_cast<int8_t*>(
      micro_context->GetScratchBuffer(data->filter_buffer_idx));
  auto bias_buffer = reinterpret_cast<int32_t*>(
      micro_context->GetScratchBuffer(data->bias_buffer_idx));

  beco::hwc2chw<int8_t>(io_buffer, tflite::micro::GetTensorData<int8_t>(input),
                        input->dims->data[1], input->dims->data[2],
                        input->dims->data[3]);
  std::memcpy(input->data.data, io_buffer, ElementCount(*input->dims));

  beco::ohwi2ihwo<int8_t>(
      filter_buffer, tflite::micro::GetTensorData<int8_t>(filter),
      data->padding, filter->dims->data[0], filter->dims->data[1],
      filter->dims->data[2], filter->dims->data[3]);

  if (conv_params.input_offset != 0) {
    beco::addOffset2Bias<int32_t, int8_t>(
        bias_buffer, tflite::micro::GetOptionalTensorData<int32_t>(bias),
        conv_params.input_offset, filter_buffer, data->padding,
        filter->dims->data[0], filter->dims->data[1], filter->dims->data[2],
        filter->dims->data[3]);
  } else {
    std::memcpy(bias_buffer, tflite::micro::GetTensorData<int32_t>(bias),
                bias->dims->data[0] * sizeof(int32_t));
  }

  // Do the real job.
  BECO_INIT();
  beco_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims,
                   tflite::micro::GetTensorData<int8_t>(input), &filter_dims,
                   filter_buffer, &bias_dims, bias_buffer, &output_dims,
                   io_buffer);
  BECO_EXIT(0);

  beco::chw2hwc<int8_t>(tflite::micro::GetTensorData<int8_t>(output), io_buffer,
                        output->dims->data[1], output->dims->data[2],
                        output->dims->data[3]);

  return kTfLiteOk;
}
#endif

}  // namespace

#ifdef TFLITE_MODEL_COMPILER
TFLMRegistration Register_BECO_CONV_2D() { return Register_BECO_CONV_2D_INT8(); }

TFLMRegistration Register_BECO_CONV_2D_INT8() {
  return tflite::micro::CompileOp(Init, Prepare, nullptr, Compile);
}
#else
TFLMRegistration Register_CONV_2D() { return Register_CONV_2D_INT8(); }

TFLMRegistration Register_CONV_2D_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
#endif

}  // namespace tflite
