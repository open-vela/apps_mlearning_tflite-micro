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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "beco_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/beco/utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct OpData {
  OpDataConv reference_op_data;
  int buffer_idx_input;
  int buffer_idx_filter;
  int buffer_idx_bias;
  int buffer_idx_output;
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

  // Request temp convert buffers.
  data->padding = (8 - (output->dims->data[3] & 7)) & 7;
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      input->dims->data[0] * input->dims->data[1] * input->dims->data[2] *
          input->dims->data[3],
      &data->buffer_idx_input));
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      (filter->dims->data[0] + data->padding) * filter->dims->data[1] *
          filter->dims->data[2] * filter->dims->data[3],
      &data->buffer_idx_filter));
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      (bias->dims->data[0] + data->padding) * sizeof(int32_t),
      &data->buffer_idx_bias));
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      output->dims->data[0] * output->dims->data[1] * output->dims->data[2] *
          (output->dims->data[3] + data->padding),
      &data->buffer_idx_output));

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

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

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

  // Initialize beco conv parameters.
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

  // Fetch buffers.
  auto input_buffer = reinterpret_cast<int8_t*>(
      micro_context->GetScratchBuffer(data->buffer_idx_input));
  auto filter_buffer = reinterpret_cast<int8_t*>(
      micro_context->GetScratchBuffer(data->buffer_idx_filter));
  auto bias_buffer = reinterpret_cast<int32_t*>(
      micro_context->GetScratchBuffer(data->buffer_idx_bias));
  auto output_buffer = reinterpret_cast<int8_t*>(
      micro_context->GetScratchBuffer(data->buffer_idx_output));

  cmsis_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input->dims->data[3];
  beco::hwc2chw<int8_t>(
      input_buffer, tflite::micro::GetTensorData<int8_t>(input),
      input->dims->data[1], input->dims->data[2], input->dims->data[3]);

  cmsis_nn_dims filter_dims;
  filter_dims.n = filter->dims->data[0] + data->padding;
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];
  beco::ohwi2ihwo<int8_t>(
      filter_buffer, tflite::micro::GetTensorData<int8_t>(filter),
      data->padding, filter->dims->data[0], filter->dims->data[1],
      filter->dims->data[2], filter->dims->data[3]);

  // XXX: should consider empty bias.
  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = bias->dims->data[0] + data->padding;
  if (conv_params.input_offset != 0) {
    beco::addOffset2Bias<int32_t, int8_t>(
        bias_buffer, tflite::micro::GetOptionalTensorData<int32_t>(bias),
        conv_params.input_offset, filter_buffer, data->padding,
        filter->dims->data[0], filter->dims->data[1], filter->dims->data[2],
        filter->dims->data[3]);
  } else {
    std::memcpy(bias_buffer,
                tflite::micro::GetOptionalTensorData<int32_t>(bias),
                bias->dims->data[0] * sizeof(int32_t));
  }

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3] + data->padding;

  // Do the real job.
  BECO_INIT();
  beco_convolve_s8(NULL, &conv_params, &quant_params, &input_dims, input_buffer,
                   &filter_dims, filter_buffer, &bias_dims, bias_buffer,
                   &output_dims, output_buffer);
  BECO_EXIT(0);

  beco::chw2hwc<int8_t>(tflite::micro::GetTensorData<int8_t>(output),
                        output_buffer, output->dims->data[1],
                        output->dims->data[2], output->dims->data[3]);

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

TFLMRegistration Register_CONV_2D_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
