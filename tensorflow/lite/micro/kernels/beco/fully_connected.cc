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

#include "arm_nn_types.h"
#include "beco_nnfunctions.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/beco/utils.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct OpData {
  OpDataFullyConnected reference_op_data;
  int buffer_idx_filter;
  int buffer_idx_bias;
  int buffer_idx_hw;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  // Consistency check.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (input->type != kTfLiteInt8) {
    MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                input->type);
    return kTfLiteError;
  }

  // Request temp convert buffers.
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      filter->dims->data[0] * filter->dims->data[1],
      &data->buffer_idx_filter));
  TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      bias->dims->data[0] * sizeof(int32_t),
      &data->buffer_idx_bias));

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = 1;
  output_dims.w = 1;
  output_dims.c = output->dims->data[1];

  auto buffer_size = beco_fully_connected_s8_get_buffer_size(&output_dims);
  if (buffer_size > 0) {
    TF_LITE_ENSURE_STATUS(micro_context->RequestScratchBufferInArena(
      buffer_size, &data->buffer_idx_hw));
  } else {
    data->buffer_idx_hw = -1;
  }

  // Initialize fc reference parameters.
  TF_LITE_ENSURE_STATUS(CalculateOpDataFullyConnected(
      context, params->activation, input->type, input, filter, bias, output,
      &data->reference_op_data));

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));
  MicroContext* micro_context = GetMicroContext(context);

  // Initialize beco fc parameters.
  cmsis_nn_fc_params fc_params;
  fc_params.input_offset = -data.reference_op_data.input_zero_point;
  fc_params.filter_offset = -data.reference_op_data.filter_zero_point;
  fc_params.output_offset = data.reference_op_data.output_zero_point;
  fc_params.activation.min = data.reference_op_data.output_activation_min;
  fc_params.activation.max = data.reference_op_data.output_activation_max;

  cmsis_nn_per_tensor_quant_params quant_params;
  quant_params.multiplier = data.reference_op_data.output_multiplier;
  quant_params.shift = data.reference_op_data.output_shift;

  // Fetch buffers.
  auto filter_buffer = reinterpret_cast<int8_t*>(
      micro_context->GetScratchBuffer(data.buffer_idx_filter));
  auto bias_buffer = reinterpret_cast<int32_t*>(
      micro_context->GetScratchBuffer(data.buffer_idx_bias));

  // XXX: should consider other dims.
  cmsis_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = 1;
  input_dims.w = 1;
  input_dims.c = input->dims->data[1];

  cmsis_nn_dims filter_dims;
  filter_dims.n = filter->dims->data[0];
  filter_dims.h = 1;
  filter_dims.w = 1;
  filter_dims.c = filter->dims->data[1];
  beco::ohwi2ihwo<int8_t>(
    filter_buffer, tflite::micro::GetTensorData<int8_t>(filter),
    0, filter->dims->data[0], 1, 1, filter->dims->data[1]);

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output->dims->data[1];
  if (fc_params.input_offset != 0) {
    beco::addOffset2Bias<int32_t,int8_t>(
      bias_buffer, tflite::micro::GetOptionalTensorData<int32_t>(bias),
      fc_params.input_offset, filter_buffer, 0,
      filter->dims->data[0], 1, 1, filter->dims->data[1]);
  } else {
    std::memcpy(bias_buffer, tflite::micro::GetOptionalTensorData<int32_t>(bias),
      bias->dims->data[1] * sizeof(int32_t));
  }

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = 1;
  output_dims.w = 1;
  output_dims.c = output->dims->data[1];

  cmsis_nn_context ctx;
  ctx.size = 0; // NOTE: not used in BECO
  if (data.buffer_idx_hw > 0) {
    ctx.buf = micro_context->GetScratchBuffer(data.buffer_idx_hw);
  } else {
    ctx.buf = nullptr;
  }

  // Do the real job.
  BECO_INIT();
  beco_fully_connected_s8(&ctx, &fc_params, &quant_params,
    &input_dims,  tflite::micro::GetTensorData<int8_t>(input),
    &filter_dims, filter_buffer,
    &bias_dims,   bias_buffer,
    &output_dims, tflite::micro::GetTensorData<int8_t>(output));
  BECO_EXIT(0);

  return kTfLiteOk;
}

}   // namespace

#ifdef TFLITE_MODEL_COMPILER
TFLMRegistration Register_BECO_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

TFLMRegistration Register_BECO_FULLY_CONNECTED_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
#else
TFLMRegistration Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

TFLMRegistration Register_FULLY_CONNECTED_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
#endif

}   // namespace tflite
