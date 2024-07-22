/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/mul.h"

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/mul.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   const OpDataMul* data, const TfLiteEvalTensor* input1,
                   const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params = {};

  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.float_activation_max = data->output_activation_max_f32;
  op_params.input1_offset = -data->input1_zero_point;
  op_params.input2_offset = -data->input2_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;

  bool need_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

  if (need_broadcast) {
    if (input1->type == kTfLiteInt8) {
      reference_integer_ops::BroadcastMul4DSlow(
          op_params, tflite::micro::GetTensorShape(input1),
          tflite::micro::GetTensorData<int8_t>(input1),
          tflite::micro::GetTensorShape(input2),
          tflite::micro::GetTensorData<int8_t>(input2),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
    } else if (input1->type == kTfLiteInt16) {
      reference_integer_ops::BroadcastMul4DSlow(
          op_params, tflite::micro::GetTensorShape(input1),
          tflite::micro::GetTensorData<int16_t>(input1),
          tflite::micro::GetTensorShape(input2),
          tflite::micro::GetTensorData<int16_t>(input2),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
    }

  } else {
    if (input1->type == kTfLiteInt8) {
      arm_elementwise_mul_s8(
          tflite::micro::GetTensorData<int8_t>(input1),
          tflite::micro::GetTensorData<int8_t>(input2), op_params.input1_offset,
          op_params.input2_offset, tflite::micro::GetTensorData<int8_t>(output),
          op_params.output_offset, op_params.output_multiplier,
          op_params.output_shift, op_params.quantized_activation_min,
          op_params.quantized_activation_max,
          MatchingElementsSize(tflite::micro::GetTensorShape(input1),
                               tflite::micro::GetTensorShape(input2),
                               tflite::micro::GetTensorShape(output)));
    } else if (input1->type == kTfLiteInt16) {
      arm_elementwise_mul_s16(
          tflite::micro::GetTensorData<int16_t>(input1),
          tflite::micro::GetTensorData<int16_t>(input2),
          op_params.input1_offset, op_params.input2_offset,
          tflite::micro::GetTensorData<int16_t>(output),
          op_params.output_offset, op_params.output_multiplier,
          op_params.output_shift, op_params.quantized_activation_min,
          op_params.quantized_activation_max,
          MatchingElementsSize(tflite::micro::GetTensorShape(input1),
                               tflite::micro::GetTensorShape(input2),
                               tflite::micro::GetTensorShape(output)));
    }
  }
}

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);

  switch (input1->type) {
    case kTfLiteInt8:
      EvalQuantized(context, node, data, input1, input2, output);
      break;
    case kTfLiteInt16:
      EvalQuantized(context, node, data, input1, input2, output);
      break;
    case kTfLiteInt32:
      EvalMulQuantizedReference(context, node, data, input1, input2, output);
      break;
    case kTfLiteFloat32:
      EvalMulFloatReference(context, node, params, data, input1, input2,
                            output);
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.",
                  TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);

  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);
  TFLITE_DCHECK(input1->type == kTfLiteInt8);

  EvalQuantized(context, node, data, input1, input2, output);

  return kTfLiteOk;
}

#ifdef TFLITE_MODEL_COMPILER
TfLiteStatus CompileInt8(TfLiteContext* context, TfLiteNode* node,
                         TfLiteCompileStep step, std::ofstream& ofs) {
  switch (step) {
    case kTfLiteCompileStepInclude:
      ofs << "#include \"Include/arm_nnfunctions.h\""
          << std::endl
          << "#include \"tensorflow/lite/kernels/internal/reference/mul.h\""
          << std::endl
          << "#include \"tensorflow/lite/kernels/internal/reference/"
             "integer_ops/mul.h\""
          << std::endl;
      break;

    case kTfLiteCompileStepEval: {
      TFLITE_DCHECK(node->builtin_data != nullptr);
      TFLITE_DCHECK(node->user_data != nullptr);

      const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);
      const TfLiteEvalTensor* input1 =
          tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
      const TfLiteEvalTensor* input2 =
          tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);
      TFLITE_DCHECK(input1->type == kTfLiteInt8);

      tflite::ArithmeticParams op_params = {};

      op_params.quantized_activation_min = data->output_activation_min;
      op_params.quantized_activation_max = data->output_activation_max;
      op_params.float_activation_max = data->output_activation_max_f32;
      op_params.input1_offset = -data->input1_zero_point;
      op_params.input2_offset = -data->input2_zero_point;
      op_params.output_offset = data->output_zero_point;
      op_params.output_multiplier = data->output_multiplier;
      op_params.output_shift = data->output_shift;

      bool need_broadcast = reference_ops::ProcessBroadcastShapes(
          tflite::micro::GetTensorShape(input1),
          tflite::micro::GetTensorShape(input2), &op_params);

      ofs << "{ // mul int8" << std::endl;

      tflite::micro::CompileAddress(
          ofs, "input1", tflite::micro::GetTensorData<int8_t>(input1));
      tflite::micro::CompileAddress(
          ofs, "input2", tflite::micro::GetTensorData<int8_t>(input2));
      tflite::micro::CompileAddress(
          ofs, "output", tflite::micro::GetTensorData<int8_t>(output));

      if (need_broadcast) {
        ofs << "tflite::ArithmeticParams op_params = {"
            << ".broadcast_category="
            << "static_cast<tflite::BroadcastableOpCategory>("
            << static_cast<unsigned int>(op_params.broadcast_category)
            << "), "
            << ".input1_offset=" << op_params.input1_offset << ", "
            << ".input2_offset=" << op_params.input2_offset << ", "
            << ".output_offset=" << op_params.output_offset << ", "
            << ".output_multiplier=" << op_params.output_multiplier << ", "
            << ".output_shift=" << op_params.output_shift << ", "
            << ".left_shift=" << op_params.left_shift << ", "
            << ".input1_multiplier=" << op_params.input1_multiplier << ", "
            << ".input1_shift=" << op_params.input1_shift << ", "
            << ".input2_multiplier=" << op_params.input2_multiplier << ", "
            << ".input2_shift=" << op_params.input2_shift << ", "
            << ".quantized_activation_min="
            << op_params.quantized_activation_min << ", "
            << ".quantized_activation_max="
            << op_params.quantized_activation_max << ", "
            << ".float_activation_min="
            << op_params.float_activation_min << ", "
            << ".float_activation_max="
            << op_params.float_activation_max << ", "
            << ".int64_activation_min="
            << op_params.int64_activation_min << ", "
            << ".int64_activation_max="
            << op_params.int64_activation_max << ", "
            << ".int16_activation_min="
            << op_params.int16_activation_min << ", "
            << ".int16_activation_max="
            << op_params.int16_activation_max << ", "
            << ".broadcast_shape={" << op_params.broadcast_shape[0] << ", "
            << op_params.broadcast_shape[1] << ", "
            << op_params.broadcast_shape[2] << ", "
            << op_params.broadcast_shape[3] << ", "
            << op_params.broadcast_shape[4] << "}};"
            << std::endl;

        tflite::micro::CompileArray(ofs, "const int32_t", "input1_dims_data",
                            input1->dims->data, input1->dims->size);
        tflite::micro::CompileArray(ofs, "const int32_t", "input2_dims_data",
                            input2->dims->data, input2->dims->size);
        tflite::micro::CompileArray(ofs, "const int32_t", "output_dims_data",
                                    output->dims->data, output->dims->size);

        ofs << "tflite::reference_integer_ops::BroadcastMul4DSlow(op_params, "
            << "tflite::RuntimeShape(" << input1->dims->size
            << ", input1_dims_data), " << "input1, "
            << "tflite::RuntimeShape(" << input2->dims->size
            << ", input2_dims_data), " << "input2, "
            << "tflite::RuntimeShape(" << output->dims->size
            << ", output_dims_data), " << "output);"
            << std::endl;
      } else {
        const auto shape1 = tflite::micro::GetTensorShape(input1);
        const auto shape2 = tflite::micro::GetTensorShape(input2);
        const auto shape3 = tflite::micro::GetTensorShape(output);
        const int32_t block_size = MatchingElementsSize(shape1, shape2, shape3);

        ofs << "arm_elementwise_mul_s8(input1, input2, "
            << op_params.input1_offset << ", "
            << op_params.input2_offset << ", output, "
            << op_params.output_offset << ", "
            << op_params.output_multiplier << ", "
            << op_params.output_shift << ", "
            << op_params.quantized_activation_min << ", "
            << op_params.quantized_activation_max << ", "
            << block_size << ");" << std::endl;
      }
      ofs << "}" << std::endl;

    } break;

    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif

TfLiteStatus EvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);

  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);
  TFLITE_DCHECK(input1->type == kTfLiteInt16);

  EvalQuantized(context, node, data, input1, input2, output);

  return kTfLiteOk;
}

TFLMRegistration Register_MUL() {
  return tflite::micro::RegisterOp(MulInit, MulPrepare, Eval);
}

TFLMRegistration Register_MUL_INT8() {
#ifdef TFLITE_MODEL_COMPILER
  return tflite::micro::CompileOp(MulInit, MulPrepare, EvalInt8, CompileInt8);
#else
  return tflite::micro::RegisterOp(MulInit, MulPrepare, EvalInt8);
#endif
}

TFLMRegistration Register_MUL_INT16() {
  return tflite::micro::RegisterOp(MulInit, MulPrepare, EvalInt16);
}

}  // namespace tflite
