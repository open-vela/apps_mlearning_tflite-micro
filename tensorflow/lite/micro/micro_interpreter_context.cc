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

#include "tensorflow/lite/micro/micro_interpreter_context.h"

#include <cstdint>
#include <stdio.h>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
MicroInterpreterContext::MicroInterpreterContext(MicroAllocator* allocator,
                                                 const Model* model,
                                                 MicroInterpreterGraph* graph)
    : allocator_(*allocator),
      graph_(*graph),
      model_(model),
      state_(InterpreterState::kInit) {}

MicroInterpreterContext::~MicroInterpreterContext() {}

void* MicroInterpreterContext::AllocatePersistentBuffer(size_t bytes) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInit);
  return allocator_.AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroInterpreterContext::RequestScratchBufferInArena(
    size_t bytes, int* buffer_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  return allocator_.RequestScratchBufferInArena(
      bytes, graph_.GetCurrentSubgraphIndex(), buffer_idx);
}

void* MicroInterpreterContext::GetScratchBuffer(int buffer_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kInvoke);
  ScratchBufferHandle* handle = scratch_buffer_handles_ + buffer_idx;
  return handle->data;
}

TfLiteTensor* MicroInterpreterContext::AllocateTempTfLiteTensor(
    int tensor_idx) {
  return allocator_.AllocateTempTfLiteTensor(model_, graph_.GetAllocations(),
                                             tensor_idx,
                                             graph_.GetCurrentSubgraphIndex());
}

void MicroInterpreterContext::DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
  return allocator_.DeallocateTempTfLiteTensor(tensor);
}

uint8_t* MicroInterpreterContext::AllocateTempBuffer(size_t size,
                                                     size_t alignment) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  return allocator_.AllocateTempBuffer(size, alignment);
}

void MicroInterpreterContext::DeallocateTempBuffer(uint8_t* buffer) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  allocator_.DeallocateTempBuffer(buffer);
}

TfLiteEvalTensor* MicroInterpreterContext::GetEvalTensor(int tensor_idx) {
  return &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()]
              .tensors[tensor_idx];
}

void MicroInterpreterContext::SetScratchBufferHandles(
    ScratchBufferHandle* scratch_buffer_handles) {
  scratch_buffer_handles_ = scratch_buffer_handles;
}

TfLiteStatus MicroInterpreterContext::set_external_context(
    void* external_context_payload) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);
  if (external_context_payload == nullptr ||
      external_context_payload_ != nullptr) {
    MicroPrintf(
        "Attempting to set external context to %x but it was %x already",
        external_context_payload, external_context_payload_);
    return kTfLiteError;
  }

  external_context_payload_ = external_context_payload;
  return kTfLiteOk;
}

#ifdef CONFIG_MICRO_DELEGATE
TfLiteStatus MicroInterpreterContext::GetNodeAndRegistration(
    int node_index, TfLiteNode** node,
    const struct TFLMRegistration** registration) {
  *node =
      &(graph_.GetAllocations()[0].node_and_registrations[node_index].node);
  *registration = graph_.GetAllocations()[0]
                        .node_and_registrations[node_index]
                        .registration;
  if (*node == nullptr || *registration == nullptr)
    return kTfLiteError;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterContext::GetExecutionPlan(
      TfLiteIntArray** execution_plan) {
  if (*execution_plan == nullptr) {
    *execution_plan = TfLiteIntArrayCreate(NumSubgraphOperators(model_, 0));

    for (int i = 0; i < (*execution_plan)->size; ++i) {
      (*execution_plan)->data[i] = i;
    }

    if (*execution_plan == nullptr)
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterContext::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteContext* context,
    struct TFLMRegistration *registration,
    const TfLiteIntArray* nodes_to_replace,
    std::vector<TfLiteDelegateParams*> partitions,
    struct TfLiteDelegate* delegate) {
  registration->builtin_code = BuiltinOperator_DELEGATE;
  if (!nodes_to_replace->size) return kTfLiteOk;

  int start_index = 0;

  for (auto& partition : partitions) {
    if (partition->nodes_to_replace->data[0] ==
        nodes_to_replace->data[start_index]) {
      partition->delegate = delegate;

      for (int i = 0; i < partition->nodes_to_replace->size; ++i) {
        TfLiteNode *node =
            &(graph_.GetAllocations()[0]
              .node_and_registrations[partition->nodes_to_replace->data[i]]
              .node);

        node->delegate = delegate;

        auto *tmp = registration->init(context, (char *)partition,
                                      sizeof(*partition));

         if (tmp == nullptr) return kTfLiteDelegateError;

        TfLiteStatus status = registration->prepare(context, node);

        if (status != kTfLiteOk) return kTfLiteDelegateError;

        const struct TFLMRegistration **tmp_registration =
            &(graph_.GetAllocations()[0]
              .node_and_registrations[partition->nodes_to_replace->data[i]]
              .registration);

        registration->free = (*tmp_registration)->free;
        registration->reset = (*tmp_registration)->reset;
        registration->builtin_code = (*tmp_registration)->builtin_code;
        *tmp_registration = registration;
      }

      start_index += partition->nodes_to_replace->size;
    }
    if (start_index > nodes_to_replace->size)
      break;
  }

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterContext::PreviewDelegatePartitioning(
    const TfLiteIntArray* execution_plan,
    const TfLiteIntArray* nodes_to_replace,
    std::vector<TfLiteDelegateParams*>* partitions) {
  if (!nodes_to_replace->size || execution_plan->size <= 0) return kTfLiteOk;

  int start_index = 0;
  for (int i = 0; i < nodes_to_replace->size; ++i) {
    int temp_size = 1;
    int tmp_index = start_index;
    if (nodes_to_replace->data[i] == execution_plan->data[tmp_index]) {
      while (i + 1 < nodes_to_replace->size &&
            nodes_to_replace->data[i + 1] ==
            execution_plan->data[++tmp_index]) {
        i++;
        temp_size++;
      }
      TfLiteDelegateParams *temp_para =
          (TfLiteDelegateParams *)malloc(sizeof(TfLiteDelegateParams));
      if (temp_para == nullptr) return kTfLiteDelegateError;

      temp_para->delegate = nullptr;
      temp_para->nodes_to_replace = TfLiteIntArrayCreate(temp_size);
      temp_para->input_tensors = nullptr;
      temp_para->output_tensors = nullptr;

      for (int j = 0; j < temp_size; ++j) {
        temp_para->nodes_to_replace->data[j] =
            execution_plan->data[start_index + j];
      }

      partitions->push_back(temp_para);

      if (i + 1 == nodes_to_replace->size)
        tmp_index++;
      start_index = tmp_index;
    } else {
      while (nodes_to_replace->data[i] != execution_plan->data[++tmp_index]) {
        temp_size++;
      }
      TfLiteDelegateParams *temp_para =
          (TfLiteDelegateParams *)malloc(sizeof(TfLiteDelegateParams));
      if (temp_para == nullptr) return kTfLiteDelegateError;

      temp_para->delegate = nullptr;
      temp_para->nodes_to_replace = TfLiteIntArrayCreate(temp_size);
      temp_para->input_tensors = nullptr;
      temp_para->output_tensors = nullptr;

      for (int j = 0; j < temp_size; ++j) {
        temp_para->nodes_to_replace->data[j] =
            execution_plan->data[start_index + j];
      }

      partitions->push_back(temp_para);

      start_index = tmp_index;
      i--;
    }
  }

  int remain_size = 0;
  while (start_index + remain_size < execution_plan->size) {
    remain_size++;
  }
  if (remain_size > 0) {
    TfLiteDelegateParams *temp_para =
        (TfLiteDelegateParams *)malloc(sizeof(TfLiteDelegateParams));
    if (temp_para == nullptr) return kTfLiteDelegateError;

    temp_para->delegate = nullptr;
    temp_para->nodes_to_replace = TfLiteIntArrayCreate(remain_size);
    temp_para->input_tensors = nullptr;
    temp_para->output_tensors = nullptr;

    for (int j = 0; j < remain_size; ++j)
      temp_para->nodes_to_replace->data[j] =
          execution_plan->data[start_index + j];

    partitions->push_back(temp_para);
  }

  return kTfLiteOk;
}
#endif

void MicroInterpreterContext::SetInterpreterState(InterpreterState state) {
  state_ = state;
}

MicroInterpreterContext::InterpreterState
MicroInterpreterContext::GetInterpreterState() const {
  return state_;
}

}  // namespace tflite
