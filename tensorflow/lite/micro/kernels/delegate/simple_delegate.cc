/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/

#include <limits>
#include <memory>
#include <string>

#ifdef CONFIG_MICRO_DELEGATE_DEBUG
#include <syslog.h>
#endif

#include "tensorflow/lite/micro/kernels/delegate/simple_delegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter_context.h"

namespace tflite {
namespace {
struct TFLMRegistration *GetDelegateKernelRegistration(
                            SimpleDelegateInterface* delegate)
{
  struct TFLMRegistration *kernel_registration =
      (struct TFLMRegistration *)malloc(sizeof(struct TFLMRegistration));
  kernel_registration->builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration->custom_name = delegate->Name();
  kernel_registration->free =
      [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<SimpleDelegateKernelInterface*>(buffer);
  };

  kernel_registration->init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    if (params == nullptr) {
      TF_LITE_KERNEL_LOG(context, "NULL TfLiteDelegateParams passed.");
      return nullptr;
    }
    auto* delegate =
        reinterpret_cast<SimpleDelegateInterface*>(params->delegate->data_);
    auto *delegate_kernel = delegate->CreateDelegateKernelInterface();
    if (delegate_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return delegate_kernel;
  };

  kernel_registration->prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->delegate->data_ == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Delegate kernel was not initialized");
      return kTfLiteError;
    }
    SimpleDelegateInterface* delegate =
        reinterpret_cast<SimpleDelegateInterface*>(node->delegate->data_);
    return delegate->GetDelegateKernelInterface()->Prepare(context, node);
  };

  kernel_registration->invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    SimpleDelegateInterface* delegate =
        reinterpret_cast<SimpleDelegateInterface*>(node->delegate->data_);
    TFLITE_DCHECK(delegate->GetDelegateKernelInterface() != nullptr);
    return delegate->GetDelegateKernelInterface()->Eval(context, node);
  };

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context,
                            TfLiteDelegate* base_delegate) {
  auto* delegate =
      reinterpret_cast<SimpleDelegateInterface*>(base_delegate->data_);

  TF_LITE_ENSURE_STATUS(delegate->Initialize(context));
  IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
         const struct TFLMRegistration* registration) -> bool {
    return delegate->IsNodeSupportedByDelegate(registration, node, context);
  };

  GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition());

#ifdef CONFIG_MICRO_DELEGATE_DEBUG
  MicroPrintf(
        "%s delegate: %d nodes delegated of %d nodes with %d partitions.\n",
            delegate->Name(), helper.GetNumSupportedNodes(),
            helper.num_total_nodes(), helper.num_partitions());
#endif

  struct TFLMRegistration *delegate_kernel_registration =
      GetDelegateKernelRegistration(delegate);

  delegate->SetDelegateKernelRegistration(delegate_kernel_registration);

  return static_cast<MicroInterpreterContext*>(context->impl_)
            ->ReplaceNodeSubsetsWithDelegateKernels(
                context,
                delegate_kernel_registration,
                helper.GetSupportedNodes(),
                helper.GetPartitions(),
                base_delegate);
}
}  // namespace

TfLiteDelegate* TfLiteDelegateFactory::CreateSimpleDelegate(
    std::unique_ptr<SimpleDelegateInterface> simple_delegate, int64_t flag) {
  if (simple_delegate == nullptr) {
    return nullptr;
  }
  auto delegate = new TfLiteDelegate();
  delegate->Prepare = &DelegatePrepare;
  delegate->flags = flag;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;
  delegate->data_ = simple_delegate.release();

  return delegate;
}

void TfLiteDelegateFactory::DeleteSimpleDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  SimpleDelegateInterface* simple_delegate =
      reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
  delete simple_delegate;
  delete delegate;
}

TfLiteStatus GraphPartitionHelper::PrepareSupportedNodes(int start_node_index,
                                                        int end_node_index) {
  if (!is_node_supported_fn_) return kTfLiteOk;

  auto status = static_cast<MicroInterpreterContext*>(context_->impl_)
                    ->GetExecutionPlan(&original_execution_plan_);
  if (status != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context_, "Unable to get graph execution plan.\n");
    return kTfLiteDelegateError;
  }
  num_total_nodes_ = original_execution_plan_->size;

  supported_nodes_ = TfLiteIntArrayCreate(num_total_nodes_);
  supported_nodes_->size = 0;

  for (int node_id : TfLiteIntArrayView(original_execution_plan_)) {
    if (node_id < start_node_index) continue;
    if (node_id > end_node_index) break;

    TfLiteNode* node;
    const struct TFLMRegistration* registration;

    status = static_cast<MicroInterpreterContext*>(context_->impl_)
                ->GetNodeAndRegistration(node_id, &node, &registration);
    if (status != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context_,
                        "Couldn't get node and registration info for op: %d\n",
                        node_id);
      supported_nodes_->size = 0;
      return kTfLiteDelegateError;
    }

    if (IsNodeSupported(context_, node, registration, node_id)) {
      supported_nodes_->data[supported_nodes_->size++] = node_id;
    }
  }

  num_supported_nodes_ = supported_nodes_->size;
  return kTfLiteOk;
}

TfLiteStatus GraphPartitionHelper::PartitionImpl(int start_node_index,
                                                int end_node_index) {
  const auto prepare_status = PrepareSupportedNodes(start_node_index,
                                                   end_node_index);
  if (prepare_status != kTfLiteOk) return prepare_status;

  if (static_cast<MicroInterpreterContext*>(context_->impl_)
        ->PreviewDelegatePartitioning(original_execution_plan_,
                                     supported_nodes_,
                                     &partitions_) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context_, "Unable to preview delegate partition.\n");
    return kTfLiteDelegateError;
  }

  return kTfLiteOk;
}

}  // namespace tflite