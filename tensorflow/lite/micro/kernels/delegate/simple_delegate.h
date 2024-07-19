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
==============================================================================*/

// This file has utilities that facilitates creating new delegates.
// - SimpleDelegateKernelInterface: Represents a Kernel which handles a subgraph
// to be delegated. It has Init/Prepare/Invoke which are going to be called
// during inference, similar to TFLite Kernels. Delegate owner should implement
// this interface to build/prepare/invoke the delegated subgraph.
// - SimpleDelegateInterface:
// This class wraps TFLiteDelegate and users need to implement the interface and
// then call TfLiteDelegateFactory::CreateSimpleDelegate(...) to get
// TfLiteDelegate* that can be passed to ModifyGraphWithDelegate and free it via
// TfLiteDelegateFactory::DeleteSimpleDelegate(...).
// or call TfLiteDelegateFactory::Create(...) to get a std::unique_ptr
// TfLiteDelegate that can also be passed to ModifyGraphWithDelegate, in which
// case TfLite interpereter takes the memory ownership of the delegate.

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/micro/micro_common.h"

#include <memory>
#include <functional>
#include <set>
#include <vector>
#include <limits>

namespace tflite {

using TfLiteDelegateUniquePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

using IsNodeSupportedFn =
    std::function<bool(TfLiteContext*, TfLiteNode*,
                      const struct TFLMRegistration*)>;

// Users should inherit from this class and implement the interface below.
// Each instance represents a single part of the graph (subgraph).
class SimpleDelegateKernelInterface {
  public:
    virtual ~SimpleDelegateKernelInterface() {}

    // Initializes a delegated subgraph.
    // The nodes in the subgraph are inside
    // TfLiteDelegateParams->nodes_to_replace
    virtual TfLiteStatus Init(TfLiteContext* context,
                              const TfLiteDelegateParams* params) = 0;

    // Will be called by the framework. Should handle any needed preparation
    // for the subgraph e.g. allocating buffers, compiling model.
    // Returns status, and signalling any errors.
    virtual TfLiteStatus Prepare(TfLiteContext* context,
                                TfLiteNode* node) = 0;

    // Actual subgraph inference should happen on this call.
    // Returns status, and signalling any errors.
    // NOTE: Tensor data pointers (tensor->data) can change every inference,
    // so the implementation of this method needs to take that into account.
    virtual TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) = 0;
};

// Pure Interface that clients should implement.
// The Interface represents a delegate's capabilities and provides a factory
// for SimpleDelegateKernelInterface.
//
// Clients should implement the following methods:
// - IsNodeSupportedByDelegate
// - Initialize
// - Name
// - CreateDelegateKernelInterface
class SimpleDelegateInterface {
  public:
    virtual ~SimpleDelegateInterface() {}

    // Returns true if 'node' is supported by the delegate. False otherwise.
    virtual bool IsNodeSupportedByDelegate(
        const struct TFLMRegistration* registration,
        const TfLiteNode* node,
        TfLiteContext* context) = 0;

    // Initialize the delegate before finding and replacing TfLite nodes with
    // delegate kernels, for example, retrieving some TFLite settings from
    // 'context'.
    virtual TfLiteStatus Initialize(TfLiteContext* context) = 0;

    // Returns a name that identifies the delegate.
    // This name is used for debugging/logging/profiling.
    virtual const char* Name() const = 0;

    // Returns instance of an object that implements the interface
    // SimpleDelegateKernelInterface.
    // An instance of SimpleDelegateKernelInterface represents one subgraph to
    // be delegated.
    // Caller takes ownership of the returned object.
    virtual SimpleDelegateKernelInterface *CreateDelegateKernelInterface() = 0;

    SimpleDelegateKernelInterface *GetDelegateKernelInterface() {
      return delegate_kernel_;
    }

    void SetDelegateKernelRegistration(
        struct TFLMRegistration *delegate_kernel_registration) {
      delegate_kernel_registration_ = delegate_kernel_registration;
    }

  protected:
    SimpleDelegateKernelInterface *delegate_kernel_ = nullptr;
    struct TFLMRegistration *delegate_kernel_registration_ = nullptr;
};

// Factory class that provides static methods to deal with SimpleDelegate
// creation and deletion.
class TfLiteDelegateFactory {
  public:
    // Creates TfLiteDelegate from the provided SimpleDelegateInterface.
    // The returned TfLiteDelegate should be deleted using DeleteSimpleDelegate.
    // A simple usage of the flags bit mask:
    // CreateSimpleDelegate(..., kTfLiteDelegateFlagsAllowDynamicTensors |
    // kTfLiteDelegateFlagsRequirePropagatedShapes)
    static TfLiteDelegate* CreateSimpleDelegate(
        std::unique_ptr<SimpleDelegateInterface> simple_delegate,
        int64_t flags = kTfLiteDelegateFlagsNone);

    // Deletes 'delegate' the passed pointer must be the one returned
    // from CreateSimpleDelegate.
    // This function will destruct the SimpleDelegate object too.
    static void DeleteSimpleDelegate(TfLiteDelegate* delegate);

    // A convenient function wrapping the above two functions and returning a
    // std::unique_ptr type for auto memory management.
    inline static TfLiteDelegateUniquePtr Create(
        std::unique_ptr<SimpleDelegateInterface> simple_delegate) {
      return TfLiteDelegateUniquePtr(
          CreateSimpleDelegate(std::move(simple_delegate)),
                              DeleteSimpleDelegate);
    }
};

// A utility class to help model graph parition.
// Note the class *needs* to be used in TfLiteDelegate::Prepare.
class GraphPartitionHelper {
  public:
    GraphPartitionHelper(TfLiteContext* context,
                        IsNodeSupportedFn is_node_supported_fn)
        : context_(context), is_node_supported_fn_(is_node_supported_fn){}

    virtual ~GraphPartitionHelper() {
      free(supported_nodes_);
      free(original_execution_plan_);
      for (auto partition: partitions_) {
        free(partition->nodes_to_replace);
        free(partition);
      }
    }

    // Partition the graph into node subsets such that each subset could be
    // replaced with one delegate kernel (i.e. a kTfLiteBuiltinDelegate op).
    // If 'unsupported_nodes_info' is provided, it will be populated with
    // information about all different unsupported nodes.
    virtual TfLiteStatus Partition() {
      return PartitionImpl(0, std::numeric_limits<int>::max());
    }

    int num_total_nodes() const { return num_total_nodes_; }
    int num_supported_nodes() const { return num_supported_nodes_; }
    int num_partitions() const { return partitions_.size(); }
    int GetNumSupportedNodes() { return num_supported_nodes_; }
    TfLiteIntArray* GetSupportedNodes() { return supported_nodes_; }
    std::vector<TfLiteDelegateParams*> GetPartitions() { return partitions_; }

  protected:
    virtual bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                                const struct TFLMRegistration* registration,
                                int node_id) {
      return is_node_supported_fn_(context, node, registration);
    }

    virtual TfLiteStatus PartitionImpl(int start_node_index,
                                      int end_node_index);

    TfLiteContext* const context_ = nullptr;

    // Doesn't own the memory of each TfLiteDelegateParams object as it's
    // managed by the TfLite runtime itself. See
    // TfLiteContext::PreviewDelegatePartitioning for details.
    std::vector<TfLiteDelegateParams*> partitions_;

    // Copy of (pre-delegation) execution plan obtained from TfLiteContext in
    // PrepareSupportedNodes
    TfLiteIntArray* original_execution_plan_ = nullptr;

  private:
    // Generate a list of supported nodes (i.e. populating 'supported_nodes_')
    // by iterating over all nodes (i,e. those listed in the execution_plan
    // associated with 'context_').
    // If 'unsupported_nodes_info' is provided, it will be populated with
    // information about all different unsupported nodes.
    // The 'start_node_index' and 'end_node_index' define the range of nodes
    // that could be delegated.
    TfLiteStatus PrepareSupportedNodes(
                    int start_node_index = 0,
                    int end_node_index = std::numeric_limits<int>::max());

    // The number of total nodes passed in for partitioning (i.e. the
    // execution_plan size associated w/ 'context_')
    int num_total_nodes_ = 0;

    int num_supported_nodes_ = 0;

    // Tells if a node is supported as it could be delegated.
    const IsNodeSupportedFn is_node_supported_fn_ = nullptr;

    // Contains an array of supported node indices.
    TfLiteIntArray* supported_nodes_ = nullptr;  // owns the memory
  };

}  // namespace tflite