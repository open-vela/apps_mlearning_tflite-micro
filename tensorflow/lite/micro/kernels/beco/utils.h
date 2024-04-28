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
#ifndef TENSORFLOW_LITE_MICROpKERNELS_BECO_UTILS_H_
#define TENSORFLOW_LITE_MICROpKERNELS_BECO_UTILS_H_

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace beco {

template <typename Dtype>
void chw2hwc(Dtype* dst, const Dtype* src, int H, int W, int C) {
  const int HW = H * W;
  int idx_hwc = 0;

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      for (int c = 0; c < C; ++c) {
        const int idx_chw = c * HW + h * W + w;
        dst[idx_hwc] = src[idx_chw];
        ++idx_hwc;
      }
    }
  }
}

template <typename Dtype>
void hwc2chw(Dtype* dst, const Dtype* src, int H, int W, int C) {
  const int HW = H * W;
  int idx_hwc = 0;

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      for (int c = 0; c < C; ++c) {
        const int idx_chw = c * HW + h * W + w;
        dst[idx_chw] = src[idx_hwc];
        ++idx_hwc;
      }
    }
  }
}

template <typename Dtype>
void ohwi2ihwo(Dtype* dst, const Dtype* src,
               int p, int O, int H, int W, int I) {
  const int Op = O + p;
  const int WOp = W * Op;
  const int HWOp = H * WOp;
  int idx_ohwi = 0;

  for (int o = 0; o < O; ++o) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int i = 0; i < I; ++i) {
          const int idx_ihwo = i * HWOp + h * WOp + w * Op + o;
          dst[idx_ihwo] = src[idx_ohwi];
          ++idx_ohwi;
        }
      }
    }
  }
}

template <typename Dtype, typename WType>
void addOffset2Bias(Dtype* dst, const Dtype* src, Dtype offset,
                    const WType* weight, int p, int O, int H, int W, int I) {
  const int Op = O + p;
  const int IHW = I * H * W;

  for (int o = 0; o < O; ++o) {
    Dtype sum = 0;
    for (int idx = 0; idx < IHW; ++idx) {
      sum += weight[idx * Op + o] * offset;
    }
    dst[o] = src[o] + sum;
  }
}

}  // namespace beco
}  // namespace tflite

#endif
