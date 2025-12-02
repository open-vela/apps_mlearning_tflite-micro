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

// Reference implementation of timer functions.  Platforms are not required to
// implement these timer methods, but they are required to enable profiling.

// On platforms that have a POSIX stack or C library, it can be written using
// methods from <sys/time.h> or clock() from <time.h>.

// To add an equivalent function for your own platform, create your own
// implementation file, and place it in a subfolder with named after the OS
// you're targeting. For example, see the Cortex M bare metal version in
// tensorflow/lite/micro/bluepill/micro_time.cc

#include "tensorflow/lite/micro/micro_time.h"

#if defined(TF_LITE_USE_CTIME)
#include <ctime>
#endif

namespace tflite {

#if !defined(TF_LITE_USE_CTIME)

// Reference implementation of the ticks_per_second() function that's required
// for a platform to support Tensorflow Lite for Microcontrollers profiling.
// This returns 0 by default because timing is an optional feature that builds
// without errors on platforms that do not need it.
uint32_t ticks_per_second() { return 0; }

// Reference implementation of the GetCurrentTimeTicks() function that's
// required for a platform to support Tensorflow Lite for Microcontrollers
// profiling. This returns 0 by default because timing is an optional feature
// that builds without errors on platforms that do not need it.
uint32_t GetCurrentTimeTicks() { return 0; }

#else  // defined(TF_LITE_USE_CTIME)

uint32_t ticks_per_second() {
    return 1000000;  // 1微秒 = 1 tick
}

uint32_t GetCurrentTimeTicks() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // 将时间转换为微秒，并确保在 uint32_t 范围内
    uint64_t micros = static_cast<uint64_t>(ts.tv_sec) * 1000000ULL +
                      static_cast<uint64_t>(ts.tv_nsec) / 1000ULL;
    return static_cast<uint32_t>(micros);
}

#endif

}  // namespace tflite
