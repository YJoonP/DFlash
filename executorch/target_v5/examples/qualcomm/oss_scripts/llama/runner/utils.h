/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <fstream>
#include <cstdlib>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <cstddef>
#include <memory>
#include <vector>

// Template struct to hold tensor data and tensor

// TODO: Refactor these struct to use TensorPtr
// see https://docs.pytorch.org/executorch/stable/extension-tensor.html

// TensorStruct whose dtype known in compile time
template <typename T>
struct TensorStruct {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  std::shared_ptr<std::vector<T>> buffer;
  T* data;
  // data size in bytes
  size_t size;
};

// QNN_FORCE_TOKENS=<int64 raw file>: teacher forcing for accuracy evaluation. After the
// prompt, token i of the file is fed as the i-th decode input instead of the sampled token.
inline const std::vector<int64_t>& forced_tokens() {
  static const std::vector<int64_t> tokens = [] {
    std::vector<int64_t> out;
    const char* path = std::getenv("QNN_FORCE_TOKENS");
    if (path == nullptr) {
      return out;
    }
    std::ifstream f(path, std::ios::binary);
    int64_t t;
    while (f.read(reinterpret_cast<char*>(&t), sizeof(t))) {
      out.push_back(t);
    }
    return out;
  }();
  return tokens;
}

inline size_t getDtypeSize(executorch::aten::ScalarType dtype) {
  switch (dtype) {
    case executorch::aten::ScalarType::Float:
      return sizeof(float);
    case executorch::aten::ScalarType::Double:
      return sizeof(double);
    case executorch::aten::ScalarType::Int:
      return sizeof(int32_t);
    case executorch::aten::ScalarType::Long:
      return sizeof(int64_t);
    case executorch::aten::ScalarType::Byte:
      return sizeof(uint8_t);
    case executorch::aten::ScalarType::UInt16:
      return sizeof(uint16_t);
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported scalar type %s",
          executorch::runtime::toString(dtype));
      break;
  }
}

// TensorStruct whose dtype known in runtime, and raw file is used
struct TensorStructRaw {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  std::byte* data;
  // data size in bytes
  size_t size;
  executorch::aten::ScalarType dtype;
  size_t getElementSize() const {
    return getDtypeSize(dtype);
  }
};
