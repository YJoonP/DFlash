/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <unordered_map>
#include "HTP/QnnHtpMem.h"

namespace executorch {
namespace backends {
namespace qnn {

// 진단용: 프로세스 전체 QnnMem_register 호출 수 (2026-09-15 KST)
uint64_t QnnMemRegisterCallCount();

class QnnMemManager {
 public:
  explicit QnnMemManager(
      QnnImplementation* implementation,
      QnnContext* context,
      QnnExecuTorchLogLevel log_level)
      : implementation_(implementation),
        context_(context),
        log_level_(log_level) {}
  ~QnnMemManager() {
    DeRegisterMem();
  }

  executorch::runtime::Error RegisterIonMem(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper,
      int32_t mem_fd,
      void* mem_ptr);

  executorch::runtime::Error RegisterCustomMem(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper,
      int32_t mem_fd,
      void* mem_ptr,
      size_t total_custom_mem_size,
      size_t tensor_offset,
      const CustomMemTensorInfo& info);

  bool IsRegistered(Qnn_MemHandle_t handle, void* mem_ptr);

  void* GetPreRegisteredHandle(const CustomMemTensorInfo& info);

  executorch::runtime::Error SetMemHandle(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper,
      void* mem_ptr,
      Qnn_MemHandle_t handle);

  // zero-copy 대형 모델 지원 (2026-09-15 KST). context 생성 직후 이 context 의 cDSP 세션에
  // 더미 공유 버퍼를 매핑해 I/O 매핑용 주소공간을 잡아 두고, 실제 I/O 등록 직전에 푼다.
  // (SDK 문서 "Multiple cDSP Sessions" 의 dummy registration)
  // 주의: 다른 정적 라이브러리 객체와 클래스 배치를 맞추기 위해 데이터 멤버는 추가하지 않는다.
  // 예약 상태는 QnnMemManager.cpp 의 별도 표에 둔다.
  executorch::runtime::Error ReserveVa(size_t bytes);
  void ReleaseVaReservation();

 private:
  void DeRegisterMem();

  QnnImplementation* implementation_;
  QnnContext* context_;
  QnnExecuTorchLogLevel log_level_;
  // Store the registered Qnn_MemHandle_t for de-registration
  std::unordered_map<Qnn_MemHandle_t, void*> registered_map_;
  // Store the pre-registered custom mem handles
  std::unordered_map<CustomMemTensorInfo, void*> pre_registered_handles_;
  std::unordered_map<executorch::aten::ScalarType, Qnn_DataType_t>
      scalar_type_to_qnn_dtype_ = {
          {executorch::aten::ScalarType::Int,
           Qnn_DataType_t::QNN_DATATYPE_INT_32},
          {executorch::aten::ScalarType::Float,
           Qnn_DataType_t::QNN_DATATYPE_FLOAT_32},
          {executorch::aten::ScalarType::Char,
           Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8},
          {executorch::aten::ScalarType::Short,
           Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16},
          {executorch::aten::ScalarType::Byte,
           Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8},
          {executorch::aten::ScalarType::UInt16,
           Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16},
  };
};
} // namespace qnn
} // namespace backends
} // namespace executorch
