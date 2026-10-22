/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/htp/HtpContext.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpContextCustomConfig.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>

#include <cstdlib>
#include <cstring>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnContext_CustomConfig_t>
HtpContextCustomConfig::CreateContextCustomConfig() {
  std::vector<QnnContext_CustomConfig_t> ret;
  QnnHtpContext_CustomConfig_t* p_custom_config = nullptr;
  const HtpContext* htp_ctx = static_cast<const HtpContext*>(context_);

  // TODO: Verify heap profile works with kProfileBasic once enabled.
  if (profile_level_ != QnnExecuTorchProfileLevel::kProfileOff) {
    QnnHtpContext_CustomConfig_t* p_custom_config_profile = nullptr;
    p_custom_config_profile = AllocContextCustomConfig();
    p_custom_config_profile->option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED;
    p_custom_config_profile->dspMemoryProfilingEnabled = true;
    ret.push_back(
        static_cast<QnnContext_CustomConfig_t>(p_custom_config_profile));
  }

  if (htp_options_->use_multi_contexts() &&
      htp_options_->max_sf_buf_size() != 0) {
    p_custom_config = AllocContextCustomConfig();
    p_custom_config->option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    QnnHtpContext_GroupRegistration_t group_info;
    group_info.firstGroupHandle = htp_ctx->GetSpillFillHandle();
    group_info.maxSpillFillBuffer = htp_options_->max_sf_buf_size();
    p_custom_config->groupRegistration = group_info;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  }

  // zero-copy(shared buffer) 대형 모델 지원 (2026-09-15 KST).
  // QNN HTP 는 context binary 를 cDSP 세션(주소공간 3.75GB)에 가능한 한 빽빽이 채운다.
  // 그러면 나중에 QnnMem_register 로 I/O 버퍼를 같은 세션에 매핑할 공간이 없어
  // 첫 실행에서 skel 오류(1100)나 SSR 이 난다. SDK 문서 "Multiple cDSP Sessions" 가
  // 권하는 해결책이 IO_MEM_ESTIMATION 이다: 세션 배치 때 I/O 크기를 미리 계산해 공간을 남긴다.
  // 공유 버퍼를 쓰지 않는 기존 실행의 세션 배치를 바꾸지 않도록 환경변수로만 켠다.
  const char* io_mem_estimation = std::getenv("QNN_IO_MEM_ESTIMATION");
  if (io_mem_estimation != nullptr && std::strcmp(io_mem_estimation, "1") == 0) {
    QnnHtpContext_CustomConfig_t* p_io_mem = AllocContextCustomConfig();
    p_io_mem->option = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    p_io_mem->ioMemEstimation = true;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_io_mem));
    QNN_EXECUTORCH_LOG_WARN("QNN_IO_MEM_ESTIMATION=1: context I/O memory estimation enabled");
  }

  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
