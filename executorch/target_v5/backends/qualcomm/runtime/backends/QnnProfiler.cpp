/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/QnnProfiler.h>

#include <cinttypes>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <vector>

#include <cstdio>
#include <memory>

#include "HTP/QnnHtpProfile.h"

namespace executorch {
namespace backends {
namespace qnn {

#define DEFINE_HEAP_BEFORE_CREATION "DSP:before_context_created"
#define DEFINE_HEAP_AFTER_FREED "DSP:after_context_freed"

namespace {
const char* get_event_unit(QnnProfile_EventUnit_t unit) {
  switch (unit) {
    case QNN_PROFILE_EVENTUNIT_MICROSEC:
      return " (us)";
    case QNN_PROFILE_EVENTUNIT_BYTES:
      return " (bytes)";
    case QNN_PROFILE_EVENTUNIT_COUNT:
      return " (count)";
    case QNN_PROFILE_EVENTUNIT_BACKEND:
    // cycle unit is default appeared
    case QNN_PROFILE_EVENTUNIT_CYCLES:
    default:
      return "";
  }
}
} // namespace

QnnProfile::QnnProfile(
    QnnImplementation* implementation,
    QnnBackend* backend,
    const QnnExecuTorchProfileLevel& profile_level)
    : handle_(nullptr), implementation_(implementation), backend_(backend) {
  const bool compact = std::getenv("QNN_PROFILE_COMPACT") != nullptr;
  if (compact || profile_level != QnnExecuTorchProfileLevel::kProfileOff) {
    const QnnInterface& qnn_interface = implementation_->GetQnnInterface();

    QnnProfile_Level_t qnnProfileLevel = 0;
    if (compact) {
      // QNN_PROFILE_COMPACT=basic|detailed|linting (linting: per-op main-thread cycles + wait/overlap)
      const std::string mode(std::getenv("QNN_PROFILE_COMPACT"));
      qnnProfileLevel = mode == "basic" ? QNN_PROFILE_LEVEL_BASIC
          : mode == "linting" ? static_cast<QnnProfile_Level_t>(QNN_HTP_PROFILE_LEVEL_LINTING)
                              : QNN_PROFILE_LEVEL_DETAILED;
    } else if (profile_level == QnnExecuTorchProfileLevel::kProfileBasic) {
      qnnProfileLevel = QNN_PROFILE_LEVEL_BASIC;
    } else if (
        profile_level == QnnExecuTorchProfileLevel::kProfileDetailed ||
        profile_level == QnnExecuTorchProfileLevel::kProfileOptrace) {
      qnnProfileLevel = QNN_PROFILE_LEVEL_DETAILED;
    } else {
      QNN_EXECUTORCH_LOG_WARN("Invalid profile level");
      return;
    }

    Qnn_ErrorHandle_t error = qnn_interface.qnn_profile_create(
        backend_->GetHandle(), qnnProfileLevel, &handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Failed to create profile_handle for backend "
          " %u, error=%d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));

      // ignore error and continue to create backend handle...
      handle_ = nullptr;
    }

    if (profile_level == QnnExecuTorchProfileLevel::kProfileOptrace) {
      if (handle_ == nullptr) {
        QNN_EXECUTORCH_LOG_WARN(
            "Prfoile handle is null, cannot enable optrace");
        return;
      }

      QnnProfile_Config_t qnnProfileConfig = QNN_PROFILE_CONFIG_INIT;
      qnnProfileConfig.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
      qnnProfileConfig.enableOptrace = true;
      std::array<const QnnProfile_Config_t*, 2> profileConfigs = {
          &qnnProfileConfig, nullptr};
      error =
          qnn_interface.qnn_profile_set_config(handle_, profileConfigs.data());

      if (error != QNN_SUCCESS) {
        QNN_EXECUTORCH_LOG_WARN(
            "Failed to set optrace for backend "
            " %u, error=%d",
            qnn_interface.GetBackendId(),
            QNN_GET_ERROR_CODE(error));
      }
    }
  }
}

Qnn_ErrorHandle_t QnnProfile::FetchEvents(
    const QnnProfile_EventId_t** events_ptr,
    std::uint32_t* num_events) {
  if (handle_ == nullptr) {
    QNN_EXECUTORCH_LOG_WARN("Profile handle is null, skipping FetchEvents");
    *num_events = 0;
    return QNN_SUCCESS;
  }
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_profile_get_events(handle_, events_ptr, num_events);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to get profile events: %d", QNN_GET_ERROR_CODE(error));
  }
  return error;
}

Qnn_ErrorHandle_t QnnProfile::ProfileData(
    executorch::runtime::EventTracer* event_tracer) {
  // A bounded text snapshot avoids transferring large traces over device USB.
  const bool compact = std::getenv("QNN_PROFILE_COMPACT") != nullptr;
  static thread_local uint64_t compact_calls = 0;
  if (compact) ++compact_calls;
  // Capture one complete eight-shard step; start=1 selects prefill.
  const char* start_env = std::getenv("QNN_PROFILE_COMPACT_START");
  const uint64_t start = start_env ? std::max(1, std::atoi(start_env)) : 65;
  // QNN_PROFILE_COMPACT_SPAN=<n> calls in the window (default 8 = one Qwen3.5 step).
  // QNN_PROFILE_COMPACT_DUMP=<path> appends every node row of the window as TSV:
  //   call, parent_type, unit, value, name, wait, overlap, wait_overlap (linting sub-events).
  const char* span_env = std::getenv("QNN_PROFILE_COMPACT_SPAN");
  const uint64_t span = span_env ? std::max(1, std::atoi(span_env)) : 8;
  const char* dump_path = std::getenv("QNN_PROFILE_COMPACT_DUMP");
  const bool snapshot = compact && compact_calls >= start && compact_calls < start + span;
  std::unique_ptr<FILE, int (*)(FILE*)> dump_holder(
      (snapshot && dump_path != nullptr) ? std::fopen(dump_path, "a") : nullptr, &std::fclose);
  FILE* dump = dump_holder.get();
  if (compact && !snapshot && event_tracer == nullptr) return QNN_SUCCESS;
  const QnnProfile_EventId_t* events_ptr = nullptr;
  std::uint32_t num_events = 0;
  Qnn_ErrorHandle_t error = FetchEvents(&events_ptr, &num_events);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to profile data in function FetchEvents: %d",
        QNN_GET_ERROR_CODE(error));
    return error;
  }
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  QnnProfile_EventData_t event_data;
  for (std::uint32_t i = 0; i < num_events; ++i) {
    error =
        qnn_interface.qnn_profile_get_event_data(events_ptr[i], &event_data);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "ProfileData failed to get event data "
          "for event %d: %d",
          i,
          QNN_GET_ERROR_CODE(error));
      return error;
    }
    // add events for other important metrics, e.g. RPC execution time
    std::string identifier =
        std::string(event_data.identifier) + get_event_unit(event_data.unit);
    if (snapshot) {
      QNN_EXECUTORCH_LOG_INFO("[QNN_COMPACT] parent %s: %llu", identifier.c_str(),
          static_cast<unsigned long long>(event_data.value));
    }
    executorch::runtime::event_tracer_log_profiling_delegate(
        event_tracer,
        identifier.c_str(),
        /*delegate_debug_id=*/
        static_cast<executorch::runtime::DebugHandle>(-1),
        0,
        event_data.value);
    // Check an event's sub events only if it relates to graph execution time
    // (and its sub events are the individual op executions):
    const bool node_parent = backend_->IsProfileEventTypeParentOfNodeTime(event_data.type) ||
        (snapshot &&
         (event_data.type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC ||
          event_data.type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_CRITICAL_ACCEL_TIME_CYCLE));
    if (!node_parent) {
      continue;
    }
    const QnnProfile_EventId_t* sub_events_ptr = nullptr;
    std::uint32_t num_sub_events = 0;
    error = qnn_interface.qnn_profile_get_sub_events(
        events_ptr[i], &sub_events_ptr, &num_sub_events);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "ProfileData failed to get sub events "
          "for event %d: %d",
          i,
          QNN_GET_ERROR_CODE(error));
      return error;
    }

    QnnProfile_EventData_t sub_event_data;
    std::vector<std::pair<uint64_t, std::string>> nodes;
    for (std::uint32_t j = 0; j < num_sub_events; ++j) {
      error = qnn_interface.qnn_profile_get_event_data(
          sub_events_ptr[j], &sub_event_data);
      if (error != QNN_SUCCESS) {
        QNN_EXECUTORCH_LOG_ERROR(
            "ProfileData failed to get sub "
            "event data for sub event %d of event %d: %d",
            j,
            i,
            QNN_GET_ERROR_CODE(error));
        return error;
      }
      if (sub_event_data.type == QNN_PROFILE_EVENTTYPE_NODE &&
          (sub_event_data.unit == QNN_PROFILE_EVENTUNIT_MICROSEC ||
           sub_event_data.unit == QNN_PROFILE_EVENTUNIT_CYCLES)) {
        if (snapshot) nodes.emplace_back(sub_event_data.value,
            std::string(sub_event_data.identifier) + get_event_unit(sub_event_data.unit));
        if (dump != nullptr) {
          unsigned long long lint[3] = {0, 0, 0};
          const QnnProfile_EventId_t* lint_ptr = nullptr;
          std::uint32_t lint_n = 0;
          if (qnn_interface.qnn_profile_get_sub_events(sub_events_ptr[j], &lint_ptr, &lint_n) ==
              QNN_SUCCESS) {
            QnnProfile_EventData_t lint_data;
            for (std::uint32_t k = 0; k < lint_n; ++k) {
              if (qnn_interface.qnn_profile_get_event_data(lint_ptr[k], &lint_data) != QNN_SUCCESS)
                continue;
              if (lint_data.type == QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT) lint[0] = lint_data.value;
              if (lint_data.type == QNN_HTP_PROFILE_EVENTTYPE_NODE_OVERLAP) lint[1] = lint_data.value;
              if (lint_data.type == QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT_OVERLAP) lint[2] = lint_data.value;
            }
          }
          std::fprintf(dump, "%llu\t%u\t%u\t%llu\t%s\t%llu\t%llu\t%llu\n",
              static_cast<unsigned long long>(compact_calls), static_cast<unsigned>(event_data.type),
              static_cast<unsigned>(sub_event_data.unit),
              static_cast<unsigned long long>(sub_event_data.value), sub_event_data.identifier,
              lint[0], lint[1], lint[2]);
        }
        executorch::runtime::event_tracer_log_profiling_delegate(
            event_tracer,
            sub_event_data.identifier,
            /*delegate_debug_id=*/
            static_cast<executorch::runtime::DebugHandle>(-1),
            0,
            sub_event_data.value);
      }
    }
    if (snapshot) {
      std::sort(nodes.begin(), nodes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      // QNN_PROFILE_COMPACT_TOPN=<n> prints the n most expensive nodes per graph execute (default 24, 0 = all).
      const char* topn_env = std::getenv("QNN_PROFILE_COMPACT_TOPN");
      const size_t topn_req = topn_env ? std::strtoul(topn_env, nullptr, 10) : 24;
      const size_t topn = topn_req == 0 ? nodes.size() : std::min<size_t>(nodes.size(), topn_req);
      for (size_t n = 0; n < topn; ++n)
        QNN_EXECUTORCH_LOG_INFO("[QNN_COMPACT] node %s: %llu", nodes[n].second.c_str(),
            static_cast<unsigned long long>(nodes[n].first));
      std::map<std::string, std::pair<uint64_t, uint64_t>> totals;
      for (const auto& node : nodes) {
        // Preserve the unit: cycle totals are not wall-clock execution time.
        const auto unit = node.second.find(" (");
        auto name = node.second.substr(0, std::min({unit,
            node.second.find("_default"), node.second.find(":OpId")}));
        const auto suffix = name.find_last_not_of("0123456789");
        if (suffix != std::string::npos && name[suffix] == '_' && suffix + 1 < name.size())
          name.resize(suffix);
        const auto key = name +
            (unit == std::string::npos ? "" : node.second.substr(unit));
        auto& total = totals[key];
        ++total.first;
        total.second += node.first;
      }
      for (const auto& entry : totals) {
        QNN_EXECUTORCH_LOG_INFO("[QNN_COMPACT] aggregate %s count=%llu sum=%llu",
            entry.first.c_str(), static_cast<unsigned long long>(entry.second.first),
            static_cast<unsigned long long>(entry.second.second));
      }
    }
  }
  return error;
}

Qnn_ErrorHandle_t QnnProfile::ProfileDataToFile(
    const std::string& profile_filename) {
  if (handle_ == nullptr) {
    QNN_EXECUTORCH_LOG_WARN(
        "Profile handle is null, skipping ProfileDataToFile");
    return QNN_SUCCESS;
  }
  if (profile_filename.empty()) {
    QNN_EXECUTORCH_LOG_WARN(
        "Heap profiling path is empty. Please provide profiling filename from runtime option.");
    return QNN_SUCCESS;
  }
  const QnnProfile_EventId_t* events_ptr = nullptr;
  std::uint32_t num_events = 0;
  Qnn_ErrorHandle_t error = FetchEvents(&events_ptr, &num_events);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to profile data in function FetchEvents: %d",
        QNN_GET_ERROR_CODE(error));
    return error;
  }
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  QnnProfile_EventData_t event_data;
  std::uint32_t count_num_events = 0;
  for (std::uint32_t i = 0; i < num_events; ++i) {
    error =
        qnn_interface.qnn_profile_get_event_data(events_ptr[i], &event_data);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "ProfileData failed to get event data "
          "for event %d: %d",
          i,
          QNN_GET_ERROR_CODE(error));
      return error;
    }

    std::ios_base::openmode open_mode = std::ios::app;
    if (strcmp(event_data.identifier, DEFINE_HEAP_BEFORE_CREATION) == 0) {
      open_mode = std::ios::trunc;
    } else if (strcmp(event_data.identifier, DEFINE_HEAP_AFTER_FREED) == 0) {
      open_mode = std::ios::app;
    } else {
      count_num_events++;
      continue;
    }
    std::string identifier =
        std::string(event_data.identifier) + get_event_unit(event_data.unit);
    std::ofstream ofs(profile_filename, open_mode);
    if (!ofs) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Error when opening profile file: %s", profile_filename.c_str());
      return QNN_COMMON_ERROR_GENERAL;
    }
    ofs << identifier << ", " << event_data.value << "\n";
  }
  if (count_num_events == num_events) {
    QNN_EXECUTORCH_LOG_WARN(
        "Not HTP backend but enable htp profiling. Please check setting.");
    return QNN_SUCCESS;
  }
  return QNN_SUCCESS;
}

QnnProfile::~QnnProfile() {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  if (handle_ != nullptr) {
    Qnn_ErrorHandle_t error = qnn_interface.qnn_profile_free(handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN profile_handle. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}
} // namespace qnn
} // namespace backends
} // namespace executorch
