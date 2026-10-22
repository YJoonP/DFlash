#include "HTP/QnnHtpCommon.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include <cstring>

namespace {
bool initialized = false;
const char* names[] = {"DeltaRule", "DeltaRuleState16", "DeltaRuleDecode16", "DeltaRuleState16Masked", "DeltaRuleSnapshots16", "DeltaRuleReplay16"};
Qnn_ApiVersion_t version = QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t info = QNN_OP_PACKAGE_INFO_INIT;
Qnn_ErrorHandle_t init(QnnOpPackage_GlobalInfrastructure_t) {
  if (initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  info.packageName = "DeltaRuleOpPackage";
  info.operationNames = names;
  info.numOperations = 6;
  info.sdkBuildId = QNN_SDK_BUILD_ID;
  info.sdkApiVersion = &version;
  initialized = true;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t terminate() { initialized = false; return QNN_SUCCESS; }
Qnn_ErrorHandle_t get_info(const QnnOpPackage_Info_t** out) {
  if (!initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (!out) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  *out = &info; return QNN_SUCCESS;
}
Qnn_ErrorHandle_t validate(Qnn_OpConfig_t c) {
  if (c.version != QNN_OPCONFIG_VERSION_1 || !c.v1.packageName || !c.v1.typeName ||
      std::strcmp(c.v1.packageName, "DeltaRuleOpPackage") ||
      c.v1.numOfParams)
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  const auto is = [&](const char* type, uint32_t inputs, uint32_t outputs) {
    return std::strcmp(c.v1.typeName, type) == 0 && c.v1.numOfInputs == inputs && c.v1.numOfOutputs == outputs;
  };
  if (!is("DeltaRule", 6, 2) && !is("DeltaRuleState16", 6, 3) && !is("DeltaRuleDecode16", 6, 2) &&
      !is("DeltaRuleState16Masked", 7, 3) && !is("DeltaRuleSnapshots16", 7, 3) && !is("DeltaRuleReplay16", 3, 1))
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t create(QnnOpPackage_GraphInfrastructure_t, QnnOpPackage_Node_t, QnnOpPackage_OpImpl_t*) {
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}
Qnn_ErrorHandle_t free_op(QnnOpPackage_OpImpl_t) { return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE; }
Qnn_ErrorHandle_t log_init(QnnLog_Callback_t, QnnLog_Level_t) { return QNN_SUCCESS; }
Qnn_ErrorHandle_t log_level(QnnLog_Level_t) { return QNN_SUCCESS; }
Qnn_ErrorHandle_t log_end() { return QNN_SUCCESS; }
}
extern "C" Qnn_ErrorHandle_t DeltaRuleOpPackageInterfaceProvider(QnnOpPackage_Interface_t* p) {
  if (!p) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  p->interfaceVersion = {1,4,0};
  p->v1_4.init = init; p->v1_4.terminate = terminate; p->v1_4.getInfo = get_info;
  p->v1_4.validateOpConfig = validate; p->v1_4.createOpImpl = create; p->v1_4.freeOpImpl = free_op;
  p->v1_4.logInitialize = log_init; p->v1_4.logSetLevel = log_level; p->v1_4.logTerminate = log_end;
  return QNN_SUCCESS;
}
