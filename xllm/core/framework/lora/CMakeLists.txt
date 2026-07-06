include(cc_library)
include(cc_test)

# Multi-tenant LoRA support.
#
#   M1 lora_config      - CLI flags + LoRAConfig struct
#   M3 lora_request     - LoRARequest immutable descriptor
#   M3 lora_registry    - name<->int_id map + refcount drain semantics
#   M2 adapter_loader   - PEFT (adapter_config.json + safetensors) reader
#   MVP lora_runtime    - process singleton wiring loader + registry +
#                         "currently active" whole-block A/B tensors so
#                         forward paths can pick them up in one lookup.
#                         Replaced by proper LoraCacheManager (M4) once
#                         per-adapter and per-layer routing lands.
cc_library(
  NAME
    lora
  HDRS
    lora_config.h
    lora_request.h
    lora_registry.h
    adapter_loader.h
    lora_runtime.h
  SRCS
    lora_config.cpp
    lora_registry.cpp
    adapter_loader.cpp
    lora_runtime.cpp
  DEPS
    :state_dict
    glog::glog
    gflags::gflags
    torch
)

# Route B: raw aclrtMemcpy is used by lora_runtime.cpp to bypass the
# per-thread torch_npu opapi memcpy stream. This is a NPU-only path.
# The root CMakeLists.txt gates the ACL include on $ENV{NPU_HOME_PATH}
# which may be empty in interactive shells; explicitly attach the
# canonical CANN 8.5 include here so the target compiles in either mode.
if(USE_NPU)
  target_include_directories(lora PRIVATE
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include
  )
endif()
