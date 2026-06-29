# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Probes are developer-driven validation binaries that need real hardware
# (NPU + mpirun) to run. They MUST stay out of ctest to avoid breaking CI on
# build hosts without NPUs.
#
# Build a probe with:
#   cmake --build build --target comm_switch_probe
# (or `--target all_probes` to build every probe in this folder)
#
# Run with:
#   mpirun -n 4 ./build/tests/probes/comm_switch_probe \
#       --rank_tablefile=/path/to/82_4card.json --iters=50

if(NOT USE_NPU)
  return()
endif()

# Umbrella target so future probes can be batched. Re-using a custom target
# (not add_test) keeps these binaries off `ctest` while letting devs trigger
# the whole set with one build command.
if(NOT TARGET all_probes)
  add_custom_target(all_probes)
endif()

add_executable(comm_switch_probe comm_switch_probe.cpp)

target_include_directories(comm_switch_probe
  PRIVATE
    "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"
    ${PROJECT_SOURCE_DIR}/xllm
)

# Top-level CMakeLists.txt declares link_directories() against
# $ENV{NPU_HOME_PATH}/lib64 / $ENV{ATB_HOME_PATH}/lib for the production
# targets. CMake's link_directories() only applies to targets created in
# the same directory or below, so we re-state the NPU shared-lib search
# paths here -- otherwise the final ld step cannot resolve -lhccl /
# -lascendcl / -lnnopbase / -lc_sec / -lhcomm / -lms_tools_ext.
set(_PROBE_NPU_HOME "$ENV{NPU_HOME_PATH}")
set(_PROBE_ATB_HOME "$ENV{ATB_HOME_PATH}")
if(NOT _PROBE_NPU_HOME)
  message(FATAL_ERROR
    "NPU_HOME_PATH must be set when configuring comm_switch_probe; "
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh first.")
endif()
if(NOT _PROBE_ATB_HOME)
  message(FATAL_ERROR
    "ATB_HOME_PATH must be set when configuring comm_switch_probe; "
    "source /usr/local/Ascend/nnal/atb/set_env.sh first.")
endif()
target_link_directories(comm_switch_probe
  PRIVATE
    "${_PROBE_NPU_HOME}/lib64"
    "${_PROBE_NPU_HOME}/tools/mstx/lib64"
    "${_PROBE_ATB_HOME}/lib"
)

# xllm_atb_layers (and its transitive deps) declare bare "-lhccl"/"-lascendcl"
# style INTERFACE link entries, so we must also push the cann lib64 onto
# the linker's library search path. target_link_directories(PRIVATE) above
# adds it for THIS target, but the bare names from the IMPORTED interface
# get re-emitted at the end of the link line where the search dirs may
# already have been forgotten -- belt-and-braces with -Wl,-rpath-link to
# keep the resolver happy through the whole link. ms_tools_ext (mstx
# profiler) is bundled under tools/mstx/lib64 in CANN 8.5+, NOT lib64.
target_link_options(comm_switch_probe
  PRIVATE
    "LINKER:-rpath-link=${_PROBE_NPU_HOME}/lib64"
    "LINKER:-rpath-link=${_PROBE_NPU_HOME}/tools/mstx/lib64"
    "LINKER:-rpath-link=${_PROBE_ATB_HOME}/lib"
    "-L${_PROBE_NPU_HOME}/lib64"
    "-L${_PROBE_NPU_HOME}/tools/mstx/lib64"
    "-L${_PROBE_ATB_HOME}/lib"
)

# torch.so on NPU pulls in libopenblasp (libtorch_cpu transitive dependency)
# which in turn needs libgfortran from torch.libs/. The production xllm
# build resolves it via the parent torch package; we need to plumb the
# torch.libs directory onto the linker's rpath-link search path so the
# dependency-of-dependency check during the static link succeeds.
get_filename_component(_PROBE_TORCH_LIB_DIR
  "${TORCH_INSTALL_PREFIX}/lib" ABSOLUTE)
get_filename_component(_PROBE_TORCH_PKG_LIBS
  "${_PROBE_TORCH_LIB_DIR}/../../torch.libs" ABSOLUTE)
if(EXISTS "${_PROBE_TORCH_PKG_LIBS}")
  target_link_options(comm_switch_probe
    PRIVATE
      "LINKER:-rpath-link=${_PROBE_TORCH_PKG_LIBS}"
  )
endif()

# parallel_state already pulls in hccl / xllm_atb_layers / torch_npu through
# its own DEPS list (see core/framework/parallel_state/CMakeLists.txt). We
# still re-list ascendcl / atb / nnopbase / c_sec because cmake's transitive
# linking does not always propagate the platform-specific shared libs in this
# tree (mirrors the pattern in mapping_npu_test). The absolute paths via
# ${_PROBE_NPU_HOME} make the link step independent of the directory-level
# link_directories() in the top-level CMakeLists.txt, which does not
# propagate into nested add_executable targets.
#
# We do NOT link ms_tools_ext (mstx profiler shim) here -- the production
# build pulls it in for runtime profiling, but the probe doesn't profile
# anything and the .so location varies by CANN release. Skipping it keeps
# the probe's link command portable across boxes.
target_link_libraries(comm_switch_probe
  PRIVATE
    parallel_state
    platform
    Python::Python
    torch
    torch_npu
    torch_python
    "${_PROBE_NPU_HOME}/lib64/libhccl.so"
    "${_PROBE_NPU_HOME}/lib64/libhcomm.so"
    xllm_atb_layers
    "${_PROBE_NPU_HOME}/lib64/libascendcl.so"
    "${_PROBE_ATB_HOME}/lib/libatb.so"
    "${_PROBE_NPU_HOME}/lib64/libnnopbase.so"
    "${_PROBE_NPU_HOME}/lib64/libc_sec.so"
    gflags::gflags
    glog::glog
)

add_dependencies(all_probes comm_switch_probe)
