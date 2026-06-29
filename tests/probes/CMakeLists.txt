# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Probes are developer-driven validation binaries that need real hardware
# (NPU + mpirun) to run. They MUST stay out of ctest to avoid breaking CI on
# build hosts without NPUs.
#
# Build a probe with:
#   cmake --build build --target comm_switch_probe
#   cmake --build build --target dual_graph_probe
# (or `--target all_probes` to build every probe in this folder)
#
# Run with:
#   bash tests/probes/run_comm_switch_probe.sh --iters=50
#   bash tests/probes/run_dual_graph_probe.sh --rounds=20

if(NOT USE_NPU)
  return()
endif()

# Umbrella target so future probes can be batched. Re-using a custom target
# (not add_test) keeps these binaries off `ctest` while letting devs trigger
# the whole set with one build command.
if(NOT TARGET all_probes)
  add_custom_target(all_probes)
endif()

# Top-level CMakeLists.txt declares link_directories() against
# $ENV{NPU_HOME_PATH}/lib64 / $ENV{ATB_HOME_PATH}/lib for the production
# targets. CMake's link_directories() only applies to targets created in
# the same directory or below, so we re-state the NPU shared-lib search
# paths here -- otherwise the final ld step cannot resolve -lhccl /
# -lascendcl / -lnnopbase / -lc_sec / -lhcomm.
set(_PROBE_NPU_HOME "$ENV{NPU_HOME_PATH}")
set(_PROBE_ATB_HOME "$ENV{ATB_HOME_PATH}")
if(NOT _PROBE_NPU_HOME)
  message(FATAL_ERROR
    "NPU_HOME_PATH must be set when configuring xllm probes; "
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh first.")
endif()
if(NOT _PROBE_ATB_HOME)
  message(FATAL_ERROR
    "ATB_HOME_PATH must be set when configuring xllm probes; "
    "source /usr/local/Ascend/nnal/atb/set_env.sh first.")
endif()

# torch.so on NPU pulls in libopenblasp (libtorch_cpu transitive dep) which
# needs libgfortran from torch.libs/. The production xllm build resolves it
# via the parent torch package; we need to plumb the torch.libs directory
# onto the linker's rpath-link search path so the dep-of-dep check during
# the static link succeeds.
get_filename_component(_PROBE_TORCH_LIB_DIR
  "${TORCH_INSTALL_PREFIX}/lib" ABSOLUTE)
get_filename_component(_PROBE_TORCH_PKG_LIBS
  "${_PROBE_TORCH_LIB_DIR}/../../torch.libs" ABSOLUTE)

# Helper: apply the (verbose) NPU/ATB link configuration to a probe target.
# All probes share these settings; the only thing that differs between
# them is the source file and target name.
#
# We do NOT link ms_tools_ext (mstx profiler shim) -- production builds
# pull it in for runtime profiling, but probes don't profile anything and
# the .so location varies by CANN release. ms_tools_ext is still on the
# resolver search path via `tools/mstx/lib64` for transitive references
# from xllm_atb_layers.
function(xllm_probe_configure target)
  target_include_directories(${target}
    PRIVATE
      "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"
      ${PROJECT_SOURCE_DIR}/xllm
  )

  target_link_directories(${target}
    PRIVATE
      "${_PROBE_NPU_HOME}/lib64"
      "${_PROBE_NPU_HOME}/tools/mstx/lib64"
      "${_PROBE_ATB_HOME}/lib"
  )

  target_link_options(${target}
    PRIVATE
      "LINKER:-rpath-link=${_PROBE_NPU_HOME}/lib64"
      "LINKER:-rpath-link=${_PROBE_NPU_HOME}/tools/mstx/lib64"
      "LINKER:-rpath-link=${_PROBE_ATB_HOME}/lib"
      "-L${_PROBE_NPU_HOME}/lib64"
      "-L${_PROBE_NPU_HOME}/tools/mstx/lib64"
      "-L${_PROBE_ATB_HOME}/lib"
  )

  if(EXISTS "${_PROBE_TORCH_PKG_LIBS}")
    target_link_options(${target}
      PRIVATE
        "LINKER:-rpath-link=${_PROBE_TORCH_PKG_LIBS}"
    )
  endif()

  target_link_libraries(${target}
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

  add_dependencies(all_probes ${target})
endfunction()

add_executable(comm_switch_probe comm_switch_probe.cpp)
xllm_probe_configure(comm_switch_probe)

add_executable(dual_graph_probe dual_graph_probe.cpp)
xllm_probe_configure(dual_graph_probe)

add_executable(dual_graph_probe_atb dual_graph_probe_atb.cpp)
xllm_probe_configure(dual_graph_probe_atb)
