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

# parallel_state already pulls in hccl / xllm_atb_layers / torch_npu through
# its own DEPS list (see core/framework/parallel_state/CMakeLists.txt). We
# still re-list ascendcl / atb / nnopbase / c_sec because cmake's transitive
# linking does not always propagate the platform-specific shared libs in this
# tree (mirrors the pattern in mapping_npu_test).
target_link_libraries(comm_switch_probe
  PRIVATE
    parallel_state
    platform
    Python::Python
    torch
    torch_npu
    torch_python
    hccl
    xllm_atb_layers
    ascendcl
    atb
    nnopbase
    c_sec
    gflags::gflags
    glog::glog
)

add_dependencies(all_probes comm_switch_probe)
