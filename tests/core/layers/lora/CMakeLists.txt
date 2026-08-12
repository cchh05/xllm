include(cc_test)

if(USE_NPU)
  cc_test(
    NAME
      lora_grouped_matmul_test
    SRCS
      lora_grouped_matmul_test.cpp
    DEPS
      :common_layers
      :lora
      :kernels
      torch
      glog::glog
      GTest::gtest_main
  )

  target_compile_definitions(lora_grouped_matmul_test PRIVATE
                             XLLM_LORA_ENABLE_TEST_HOOKS
                             USE_NPU)

  target_link_libraries(lora_grouped_matmul_test PRIVATE
                        ascendcl
                        hccl
                        c_sec
                        nnopbase
                        atb)

  target_link_options(lora_grouped_matmul_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")
endif()
