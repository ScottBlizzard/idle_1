#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

extern "C" int green_v400_native_plan_envelope_open_v1(
    const char*, const char*, const char*, const char*, const char*, const char*,
    const char*, std::uint64_t, std::uint32_t, std::uint64_t*);
extern "C" int green_v400_native_plan_envelope_info_v1(
    std::uint64_t, std::uint64_t*, std::uint64_t*, std::uint32_t*,
    std::uint32_t*, std::uint32_t*, std::uint32_t*);
extern "C" int green_v400_native_plan_envelope_close_v1(std::uint64_t);
extern "C" int green_v400_native_precision_context_open_v1(
    std::uint64_t, std::uint32_t, std::uint64_t*);
extern "C" int green_v400_native_precision_context_info_v1(
    std::uint64_t, std::uint32_t*, std::uint32_t*, std::uint64_t*,
    std::uint32_t*, std::uint32_t*);
extern "C" int green_v400_native_precision_context_close_v1(std::uint64_t);
extern "C" int green_v400_native_dispatch_concurrency_reset_v1();
extern "C" int green_v400_native_dispatch_concurrency_info_v1(
    std::uint64_t*, std::uint32_t*, std::uint32_t*);
extern "C" int green_v400_native_audit_after_find_hook_enable_v1();
extern "C" int green_v400_native_audit_after_find_hook_reached_v1();
extern "C" int green_v400_native_audit_after_find_hook_release_v1();
extern "C" int green_v400_native_precision_context_dispatch_cell_v1(
    std::uint64_t, const char*, std::int64_t, const char*, std::int64_t,
    char*, std::uint64_t);

namespace {

constexpr char kDescriptorSha[] =
    "bc673467ac237e59e542634d38d02b8eaa12053cbb0abfc39e4dcaa6659ba3ee";
constexpr char kProgramSha[] =
    "38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62";
constexpr char kDispatchSha[] =
    "eb4c907ab4a86f3aac2fda445deed67099f2831c41e9712463688cccf1b6f008";
constexpr char kBlobSha[] =
    "34bcd45371c08720c23f66d8f723dfc0249779e9e47eee5499c04d6064dc3560";
constexpr char kFusionSha[] =
    "bd734f457bd3baee252af47f1c048dbd606ec15bf6a1b6533751c7bb943319c1";
constexpr std::uint64_t kBlobNbytes = 28517632ULL;
constexpr std::uint32_t kFusionWeights = 768U;
constexpr std::uint32_t kPrecisionBits = 384U;

struct AuditResult {
  int context_info_status = -999;
  int context_open_status = -999;
  bool dispatch_done_before_release = false;
  int dispatch_status = -999;
  int hook_enable_status = -999;
  int hook_reached_status = -999;
  int hook_release_status = -999;
  std::uint64_t metric_dispatch_entries = 0;
  std::uint32_t metric_active_dispatches = 0;
  int metric_info_status = -999;
  std::uint32_t metric_peak_dispatches = 0;
  int metric_reset_status = -999;
  int plan_close_status = -999;
  int plan_info_status = -999;
  int plan_open_status = -999;
  int post_close_info_status = -999;
  int context_close_status = -999;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int emit(const AuditResult& result, bool pass, const char* status) {
  // Keys are emitted in bytewise lexicographic order.  The document contains
  // only protocol/lifecycle evidence and deliberately contains no cell output.
  std::printf(
      "{\"audit_only_backend_build\":true,"
      "\"close_completed_before_hook_release\":%s,"
      "\"contains_scientific_outcome\":false,"
      "\"context_close_status\":%d,"
      "\"context_info_status\":%d,"
      "\"context_open_status\":%d,"
      "\"dispatch_done_before_release\":%s,"
      "\"dispatch_status\":%d,"
      "\"hook_enable_status\":%d,"
      "\"hook_reached_status\":%d,"
      "\"hook_release_status\":%d,"
      "\"metric_active_dispatches\":%u,"
      "\"metric_dispatch_entries\":%llu,"
      "\"metric_info_status\":%d,"
      "\"metric_peak_dispatches\":%u,"
      "\"metric_reset_status\":%d,"
      "\"pass\":%s,"
      "\"plan_close_status\":%d,"
      "\"plan_info_status\":%d,"
      "\"plan_open_status\":%d,"
      "\"post_close_info_status\":%d,"
      "\"schema_version\":\"green-v400-native-close-first-tsan-v1\","
      "\"scientific_threshold_applied\":false,"
      "\"status\":\"%s\"}\n",
      json_bool(result.context_close_status == 0 &&
                result.hook_release_status != -999),
      result.context_close_status, result.context_info_status,
      result.context_open_status,
      json_bool(result.dispatch_done_before_release), result.dispatch_status,
      result.hook_enable_status, result.hook_reached_status,
      result.hook_release_status, result.metric_active_dispatches,
      static_cast<unsigned long long>(result.metric_dispatch_entries),
      result.metric_info_status, result.metric_peak_dispatches,
      result.metric_reset_status, json_bool(pass), result.plan_close_status,
      result.plan_info_status, result.plan_open_status,
      result.post_close_info_status, status);
  std::fflush(stdout);
  return pass ? 0 : 2;
}

[[noreturn]] void emit_and_exit(const AuditResult& result,
                                const char* status) {
  emit(result, false, status);
  std::_Exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  AuditResult result;
  if (argc != 3) {
    return emit(result, false, "FAIL_USAGE_EXPECT_DESCRIPTOR_AND_BLOB");
  }

  std::uint64_t plan_handle = 0;
  result.plan_open_status = green_v400_native_plan_envelope_open_v1(
      argv[1], argv[2], kDescriptorSha, kProgramSha, kDispatchSha, kBlobSha,
      kFusionSha, kBlobNbytes, kFusionWeights, &plan_handle);
  if (result.plan_open_status != 0 || plan_handle == 0) {
    return emit(result, false, "FAIL_PLAN_OPEN");
  }

  std::uint64_t descriptor_nbytes = 0;
  std::uint64_t blob_nbytes = 0;
  std::uint32_t record_count = 0;
  std::uint32_t node_count = 0;
  std::uint32_t binding_count = 0;
  std::uint32_t fusion_weight_count = 0;
  result.plan_info_status = green_v400_native_plan_envelope_info_v1(
      plan_handle, &descriptor_nbytes, &blob_nbytes, &record_count, &node_count,
      &binding_count, &fusion_weight_count);
  if (result.plan_info_status != 0 || descriptor_nbytes == 0 ||
      blob_nbytes != kBlobNbytes || record_count != 32 || node_count != 81 ||
      binding_count != 150 || fusion_weight_count != kFusionWeights) {
    result.plan_close_status =
        green_v400_native_plan_envelope_close_v1(plan_handle);
    return emit(result, false, "FAIL_PLAN_INFO");
  }

  std::uint64_t context_handle = 0;
  result.context_open_status = green_v400_native_precision_context_open_v1(
      plan_handle, kPrecisionBits, &context_handle);
  if (result.context_open_status != 0 || context_handle == 0) {
    result.plan_close_status =
        green_v400_native_plan_envelope_close_v1(plan_handle);
    return emit(result, false, "FAIL_CONTEXT_OPEN");
  }

  std::uint32_t precision_bits = 0;
  std::uint32_t static_buffer_count = 0;
  std::uint64_t static_jet_count = 0;
  std::uint32_t context_node_count = 0;
  std::uint32_t context_binding_count = 0;
  result.context_info_status = green_v400_native_precision_context_info_v1(
      context_handle, &precision_bits, &static_buffer_count, &static_jet_count,
      &context_node_count, &context_binding_count);
  if (result.context_info_status != 0 || precision_bits != kPrecisionBits ||
      static_buffer_count != 5 || static_jet_count == 0 ||
      context_node_count != 81 || context_binding_count != 150) {
    result.context_close_status =
        green_v400_native_precision_context_close_v1(context_handle);
    result.plan_close_status =
        green_v400_native_plan_envelope_close_v1(plan_handle);
    return emit(result, false, "FAIL_CONTEXT_INFO");
  }

  result.metric_reset_status =
      green_v400_native_dispatch_concurrency_reset_v1();
  result.hook_enable_status =
      green_v400_native_audit_after_find_hook_enable_v1();
  if (result.metric_reset_status != 0 || result.hook_enable_status != 0) {
    result.context_close_status =
        green_v400_native_precision_context_close_v1(context_handle);
    result.plan_close_status =
        green_v400_native_plan_envelope_close_v1(plan_handle);
    return emit(result, false, "FAIL_HOOK_SETUP");
  }

  std::atomic<bool> dispatch_done{false};
  std::atomic<int> dispatch_status{-999};
  std::thread dispatch_thread([&] {
    std::array<char, 1024> output{};
    dispatch_status.store(green_v400_native_precision_context_dispatch_cell_v1(
                              context_handle, "-1", -14, "0", 0,
                              output.data(), output.size()),
                          std::memory_order_release);
    dispatch_done.store(true, std::memory_order_release);
  });

  const auto reached_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (true) {
    result.hook_reached_status =
        green_v400_native_audit_after_find_hook_reached_v1();
    if (result.hook_reached_status == 1) break;
    if (dispatch_done.load(std::memory_order_acquire)) {
      dispatch_thread.join();
      result.dispatch_status = dispatch_status.load(std::memory_order_acquire);
      result.context_close_status =
          green_v400_native_precision_context_close_v1(context_handle);
      result.plan_close_status =
          green_v400_native_plan_envelope_close_v1(plan_handle);
      return emit(result, false, "FAIL_DISPATCH_BYPASSED_HOOK");
    }
    if (result.hook_reached_status != 0 ||
        std::chrono::steady_clock::now() >= reached_deadline) {
      // A join is not safe if the worker is or later becomes blocked in the
      // hook.  Terminate this failed evidence process; the outer timeout is an
      // additional independent liveness bound.
      emit_and_exit(result, "FAIL_HOOK_NOT_REACHED");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // This call is intentionally synchronous.  Returning before the release
  // below is the close-first property under test.
  result.context_close_status =
      green_v400_native_precision_context_close_v1(context_handle);
  result.dispatch_done_before_release =
      dispatch_done.load(std::memory_order_acquire);
  result.hook_release_status =
      green_v400_native_audit_after_find_hook_release_v1();
  if (result.hook_release_status != 0) {
    emit_and_exit(result, "FAIL_HOOK_RELEASE");
  }

  dispatch_thread.join();
  result.dispatch_status = dispatch_status.load(std::memory_order_acquire);
  result.metric_info_status = green_v400_native_dispatch_concurrency_info_v1(
      &result.metric_dispatch_entries, &result.metric_active_dispatches,
      &result.metric_peak_dispatches);

  std::uint32_t stale_precision = 0;
  result.post_close_info_status = green_v400_native_precision_context_info_v1(
      context_handle, &stale_precision, nullptr, nullptr, nullptr, nullptr);
  result.plan_close_status =
      green_v400_native_plan_envelope_close_v1(plan_handle);

  const bool pass =
      result.context_close_status == 0 &&
      !result.dispatch_done_before_release && result.dispatch_status == 2 &&
      result.metric_info_status == 0 && result.metric_dispatch_entries == 0 &&
      result.metric_active_dispatches == 0 &&
      result.metric_peak_dispatches == 0 &&
      result.post_close_info_status == 2 && result.plan_close_status == 0;
  return emit(result, pass,
              pass ? "PASS_CLOSE_FIRST_PRELOCK_WAITER"
                   : "FAIL_CLOSE_FIRST_CONTRACT");
}
