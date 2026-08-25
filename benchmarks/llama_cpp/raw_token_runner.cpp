#include "llama.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>

#if !defined(O_NOFOLLOW) || !defined(O_CLOEXEC)
#error "This runner requires POSIX O_NOFOLLOW and O_CLOEXEC support"
#endif

namespace {

constexpr std::array<llama_token, 13> kPromptTokens = {
    248045, 846, 198,    9419, 248046, 198, 248045,
    74455,  198, 248068, 271,  248069, 271,
};
constexpr uint32_t kContextSize = 142;
constexpr uint32_t kBatchSize = 13;
constexpr uint32_t kMicroBatchSize = 13;
constexpr uint32_t kSequenceCount = 1;
constexpr int kGeneratedTokenCount = 128;
constexpr int32_t kVocabularySize = 248320;

using Clock = std::chrono::steady_clock;

struct Options {
  std::filesystem::path model_path;
  std::string gpu_device;
  int32_t n_gpu_layers = 0;
  int32_t n_threads = 4;
};

struct BackendGuard {
  BackendGuard() { llama_backend_init(); }
  ~BackendGuard() { llama_backend_free(); }

  BackendGuard(const BackendGuard &) = delete;
  BackendGuard &operator=(const BackendGuard &) = delete;
};

struct ModelDeleter {
  void operator()(llama_model *value) const noexcept {
    llama_model_free(value);
  }
};

struct ContextDeleter {
  void operator()(llama_context *value) const noexcept { llama_free(value); }
};

struct SamplerDeleter {
  void operator()(llama_sampler *value) const noexcept {
    llama_sampler_free(value);
  }
};

struct FileDeleter {
  void operator()(FILE *value) const noexcept { std::fclose(value); }
};

class UniqueFd {
public:
  explicit UniqueFd(int value) : value_(value) {}
  ~UniqueFd() {
    if (value_ >= 0) {
      ::close(value_);
    }
  }

  UniqueFd(const UniqueFd &) = delete;
  UniqueFd &operator=(const UniqueFd &) = delete;

  int get() const { return value_; }
  int release() {
    const int value = value_;
    value_ = -1;
    return value;
  }

private:
  int value_;
};

struct FileIdentity {
  uint64_t device;
  uint64_t inode;
  uint64_t size_bytes;
  uint64_t hard_link_count;
  int64_t change_time_seconds;
  int64_t change_time_nanoseconds;
};

struct MemoryTotals {
  uint64_t model = 0;
  uint64_t context = 0;
  uint64_t compute = 0;

  uint64_t total() const { return model + context + compute; }
};

struct PlacementAttestation {
  int32_t layer_count = 0;
  int32_t layers_on_selected_device = 0;
  int32_t layers_on_cpu = 0;
  bool output_on_selected_device = false;
  bool output_on_cpu = false;
  int32_t model_selected_device_count = 0;
  MemoryTotals gpu;
  MemoryTotals cpu;
  MemoryTotals accelerator;
  MemoryTotals other;
  llama_memory_breakdown memory_breakdown;
};

struct ExecutionProof {
  ggml_backend_sched_t sched = nullptr;
  ggml_backend_dev_t selected_gpu = nullptr;
  ggml_backend_dev_t cpu = nullptr;
  std::array<bool, 24> completed_layers{};
  bool completed_input_embedding = false;
  bool completed_output = false;
  bool duplicate_or_unexpected_callback = false;
  bool backend_mismatch = false;
  size_t requested_sentinels = 0;
  size_t completed_sentinels = 0;
  size_t completed_on_selected_gpu = 0;
  size_t completed_on_cpu = 0;
  int64_t elapsed_ns = 0;
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;
using FilePtr = std::unique_ptr<FILE, FileDeleter>;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void reject_environment_variable(const char *name) {
  if (std::getenv(name) != nullptr) {
    fail(std::string(name) +
         " must be absent because this runner permits linked static backends "
         "only");
  }
}

std::string json_escape(std::string_view input) {
  std::ostringstream out;
  for (const char character : input) {
    const auto value = static_cast<unsigned char>(character);
    switch (value) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (value < 0x20) {
        out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<unsigned int>(value) << std::dec;
      } else {
        out << static_cast<char>(value);
      }
    }
  }
  return out.str();
}

template <typename Integer>
Integer parse_integer(std::string_view text, std::string_view option_name) {
  Integer value{};
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end) {
    fail("invalid integer for " + std::string(option_name) + ": " +
         std::string(text));
  }
  return value;
}

void print_usage(const char *program) {
  std::fprintf(stderr,
               "Usage: %s --model MODEL.gguf --gpu-layers {0|-1} "
               "[--gpu-device NAME] [--threads N]\n"
               "       %s MODEL.gguf --gpu-layers {0|-1} "
               "[--gpu-device NAME] [--threads N]\n",
               program, program);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto require_value = [&](std::string_view option_name) -> std::string_view {
      if (index + 1 >= argc) {
        fail("missing value for " + std::string(option_name));
      }
      return argv[++index];
    };

    if (argument == "--model" || argument == "-m") {
      options.model_path = require_value(argument);
    } else if (argument == "--gpu-layers" || argument == "-ngl") {
      options.n_gpu_layers =
          parse_integer<int32_t>(require_value(argument), argument);
    } else if (argument == "--threads" || argument == "-t") {
      options.n_threads =
          parse_integer<int32_t>(require_value(argument), argument);
    } else if (argument == "--gpu-device") {
      options.gpu_device = require_value(argument);
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (!argument.empty() && argument.front() == '-') {
      fail("unknown option: " + std::string(argument));
    } else if (options.model_path.empty()) {
      options.model_path = argument;
    } else {
      fail("unexpected positional argument: " + std::string(argument));
    }
  }

  if (options.model_path.empty()) {
    fail("--model is required");
  }
  if (options.n_threads <= 0) {
    fail("--threads must be positive");
  }
  if (options.n_gpu_layers != 0 && options.n_gpu_layers != -1) {
    fail("--gpu-layers must be exactly 0 (CPU) or -1 (all layers)");
  }
  if (options.n_gpu_layers == -1 && options.gpu_device.empty()) {
    fail("--gpu-device is required when --gpu-layers is -1");
  }
  if (options.n_gpu_layers == 0 && !options.gpu_device.empty()) {
    fail("--gpu-device is forbidden when --gpu-layers is 0");
  }
  return options;
}

int64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

FileIdentity capture_file_identity(int fd) {
  struct stat attributes {};
  if (::fstat(fd, &attributes) != 0) {
    fail("fstat on pinned model failed: " + std::string(std::strerror(errno)));
  }
  if (!S_ISREG(attributes.st_mode)) {
    fail("model must be a regular file");
  }
  if (attributes.st_nlink != 1) {
    fail("model must have exactly one hard link");
  }
  if (attributes.st_size <= 0) {
    fail("model file must be non-empty");
  }

#if defined(__APPLE__)
  const auto change_time_seconds = attributes.st_ctimespec.tv_sec;
  const auto change_time_nanoseconds = attributes.st_ctimespec.tv_nsec;
#else
  const auto change_time_seconds = attributes.st_ctim.tv_sec;
  const auto change_time_nanoseconds = attributes.st_ctim.tv_nsec;
#endif

  return {
      static_cast<uint64_t>(attributes.st_dev),
      static_cast<uint64_t>(attributes.st_ino),
      static_cast<uint64_t>(attributes.st_size),
      static_cast<uint64_t>(attributes.st_nlink),
      static_cast<int64_t>(change_time_seconds),
      static_cast<int64_t>(change_time_nanoseconds),
  };
}

void require_same_file_identity(const FileIdentity &expected,
                                const FileIdentity &actual,
                                std::string_view phase) {
  if (expected.device != actual.device || expected.inode != actual.inode ||
      expected.size_bytes != actual.size_bytes ||
      expected.hard_link_count != actual.hard_link_count ||
      expected.change_time_seconds != actual.change_time_seconds ||
      expected.change_time_nanoseconds != actual.change_time_nanoseconds) {
    fail("pinned model identity changed " + std::string(phase));
  }
}

void append_file_identity(std::ostringstream &out,
                          const FileIdentity &identity) {
  out << "{\"device\":" << identity.device << ",\"inode\":" << identity.inode
      << ",\"size_bytes\":" << identity.size_bytes
      << ",\"hard_link_count\":" << identity.hard_link_count
      << ",\"change_time_seconds\":" << identity.change_time_seconds
      << ",\"change_time_nanoseconds\":" << identity.change_time_nanoseconds
      << '}';
}

std::string backend_type_name(enum ggml_backend_dev_type type) {
  switch (type) {
  case GGML_BACKEND_DEVICE_TYPE_CPU:
    return "cpu";
  case GGML_BACKEND_DEVICE_TYPE_GPU:
    return "gpu";
  case GGML_BACKEND_DEVICE_TYPE_IGPU:
    return "integrated-gpu";
  case GGML_BACKEND_DEVICE_TYPE_ACCEL:
    return "accelerator";
  case GGML_BACKEND_DEVICE_TYPE_META:
    return "meta";
  }
  return "unknown";
}

bool is_gpu_device(ggml_backend_dev_t device) {
  if (device == nullptr) {
    return false;
  }
  const auto type = ggml_backend_dev_type(device);
  return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
         type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

bool is_cpu_device(ggml_backend_dev_t device) {
  return device != nullptr &&
         ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

ggml_backend_dev_t find_unique_gpu_device(std::string_view expected_name) {
  ggml_backend_dev_t result = nullptr;
  size_t matches = 0;
  for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    const char *name = ggml_backend_dev_name(device);
    if (is_gpu_device(device) && name != nullptr && expected_name == name) {
      result = device;
      ++matches;
    }
  }
  if (matches != 1) {
    fail("--gpu-device must identify exactly one registered GPU device: " +
         std::string(expected_name));
  }
  return result;
}

void append_backend_device(std::ostringstream &out,
                           ggml_backend_dev_t device) {
  if (device == nullptr) {
    out << "null";
    return;
  }
  ggml_backend_dev_props properties{};
  ggml_backend_dev_get_props(device, &properties);
  out << "{\"name\":\""
      << json_escape(properties.name == nullptr ? "" : properties.name) << '"'
      << ",\"description\":\""
      << json_escape(properties.description == nullptr ? ""
                                                        : properties.description)
      << '"' << ",\"type\":\"" << backend_type_name(properties.type) << '"'
      << ",\"type_code\":" << static_cast<int>(properties.type)
      << ",\"device_id\":";
  if (properties.device_id == nullptr) {
    out << "null";
  } else {
    out << '"' << json_escape(properties.device_id) << '"';
  }
  out << '}';
}

void add_memory(MemoryTotals &target,
                const llama_memory_breakdown_data &source) {
  target.model += static_cast<uint64_t>(source.model);
  target.context += static_cast<uint64_t>(source.context);
  target.compute += static_cast<uint64_t>(source.compute);
}

PlacementAttestation attest_placement(const llama_model *model,
                                      const llama_context *context,
                                      ggml_backend_dev_t selected_gpu,
                                      bool gpu_lane) {
  PlacementAttestation result;
  if (model->tok_embd == nullptr || model->tok_embd->buffer == nullptr) {
    fail("model input embedding has no allocated backend buffer");
  }
  result.layer_count = llama_model_n_layer(model);
  result.model_selected_device_count = llama_model_n_devices(model);
  for (int32_t layer = 0; layer < result.layer_count; ++layer) {
    ggml_backend_dev_t device = model->dev_layer(layer);
    if (device == selected_gpu && selected_gpu != nullptr) {
      ++result.layers_on_selected_device;
    }
    if (is_cpu_device(device)) {
      ++result.layers_on_cpu;
    }
  }
  const ggml_backend_dev_t output_device = model->dev_output();
  result.output_on_selected_device =
      selected_gpu != nullptr && output_device == selected_gpu;
  result.output_on_cpu = is_cpu_device(output_device);

  result.memory_breakdown = llama_get_memory_breakdown(context);
  for (const auto &[buffer_type, memory] : result.memory_breakdown) {
    ggml_backend_dev_t device = ggml_backend_buft_get_device(buffer_type);
    if (device == nullptr || is_cpu_device(device)) {
      add_memory(result.cpu, memory);
      continue;
    }
    if (is_gpu_device(device)) {
      if (memory.total() != 0 && selected_gpu != nullptr &&
          device != selected_gpu) {
        fail("memory was allocated on an unselected GPU backend");
      }
      add_memory(result.gpu, memory);
      continue;
    }
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
      add_memory(result.accelerator, memory);
    } else {
      add_memory(result.other, memory);
    }
  }

  if (gpu_lane) {
    if (selected_gpu == nullptr || result.model_selected_device_count != 1 ||
        llama_model_get_device(model, 0) != selected_gpu ||
        result.layers_on_selected_device != result.layer_count ||
        !result.output_on_selected_device || result.gpu.model == 0 ||
        result.gpu.context == 0 || result.gpu.compute == 0) {
      fail("GPU lane did not place every transformer/output layer and active "
           "model/context/compute buffers on the selected GPU");
    }
  } else if (result.model_selected_device_count != 0 ||
             result.layers_on_cpu != result.layer_count ||
             !result.output_on_cpu || result.gpu.total() != 0 ||
             result.cpu.model == 0 || result.cpu.context == 0 ||
             result.cpu.compute == 0) {
    fail("CPU lane placement attestation found non-CPU execution state");
  }
  if (result.other.total() != 0) {
    fail("placement attestation found memory on an unknown backend type");
  }
  return result;
}

int execution_proof_layer_index(std::string_view name) noexcept {
  constexpr std::string_view prefix = "l_out-";
  if (name.size() <= prefix.size() || name.substr(0, prefix.size()) != prefix) {
    return -1;
  }
  int value = -1;
  const char *begin = name.data() + prefix.size();
  const char *end = name.data() + name.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end || value < 0 ||
      value >= 24) {
    return -1;
  }
  return value;
}

bool execution_proof_callback(ggml_tensor *tensor, bool ask,
                              void *user_data) noexcept {
  auto *proof = static_cast<ExecutionProof *>(user_data);
  const char *raw_name = ggml_get_name(tensor);
  const std::string_view name = raw_name == nullptr ? "" : raw_name;
  const int layer_index = execution_proof_layer_index(name);
  const bool is_input = name == "model.input_embed";
  const bool is_output = name == "result_output";
  const bool is_sentinel = is_input || is_output || layer_index >= 0;
  if (ask) {
    if (is_sentinel) {
      ++proof->requested_sentinels;
    }
    return is_sentinel;
  }
  if (!is_sentinel) {
    proof->duplicate_or_unexpected_callback = true;
    return true;
  }

  ++proof->completed_sentinels;
  ggml_backend_t backend =
      ggml_backend_sched_get_tensor_backend(proof->sched, tensor);
  ggml_backend_dev_t device =
      backend == nullptr ? nullptr : ggml_backend_get_device(backend);
  if (device == proof->selected_gpu && proof->selected_gpu != nullptr) {
    ++proof->completed_on_selected_gpu;
  } else if (device == proof->cpu) {
    ++proof->completed_on_cpu;
  }

  const ggml_backend_dev_t expected_device =
      is_input || proof->selected_gpu == nullptr ? proof->cpu
                                                 : proof->selected_gpu;
  if (device != expected_device) {
    proof->backend_mismatch = true;
  }
  if (is_input) {
    if (proof->completed_input_embedding) {
      proof->duplicate_or_unexpected_callback = true;
    }
    proof->completed_input_embedding = true;
  } else if (is_output) {
    if (proof->completed_output) {
      proof->duplicate_or_unexpected_callback = true;
    }
    proof->completed_output = true;
  } else {
    const size_t index = static_cast<size_t>(layer_index);
    if (proof->completed_layers[index]) {
      proof->duplicate_or_unexpected_callback = true;
    }
    proof->completed_layers[index] = true;
  }
  return true;
}

ExecutionProof run_post_measurement_execution_proof(
    llama_context *context, llama_token proof_token,
    ggml_backend_dev_t selected_gpu) {
  ExecutionProof proof;
  proof.sched = context->get_sched();
  proof.selected_gpu = selected_gpu;
  proof.cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (proof.sched == nullptr || proof.cpu == nullptr) {
    fail("post-measurement execution proof requires scheduler and CPU device");
  }

  llama_batch proof_batch = llama_batch_get_one(&proof_token, 1);
  const auto proof_start = Clock::now();
  ggml_backend_sched_set_eval_callback(proof.sched, execution_proof_callback,
                                       &proof);
  const int32_t decode_result = llama_decode(context, proof_batch);
  ggml_backend_sched_set_eval_callback(proof.sched, nullptr, nullptr);
  const auto proof_end = Clock::now();
  proof.elapsed_ns = elapsed_ns(proof_start, proof_end);
  if (decode_result != 0) {
    fail("post-measurement execution proof decode failed with code " +
         std::to_string(decode_result));
  }

  const bool all_layers =
      std::all_of(proof.completed_layers.begin(), proof.completed_layers.end(),
                  [](bool value) { return value; });
  const size_t expected_gpu_sentinels = selected_gpu == nullptr ? 0 : 25;
  const size_t expected_cpu_sentinels = selected_gpu == nullptr ? 26 : 1;
  if (proof.requested_sentinels != 26 || proof.completed_sentinels != 26 ||
      !all_layers || !proof.completed_input_embedding ||
      !proof.completed_output || proof.duplicate_or_unexpected_callback ||
      proof.backend_mismatch ||
      proof.completed_on_selected_gpu != expected_gpu_sentinels ||
      proof.completed_on_cpu != expected_cpu_sentinels ||
      proof.elapsed_ns <= 0) {
    fail("post-measurement execution proof did not complete every expected "
         "CPU/Metal sentinel on its required backend");
  }
  return proof;
}

void append_memory_totals(std::ostringstream &out, const MemoryTotals &value) {
  out << "{\"model_bytes\":" << value.model
      << ",\"context_bytes\":" << value.context
      << ",\"compute_bytes\":" << value.compute
      << ",\"total_bytes\":" << value.total() << '}';
}

void append_memory_breakdown(std::ostringstream &out,
                             const llama_memory_breakdown &breakdown) {
  out << '[';
  size_t index = 0;
  for (const auto &[buffer_type, memory] : breakdown) {
    if (index++ != 0) {
      out << ',';
    }
    const ggml_backend_dev_t device =
        ggml_backend_buft_get_device(buffer_type);
    const char *buffer_name = ggml_backend_buft_name(buffer_type);
    out << "{\"buffer_type\":\""
        << json_escape(buffer_name == nullptr ? "" : buffer_name) << '"'
        << ",\"device\":";
    append_backend_device(out, device);
    out << ",\"model_bytes\":" << memory.model
        << ",\"context_bytes\":" << memory.context
        << ",\"compute_bytes\":" << memory.compute
        << ",\"total_bytes\":" << memory.total() << '}';
  }
  out << ']';
}

std::string model_description(const llama_model *model) {
  std::vector<char> buffer(512, '\0');
  int32_t result = llama_model_desc(model, buffer.data(), buffer.size());
  if (result < 0) {
    return "unavailable";
  }
  if (static_cast<size_t>(result) >= buffer.size()) {
    buffer.assign(static_cast<size_t>(result) + 1, '\0');
    result = llama_model_desc(model, buffer.data(), buffer.size());
    if (result < 0) {
      return "unavailable";
    }
  }
  return buffer.data();
}

template <typename Container>
void append_token_array(std::ostringstream &out, const Container &values) {
  out << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << values[index];
  }
  out << ']';
}

void append_backend_devices(std::ostringstream &out) {
  out << '[';
  const size_t count = ggml_backend_dev_count();
  for (size_t index = 0; index < count; ++index) {
    if (index != 0) {
      out << ',';
    }
    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    out << "{\"index\":" << index << ",\"name\":\""
        << json_escape(properties.name == nullptr ? "" : properties.name) << '"'
        << ",\"description\":\""
        << json_escape(
               properties.description == nullptr ? "" : properties.description)
        << '"' << ",\"type\":\"" << backend_type_name(properties.type) << '"'
        << ",\"type_code\":" << static_cast<int>(properties.type)
        << ",\"device_id\":";
    if (properties.device_id == nullptr) {
      out << "null";
    } else {
      out << '"' << json_escape(properties.device_id) << '"';
    }
    out << ",\"memory_free_bytes\":" << properties.memory_free
        << ",\"memory_total_bytes\":" << properties.memory_total
        << ",\"capabilities\":{"
        << "\"async\":" << (properties.caps.async ? "true" : "false")
        << ",\"host_buffer\":"
        << (properties.caps.host_buffer ? "true" : "false")
        << ",\"buffer_from_host_ptr\":"
        << (properties.caps.buffer_from_host_ptr ? "true" : "false")
        << ",\"events\":" << (properties.caps.events ? "true" : "false")
        << "}}";
  }
  out << ']';
}

std::string run(const Options &options) {
  const auto full_start = Clock::now();
  const bool gpu_lane = options.n_gpu_layers == -1;

  std::error_code filesystem_error;
  const auto display_model_path =
      std::filesystem::absolute(options.model_path, filesystem_error)
          .lexically_normal();
  if (filesystem_error) {
    fail("cannot make model path absolute: " + filesystem_error.message());
  }

  int raw_model_fd;
  do {
    raw_model_fd = ::open(options.model_path.string().c_str(),
                          O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  } while (raw_model_fd < 0 && errno == EINTR);
  if (raw_model_fd < 0) {
    fail("open of model failed: " + std::string(std::strerror(errno)));
  }
  UniqueFd model_fd_owner(raw_model_fd);
  const FileIdentity model_identity_start =
      capture_file_identity(model_fd_owner.get());

  FILE *raw_model_file = ::fdopen(model_fd_owner.get(), "rb");
  if (raw_model_file == nullptr) {
    fail("fdopen of pinned model failed: " + std::string(std::strerror(errno)));
  }
  model_fd_owner.release();
  FilePtr model_file(raw_model_file);
  const int pinned_model_fd = ::fileno(model_file.get());

  if (ggml_backend_dev_count() == 0) {
    fail("llama.cpp registered no backend devices");
  }
  if (gpu_lane && !llama_supports_gpu_offload()) {
    fail("GPU lane requested but llama.cpp does not support GPU offload");
  }
  const ggml_backend_dev_t selected_gpu =
      gpu_lane ? find_unique_gpu_device(options.gpu_device) : nullptr;
  std::array<ggml_backend_dev_t, 2> selected_devices = {
      selected_gpu,
      nullptr,
  };

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = options.n_gpu_layers;
  model_params.devices = selected_devices.data();
  model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
  model_params.main_gpu = 0;
#if defined(APXINF_LLAMA_CPP_HAS_LOAD_MODE)
  model_params.load_mode = LLAMA_LOAD_MODE_NONE;
#else
  model_params.use_mmap = false;
  model_params.use_direct_io = false;
  model_params.use_mlock = false;
#endif
  model_params.check_tensors = false;

  const auto model_load_start = Clock::now();
  ModelPtr model(
      llama_model_load_from_file_ptr(model_file.get(), model_params));
  const auto model_load_end = Clock::now();
  if (!model) {
    fail("llama_model_load_from_file_ptr failed");
  }
  const FileIdentity model_identity_after_load =
      capture_file_identity(pinned_model_fd);
  require_same_file_identity(model_identity_start, model_identity_after_load,
                             "during model load");
  if (llama_model_has_encoder(model.get()) ||
      !llama_model_has_decoder(model.get())) {
    fail("runner requires a decoder-only model");
  }

  const llama_vocab *vocab = llama_model_get_vocab(model.get());
  if (vocab == nullptr) {
    fail("model has no vocabulary");
  }
  const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
  if (vocabulary_size != kVocabularySize) {
    fail("model vocabulary size must be exactly " +
         std::to_string(kVocabularySize));
  }
  for (const llama_token token : kPromptTokens) {
    if (token < 0 || token >= vocabulary_size) {
      fail("raw prompt token is outside the model vocabulary: " +
           std::to_string(token));
    }
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = kContextSize;
  context_params.n_batch = kBatchSize;
  context_params.n_ubatch = kMicroBatchSize;
  context_params.n_seq_max = kSequenceCount;
  context_params.n_threads = options.n_threads;
  context_params.n_threads_batch = options.n_threads;
  context_params.type_k = gpu_lane ? GGML_TYPE_F16 : GGML_TYPE_F32;
  context_params.type_v = gpu_lane ? GGML_TYPE_F16 : GGML_TYPE_F32;
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
  context_params.embeddings = false;
  context_params.offload_kqv = gpu_lane;
  context_params.no_perf = false;
  context_params.op_offload = gpu_lane;
  context_params.swa_full = false;
  context_params.kv_unified = false;

  const auto context_init_start = Clock::now();
  ContextPtr context(llama_init_from_model(model.get(), context_params));
  const auto context_init_end = Clock::now();
  if (!context) {
    fail("llama_init_from_model failed");
  }

  if (llama_n_ctx(context.get()) < kContextSize ||
      llama_n_batch(context.get()) != kBatchSize ||
      llama_n_ubatch(context.get()) != kMicroBatchSize ||
      llama_n_seq_max(context.get()) != kSequenceCount) {
    fail("llama.cpp could not honor the minimum context or strict batch "
         "parameters");
  }

  llama_sampler_chain_params sampler_params =
      llama_sampler_chain_default_params();
  sampler_params.no_perf = false;
  SamplerPtr sampler(llama_sampler_chain_init(sampler_params));
  if (!sampler) {
    fail("llama_sampler_chain_init failed");
  }
  llama_sampler *greedy = llama_sampler_init_greedy();
  if (greedy == nullptr) {
    fail("llama_sampler_init_greedy failed");
  }
  llama_sampler_chain_add(sampler.get(), greedy);

  std::array<llama_token, kPromptTokens.size()> mutable_prompt = kPromptTokens;
  llama_batch batch = llama_batch_get_one(
      mutable_prompt.data(), static_cast<int32_t>(mutable_prompt.size()));
  std::vector<llama_token> generated_tokens;
  std::vector<int64_t> token_ready_elapsed_ns;
  generated_tokens.reserve(kGeneratedTokenCount);
  token_ready_elapsed_ns.reserve(kGeneratedTokenCount);

  const auto generation_start = Clock::now();
  llama_token previous_token = LLAMA_TOKEN_NULL;
  for (int index = 0; index < kGeneratedTokenCount; ++index) {
    const int32_t decode_result = llama_decode(context.get(), batch);
    if (decode_result != 0) {
      fail("llama_decode failed at sampled token " + std::to_string(index) +
           " with code " + std::to_string(decode_result));
    }

    previous_token = llama_sampler_sample(sampler.get(), context.get(), -1);
    if (previous_token < 0 || previous_token >= vocabulary_size) {
      fail("sampled token is outside the model vocabulary at position " +
           std::to_string(index));
    }
    const auto token_ready = Clock::now();
    generated_tokens.push_back(previous_token);
    token_ready_elapsed_ns.push_back(elapsed_ns(generation_start, token_ready));

    if (index + 1 < kGeneratedTokenCount) {
      batch = llama_batch_get_one(&previous_token, 1);
    }
  }
  const auto generation_end = Clock::now();

  const llama_perf_context_data context_perf =
      llama_perf_context(context.get());
  const llama_perf_sampler_data sampler_perf =
      llama_perf_sampler(sampler.get());
  if (generated_tokens.size() != static_cast<size_t>(kGeneratedTokenCount) ||
      token_ready_elapsed_ns.size() !=
          static_cast<size_t>(kGeneratedTokenCount)) {
    fail("generation did not produce exactly 128 token receipts");
  }
  int64_t previous_ready_ns = -1;
  for (const int64_t ready_ns : token_ready_elapsed_ns) {
    if (ready_ns < 0 || ready_ns <= previous_ready_ns) {
      fail("token-ready elapsed times must be non-negative and strictly "
           "increasing");
    }
    previous_ready_ns = ready_ns;
  }
  if (context_perf.n_p_eval != static_cast<int32_t>(kPromptTokens.size()) ||
      context_perf.n_eval != kGeneratedTokenCount - 1 ||
      context_perf.n_reused != kGeneratedTokenCount - 2 ||
      sampler_perf.n_sample != kGeneratedTokenCount) {
    fail("llama.cpp performance counters violate the raw-token contract");
  }
  const auto require_finite_nonnegative = [](double value,
                                             std::string_view label) {
    if (!std::isfinite(value) || value < 0.0) {
      fail("llama.cpp performance value must be finite and non-negative: " +
           std::string(label));
    }
  };
  require_finite_nonnegative(context_perf.t_start_ms, "context.t_start_ms");
  require_finite_nonnegative(context_perf.t_load_ms, "context.t_load_ms");
  require_finite_nonnegative(context_perf.t_p_eval_ms,
                             "context.t_prompt_eval_ms");
  require_finite_nonnegative(context_perf.t_eval_ms, "context.t_eval_ms");
  require_finite_nonnegative(sampler_perf.t_sample_ms, "sampler.t_sample_ms");
  const PlacementAttestation placement =
      attest_placement(model.get(), context.get(), selected_gpu, gpu_lane);
  const auto measurement_end = Clock::now();
  const ExecutionProof execution_proof = run_post_measurement_execution_proof(
      context.get(), previous_token, selected_gpu);
  const FileIdentity model_identity_before_receipt =
      capture_file_identity(pinned_model_fd);
  require_same_file_identity(model_identity_start,
                             model_identity_before_receipt,
                             "before receipt publication");
  const auto receipt_ready_end = Clock::now();

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"apxinf.llama-cpp.raw-token-diagnostic.v2\""
      << ",\"ok\":true"
      << ",\"contract\":{"
      << "\"prompt_token_ids\":";
  append_token_array(out, kPromptTokens);
  out << ",\"sampling\":\"greedy-argmax\""
      << ",\"generated_token_count\":" << kGeneratedTokenCount
      << ",\"eog_termination\":false"
      << ",\"token_ready_elapsed_ns_origin\":\"immediately-before-prompt-"
         "decode\""
      << ",\"final_sampled_token_is_not_decoded_in_timed_workload\":true"
      << ",\"final_sampled_token_decoded_once_post_measurement_for_execution_"
         "proof\":true}"
      << ",\"model\":{"
      << "\"requested_path\":\"" << json_escape(display_model_path.string())
      << '"' << ",\"load_binding\":\"pinned-file-descriptor\""
      << ",\"open_flags\":\"O_RDONLY|O_NOFOLLOW|O_CLOEXEC\""
      << ",\"file_identity_start\":";
  append_file_identity(out, model_identity_start);
  out << ",\"file_identity_after_load\":";
  append_file_identity(out, model_identity_after_load);
  out << ",\"file_identity_before_receipt\":";
  append_file_identity(out, model_identity_before_receipt);
  out << ",\"file_identity_unchanged\":true"
      << ",\"file_size_bytes\":" << model_identity_start.size_bytes
      << ",\"description\":\"" << json_escape(model_description(model.get()))
      << '"' << ",\"parameter_count\":" << llama_model_n_params(model.get())
      << ",\"tensor_size_bytes\":" << llama_model_size(model.get())
      << ",\"file_type\":\""
      << json_escape(llama_ftype_name(llama_model_ftype(model.get()))) << '"'
      << ",\"file_type_code\":"
      << static_cast<int>(llama_model_ftype(model.get()))
      << ",\"vocabulary_size\":" << vocabulary_size
      << ",\"layer_count\":" << llama_model_n_layer(model.get())
      << ",\"is_recurrent\":"
      << (llama_model_is_recurrent(model.get()) ? "true" : "false")
      << ",\"is_hybrid\":"
      << (llama_model_is_hybrid(model.get()) ? "true" : "false") << '}'
      << ",\"parameters\":{"
      << "\"n_ctx_requested\":" << kContextSize
      << ",\"n_ctx_effective\":" << llama_n_ctx(context.get())
      << ",\"n_ctx_per_sequence_effective\":" << llama_n_ctx_seq(context.get())
      << ",\"n_batch_requested\":" << kBatchSize
      << ",\"n_batch_effective\":" << llama_n_batch(context.get())
      << ",\"n_ubatch_requested\":" << kMicroBatchSize
      << ",\"n_ubatch_effective\":" << llama_n_ubatch(context.get())
      << ",\"n_seq_max_requested\":" << kSequenceCount
      << ",\"n_seq_max_effective\":" << llama_n_seq_max(context.get())
      << ",\"n_threads\":" << options.n_threads
      << ",\"n_threads_batch\":" << options.n_threads
      << ",\"lane\":\"" << (gpu_lane ? "gpu-all-layers" : "cpu-only")
      << '"'
      << ",\"n_gpu_layers\":" << options.n_gpu_layers
      << ",\"kv_cache_type_k\":\"" << ggml_type_name(context_params.type_k)
      << '"' << ",\"kv_cache_type_v\":\""
      << ggml_type_name(context_params.type_v) << '"'
      << ",\"flash_attention\":\""
      << llama_flash_attn_type_name(context_params.flash_attn_type) << '"'
      << ",\"offload_kqv\":" << (context_params.offload_kqv ? "true" : "false")
      << ",\"op_offload\":" << (context_params.op_offload ? "true" : "false")
      << ",\"swa_full\":" << (context_params.swa_full ? "true" : "false")
      << ",\"kv_unified\":" << (context_params.kv_unified ? "true" : "false")
      << ",\"model_load_mode\":\"none-from-pinned-file-pointer\""
      << ",\"use_mmap\":false"
      << ",\"use_direct_io\":false"
      << ",\"use_mlock\":false"
      << ",\"check_tensors\":"
      << (model_params.check_tensors ? "true" : "false") << '}'
      << ",\"output\":{\"token_ids\":";
  append_token_array(out, generated_tokens);
  out << ",\"token_ready_elapsed_ns\":";
  append_token_array(out, token_ready_elapsed_ns);
  out << '}' << ",\"timings\":{"
      << "\"model_load_elapsed_ns\":"
      << elapsed_ns(model_load_start, model_load_end)
      << ",\"context_init_elapsed_ns\":"
      << elapsed_ns(context_init_start, context_init_end)
      << ",\"generation_elapsed_ns\":"
      << elapsed_ns(generation_start, generation_end)
      << ",\"measurement_scope_elapsed_ns\":"
      << elapsed_ns(full_start, measurement_end)
      << ",\"post_measurement_execution_proof_elapsed_ns\":"
      << execution_proof.elapsed_ns
      << ",\"receipt_ready_elapsed_ns\":"
      << elapsed_ns(full_start, receipt_ready_end) << '}'
      << ",\"llama_perf\":{\"context\":{"
      << "\"t_start_ms\":" << context_perf.t_start_ms
      << ",\"t_load_ms\":" << context_perf.t_load_ms
      << ",\"t_prompt_eval_ms\":" << context_perf.t_p_eval_ms
      << ",\"t_eval_ms\":" << context_perf.t_eval_ms
      << ",\"n_prompt_eval\":" << context_perf.n_p_eval
      << ",\"n_eval\":" << context_perf.n_eval
      << ",\"n_reused\":" << context_perf.n_reused << "},\"sampler\":{"
      << "\"t_sample_ms\":" << sampler_perf.t_sample_ms
      << ",\"n_sample\":" << sampler_perf.n_sample
      << "},\"captured_before_post_measurement_execution_proof\":true}"
      << ",\"backend\":{"
      << "\"registration_mode\":\"linked-static-registry-only\""
      << ",\"dynamic_backend_scan_invoked\":false"
      << ",\"backend_directory_option_supported\":false"
      << ",\"ggml_backend_path_present\":false"
      << ",\"supports_gpu_offload\":"
      << (llama_supports_gpu_offload() ? "true" : "false")
      << ",\"selected_gpu_device\":";
  append_backend_device(out, selected_gpu);
  out << ",\"registered_devices_after_generation\":";
  append_backend_devices(out);
  out << ",\"system_info\":\"" << json_escape(llama_print_system_info())
      << "\"}"
      << ",\"placement_attestation\":{"
      << "\"method\":\"pinned-llama-internal-layer-assignments-plus-memory-"
         "breakdown-v1\""
      << ",\"passed\":true"
      << ",\"model_selected_device_count\":"
      << placement.model_selected_device_count
      << ",\"transformer_layer_count\":" << placement.layer_count
      << ",\"layers_on_selected_gpu\":"
      << placement.layers_on_selected_device
      << ",\"layers_on_cpu\":" << placement.layers_on_cpu
      << ",\"output_on_selected_gpu\":"
      << (placement.output_on_selected_device ? "true" : "false")
      << ",\"output_on_cpu\":"
      << (placement.output_on_cpu ? "true" : "false")
      << ",\"input_embedding_buffer_type\":\""
      << json_escape(ggml_backend_buft_name(
             ggml_backend_buffer_get_type(model->tok_embd->buffer)))
      << '"' << ",\"input_embedding_device\":";
  append_backend_device(
      out, ggml_backend_buft_get_device(
               ggml_backend_buffer_get_type(model->tok_embd->buffer)));
  out << ",\"memory_by_device_class\":{\"gpu\":";
  append_memory_totals(out, placement.gpu);
  out << ",\"cpu\":";
  append_memory_totals(out, placement.cpu);
  out << ",\"accelerator\":";
  append_memory_totals(out, placement.accelerator);
  out << ",\"other\":";
  append_memory_totals(out, placement.other);
  out << "},\"memory_by_buffer_type\":";
  append_memory_breakdown(out, placement.memory_breakdown);
  out << '}'
      << ",\"post_measurement_execution_proof\":{"
      << "\"method\":\"scheduler-callback-completed-sentinels-v1\""
      << ",\"passed\":true"
      << ",\"timing_excluded\":true"
      << ",\"decode_count\":1"
      << ",\"proof_token_id\":" << previous_token
      << ",\"requested_sentinel_count\":"
      << execution_proof.requested_sentinels
      << ",\"completed_sentinel_count\":"
      << execution_proof.completed_sentinels
      << ",\"completed_input_embedding_on_cpu\":"
      << (execution_proof.completed_input_embedding ? "true" : "false")
      << ",\"completed_transformer_layer_endpoints\":24"
      << ",\"completed_output_head\":"
      << (execution_proof.completed_output ? "true" : "false")
      << ",\"completed_on_selected_gpu\":"
      << execution_proof.completed_on_selected_gpu
      << ",\"completed_on_cpu\":" << execution_proof.completed_on_cpu
      << ",\"backend_mismatch\":false"
      << ",\"duplicate_or_unexpected_callback\":false}"
      << ",\"build\":{"
      << "\"llama_cpp_source_id\":\"" << json_escape(APXINF_LLAMA_CPP_SOURCE_ID)
      << '"' << ",\"llama_cpp_source_id_provenance\":\""
      << json_escape(APXINF_LLAMA_CPP_SOURCE_ID_PROVENANCE) << '"'
#if defined(APXINF_LLAMA_CPP_HAS_VERSION_API)
      << ",\"llama_cpp_version\":\"" << json_escape(llama_version()) << '"'
#endif
      << ",\"cmake_version\":\"" << json_escape(APXINF_CMAKE_VERSION) << '"'
      << ",\"cxx_compiler_id\":\"" << json_escape(APXINF_CXX_COMPILER_ID) << '"'
      << ",\"cxx_compiler_version\":\""
      << json_escape(APXINF_CXX_COMPILER_VERSION) << '"'
      << ",\"cmake_build_type\":\"" << json_escape(APXINF_BUILD_TYPE) << '"'
      << ",\"build_shared_libs\":false"
      << ",\"ggml_backend_dl\":false"
      << ",\"ggml_metal\":true"
      << ",\"ggml_metal_embed_library\":true"
      << ",\"ggml_accelerate\":true"
      << ",\"ggml_native\":true"
#if defined(__VERSION__)
      << ",\"cxx_compiler_banner\":\"" << json_escape(__VERSION__) << '"'
#endif
      << "}}";
  return out.str();
}

} // namespace

int main(int argc, char **argv) {
  std::locale::global(std::locale::classic());
  try {
    std::cout.exceptions(std::ios::badbit | std::ios::failbit);
    const Options options = parse_options(argc, argv);
    reject_environment_variable("GGML_BACKEND_PATH");
    BackendGuard backend;
    std::cout << run(options) << '\n' << std::flush;
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    try {
      std::cout.exceptions(std::ios::goodbit);
      std::cout.clear();
      std::cout << "{\"schema\":\"apxinf.llama-cpp.raw-token-diagnostic.v2\","
                   "\"ok\":false,\"error\":\""
                << json_escape(error.what()) << "\"}\n"
                << std::flush;
    } catch (...) {
    }
    return EXIT_FAILURE;
  }
}
