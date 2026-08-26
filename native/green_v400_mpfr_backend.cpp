#include <mpfr.h>
#include <gmp.h>

#include <cstdint>
#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

class MpfrValue {
 public:
  explicit MpfrValue(mpfr_prec_t precision) { mpfr_init2(value_, precision); }
  ~MpfrValue() { mpfr_clear(value_); }
  MpfrValue(const MpfrValue&) = delete;
  MpfrValue& operator=(const MpfrValue&) = delete;
  MpfrValue(MpfrValue&& other) noexcept {
    mpfr_init2(value_, mpfr_get_prec(other.value_));
    mpfr_swap(value_, other.value_);
  }
  MpfrValue& operator=(MpfrValue&& other) noexcept {
    if (this != &other) mpfr_swap(value_, other.value_);
    return *this;
  }
  mpfr_ptr get() { return value_; }
  mpfr_srcptr get() const { return value_; }

 private:
  mpfr_t value_;
};

float float_from_bits(std::uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

double double_from_bits(std::uint64_t bits) {
  double result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void pairwise_sum(std::vector<MpfrValue*>& terms, mpfr_ptr output,
                  mpfr_rnd_t rounding) {
  if (terms.empty()) {
    mpfr_set_zero(output, 0);
    return;
  }
  std::size_t count = terms.size();
  while (count > 1) {
    std::size_t next = 0;
    for (std::size_t index = 0; index + 1 < count; index += 2) {
      mpfr_add(terms[next]->get(), terms[index]->get(), terms[index + 1]->get(), rounding);
      ++next;
    }
    if (count & 1U) {
      mpfr_set(terms[next]->get(), terms[count - 1]->get(), rounding);
      ++next;
    }
    count = next;
  }
  mpfr_set(output, terms[0]->get(), rounding);
}

std::string exact_binary(mpfr_srcptr value, mpfr_prec_t precision) {
  mpz_t significand;
  mpz_init(significand);
  const mpfr_exp_t exponent = mpfr_get_z_2exp(significand, value);
  char* raw = mpz_get_str(nullptr, 16, significand);
  std::ostringstream stream;
  stream << "{\"significand_hex\":\"" << raw << "\",\"exponent_2\":"
         << exponent << ",\"precision_bits\":" << precision << "}";
  void (*free_function)(void*, size_t) = nullptr;
  mp_get_memory_functions(nullptr, nullptr, &free_function);
  free_function(raw, std::strlen(raw) + 1);
  mpz_clear(significand);
  return stream.str();
}

int affine_component(mpfr_ptr lower_output, mpfr_ptr upper_output,
                     const std::uint32_t* weight_bits,
                     const std::uint64_t* lower_bits,
                     const std::uint64_t* upper_bits,
                     std::size_t count, mpfr_prec_t precision) {
  std::vector<MpfrValue> lower_storage;
  std::vector<MpfrValue> upper_storage;
  lower_storage.reserve(count);
  upper_storage.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    lower_storage.emplace_back(precision);
    upper_storage.emplace_back(precision);
  }
  std::vector<MpfrValue*> lower_terms;
  std::vector<MpfrValue*> upper_terms;
  lower_terms.reserve(count);
  upper_terms.reserve(count);
  MpfrValue weight(precision), lower(precision), upper(precision);
  for (std::size_t index = 0; index < count; ++index) {
    const float raw_weight = float_from_bits(weight_bits[index]);
    const double raw_lower = double_from_bits(lower_bits[index]);
    const double raw_upper = double_from_bits(upper_bits[index]);
    if (!(raw_lower <= raw_upper)) return 3;
    mpfr_set_flt(weight.get(), raw_weight, MPFR_RNDN);
    mpfr_set_d(lower.get(), raw_lower, MPFR_RNDN);
    mpfr_set_d(upper.get(), raw_upper, MPFR_RNDN);
    if (mpfr_sgn(weight.get()) >= 0) {
      mpfr_mul(lower_storage[index].get(), weight.get(), lower.get(), MPFR_RNDD);
      mpfr_mul(upper_storage[index].get(), weight.get(), upper.get(), MPFR_RNDU);
    } else {
      mpfr_mul(lower_storage[index].get(), weight.get(), upper.get(), MPFR_RNDD);
      mpfr_mul(upper_storage[index].get(), weight.get(), lower.get(), MPFR_RNDU);
    }
    lower_terms.push_back(&lower_storage[index]);
    upper_terms.push_back(&upper_storage[index]);
  }
  pairwise_sum(lower_terms, lower_output, MPFR_RNDD);
  pairwise_sum(upper_terms, upper_output, MPFR_RNDU);
  return 0;
}

int set_exact_binary(mpfr_ptr output, const char* significand_hex,
                     std::int64_t exponent_2) {
  if (significand_hex == nullptr) return 2;
  mpz_t significand;
  mpz_init(significand);
  if (mpz_set_str(significand, significand_hex, 16) != 0) {
    mpz_clear(significand);
    return 3;
  }
  mpfr_set_z(output, significand, MPFR_RNDN);
  mpfr_mul_2si(output, output, static_cast<long>(exponent_2), MPFR_RNDN);
  mpz_clear(significand);
  return 0;
}

int affine_component_exact(mpfr_ptr lower_output, mpfr_ptr upper_output,
                           const std::uint32_t* weight_bits,
                           const char* const* lower_significands,
                           const std::int64_t* lower_exponents,
                           const char* const* upper_significands,
                           const std::int64_t* upper_exponents,
                           std::size_t count, mpfr_prec_t precision) {
  std::vector<MpfrValue> lower_storage;
  std::vector<MpfrValue> upper_storage;
  lower_storage.reserve(count);
  upper_storage.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    lower_storage.emplace_back(precision);
    upper_storage.emplace_back(precision);
  }
  std::vector<MpfrValue*> lower_terms;
  std::vector<MpfrValue*> upper_terms;
  lower_terms.reserve(count);
  upper_terms.reserve(count);
  MpfrValue weight(precision), lower(precision), upper(precision);
  for (std::size_t index = 0; index < count; ++index) {
    mpfr_set_flt(weight.get(), float_from_bits(weight_bits[index]), MPFR_RNDN);
    int status = set_exact_binary(lower.get(), lower_significands[index], lower_exponents[index]);
    if (status != 0) return status;
    status = set_exact_binary(upper.get(), upper_significands[index], upper_exponents[index]);
    if (status != 0 || mpfr_greater_p(lower.get(), upper.get())) return 3;
    if (mpfr_sgn(weight.get()) >= 0) {
      mpfr_mul(lower_storage[index].get(), weight.get(), lower.get(), MPFR_RNDD);
      mpfr_mul(upper_storage[index].get(), weight.get(), upper.get(), MPFR_RNDU);
    } else {
      mpfr_mul(lower_storage[index].get(), weight.get(), upper.get(), MPFR_RNDD);
      mpfr_mul(upper_storage[index].get(), weight.get(), lower.get(), MPFR_RNDU);
    }
    lower_terms.push_back(&lower_storage[index]);
    upper_terms.push_back(&upper_storage[index]);
  }
  pairwise_sum(lower_terms, lower_output, MPFR_RNDD);
  pairwise_sum(upper_terms, upper_output, MPFR_RNDU);
  return 0;
}

std::string serialize_jet(mpfr_srcptr value_lower, mpfr_srcptr value_upper,
                          mpfr_srcptr first_lower, mpfr_srcptr first_upper,
                          mpfr_srcptr second_lower, mpfr_srcptr second_upper,
                          mpfr_prec_t precision) {
  std::ostringstream stream;
  stream << "{\"schema_version\":\"green-v400-compiled-affine-jet2-v1\",";
  stream << "\"precision_bits\":" << precision << ",";
  stream << "\"value\":{\"lower\":" << exact_binary(value_lower, precision)
         << ",\"upper\":" << exact_binary(value_upper, precision) << "},";
  stream << "\"first\":{\"lower\":" << exact_binary(first_lower, precision)
         << ",\"upper\":" << exact_binary(first_upper, precision) << "},";
  stream << "\"second\":{\"lower\":" << exact_binary(second_lower, precision)
         << ",\"upper\":" << exact_binary(second_upper, precision) << "}}";
  return stream.str();
}

std::uint64_t mix_checksum(std::uint64_t state, mpfr_srcptr value) {
  const double converted = mpfr_get_d(value, MPFR_RNDN);
  std::uint64_t bits;
  std::memcpy(&bits, &converted, sizeof(bits));
  state ^= bits + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
  return state;
}

}  // namespace

extern "C" const char* green_v400_mpfr_backend_version() {
  return "green-v400-compiled-mpfr-v1";
}

extern "C" int green_v400_affine_jet2_f32(
    std::uint32_t precision_bits, std::uint64_t count,
    const std::uint32_t* weight_bits, std::uint32_t bias_bits,
    const std::uint64_t* value_lower_bits, const std::uint64_t* value_upper_bits,
    const std::uint64_t* first_lower_bits, const std::uint64_t* first_upper_bits,
    const std::uint64_t* second_lower_bits, const std::uint64_t* second_upper_bits,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || count == 0 || count > 100000000ULL
      || weight_bits == nullptr || output_json == nullptr || output_capacity == 0) {
    return 2;
  }
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  MpfrValue value_lower(precision), value_upper(precision), first_lower(precision),
      first_upper(precision), second_lower(precision), second_upper(precision);
  int status = affine_component(value_lower.get(), value_upper.get(), weight_bits,
                                value_lower_bits, value_upper_bits, count, precision);
  if (status != 0) return status;
  status = affine_component(first_lower.get(), first_upper.get(), weight_bits,
                            first_lower_bits, first_upper_bits, count, precision);
  if (status != 0) return status;
  status = affine_component(second_lower.get(), second_upper.get(), weight_bits,
                            second_lower_bits, second_upper_bits, count, precision);
  if (status != 0) return status;

  MpfrValue bias(precision);
  mpfr_set_flt(bias.get(), float_from_bits(bias_bits), MPFR_RNDN);
  mpfr_add(value_lower.get(), value_lower.get(), bias.get(), MPFR_RNDD);
  mpfr_add(value_upper.get(), value_upper.get(), bias.get(), MPFR_RNDU);

  const std::string serialized = serialize_jet(
      value_lower.get(), value_upper.get(), first_lower.get(), first_upper.get(),
      second_lower.get(), second_upper.get(), precision);
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_affine_jet2_exact(
    std::uint32_t precision_bits, std::uint64_t count,
    const std::uint32_t* weight_bits, std::uint32_t bias_bits,
    const char* const* value_lower_significands, const std::int64_t* value_lower_exponents,
    const char* const* value_upper_significands, const std::int64_t* value_upper_exponents,
    const char* const* first_lower_significands, const std::int64_t* first_lower_exponents,
    const char* const* first_upper_significands, const std::int64_t* first_upper_exponents,
    const char* const* second_lower_significands, const std::int64_t* second_lower_exponents,
    const char* const* second_upper_significands, const std::int64_t* second_upper_exponents,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || count == 0 || count > 100000000ULL
      || weight_bits == nullptr || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  MpfrValue value_lower(precision), value_upper(precision), first_lower(precision),
      first_upper(precision), second_lower(precision), second_upper(precision);
  int status = affine_component_exact(
      value_lower.get(), value_upper.get(), weight_bits,
      value_lower_significands, value_lower_exponents,
      value_upper_significands, value_upper_exponents, count, precision);
  if (status != 0) return status;
  status = affine_component_exact(
      first_lower.get(), first_upper.get(), weight_bits,
      first_lower_significands, first_lower_exponents,
      first_upper_significands, first_upper_exponents, count, precision);
  if (status != 0) return status;
  status = affine_component_exact(
      second_lower.get(), second_upper.get(), weight_bits,
      second_lower_significands, second_lower_exponents,
      second_upper_significands, second_upper_exponents, count, precision);
  if (status != 0) return status;
  MpfrValue bias(precision);
  mpfr_set_flt(bias.get(), float_from_bits(bias_bits), MPFR_RNDN);
  mpfr_add(value_lower.get(), value_lower.get(), bias.get(), MPFR_RNDD);
  mpfr_add(value_upper.get(), value_upper.get(), bias.get(), MPFR_RNDU);
  const std::string serialized = serialize_jet(
      value_lower.get(), value_upper.get(), first_lower.get(), first_upper.get(),
      second_lower.get(), second_upper.get(), precision);
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_benchmark_affine_jet2_layer(
    std::uint32_t precision_bits, std::uint32_t input_width,
    std::uint32_t output_width, std::uint32_t repeats,
    double* elapsed_seconds, std::uint64_t* checksum) {
  if (precision_bits < 64 || precision_bits > 4096 || input_width == 0
      || output_width == 0 || repeats == 0 || input_width > 1000000U
      || output_width > 1000000U || elapsed_seconds == nullptr || checksum == nullptr) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<MpfrValue> inputs;
  inputs.reserve(static_cast<std::size_t>(6) * input_width);
  for (std::size_t index = 0; index < static_cast<std::size_t>(6) * input_width; ++index) {
    inputs.emplace_back(precision);
    const long numerator = static_cast<long>((index * 37U) % 2048U) - 1024L;
    mpfr_set_si(inputs.back().get(), numerator, MPFR_RNDN);
    mpfr_div_2ui(inputs.back().get(), inputs.back().get(), 10U, MPFR_RNDN);
  }
  // Force each lower/upper pair into valid order without inspecting outcomes.
  for (std::size_t component = 0; component < 3; ++component) {
    const std::size_t lower_offset = component * 2U * input_width;
    const std::size_t upper_offset = lower_offset + input_width;
    for (std::size_t index = 0; index < input_width; ++index) {
      if (mpfr_greater_p(inputs[lower_offset + index].get(), inputs[upper_offset + index].get()))
        mpfr_swap(inputs[lower_offset + index].get(), inputs[upper_offset + index].get());
    }
  }
  std::vector<MpfrValue> lower_terms;
  std::vector<MpfrValue> upper_terms;
  lower_terms.reserve(input_width);
  upper_terms.reserve(input_width);
  for (std::size_t index = 0; index < input_width; ++index) {
    lower_terms.emplace_back(precision);
    upper_terms.emplace_back(precision);
  }
  std::vector<MpfrValue*> lower_pointers, upper_pointers;
  lower_pointers.reserve(input_width);
  upper_pointers.reserve(input_width);
  for (std::size_t index = 0; index < input_width; ++index) {
    lower_pointers.push_back(&lower_terms[index]);
    upper_pointers.push_back(&upper_terms[index]);
  }
  MpfrValue weight(precision), lower_output(precision), upper_output(precision);
  std::uint64_t state = 0x6a09e667f3bcc909ULL;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::uint32_t output = 0; output < output_width; ++output) {
      for (std::size_t component = 0; component < 3; ++component) {
        const std::size_t lower_offset = component * 2U * input_width;
        const std::size_t upper_offset = lower_offset + input_width;
        for (std::uint32_t index = 0; index < input_width; ++index) {
          const std::uint32_t hash = (index + 1U) * 2654435761U
              ^ (output + 17U) * 2246822519U ^ (repeat + 31U) * 3266489917U;
          const float raw_weight = (static_cast<int>(hash % 2049U) - 1024) / 1024.0f;
          mpfr_set_flt(weight.get(), raw_weight, MPFR_RNDN);
          mpfr_srcptr lower = inputs[lower_offset + index].get();
          mpfr_srcptr upper = inputs[upper_offset + index].get();
          if (mpfr_sgn(weight.get()) >= 0) {
            mpfr_mul(lower_terms[index].get(), weight.get(), lower, MPFR_RNDD);
            mpfr_mul(upper_terms[index].get(), weight.get(), upper, MPFR_RNDU);
          } else {
            mpfr_mul(lower_terms[index].get(), weight.get(), upper, MPFR_RNDD);
            mpfr_mul(upper_terms[index].get(), weight.get(), lower, MPFR_RNDU);
          }
        }
        pairwise_sum(lower_pointers, lower_output.get(), MPFR_RNDD);
        pairwise_sum(upper_pointers, upper_output.get(), MPFR_RNDU);
        state = mix_checksum(state, lower_output.get());
        state = mix_checksum(state, upper_output.get());
      }
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  *elapsed_seconds = std::chrono::duration<double>(stop - start).count();
  *checksum = state;
  return 0;
}

extern "C" int green_v400_interval_primitive_exact(
    const char* operation, std::uint32_t precision_bits,
    const char* lower_significand, std::int64_t lower_exponent,
    const char* upper_significand, std::int64_t upper_exponent,
    char* output_json, std::uint64_t output_capacity) {
  if (operation == nullptr || precision_bits < 64 || precision_bits > 4096
      || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  MpfrValue lower(precision), upper(precision), result_lower(precision), result_upper(precision);
  int status = set_exact_binary(lower.get(), lower_significand, lower_exponent);
  if (status != 0) return status;
  status = set_exact_binary(upper.get(), upper_significand, upper_exponent);
  if (status != 0 || mpfr_greater_p(lower.get(), upper.get())) return 3;
  if (std::strcmp(operation, "exp") == 0) {
    mpfr_exp(result_lower.get(), lower.get(), MPFR_RNDD);
    mpfr_exp(result_upper.get(), upper.get(), MPFR_RNDU);
  } else if (std::strcmp(operation, "tanh") == 0) {
    mpfr_tanh(result_lower.get(), lower.get(), MPFR_RNDD);
    mpfr_tanh(result_upper.get(), upper.get(), MPFR_RNDU);
  } else if (std::strcmp(operation, "sqrt") == 0) {
    if (mpfr_sgn(lower.get()) < 0) return 5;
    mpfr_sqrt(result_lower.get(), lower.get(), MPFR_RNDD);
    mpfr_sqrt(result_upper.get(), upper.get(), MPFR_RNDU);
  } else if (std::strcmp(operation, "inv_sqrt") == 0) {
    if (mpfr_sgn(lower.get()) <= 0) return 5;
    MpfrValue sqrt_lower(precision), sqrt_upper(precision), one(precision);
    mpfr_sqrt(sqrt_lower.get(), lower.get(), MPFR_RNDD);
    mpfr_sqrt(sqrt_upper.get(), upper.get(), MPFR_RNDU);
    mpfr_set_ui(one.get(), 1U, MPFR_RNDN);
    mpfr_div(result_lower.get(), one.get(), sqrt_upper.get(), MPFR_RNDD);
    mpfr_div(result_upper.get(), one.get(), sqrt_lower.get(), MPFR_RNDU);
  } else {
    return 6;
  }
  std::ostringstream stream;
  stream << "{\"schema_version\":\"green-v400-compiled-interval-primitive-v1\",";
  stream << "\"operation\":\"" << operation << "\",\"precision_bits\":"
         << precision_bits << ",\"lower\":" << exact_binary(result_lower.get(), precision)
         << ",\"upper\":" << exact_binary(result_upper.get(), precision) << "}";
  const std::string serialized = stream.str();
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}
