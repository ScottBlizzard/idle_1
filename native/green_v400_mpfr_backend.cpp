#include <mpfr.h>
#include <gmp.h>

#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
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

thread_local std::uint64_t* active_primitive_counter = nullptr;
thread_local std::uint64_t* active_dispatch_trace = nullptr;
thread_local std::uint64_t* active_dispatch_events = nullptr;
thread_local std::uint8_t* active_dispatch_tags = nullptr;
thread_local std::uint64_t active_dispatch_tag_capacity = 0;

enum DispatchKernel : std::uint64_t {
  kAffineScatter = 1U, kStaticView = 2U, kPairwiseAffine = 3U,
  kLayerNorm = 4U, kGeluNew = 5U, kCausalAttention = 6U,
  kResidualAdd = 7U, kFinalContrast = 8U, kBranchLinearCombination = 9U,
};

void record_dispatch(DispatchKernel kernel) {
  if (active_dispatch_trace == nullptr || active_dispatch_events == nullptr) return;
  if (active_dispatch_tags != nullptr && *active_dispatch_events < active_dispatch_tag_capacity)
    active_dispatch_tags[*active_dispatch_events] = static_cast<std::uint8_t>(kernel);
  *active_dispatch_trace ^= static_cast<std::uint64_t>(kernel);
  *active_dispatch_trace *= 1099511628211ULL;
  ++(*active_dispatch_events);
}

void count_primitive() {
  if (active_primitive_counter != nullptr) ++(*active_primitive_counter);
}

int counted_add(mpfr_ptr result, mpfr_srcptr left, mpfr_srcptr right, mpfr_rnd_t rounding) {
  count_primitive();
  return mpfr_add(result, left, right, rounding);
}

int counted_mul(mpfr_ptr result, mpfr_srcptr left, mpfr_srcptr right, mpfr_rnd_t rounding) {
  count_primitive();
  return mpfr_mul(result, left, right, rounding);
}

int counted_div(mpfr_ptr result, mpfr_srcptr left, mpfr_srcptr right, mpfr_rnd_t rounding) {
  count_primitive();
  return mpfr_div(result, left, right, rounding);
}

int counted_div_ui(mpfr_ptr result, mpfr_srcptr left, unsigned long right,
                   mpfr_rnd_t rounding) {
  count_primitive();
  return mpfr_div_ui(result, left, right, rounding);
}

int counted_sqrt(mpfr_ptr result, mpfr_srcptr input, mpfr_rnd_t rounding) {
  count_primitive();
  return mpfr_sqrt(result, input, rounding);
}

int counted_tanh(mpfr_ptr result, mpfr_srcptr input, mpfr_rnd_t rounding) {
  count_primitive();
  return mpfr_tanh(result, input, rounding);
}

int counted_exp(mpfr_ptr result, mpfr_srcptr input, mpfr_rnd_t rounding) {
  count_primitive();
  return mpfr_exp(result, input, rounding);
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
      counted_add(terms[next]->get(), terms[index]->get(), terms[index + 1]->get(), rounding);
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

struct IntervalMP {
  explicit IntervalMP(mpfr_prec_t precision) : lower(precision), upper(precision), precision(precision) {}
  MpfrValue lower;
  MpfrValue upper;
  mpfr_prec_t precision;
};

struct JetMP {
  explicit JetMP(mpfr_prec_t precision) : value(precision), first(precision), second(precision) {}
  IntervalMP value;
  IntervalMP first;
  IntervalMP second;
};

IntervalMP interval_point_float(float raw, mpfr_prec_t precision) {
  IntervalMP result(precision);
  mpfr_set_flt(result.lower.get(), raw, MPFR_RNDN);
  mpfr_set(result.upper.get(), result.lower.get(), MPFR_RNDN);
  return result;
}

IntervalMP interval_add(const IntervalMP& left, const IntervalMP& right) {
  IntervalMP result(left.precision);
  counted_add(result.lower.get(), left.lower.get(), right.lower.get(), MPFR_RNDD);
  counted_add(result.upper.get(), left.upper.get(), right.upper.get(), MPFR_RNDU);
  return result;
}

IntervalMP interval_neg(const IntervalMP& input) {
  IntervalMP result(input.precision);
  mpfr_neg(result.lower.get(), input.upper.get(), MPFR_RNDN);
  mpfr_neg(result.upper.get(), input.lower.get(), MPFR_RNDN);
  return result;
}

IntervalMP interval_mul(const IntervalMP& left, const IntervalMP& right) {
  IntervalMP result(left.precision);
  MpfrValue candidate(left.precision);
  bool initialized = false;
  mpfr_srcptr left_values[2] = {left.lower.get(), left.upper.get()};
  mpfr_srcptr right_values[2] = {right.lower.get(), right.upper.get()};
  for (mpfr_srcptr a : left_values) for (mpfr_srcptr b : right_values) {
    counted_mul(candidate.get(), a, b, MPFR_RNDD);
    if (!initialized || mpfr_less_p(candidate.get(), result.lower.get()))
      mpfr_set(result.lower.get(), candidate.get(), MPFR_RNDN);
    initialized = true;
  }
  initialized = false;
  for (mpfr_srcptr a : left_values) for (mpfr_srcptr b : right_values) {
    counted_mul(candidate.get(), a, b, MPFR_RNDU);
    if (!initialized || mpfr_greater_p(candidate.get(), result.upper.get()))
      mpfr_set(result.upper.get(), candidate.get(), MPFR_RNDN);
    initialized = true;
  }
  return result;
}

IntervalMP interval_square(const IntervalMP& input) {
  IntervalMP result(input.precision);
  MpfrValue lower_square(input.precision), upper_square(input.precision),
      lower_up(input.precision), upper_up(input.precision);
  counted_mul(lower_square.get(), input.lower.get(), input.lower.get(), MPFR_RNDD);
  counted_mul(upper_square.get(), input.upper.get(), input.upper.get(), MPFR_RNDD);
  counted_mul(lower_up.get(), input.lower.get(), input.lower.get(), MPFR_RNDU);
  counted_mul(upper_up.get(), input.upper.get(), input.upper.get(), MPFR_RNDU);
  if (mpfr_sgn(input.lower.get()) <= 0 && mpfr_sgn(input.upper.get()) >= 0) {
    mpfr_set_zero(result.lower.get(), 0);
  } else {
    mpfr_set(result.lower.get(), mpfr_less_p(lower_square.get(), upper_square.get())
             ? lower_square.get() : upper_square.get(), MPFR_RNDN);
  }
  mpfr_set(result.upper.get(), mpfr_greater_p(lower_up.get(), upper_up.get())
           ? lower_up.get() : upper_up.get(), MPFR_RNDN);
  return result;
}

IntervalMP interval_tanh(const IntervalMP& input) {
  IntervalMP result(input.precision);
  counted_tanh(result.lower.get(), input.lower.get(), MPFR_RNDD);
  counted_tanh(result.upper.get(), input.upper.get(), MPFR_RNDU);
  return result;
}

IntervalMP interval_exp(const IntervalMP& input) {
  IntervalMP result(input.precision);
  counted_exp(result.lower.get(), input.lower.get(), MPFR_RNDD);
  counted_exp(result.upper.get(), input.upper.get(), MPFR_RNDU);
  return result;
}

IntervalMP interval_clone(const IntervalMP& input) {
  IntervalMP result(input.precision);
  mpfr_set(result.lower.get(), input.lower.get(), MPFR_RNDN);
  mpfr_set(result.upper.get(), input.upper.get(), MPFR_RNDN);
  return result;
}

IntervalMP interval_point_rational(unsigned long numerator, unsigned long denominator,
                                   mpfr_prec_t precision) {
  IntervalMP result(precision);
  MpfrValue raw(precision);
  mpfr_set_ui(raw.get(), numerator, MPFR_RNDN);
  counted_div_ui(result.lower.get(), raw.get(), denominator, MPFR_RNDD);
  counted_div_ui(result.upper.get(), raw.get(), denominator, MPFR_RNDU);
  return result;
}

IntervalMP interval_reciprocal(const IntervalMP& input) {
  IntervalMP result(input.precision);
  MpfrValue one(input.precision), first(input.precision), second(input.precision);
  mpfr_set_ui(one.get(), 1U, MPFR_RNDN);
  counted_div(first.get(), one.get(), input.upper.get(), MPFR_RNDD);
  counted_div(second.get(), one.get(), input.lower.get(), MPFR_RNDU);
  mpfr_set(result.lower.get(), mpfr_less_p(first.get(), second.get())
           ? first.get() : second.get(), MPFR_RNDN);
  mpfr_set(result.upper.get(), mpfr_greater_p(first.get(), second.get())
           ? first.get() : second.get(), MPFR_RNDN);
  return result;
}

IntervalMP interval_inv_sqrt(const IntervalMP& input) {
  IntervalMP roots(input.precision);
  counted_sqrt(roots.lower.get(), input.lower.get(), MPFR_RNDD);
  counted_sqrt(roots.upper.get(), input.upper.get(), MPFR_RNDU);
  return interval_reciprocal(roots);
}

JetMP jet_constant(const IntervalMP& value) {
  JetMP result(value.precision);
  mpfr_set(result.value.lower.get(), value.lower.get(), MPFR_RNDN);
  mpfr_set(result.value.upper.get(), value.upper.get(), MPFR_RNDN);
  mpfr_set_zero(result.first.lower.get(), 0); mpfr_set_zero(result.first.upper.get(), 0);
  mpfr_set_zero(result.second.lower.get(), 0); mpfr_set_zero(result.second.upper.get(), 0);
  return result;
}

JetMP jet_add(const JetMP& left, const JetMP& right) {
  JetMP result(left.value.precision);
  result.value = interval_add(left.value, right.value);
  result.first = interval_add(left.first, right.first);
  result.second = interval_add(left.second, right.second);
  return result;
}

JetMP jet_clone(const JetMP& input) {
  JetMP result(input.value.precision);
  result.value = interval_clone(input.value);
  result.first = interval_clone(input.first);
  result.second = interval_clone(input.second);
  return result;
}

JetMP jet_sub(const JetMP& left, const JetMP& right) {
  JetMP negated(right.value.precision);
  negated.value = interval_neg(right.value);
  negated.first = interval_neg(right.first);
  negated.second = interval_neg(right.second);
  return jet_add(left, negated);
}

JetMP jet_mul(const JetMP& x, const JetMP& y) {
  const mpfr_prec_t precision = x.value.precision;
  JetMP result(precision);
  result.value = interval_mul(x.value, y.value);
  result.first = interval_add(interval_mul(x.first, y.value), interval_mul(x.value, y.first));
  const IntervalMP two = interval_point_float(2.0f, precision);
  IntervalMP first_term = interval_mul(x.second, y.value);
  IntervalMP middle_term = interval_mul(interval_mul(two, x.first), y.first);
  IntervalMP last_term = interval_mul(x.value, y.second);
  result.second = interval_add(interval_add(first_term, middle_term), last_term);
  return result;
}

JetMP jet_scale_float(const JetMP& value, float scalar) {
  return jet_mul(value, jet_constant(interval_point_float(scalar, value.value.precision)));
}

JetMP jet_scale_interval(const JetMP& value, const IntervalMP& scalar) {
  return jet_mul(value, jet_constant(scalar));
}

JetMP jet_square(const JetMP& x) {
  const mpfr_prec_t precision = x.value.precision;
  JetMP result(precision);
  const IntervalMP two = interval_point_float(2.0f, precision);
  result.value = interval_square(x.value);
  result.first = interval_mul(interval_mul(two, x.value), x.first);
  result.second = interval_mul(
      two, interval_add(interval_square(x.first), interval_mul(x.value, x.second)));
  return result;
}

JetMP jet_inv_sqrt(const JetMP& x) {
  const mpfr_prec_t precision = x.value.precision;
  JetMP result(precision);
  result.value = interval_inv_sqrt(x.value);
  const IntervalMP inverse_sqrt = interval_inv_sqrt(x.value);
  const IntervalMP inverse_value = interval_reciprocal(x.value);
  const IntervalMP first_factor = interval_mul(
      interval_mul(interval_point_float(-0.5f, precision), inverse_sqrt),
      inverse_value);
  const IntervalMP second_factor = interval_mul(
      interval_mul(interval_point_float(0.75f, precision), inverse_sqrt),
      interval_reciprocal(interval_square(x.value)));
  result.first = interval_mul(first_factor, x.first);
  result.second = interval_add(
      interval_mul(second_factor, interval_square(x.first)),
      interval_mul(first_factor, x.second));
  return result;
}

JetMP jet_pairwise_sum(const std::vector<JetMP>& inputs) {
  std::vector<JetMP> level;
  level.reserve(inputs.size());
  for (const JetMP& input : inputs) level.emplace_back(jet_clone(input));
  while (level.size() > 1) {
    std::vector<JetMP> next;
    next.reserve((level.size() + 1U) / 2U);
    std::size_t index = 0;
    for (; index + 1 < level.size(); index += 2)
      next.emplace_back(jet_add(level[index], level[index + 1]));
    if (index < level.size()) next.emplace_back(std::move(level[index]));
    level = std::move(next);
  }
  return std::move(level[0]);
}

JetMP jet_tanh(const JetMP& x) {
  const mpfr_prec_t precision = x.value.precision;
  JetMP result(precision);
  result.value = interval_tanh(x.value);
  const IntervalMP t = interval_tanh(x.value);
  const IntervalMP one = interval_point_float(1.0f, precision);
  const IntervalMP two = interval_point_float(2.0f, precision);
  const IntervalMP first_factor = interval_add(one, interval_neg(interval_square(t)));
  const IntervalMP second_factor = interval_mul(
      interval_neg(interval_mul(two, t)), first_factor);
  result.first = interval_mul(first_factor, x.first);
  result.second = interval_add(
      interval_mul(second_factor, interval_square(x.first)),
      interval_mul(first_factor, x.second));
  return result;
}

JetMP jet_exp(const JetMP& x) {
  const mpfr_prec_t precision = x.value.precision;
  JetMP result(precision);
  result.value = interval_exp(x.value);
  const IntervalMP first_factor = interval_exp(x.value);
  const IntervalMP second_factor = interval_exp(x.value);
  result.first = interval_mul(first_factor, x.first);
  result.second = interval_add(
      interval_mul(second_factor, interval_square(x.first)),
      interval_mul(first_factor, x.second));
  return result;
}

JetMP jet_reciprocal(const JetMP& x) {
  const mpfr_prec_t precision = x.value.precision;
  JetMP result(precision);
  const IntervalMP inverse = interval_reciprocal(x.value);
  const IntervalMP inverse2 = interval_square(inverse);
  const IntervalMP inverse3 = interval_mul(interval_square(inverse), inverse);
  const IntervalMP two = interval_point_float(2.0f, precision);
  result.value = interval_clone(inverse);
  result.first = interval_mul(interval_neg(x.first), inverse2);
  result.second = interval_add(
      interval_mul(interval_mul(two, interval_square(x.first)), inverse3),
      interval_neg(interval_mul(x.second, inverse2)));
  return result;
}

JetMP jet_gelu_new(const JetMP& x, float kappa, float lambda) {
  const mpfr_prec_t precision = x.value.precision;
  // Match the Python expression order exactly, including the nested x*x*x.
  JetMP x3 = jet_mul(jet_mul(x, x), x);
  JetMP inner = jet_add(x, jet_scale_float(x3, lambda));
  JetMP u = jet_scale_float(inner, kappa);
  JetMP tanh_u = jet_tanh(u);
  JetMP one = jet_constant(interval_point_float(1.0f, precision));
  return jet_scale_float(jet_mul(x, jet_add(one, tanh_u)), 0.5f);
}

JetMP synthetic_jet(std::size_t index, mpfr_prec_t precision) {
  JetMP result(precision);
  IntervalMP* components[3] = {&result.value, &result.first, &result.second};
  const long numerators[3] = {
      static_cast<long>((index * 37U) % 2048U) - 1024L,
      static_cast<long>((index * 17U) % 512U) - 256L,
      static_cast<long>((index * 11U) % 256U) - 128L,
  };
  const unsigned shifts[3] = {10U, 12U, 13U};
  for (std::size_t component = 0; component < 3; ++component) {
    mpfr_set_si(components[component]->lower.get(), numerators[component], MPFR_RNDN);
    mpfr_div_2ui(components[component]->lower.get(), components[component]->lower.get(),
                 shifts[component], MPFR_RNDN);
    mpfr_set(components[component]->upper.get(), components[component]->lower.get(), MPFR_RNDN);
  }
  return result;
}

IntervalMP interval_point_double(double raw, mpfr_prec_t precision) {
  IntervalMP result(precision);
  mpfr_set_d(result.lower.get(), raw, MPFR_RNDN);
  mpfr_set(result.upper.get(), result.lower.get(), MPFR_RNDN);
  return result;
}

JetMP jet_scale_double(const JetMP& value, double scalar) {
  return jet_scale_interval(value, interval_point_double(scalar, value.value.precision));
}

std::vector<JetMP> attention_final_head(
    const std::vector<JetMP>& query, const std::vector<JetMP>& keys,
    const std::vector<JetMP>& values, std::uint32_t sequence_length,
    std::uint32_t head_dim, std::uint32_t pivot) {
  const mpfr_prec_t precision = query[0].value.precision;
  const double scaling = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<JetMP> scores;
  scores.reserve(sequence_length);
  for (std::uint32_t token = 0; token < sequence_length; ++token) {
    std::vector<JetMP> products;
    products.reserve(head_dim);
    for (std::uint32_t coordinate = 0; coordinate < head_dim; ++coordinate)
      products.emplace_back(jet_mul(query[coordinate], keys[token * head_dim + coordinate]));
    scores.emplace_back(jet_scale_double(jet_pairwise_sum(products), scaling));
  }
  std::vector<JetMP> exponentials;
  exponentials.reserve(sequence_length);
  for (std::uint32_t token = 0; token < sequence_length; ++token) {
    if (token == pivot) {
      exponentials.emplace_back(jet_constant(interval_point_float(1.0f, precision)));
    } else {
      exponentials.emplace_back(jet_exp(jet_sub(scores[token], scores[pivot])));
    }
  }
  JetMP inverse_denominator = jet_reciprocal(jet_pairwise_sum(exponentials));
  std::vector<JetMP> weights;
  weights.reserve(sequence_length);
  for (const JetMP& exponential : exponentials)
    weights.emplace_back(jet_mul(exponential, inverse_denominator));
  std::vector<JetMP> output;
  output.reserve(head_dim);
  for (std::uint32_t coordinate = 0; coordinate < head_dim; ++coordinate) {
    std::vector<JetMP> terms;
    terms.reserve(sequence_length);
    for (std::uint32_t token = 0; token < sequence_length; ++token)
      terms.emplace_back(jet_mul(weights[token], values[token * head_dim + coordinate]));
    output.emplace_back(jet_pairwise_sum(terms));
  }
  return output;
}

IntervalMP fused_contrast_scalar(
    const std::uint32_t* source_bits, std::size_t stride,
    const std::int64_t* suffix_ids, const std::uint64_t* coefficient_bits,
    std::uint32_t contrast_width, std::uint32_t vocabulary_size,
    mpfr_prec_t precision) {
  std::vector<MpfrValue> lower_terms, upper_terms;
  lower_terms.reserve(contrast_width); upper_terms.reserve(contrast_width);
  for (std::uint32_t index = 0; index < contrast_width; ++index) {
    lower_terms.emplace_back(precision); upper_terms.emplace_back(precision);
  }
  std::vector<MpfrValue*> lower_pointers, upper_pointers;
  lower_pointers.reserve(contrast_width); upper_pointers.reserve(contrast_width);
  MpfrValue coefficient(precision), source(precision);
  for (std::uint32_t index = 0; index < contrast_width; ++index) {
    const std::size_t source_index = stride * static_cast<std::size_t>(suffix_ids[index]);
    mpfr_set_d(coefficient.get(), double_from_bits(coefficient_bits[index]), MPFR_RNDN);
    mpfr_set_flt(source.get(), float_from_bits(source_bits[source_index]), MPFR_RNDN);
    mpfr_mul(lower_terms[index].get(), coefficient.get(), source.get(), MPFR_RNDD);
    mpfr_mul(upper_terms[index].get(), coefficient.get(), source.get(), MPFR_RNDU);
    lower_pointers.push_back(&lower_terms[index]);
    upper_pointers.push_back(&upper_terms[index]);
  }
  IntervalMP result(precision);
  pairwise_sum(lower_pointers, result.lower.get(), MPFR_RNDD);
  pairwise_sum(upper_pointers, result.upper.get(), MPFR_RNDU);
  return result;
}

IntervalMP interval_scale_known_float(const IntervalMP& input, float scalar) {
  IntervalMP result(input.precision);
  MpfrValue weight(input.precision);
  mpfr_set_flt(weight.get(), scalar, MPFR_RNDN);
  if (mpfr_sgn(weight.get()) >= 0) {
    counted_mul(result.lower.get(), input.lower.get(), weight.get(), MPFR_RNDD);
    counted_mul(result.upper.get(), input.upper.get(), weight.get(), MPFR_RNDU);
  } else {
    counted_mul(result.lower.get(), input.upper.get(), weight.get(), MPFR_RNDD);
    counted_mul(result.upper.get(), input.lower.get(), weight.get(), MPFR_RNDU);
  }
  return result;
}

JetMP jet_scale_known_float(const JetMP& input, float scalar) {
  JetMP result(input.value.precision);
  result.value = interval_scale_known_float(input.value, scalar);
  result.first = interval_scale_known_float(input.first, scalar);
  result.second = interval_scale_known_float(input.second, scalar);
  return result;
}

std::vector<JetMP> synthetic_affine_layer(
    const std::vector<JetMP>& inputs, std::uint32_t output_width, std::uint32_t salt) {
  const mpfr_prec_t precision = inputs[0].value.precision;
  std::vector<MpfrValue> lower_terms, upper_terms;
  lower_terms.reserve(inputs.size()); upper_terms.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    lower_terms.emplace_back(precision); upper_terms.emplace_back(precision);
  }
  std::vector<MpfrValue*> lower_pointers, upper_pointers;
  lower_pointers.reserve(inputs.size()); upper_pointers.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    lower_pointers.push_back(&lower_terms[index]);
    upper_pointers.push_back(&upper_terms[index]);
  }
  MpfrValue weight_value(precision);
  std::vector<JetMP> outputs;
  outputs.reserve(output_width);
  for (std::uint32_t output = 0; output < output_width; ++output) {
    outputs.emplace_back(precision);
    IntervalMP* target_components[3] = {
        &outputs.back().value, &outputs.back().first, &outputs.back().second};
    for (std::size_t component = 0; component < 3; ++component) {
      for (std::uint32_t index = 0; index < inputs.size(); ++index) {
        const std::uint32_t hash = (index + 1U) * 2654435761U
            ^ (output + 17U) * 2246822519U ^ (salt + 31U) * 3266489917U;
        const float weight = (static_cast<int>(hash % 2049U) - 1024) / 4096.0f;
        mpfr_set_flt(weight_value.get(), weight, MPFR_RNDN);
        const IntervalMP* source_components[3] = {
            &inputs[index].value, &inputs[index].first, &inputs[index].second};
        const IntervalMP& source = *source_components[component];
        if (mpfr_sgn(weight_value.get()) >= 0) {
          counted_mul(lower_terms[index].get(), source.lower.get(), weight_value.get(), MPFR_RNDD);
          counted_mul(upper_terms[index].get(), source.upper.get(), weight_value.get(), MPFR_RNDU);
        } else {
          counted_mul(lower_terms[index].get(), source.upper.get(), weight_value.get(), MPFR_RNDD);
          counted_mul(upper_terms[index].get(), source.lower.get(), weight_value.get(), MPFR_RNDU);
        }
      }
      pairwise_sum(lower_pointers, target_components[component]->lower.get(), MPFR_RNDD);
      pairwise_sum(upper_pointers, target_components[component]->upper.get(), MPFR_RNDU);
    }
    const float bias = (static_cast<int>((output + salt * 13U) % 257U) - 128) / 8192.0f;
    mpfr_set_flt(weight_value.get(), bias, MPFR_RNDN);
    counted_add(outputs.back().value.lower.get(), outputs.back().value.lower.get(),
                weight_value.get(), MPFR_RNDD);
    counted_add(outputs.back().value.upper.get(), outputs.back().value.upper.get(),
                weight_value.get(), MPFR_RNDU);
  }
  return outputs;
}

std::vector<JetMP> packed_affine_layer(
    const std::vector<JetMP>& inputs, std::uint32_t output_width,
    const std::uint32_t* weight_bits, const std::uint32_t* bias_bits) {
  const mpfr_prec_t precision = inputs[0].value.precision;
  std::vector<MpfrValue> lower_terms, upper_terms;
  lower_terms.reserve(inputs.size()); upper_terms.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    lower_terms.emplace_back(precision); upper_terms.emplace_back(precision);
  }
  std::vector<MpfrValue*> lower_pointers, upper_pointers;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    lower_pointers.push_back(&lower_terms[index]); upper_pointers.push_back(&upper_terms[index]);
  }
  MpfrValue weight_value(precision);
  std::vector<JetMP> outputs;
  outputs.reserve(output_width);
  for (std::uint32_t output = 0; output < output_width; ++output) {
    outputs.emplace_back(precision);
    IntervalMP* targets[3] = {
        &outputs.back().value, &outputs.back().first, &outputs.back().second};
    for (std::size_t component = 0; component < 3; ++component) {
      for (std::size_t index = 0; index < inputs.size(); ++index) {
        mpfr_set_flt(weight_value.get(),
                     float_from_bits(weight_bits[index * output_width + output]), MPFR_RNDN);
        const IntervalMP* sources[3] = {
            &inputs[index].value, &inputs[index].first, &inputs[index].second};
        const IntervalMP& source = *sources[component];
        if (mpfr_sgn(weight_value.get()) >= 0) {
          counted_mul(lower_terms[index].get(), source.lower.get(), weight_value.get(), MPFR_RNDD);
          counted_mul(upper_terms[index].get(), source.upper.get(), weight_value.get(), MPFR_RNDU);
        } else {
          counted_mul(lower_terms[index].get(), source.upper.get(), weight_value.get(), MPFR_RNDD);
          counted_mul(upper_terms[index].get(), source.lower.get(), weight_value.get(), MPFR_RNDU);
        }
      }
      pairwise_sum(lower_pointers, targets[component]->lower.get(), MPFR_RNDD);
      pairwise_sum(upper_pointers, targets[component]->upper.get(), MPFR_RNDU);
    }
    mpfr_set_flt(weight_value.get(), float_from_bits(bias_bits[output]), MPFR_RNDN);
    counted_add(outputs.back().value.lower.get(), outputs.back().value.lower.get(),
                weight_value.get(), MPFR_RNDD);
    counted_add(outputs.back().value.upper.get(), outputs.back().value.upper.get(),
                weight_value.get(), MPFR_RNDU);
  }
  return outputs;
}

std::vector<JetMP> layer_norm_identity(const std::vector<JetMP>& inputs) {
  const mpfr_prec_t precision = inputs[0].value.precision;
  const IntervalMP reciprocal_width = interval_point_rational(1U, inputs.size(), precision);
  JetMP mean = jet_scale_interval(jet_pairwise_sum(inputs), reciprocal_width);
  std::vector<JetMP> centered;
  centered.reserve(inputs.size());
  for (const JetMP& input : inputs) centered.emplace_back(jet_sub(input, mean));
  std::vector<JetMP> squares;
  squares.reserve(inputs.size());
  for (const JetMP& value : centered) squares.emplace_back(jet_square(value));
  JetMP variance = jet_scale_interval(jet_pairwise_sum(squares), reciprocal_width);
  variance = jet_add(variance, jet_constant(interval_point_float(1.0e-5f, precision)));
  JetMP inverse_scale = jet_inv_sqrt(variance);
  std::vector<JetMP> output;
  output.reserve(inputs.size());
  for (const JetMP& value : centered) output.emplace_back(jet_mul(value, inverse_scale));
  return output;
}

std::vector<JetMP> gelu_vector(const std::vector<JetMP>& inputs) {
  std::vector<JetMP> output;
  output.reserve(inputs.size());
  for (const JetMP& value : inputs)
    output.emplace_back(jet_gelu_new(value, 0.7978845834732056f, 0.044715f));
  return output;
}

std::vector<JetMP> add_vectors(const std::vector<JetMP>& left,
                               const std::vector<JetMP>& right) {
  std::vector<JetMP> output;
  output.reserve(left.size());
  for (std::size_t index = 0; index < left.size(); ++index)
    output.emplace_back(jet_add(left[index], right[index]));
  return output;
}

JetMP synthetic_constant_jet(std::size_t index, mpfr_prec_t precision) {
  JetMP result = synthetic_jet(index, precision);
  mpfr_set_zero(result.first.lower.get(), 0); mpfr_set_zero(result.first.upper.get(), 0);
  mpfr_set_zero(result.second.lower.get(), 0); mpfr_set_zero(result.second.upper.get(), 0);
  return result;
}

JetMP synthetic_gpt2_tail(
    const std::vector<JetMP>& resid_post, std::uint32_t d_model,
    std::uint32_t d_mlp, std::uint32_t sequence_length,
    std::uint32_t n_heads, std::uint32_t d_head, std::uint32_t salt) {
  const mpfr_prec_t precision = resid_post[0].value.precision;
  record_dispatch(kLayerNorm);
  std::vector<JetMP> ln1 = layer_norm_identity(resid_post);
  record_dispatch(kPairwiseAffine);
  std::vector<JetMP> q = synthetic_affine_layer(ln1, d_model, salt + 101U);
  record_dispatch(kPairwiseAffine);
  std::vector<JetMP> k_final = synthetic_affine_layer(ln1, d_model, salt + 103U);
  record_dispatch(kPairwiseAffine);
  std::vector<JetMP> v_final = synthetic_affine_layer(ln1, d_model, salt + 107U);
  record_dispatch(kCausalAttention);
  std::vector<JetMP> attention;
  attention.reserve(d_model);
  for (std::uint32_t head = 0; head < n_heads; ++head) {
    const std::uint32_t start = head * d_head;
    std::vector<JetMP> query, keys, values;
    query.reserve(d_head);
    keys.reserve(static_cast<std::size_t>(sequence_length) * d_head);
    values.reserve(static_cast<std::size_t>(sequence_length) * d_head);
    for (std::uint32_t coordinate = 0; coordinate < d_head; ++coordinate)
      query.emplace_back(jet_clone(q[start + coordinate]));
    for (std::uint32_t token = 0; token < sequence_length; ++token) {
      for (std::uint32_t coordinate = 0; coordinate < d_head; ++coordinate) {
        if (token + 1U == sequence_length) {
          keys.emplace_back(jet_clone(k_final[start + coordinate]));
          values.emplace_back(jet_clone(v_final[start + coordinate]));
        } else {
          const std::size_t index = salt * 1000003ULL + token * d_model + start + coordinate;
          keys.emplace_back(synthetic_constant_jet(index + 3001U, precision));
          values.emplace_back(synthetic_constant_jet(index + 6007U, precision));
        }
      }
    }
    std::vector<JetMP> head_output = attention_final_head(
        query, keys, values, sequence_length, d_head, 0U);
    for (JetMP& value : head_output) attention.emplace_back(std::move(value));
  }
  record_dispatch(kPairwiseAffine);
  std::vector<JetMP> attention_out = synthetic_affine_layer(attention, d_model, salt + 109U);
  record_dispatch(kResidualAdd);
  std::vector<JetMP> resid_mid = add_vectors(resid_post, attention_out);
  record_dispatch(kLayerNorm);
  std::vector<JetMP> ln2 = layer_norm_identity(resid_mid);
  record_dispatch(kPairwiseAffine);
  std::vector<JetMP> pre = synthetic_affine_layer(ln2, d_mlp, salt + 113U);
  record_dispatch(kGeluNew);
  std::vector<JetMP> post = gelu_vector(pre);
  record_dispatch(kPairwiseAffine);
  std::vector<JetMP> mlp_out = synthetic_affine_layer(post, d_model, salt + 127U);
  record_dispatch(kResidualAdd);
  std::vector<JetMP> resid_final = add_vectors(resid_mid, mlp_out);
  record_dispatch(kLayerNorm);
  std::vector<JetMP> normalized = layer_norm_identity(resid_final);
  record_dispatch(kFinalContrast);
  std::vector<JetMP> contrast = synthetic_affine_layer(normalized, 1U, salt + 131U);
  return std::move(contrast[0]);
}

JetMP synthetic_joint_witness_cell(
    mpfr_prec_t precision, std::uint32_t d_model, std::uint32_t d_mlp,
    std::uint32_t sequence_length, std::uint32_t n_heads,
    std::uint32_t d_head, std::uint32_t selected_gates, std::uint32_t repeat) {
  std::vector<JetMP> roots;
  roots.reserve(4);
  for (std::uint32_t condition = 0; condition < 2; ++condition) {
    record_dispatch(kAffineScatter);
    record_dispatch(kAffineScatter);
    std::vector<JetMP> base, controlled;
    base.reserve(d_model); controlled.reserve(d_model);
    for (std::uint32_t coordinate = 0; coordinate < d_model; ++coordinate) {
      base.emplace_back(synthetic_jet(
          repeat * 10000019ULL + condition * 100003ULL + coordinate, precision));
      controlled.emplace_back(synthetic_jet(
          repeat * 10000079ULL + condition * 100019ULL + coordinate, precision));
    }
    record_dispatch(kStaticView);
    record_dispatch(kLayerNorm);
    std::vector<JetMP> ln10 = layer_norm_identity(controlled);
    std::vector<JetMP> zero_control;
    zero_control.reserve(d_model);
    for (const JetMP& value : controlled)
      zero_control.emplace_back(jet_constant(value.value));
    record_dispatch(kLayerNorm);
    std::vector<JetMP> zero_ln10 = layer_norm_identity(zero_control);
    record_dispatch(kPairwiseAffine);
    std::vector<JetMP> selected_pre = synthetic_affine_layer(
        ln10, selected_gates, 211U + condition + repeat * 17U);
    record_dispatch(kPairwiseAffine);
    std::vector<JetMP> zero_pre = synthetic_affine_layer(
        zero_ln10, selected_gates, 211U + condition + repeat * 17U);
    record_dispatch(kGeluNew);
    std::vector<JetMP> selected_live = gelu_vector(selected_pre);
    record_dispatch(kGeluNew);
    std::vector<JetMP> zero_live = gelu_vector(zero_pre);
    record_dispatch(kStaticView);
    std::vector<JetMP> selected_delta;
    selected_delta.reserve(selected_gates);
    for (std::size_t index = 0; index < selected_live.size(); ++index)
      selected_delta.emplace_back(jet_sub(selected_live[index], zero_live[index]));
    record_dispatch(kPairwiseAffine);
    std::vector<JetMP> delta_out = synthetic_affine_layer(
        selected_delta, d_model, 223U + condition + repeat * 19U);
    record_dispatch(kResidualAdd);
    std::vector<JetMP> joint = add_vectors(base, delta_out);
    roots.emplace_back(synthetic_gpt2_tail(
        joint, d_model, d_mlp, sequence_length, n_heads, d_head,
        307U + condition * 2U + repeat * 23U));
    roots.emplace_back(synthetic_gpt2_tail(
        base, d_model, d_mlp, sequence_length, n_heads, d_head,
        308U + condition * 2U + repeat * 23U));
  }
  record_dispatch(kBranchLinearCombination);
  return jet_add(jet_sub(jet_sub(roots[0], roots[1]), roots[2]), roots[3]);
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

extern "C" int green_v400_packed_affine_layer_jet2_exact(
    std::uint32_t precision_bits, std::uint32_t input_width,
    std::uint32_t output_width,
    const char* const* endpoint_significands, const std::int64_t* endpoint_exponents,
    const std::uint32_t* weight_bits, const std::uint32_t* bias_bits,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || input_width == 0
      || output_width == 0 || input_width > 100000U || output_width > 100000U
      || endpoint_significands == nullptr || endpoint_exponents == nullptr
      || weight_bits == nullptr || bias_bits == nullptr
      || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> inputs;
  inputs.reserve(input_width);
  for (std::uint32_t index = 0; index < input_width; ++index) {
    inputs.emplace_back(precision);
    IntervalMP* components[3] = {
        &inputs.back().value, &inputs.back().first, &inputs.back().second};
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t offset = static_cast<std::size_t>(index) * 6U + component * 2U;
      int status = set_exact_binary(components[component]->lower.get(),
                                    endpoint_significands[offset], endpoint_exponents[offset]);
      if (status != 0) return status;
      status = set_exact_binary(components[component]->upper.get(),
                                endpoint_significands[offset + 1U], endpoint_exponents[offset + 1U]);
      if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                       components[component]->upper.get())) return 3;
    }
  }
  std::vector<JetMP> outputs = packed_affine_layer(
      inputs, output_width, weight_bits, bias_bits);
  std::ostringstream stream;
  stream << "{\"outputs\":[";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0) stream << ',';
    const JetMP& output = outputs[index];
    stream << serialize_jet(output.value.lower.get(), output.value.upper.get(),
                            output.first.lower.get(), output.first.upper.get(),
                            output.second.lower.get(), output.second.upper.get(), precision);
  }
  stream << "]}";
  const std::string serialized = stream.str();
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

extern "C" int green_v400_benchmark_gelu_jet2(
    std::uint32_t precision_bits, std::uint32_t count, std::uint32_t repeats,
    double* elapsed_seconds, std::uint64_t* checksum) {
  if (precision_bits < 64 || precision_bits > 4096 || count == 0 || repeats == 0
      || count > 10000000U || elapsed_seconds == nullptr || checksum == nullptr) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> inputs;
  inputs.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index)
    inputs.emplace_back(synthetic_jet(index, precision));
  std::uint64_t state = 0xbb67ae8584caa73bULL;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
    for (const JetMP& input : inputs) {
      JetMP output = jet_gelu_new(input, 0.7978845834732056f, 0.044715f);
      state = mix_checksum(state, output.value.lower.get());
      state = mix_checksum(state, output.first.lower.get());
      state = mix_checksum(state, output.second.lower.get());
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  *elapsed_seconds = std::chrono::duration<double>(stop - start).count();
  *checksum = state;
  return 0;
}

extern "C" int green_v400_benchmark_layer_norm_jet2(
    std::uint32_t precision_bits, std::uint32_t width,
    std::uint32_t vector_count, std::uint32_t repeats,
    double* elapsed_seconds, std::uint64_t* checksum) {
  if (precision_bits < 64 || precision_bits > 4096 || width < 2 || vector_count == 0
      || repeats == 0 || width > 1000000U || vector_count > 1000000U
      || elapsed_seconds == nullptr || checksum == nullptr) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> inputs;
  inputs.reserve(width);
  for (std::uint32_t index = 0; index < width; ++index)
    inputs.emplace_back(synthetic_jet(index, precision));
  const IntervalMP reciprocal_width = interval_point_rational(1U, width, precision);
  const IntervalMP epsilon = interval_point_float(1.0e-5f, precision);
  std::uint64_t state = 0x3c6ef372fe94f82bULL;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::uint32_t vector_index = 0; vector_index < vector_count; ++vector_index) {
      JetMP mean = jet_scale_interval(jet_pairwise_sum(inputs), reciprocal_width);
      std::vector<JetMP> centered;
      centered.reserve(width);
      for (const JetMP& input : inputs) centered.emplace_back(jet_sub(input, mean));
      std::vector<JetMP> squares;
      squares.reserve(width);
      for (const JetMP& value : centered) squares.emplace_back(jet_square(value));
      JetMP variance = jet_scale_interval(jet_pairwise_sum(squares), reciprocal_width);
      variance = jet_add(variance, jet_constant(epsilon));
      if (mpfr_sgn(variance.value.lower.get()) <= 0) return 5;
      JetMP inverse_scale = jet_inv_sqrt(variance);
      for (const JetMP& value : centered) {
        JetMP output = jet_mul(value, inverse_scale);
        state = mix_checksum(state, output.value.lower.get());
        state = mix_checksum(state, output.first.lower.get());
        state = mix_checksum(state, output.second.lower.get());
      }
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  *elapsed_seconds = std::chrono::duration<double>(stop - start).count();
  *checksum = state;
  return 0;
}

extern "C" int green_v400_benchmark_causal_attention_jet2(
    std::uint32_t precision_bits, std::uint32_t sequence_length,
    std::uint32_t n_heads, std::uint32_t head_dim, std::uint32_t branch_count,
    std::uint32_t repeats, double* elapsed_seconds, std::uint64_t* checksum) {
  if (precision_bits < 64 || precision_bits > 4096 || sequence_length == 0
      || n_heads == 0 || head_dim == 0 || branch_count == 0 || repeats == 0
      || sequence_length > 4096U || n_heads > 4096U || head_dim > 4096U
      || branch_count > 4096U || elapsed_seconds == nullptr || checksum == nullptr) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> query, keys, values;
  query.reserve(head_dim);
  keys.reserve(static_cast<std::size_t>(sequence_length) * head_dim);
  values.reserve(static_cast<std::size_t>(sequence_length) * head_dim);
  for (std::uint32_t coordinate = 0; coordinate < head_dim; ++coordinate)
    query.emplace_back(synthetic_jet(coordinate + 101U, precision));
  for (std::uint32_t index = 0; index < sequence_length * head_dim; ++index) {
    keys.emplace_back(synthetic_jet(index + 1009U, precision));
    values.emplace_back(synthetic_jet(index + 2003U, precision));
  }
  std::uint64_t state = 0xa54ff53a5f1d36f1ULL;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::uint32_t branch = 0; branch < branch_count; ++branch) {
      for (std::uint32_t head = 0; head < n_heads; ++head) {
        std::vector<JetMP> output = attention_final_head(
            query, keys, values, sequence_length, head_dim, 0U);
        for (const JetMP& component : output) {
          state = mix_checksum(state, component.value.lower.get());
          state = mix_checksum(state, component.first.lower.get());
          state = mix_checksum(state, component.second.lower.get());
        }
      }
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  *elapsed_seconds = std::chrono::duration<double>(stop - start).count();
  *checksum = state;
  return 0;
}

extern "C" int green_v400_benchmark_gpt2_joint_witness_cell(
    std::uint32_t precision_bits, std::uint32_t d_model, std::uint32_t d_mlp,
    std::uint32_t sequence_length, std::uint32_t n_heads,
    std::uint32_t d_head, std::uint32_t selected_gates, std::uint32_t repeats,
    double* elapsed_seconds, std::uint64_t* checksum,
    std::uint64_t* primitive_count, std::uint64_t* dispatch_trace,
    std::uint64_t* dispatch_events, std::uint8_t* dispatch_tags,
    std::uint64_t dispatch_tag_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || d_model == 0 || d_mlp == 0
      || sequence_length == 0 || n_heads == 0 || d_head == 0 || selected_gates == 0
      || n_heads * d_head != d_model || selected_gates > d_mlp || repeats == 0
      || d_model > 100000U || d_mlp > 100000U || sequence_length > 4096U
      || elapsed_seconds == nullptr || checksum == nullptr
      || primitive_count == nullptr || dispatch_trace == nullptr
      || dispatch_events == nullptr || dispatch_tags == nullptr
      || dispatch_tag_capacity < 81ULL * repeats) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::uint64_t state = 0x510e527fade682d1ULL;
  *primitive_count = 0;
  *dispatch_trace = 14695981039346656037ULL;
  *dispatch_events = 0;
  active_primitive_counter = primitive_count;
  active_dispatch_trace = dispatch_trace;
  active_dispatch_events = dispatch_events;
  active_dispatch_tags = dispatch_tags;
  active_dispatch_tag_capacity = dispatch_tag_capacity;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
    JetMP output = synthetic_joint_witness_cell(
        precision, d_model, d_mlp, sequence_length, n_heads,
        d_head, selected_gates, repeat);
    state = mix_checksum(state, output.value.lower.get());
    state = mix_checksum(state, output.value.upper.get());
    state = mix_checksum(state, output.first.lower.get());
    state = mix_checksum(state, output.first.upper.get());
    state = mix_checksum(state, output.second.lower.get());
    state = mix_checksum(state, output.second.upper.get());
  }
  const auto stop = std::chrono::steady_clock::now();
  active_primitive_counter = nullptr;
  active_dispatch_trace = nullptr;
  active_dispatch_events = nullptr;
  active_dispatch_tags = nullptr;
  active_dispatch_tag_capacity = 0;
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

extern "C" int green_v400_gelu_new_jet2_exact(
    std::uint32_t precision_bits, const char* const* endpoint_significands,
    const std::int64_t* endpoint_exponents, std::uint32_t kappa_bits,
    std::uint32_t lambda_bits, char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || endpoint_significands == nullptr
      || endpoint_exponents == nullptr || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  JetMP input(precision);
  IntervalMP* components[3] = {&input.value, &input.first, &input.second};
  for (std::size_t component = 0; component < 3; ++component) {
    int status = set_exact_binary(components[component]->lower.get(),
                                  endpoint_significands[2 * component],
                                  endpoint_exponents[2 * component]);
    if (status != 0) return status;
    status = set_exact_binary(components[component]->upper.get(),
                              endpoint_significands[2 * component + 1],
                              endpoint_exponents[2 * component + 1]);
    if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                     components[component]->upper.get())) return 3;
  }
  JetMP result = jet_gelu_new(
      input, float_from_bits(kappa_bits), float_from_bits(lambda_bits));
  const std::string serialized = serialize_jet(
      result.value.lower.get(), result.value.upper.get(),
      result.first.lower.get(), result.first.upper.get(),
      result.second.lower.get(), result.second.upper.get(), precision);
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_gelu_new_layer_jet2_exact(
    std::uint32_t precision_bits, std::uint32_t width,
    const char* const* endpoint_significands, const std::int64_t* endpoint_exponents,
    std::uint32_t kappa_bits, std::uint32_t lambda_bits,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || width == 0 || width > 1000000U
      || endpoint_significands == nullptr || endpoint_exponents == nullptr
      || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> outputs;
  outputs.reserve(width);
  for (std::uint32_t index = 0; index < width; ++index) {
    JetMP input(precision);
    IntervalMP* components[3] = {&input.value, &input.first, &input.second};
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t offset = static_cast<std::size_t>(index) * 6U + 2U * component;
      int status = set_exact_binary(components[component]->lower.get(),
                                    endpoint_significands[offset], endpoint_exponents[offset]);
      if (status != 0) return status;
      status = set_exact_binary(components[component]->upper.get(),
                                endpoint_significands[offset + 1U],
                                endpoint_exponents[offset + 1U]);
      if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                       components[component]->upper.get())) return 3;
    }
    outputs.emplace_back(jet_gelu_new(
        input, float_from_bits(kappa_bits), float_from_bits(lambda_bits)));
  }
  std::ostringstream stream;
  stream << "{\"outputs\":[";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0) stream << ',';
    const JetMP& output = outputs[index];
    stream << serialize_jet(output.value.lower.get(), output.value.upper.get(),
                            output.first.lower.get(), output.first.upper.get(),
                            output.second.lower.get(), output.second.upper.get(), precision);
  }
  stream << "]}";
  const std::string serialized = stream.str();
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_layer_norm_jet2_exact(
    std::uint32_t precision_bits, std::uint32_t width,
    const char* const* endpoint_significands, const std::int64_t* endpoint_exponents,
    std::uint32_t epsilon_bits, const std::uint32_t* gamma_bits,
    const std::uint32_t* beta_bits, char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || width == 0 || width > 1000000U
      || endpoint_significands == nullptr || endpoint_exponents == nullptr
      || gamma_bits == nullptr || beta_bits == nullptr || output_json == nullptr
      || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> inputs;
  inputs.reserve(width);
  for (std::uint32_t index = 0; index < width; ++index) {
    inputs.emplace_back(precision);
    IntervalMP* components[3] = {&inputs.back().value, &inputs.back().first,
                                 &inputs.back().second};
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t offset = static_cast<std::size_t>(index) * 6U + 2U * component;
      int status = set_exact_binary(components[component]->lower.get(),
                                    endpoint_significands[offset], endpoint_exponents[offset]);
      if (status != 0) return status;
      status = set_exact_binary(components[component]->upper.get(),
                                endpoint_significands[offset + 1], endpoint_exponents[offset + 1]);
      if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                       components[component]->upper.get())) return 3;
    }
  }
  const IntervalMP reciprocal_width = interval_point_rational(1U, width, precision);
  JetMP mean = jet_scale_interval(jet_pairwise_sum(inputs), reciprocal_width);
  std::vector<JetMP> centered;
  centered.reserve(width);
  for (const JetMP& input : inputs) centered.emplace_back(jet_sub(input, mean));
  std::vector<JetMP> squares;
  squares.reserve(width);
  for (const JetMP& value : centered) squares.emplace_back(jet_square(value));
  JetMP variance = jet_scale_interval(jet_pairwise_sum(squares), reciprocal_width);
  variance = jet_add(variance, jet_constant(interval_point_float(
      float_from_bits(epsilon_bits), precision)));
  if (mpfr_sgn(variance.value.lower.get()) <= 0) return 5;
  JetMP inverse_scale = jet_inv_sqrt(variance);
  std::ostringstream stream;
  stream << "{\"schema_version\":\"green-v400-compiled-layernorm-jet2-v1\",";
  stream << "\"precision_bits\":" << precision_bits << ",\"outputs\":[";
  for (std::uint32_t index = 0; index < width; ++index) {
    JetMP normalized = jet_mul(centered[index], inverse_scale);
    JetMP scaled = jet_scale_float(normalized, float_from_bits(gamma_bits[index]));
    JetMP output = jet_add(scaled, jet_constant(interval_point_float(
        float_from_bits(beta_bits[index]), precision)));
    if (index) stream << ',';
    stream << serialize_jet(output.value.lower.get(), output.value.upper.get(),
                            output.first.lower.get(), output.first.upper.get(),
                            output.second.lower.get(), output.second.upper.get(), precision);
  }
  stream << "]}";
  const std::string serialized = stream.str();
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_causal_attention_final_head_jet2_exact(
    std::uint32_t precision_bits, std::uint32_t sequence_length,
    std::uint32_t head_dim, std::uint32_t pivot,
    const char* const* endpoint_significands, const std::int64_t* endpoint_exponents,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || sequence_length == 0
      || head_dim == 0 || pivot >= sequence_length || sequence_length > 4096U
      || head_dim > 4096U || endpoint_significands == nullptr
      || endpoint_exponents == nullptr || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  const std::size_t query_count = head_dim;
  const std::size_t matrix_count = static_cast<std::size_t>(sequence_length) * head_dim;
  const std::size_t total_count = query_count + 2U * matrix_count;
  std::vector<JetMP> all;
  all.reserve(total_count);
  for (std::size_t index = 0; index < total_count; ++index) {
    all.emplace_back(precision);
    IntervalMP* components[3] = {&all.back().value, &all.back().first, &all.back().second};
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t offset = index * 6U + component * 2U;
      int status = set_exact_binary(components[component]->lower.get(),
                                    endpoint_significands[offset], endpoint_exponents[offset]);
      if (status != 0) return status;
      status = set_exact_binary(components[component]->upper.get(),
                                endpoint_significands[offset + 1], endpoint_exponents[offset + 1]);
      if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                       components[component]->upper.get())) return 3;
    }
  }
  std::vector<JetMP> query, keys, values;
  query.reserve(query_count); keys.reserve(matrix_count); values.reserve(matrix_count);
  for (std::size_t index = 0; index < query_count; ++index)
    query.emplace_back(jet_clone(all[index]));
  for (std::size_t index = 0; index < matrix_count; ++index) {
    keys.emplace_back(jet_clone(all[query_count + index]));
    values.emplace_back(jet_clone(all[query_count + matrix_count + index]));
  }
  std::vector<JetMP> result = attention_final_head(
      query, keys, values, sequence_length, head_dim, pivot);
  std::ostringstream stream;
  stream << "{\"schema_version\":\"green-v400-compiled-causal-attention-jet2-v1\",";
  stream << "\"precision_bits\":" << precision_bits << ",\"outputs\":[";
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (index) stream << ',';
    const JetMP& output = result[index];
    stream << serialize_jet(output.value.lower.get(), output.value.upper.get(),
                            output.first.lower.get(), output.first.upper.get(),
                            output.second.lower.get(), output.second.upper.get(), precision);
  }
  stream << "]}";
  const std::string serialized = stream.str();
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_causal_attention_final_all_heads_jet2_exact(
    std::uint32_t precision_bits, std::uint32_t sequence_length,
    std::uint32_t n_heads, std::uint32_t head_dim, std::uint32_t pivot,
    const char* const* endpoint_significands, const std::int64_t* endpoint_exponents,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || sequence_length == 0
      || n_heads == 0 || head_dim == 0 || pivot >= sequence_length
      || sequence_length > 4096U || n_heads > 4096U || head_dim > 4096U
      || endpoint_significands == nullptr || endpoint_exponents == nullptr
      || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  const std::size_t d_model = static_cast<std::size_t>(n_heads) * head_dim;
  const std::size_t matrix_count = static_cast<std::size_t>(sequence_length) * d_model;
  auto parse_jet = [&](std::size_t input_index, JetMP& target) -> int {
    IntervalMP* components[3] = {&target.value, &target.first, &target.second};
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t offset = input_index * 6U + component * 2U;
      int status = set_exact_binary(components[component]->lower.get(),
                                    endpoint_significands[offset], endpoint_exponents[offset]);
      if (status != 0) return status;
      status = set_exact_binary(components[component]->upper.get(),
                                endpoint_significands[offset + 1U], endpoint_exponents[offset + 1U]);
      if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                       components[component]->upper.get())) return 3;
    }
    return 0;
  };
  std::ostringstream stream;
  stream << "{\"schema_version\":\"green-v400-compiled-causal-attention-all-heads-jet2-v1\",";
  stream << "\"precision_bits\":" << precision_bits << ",\"outputs\":[";
  std::size_t output_index = 0;
  for (std::uint32_t head = 0; head < n_heads; ++head) {
    const std::size_t head_start = static_cast<std::size_t>(head) * head_dim;
    std::vector<JetMP> query, keys, values;
    query.reserve(head_dim);
    keys.reserve(static_cast<std::size_t>(sequence_length) * head_dim);
    values.reserve(static_cast<std::size_t>(sequence_length) * head_dim);
    for (std::uint32_t coordinate = 0; coordinate < head_dim; ++coordinate) {
      query.emplace_back(precision);
      const int status = parse_jet(head_start + coordinate, query.back());
      if (status != 0) return status;
    }
    for (std::uint32_t token = 0; token < sequence_length; ++token) {
      for (std::uint32_t coordinate = 0; coordinate < head_dim; ++coordinate) {
        const std::size_t index = static_cast<std::size_t>(token) * d_model
                                  + head_start + coordinate;
        keys.emplace_back(precision);
        values.emplace_back(precision);
        int status = parse_jet(d_model + index, keys.back());
        if (status != 0) return status;
        status = parse_jet(d_model + matrix_count + index, values.back());
        if (status != 0) return status;
      }
    }
    std::vector<JetMP> head_outputs = attention_final_head(
        query, keys, values, sequence_length, head_dim, pivot);
    for (const JetMP& output : head_outputs) {
      if (output_index++) stream << ',';
      stream << serialize_jet(output.value.lower.get(), output.value.upper.get(),
                              output.first.lower.get(), output.first.upper.get(),
                              output.second.lower.get(), output.second.upper.get(), precision);
    }
  }
  stream << "]}";
  const std::string serialized = stream.str();
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_fused_contrast_jet2_exact(
    std::uint32_t precision_bits, std::uint32_t d_model,
    const char* const* endpoint_significands, const std::int64_t* endpoint_exponents,
    const char* const* weight_significands, const std::int64_t* weight_exponents,
    const char* bias_significand, std::int64_t bias_exponent,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || d_model == 0
      || d_model > 1000000U || endpoint_significands == nullptr
      || endpoint_exponents == nullptr || weight_significands == nullptr
      || weight_exponents == nullptr || bias_significand == nullptr
      || output_json == nullptr || output_capacity == 0) return 2;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> inputs;
  inputs.reserve(d_model);
  for (std::uint32_t index = 0; index < d_model; ++index) {
    inputs.emplace_back(precision);
    IntervalMP* components[3] = {
        &inputs.back().value, &inputs.back().first, &inputs.back().second};
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t offset = static_cast<std::size_t>(index) * 6U + component * 2U;
      int status = set_exact_binary(components[component]->lower.get(),
                                    endpoint_significands[offset], endpoint_exponents[offset]);
      if (status != 0) return status;
      status = set_exact_binary(components[component]->upper.get(),
                                endpoint_significands[offset + 1U], endpoint_exponents[offset + 1U]);
      if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                       components[component]->upper.get())) return 3;
    }
  }
  std::vector<JetMP> terms;
  terms.reserve(d_model);
  for (std::uint32_t index = 0; index < d_model; ++index) {
    IntervalMP weight(precision);
    int status = set_exact_binary(weight.lower.get(), weight_significands[index],
                                  weight_exponents[index]);
    if (status != 0) return status;
    mpfr_set(weight.upper.get(), weight.lower.get(), MPFR_RNDN);
    terms.emplace_back(jet_scale_interval(inputs[index], weight));
  }
  JetMP result = jet_pairwise_sum(terms);
  IntervalMP bias(precision);
  int status = set_exact_binary(bias.lower.get(), bias_significand, bias_exponent);
  if (status != 0) return status;
  mpfr_set(bias.upper.get(), bias.lower.get(), MPFR_RNDN);
  result = jet_add(result, jet_constant(bias));
  const std::string serialized = serialize_jet(
      result.value.lower.get(), result.value.upper.get(),
      result.first.lower.get(), result.first.upper.get(),
      result.second.lower.get(), result.second.upper.get(), precision);
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}

extern "C" int green_v400_final_contrast_jet2_exact(
    std::uint32_t precision_bits, std::uint32_t d_model,
    std::uint32_t vocabulary_size, std::uint32_t contrast_width,
    const char* const* endpoint_significands, const std::int64_t* endpoint_exponents,
    const std::uint32_t* unembed_bits, const std::uint32_t* bias_bits,
    const std::int64_t* suffix_ids, const std::uint64_t* coefficient_bits,
    char* output_json, std::uint64_t output_capacity) {
  if (precision_bits < 64 || precision_bits > 4096 || d_model == 0
      || vocabulary_size == 0 || contrast_width == 0 || d_model > 1000000U
      || vocabulary_size > 1000000U || contrast_width > vocabulary_size
      || endpoint_significands == nullptr || endpoint_exponents == nullptr
      || unembed_bits == nullptr || bias_bits == nullptr || suffix_ids == nullptr
      || coefficient_bits == nullptr || output_json == nullptr || output_capacity == 0) return 2;
  for (std::uint32_t index = 0; index < contrast_width; ++index)
    if (suffix_ids[index] < 0 || suffix_ids[index] >= vocabulary_size) return 3;
  const mpfr_prec_t precision = static_cast<mpfr_prec_t>(precision_bits);
  std::vector<JetMP> inputs;
  inputs.reserve(d_model);
  for (std::uint32_t index = 0; index < d_model; ++index) {
    inputs.emplace_back(precision);
    IntervalMP* components[3] = {&inputs.back().value, &inputs.back().first,
                                 &inputs.back().second};
    for (std::size_t component = 0; component < 3; ++component) {
      const std::size_t offset = static_cast<std::size_t>(index) * 6U + component * 2U;
      int status = set_exact_binary(components[component]->lower.get(),
                                    endpoint_significands[offset], endpoint_exponents[offset]);
      if (status != 0) return status;
      status = set_exact_binary(components[component]->upper.get(),
                                endpoint_significands[offset + 1], endpoint_exponents[offset + 1]);
      if (status != 0 || mpfr_greater_p(components[component]->lower.get(),
                                       components[component]->upper.get())) return 3;
    }
  }
  std::vector<JetMP> terms;
  terms.reserve(d_model);
  for (std::uint32_t coordinate = 0; coordinate < d_model; ++coordinate) {
    const IntervalMP fused_weight = fused_contrast_scalar(
        unembed_bits + static_cast<std::size_t>(coordinate) * vocabulary_size,
        1U, suffix_ids, coefficient_bits, contrast_width, vocabulary_size, precision);
    terms.emplace_back(jet_scale_interval(inputs[coordinate], fused_weight));
  }
  JetMP result = jet_pairwise_sum(terms);
  const IntervalMP fused_bias = fused_contrast_scalar(
      bias_bits, 1U, suffix_ids, coefficient_bits,
      contrast_width, vocabulary_size, precision);
  result = jet_add(result, jet_constant(fused_bias));
  const std::string serialized = serialize_jet(
      result.value.lower.get(), result.value.upper.get(),
      result.first.lower.get(), result.first.upper.get(),
      result.second.lower.get(), result.second.upper.get(), precision);
  if (serialized.size() + 1 > output_capacity) return 4;
  std::memcpy(output_json, serialized.c_str(), serialized.size() + 1);
  return 0;
}
