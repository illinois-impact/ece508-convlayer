#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <type_traits>

#ifdef __GNUC__
#define unused __attribute__((unused))
#else // __GNUC__
#define unused
#endif // __GNUC__

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T>
static bool check_success(const T &err);

template <>
bool check_success<cudaError_t>(const cudaError_t &err) {
  const auto res = err == cudaSuccess;
  if (res == true) {
    return res;
  }
  std::cout << "Failed in CUDA. Error = " << cudaGetErrorString(err) << std::endl;
  assert(res);
  return res;
}

struct shape {
  size_t num{0};    // number of images in mini-batch
  size_t depth{0};  // number of input/output feature maps
  size_t height{0}; // height of the image
  size_t width{0};  // width of the image

  shape(size_t height, size_t width) : num(1), depth(1), height(height), width(width) {
  }

  shape(size_t num, size_t depth, size_t height, size_t width) : num(num), depth(depth), height(height), width(width) {
  }

  size_t flattened_length() const {
    return num * depth * height * width;
  }
};

template <typename T, size_t N>
constexpr size_t array_size(const T (&)[N]) {
  return N;
}

template <typename SzTy, size_t N>
static size_t flattened_length(const SzTy (&idims)[N]) {
  const auto dims = std::valarray<SzTy>(idims, N);
  size_t len      = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<SzTy>());
  return len;
}

template <typename SzTy>
static size_t flattened_length(const SzTy n) {
  return static_cast<size_t>(n);
}

template <>
size_t flattened_length(const shape &shp) {
  return shp.flattened_length();
}

template <typename T, typename SzTy>
static enable_if_t<std::is_integral<SzTy>::value, T> *allocate(const SzTy len) {
  T *res = new T[len];
  return res;
}

template <typename T, typename ShapeT>
static enable_if_t<std::is_same<ShapeT, shape>::value, T *> allocate(const ShapeT &shp) {
  T *res = new T[shp.flattened_length()];
  return res;
}

template <typename T, typename SzTy, size_t N>
static T *allocate(const SzTy (&dims)[N]) {
  const auto len = flattened_length(dims);
  return allocate<T>(len);
}

template <typename T, typename SzTy, size_t N>
static T *zeros(const SzTy (&dims)[N]) {
  const auto len = flattened_length(dims);
  auto res       = allocate<T, SzTy>(len);
  std::fill(res, res + len, static_cast<T>(0));
  return res;
}

template <typename T, typename SzTy>
static enable_if_t<std::is_integral<SzTy>::value, T *> zeros(const SzTy len) {
  auto res = allocate<T, SzTy>(len);
  std::fill(res, res + len, static_cast<T>(0));
  return res;
}

template <typename T, typename ShapeT>
static enable_if_t<std::is_same<ShapeT, shape>::value, T *> zeros(const ShapeT &shp) {
  auto res = allocate<T, shape>(shp);
  std::fill(res, res + shp.flattened_length(), static_cast<T>(0));
  return res;
}

static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

/*********************************************************************/
/* Random number generator                                           */
/* https://en.wikipedia.org/wiki/Xorshift                            */
/* xorshift32                                                        */
/*********************************************************************/

static uint_fast32_t rng_uint32(uint_fast32_t *rng_state) {
  uint_fast32_t local = *rng_state;
  local ^= local << 13; // a
  local ^= local >> 17; // b
  local ^= local << 5;  // c
  *rng_state = local;
  return local;
}

static uint_fast32_t *rng_new_state(uint_fast32_t seed) {
  uint64_t *rng_state = new uint64_t;
  *rng_state          = seed;
  return rng_state;
}

static uint_fast32_t *rng_new_state() {
  return rng_new_state(88172645463325252LL);
}

static float rng_float(uint_fast32_t *state) {
  uint_fast32_t rnd = rng_uint32(state);
  const auto r      = static_cast<float>(rnd) / static_cast<float>(UINT_FAST32_MAX);
  if (std::isfinite(r)) {
    return r;
  }
  return rng_float(state);
}

#endif // __UTILS_HPP__
