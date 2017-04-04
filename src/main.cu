#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sys/time.h>
#include <valarray>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define TILE_WIDTH 4

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
shape xdims        = {FLAGS_batch_size, NUM_CHANNELS, NUM_ROWS, NUM_COLS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

shape conv1dims = {32, 1, 5, 5};
shape conv2dims = {64, 32, 5, 5};
shape fc1dims   = {128, 1024};
shape fc2dims   = {10, 128};

/******************************************************************************
 GPU Kernels
*******************************************************************************/

/******************************************************************************
 Host Functions
*******************************************************************************/

/******************************************************************************
 Sequential Functions
*******************************************************************************/
// generate input and output data with random values
static void generate_data(float *x, const shape &xdims) {
  // input dimension size
  std::cout << "generating tensor with input dimensions = " << xdims.num << " x " << xdims.depth << " x "
            << xdims.height << " x " << xdims.width << "\n";

  const float mu{0};     // mean
  const float stddev{1}; // standard deviation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(mu, stddev);

  std::generate(x, x + xdims->flattened_length(), dis(gen));
}

// generate convolution filter
static void generate_convfilters(float *conv, const shape &convdim) {
  // convolution filter dimension size
  std::cout << "filter dimensions = " << convdim.num << " x " << convdim.depth << " x " << convdim.height << " x "
            << convdim.width << "\n";

  // Set convolution filter values to 1
  std::fill(conv, conv + convdim->flattened_length(), 1);
}

// Rectified linear unit 4d
static void relu4(float *X, const shape xdims) {
  for (const auto i : range(0, xdims.num * xdims.depth * xdims.height * xdims.width)) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Rectified linear unit 2d
static void relu2(float *X, const shape xdims) {
  for (const auto i : range(0, xdims.num * xdims.depth)) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
static void average_pool(const float *X, const shape xdims, const int pool_size, float *Y, const shape ydims) {
  for (const auto i : range(0, ydims.num)) {
    for (const auto m : range(0, ydims.depth)) {
      for (const auto h : range(0, ydims.height)) {
        for (const auto w : range(0, ydims.width)) {
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset = ((i * ydims.depth + h) * ydims.height + w) * ydims.width + m;
              const auto xoffset = i * xdims.depth * xdims.height * xdims.width +
                                   (pool_size * h + p) * xdims.height * xdims.width +
                                   (pool_size * w + q) * xdims.width + m;
              Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
            }
          }
        }
      }
    }
  }
}

// From book chapter Figure 16.4
// Sequential code for the forward path of the convolution layer
static void conv_forward_valid(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
                               const shape ydims) {

  std::memset(Y, 0, ydims.num * ydims.depth * ydims.height * ydims.width);

  for (const auto i : range(0, ydims.num)) {
    for (const auto m : range(0, ydims.depth)) {    // for each output feature map
      for (const auto h : range(0, ydims.height)) { // for each output element
        for (const auto w : range(0, ydims.width)) {
          for (const auto p : range(0, wdims.height)) {    // filter height
            for (const auto q : range(0, wdims.width)) {   // filter width
              for (const auto c : range(0, xdims.depth)) { // sum over all input feature maps
                const auto yoffset = ((i * ydims.depth + h) * ydims.height + w) * ydims.width + m;
                const auto xoffset = i * xdims.depth * xdims.height * xdims.width +
                                     (h + p) * xdims.height * xdims.width + (w + q) * xdims.width + c;
                const auto woffset =
                    p * wdims.depth * wdims.height * wdims.width + q * wdims.height * wdims.width + c * wdims.width + m;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// error gradient of computed y respect to the original/correct y value
static void conv_backward_ygrad(const float *Y_orig, const float *Y, const shape ydims, float *dE_dY,
                                const shape dedydims) {
  for (const auto i : range(0, ydims.num)) {
    for (const auto m : range(0, ydims.depth)) {    // for each output feature map
      for (const auto h : range(0, ydims.height)) { // for each output element
        for (const auto w : range(0, ydims.width)) {
          const auto yoffset = ((i * ydims.depth + h) * ydims.height + w) * ydims.width + m;
          dE_dY[yoffset]     = Y[yoffset] - Y_orig[yoffset];
        }
      }
    }
  }
  relu4(dE_dY, dedydims);
}

// backward propagation for dE/dW
static void conv_backward_wgrad(const float *X, const shape xdims, const float *W, const shape wdims, const shape ydims,
                                const float *dE_dY, float *dE_dW) {
  // const auto out_h = ydims[1] - filter_h + 1;
  // const auto out_w = ydims[2] - filter_w + 1;

  std::memset(dE_dX, 0, ydims[3] * filter_h, filter_w, in_channel);

  for (const auto i : range(0, ydims.num)) {
    for (const auto m : range(0, ydims.depth)) {    // for each output feature map
      for (const auto h : range(0, ydims.height)) { // for each output element
        for (const auto w : range(0, ydims.width)) {
          for (const auto p : range(0, wdims.height)) {    // filter height
            for (const auto q : range(0, wdims.width)) {   // filter width
              for (const auto c : range(0, wdims.depth)) { // sum over all input feature maps
                const auto yoffset = ((i * ydims.depth + h) * ydims.height + w) * ydims.width + m;
                const auto xoffset = i * xdims.depth * xdims.height * xdims.width +
                                     (h + p) * xdims.height * xdims.width + (w + q) * xdims.width + c;
                const auto woffset =
                    p * wdims.depth * wdims.height * wdims.width + q * wdims.height * wdims.width + c * wdims.width + m;
                dE_dW[woffset] += X[xoffset] * dE_dY[yoffset];
              }
            }
          }
        }
      }
    }
  }
}

// backward propagation for dE/dX
static void conv_backward_xgrad(const float *X, const shape xdims, const float *W, const shape wdims, const float *Y,
                                const shape ydims, const float *dE_dY, float *dE_dX) {

  std::memset(dE_dX, 0, ydims.num * ydims.depth * ydims.height * wdims.depth);

  for (const auto i : range(0, ydims.num)) {
    for (const auto m : range(0, ydims.depth)) {    // for each output feature map
      for (const auto h : range(0, ydims.height)) { // for each output element
        for (const auto w : range(0, ydims.width)) {
          for (const auto p : range(0, wdims.height)) {    // filter height
            for (const auto q : range(0, wdims.width)) {   // filter width
              for (const auto c : range(0, xdims.depth)) { // sum over all input feature maps
                const auto yoffset = ((i * ydims.depth + h) * ydims.height + w) * ydims.width + m;
                const auto xoffset = i * xdims.depth * xdims.height * xdims.width +
                                     (h + p) * xdims.height * xdims.width + (w + q) * xdims.width + c;
                const auto woffset =
                    p * wdims.depth * wdims.height * wdims.width + q * wdims.height * wdims.width + c * wdims.width + m;
                dE_dX[xoffset] += dE_dY[yoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

void fully_forward(const float *X, const shape xdims, float *W, const shape wdims, float *Y, const shape ydims) {
  for (const auto i : range(0, xdims.num)) {
    for (const auto j : range(0, wdims.depth)) {
      float sum = 0;
      for (const auto k : range(0, xdims.depth)) {
        sum += X[i * xdims.depth + k] * W[k * wdims.depth + j];
      }
      Y[i * wdims.depth + j] = sum;
    }
  }
}

// Choose the guess with largest score
static void argmax(const float *X, const shape xdims, int *Y) {
  for (const auto i : range(0, xdims.num)) {
    auto max_idx = 0;
    auto max     = X[i * xdims.depth];
    for (const auto j : range(0, xdims.depth)) {
      const auto elem = X[(i * xdims.depth) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) {
  // conv layer
  const shape adims = {xdims.num, conv1dims.depth, (xdims.height - conv1dims.height + 1),
                       (xdims.width - conv1dims.width + 1)};
  auto a = zeros<float>(adims);
  conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

  /// relu layer
  relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const shape bdims   = {adims.num, adims.depth / pool_size, adims.height / pool_size, adims.width};
  auto b              = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);

  // conv layer
  const shape cdims = {bdims.num, conv2dims.depth, (bdims.height - conv2dims.height + 1),
                       (bdims.width - conv2dims.width + 1)};
  auto c = zeros<float>(cdims);
  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // relu
  relu4(c, cdims);

  // average pooling
  const shape ddims = {cdims.num, cdims.depth / pool_size, cdims.height / pool_size, cdims.width};
  auto d            = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);

  // reshape
  const shape ddims2 = {ddims.num, ddims.depth * ddims.height * ddims.width};

  // matrix multiplication
  const shape edims = {ddims.num, fc1dims.depth};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const shape fdims = {edims.num, fc2dims.depth};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

// Leslie: under review
// Backward operation for the CNN, a combination of conv layer + average pooling + relu
void backward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, float *y, const float *y_orig) {
  // conv layer
  const shape adims = {xdims.num, conv1dims.depth, (xdims.height - conv1dims.height + 1),
                       (xdims.width - conv1dims.width + 1)};
  auto dy = zeros<float>(dydims);
  auto dw = zeros<float>(dydims);
  auto dx = zeros<float>(dydims);
  conv_backward_ygrad(y_orig, y, dy, dydims);
  conv_backward_wgrad(x, xdims, conv1, conv1dims, ydims, dy, dw);
  conv_backward_xgrad(x, xdims, conv1, conv1dims, ydims, dy, dx);

  /// relu layer
  relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const shape bdims   = {adims.num, adims.depth / pool_size, adims.height / pool_size, adims.width};
  auto b              = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);

  // conv layer
  const shape cdims = {bdims.num, conv2dims.depth, (bdims.height - conv2dims.height + 1),
                       (bdims.width - conv2dims.width + 1)};
  auto c = zeros<float>(cdims);
  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // relu
  relu4(c, cdims);

  // average pooling
  const shape ddims = {cdims.num, cdims.depth / pool_size, cdims.height / pool_size, cdims.width};
  auto d            = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);

  // reshape
  const shape ddims2 = {ddims.num, ddims.depth * ddims.height * ddims.width};

  // matrix multiplication
  const shape edims = {ddims.num, fc1dims.depth};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const shape fdims = {edims.num, fc2dims.depth};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

static void compare_solution(float *cpu, float *gpu) {
  if (cpu.size() != gpu.size()) {
    printf("The dimensions does not match.");
    return -1;
  }
  // Element-wise comparison: only prints out the first error and halts
  for (const auto i : range(0, cpu.size()) {
    if (cpu[i] != gpu[i]) {
      printf("Element ", i, "does not match");
      return -1;
    }
  }
  printf("All the elements match!");
  return 0;
}

int main(int argc, char **argv) {

  // Generate data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  generate_data(x, xdims);

  // Generate model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  generate_convfilters(conv1, conv1dim);
  generate_convfilters(conv2, conv2dim);
  generate_convfilters(fc1, fc1dim);
  generate_convfilters(fc2, fc2dim);

  int *out = zeros<int>(FLAGS_batch_size);

  // Launch kernel
  // ----------------------------------------
  printf("Launching kernel ");

  // Sequential code
  printf("Performing CPU computation");

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);
  backward_operation(x, conv1, conv2, fc1, fc2, out, y_orig);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed
            << " milliseconds. Correctness: " << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  // Verify correctness
  // ----------------------------------------

  // Free memory
  // ----------------------------------------
  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;

  // free();

  return 0;
}
