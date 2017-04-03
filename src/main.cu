#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <numeric>
#include <sys/time.h>
#include <valarray>

// #include <hdf5.h>
#include <stdio.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define TILE_WIDTH 4

static int FLAGS_batch_size = 1; // 10000 for ece408project
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

/******************************************************************************
 GPU Kernels
*******************************************************************************/
// Basic convolution layer forward kernel
__global__ void ConvLayerForward_Basic_Kernel(const int C, const int W_grid, const int K, float *X, float *W,
                                              float *Y) {
  // INSERT KERNEL CODE HERE
}

// Convolution layer with matrix multiplication
__global__ void ConvLayerForward_MatMul_Kernel(float *X, float *W, float *Y) {
  // INSERT KERNEL CODE HERE
}

// Backward propagation on error gradient on weights
__global__ void ConvLayerBackward_WGrad_Kernel() {
  // INSERT KERNEL CODE HERE
}

// Backward propagation on error gradient on inputs
__global__ void ConvLayerBackward_XGrad_Kernel() {
  // INSERT KERNEL CODE HERE
}

/******************************************************************************
 Host Functions
*******************************************************************************/
// GPU functions to call kernels
static void ConvForward(float *X, float *W, float *Y, const int h, const int w, const int c, const int m) {
}

static void ConvBackward() {
}

// matrix formation

/******************************************************************************
 Sequential Functions
*******************************************************************************/
// generate input and output data with random values
static int generateData(float *x, float *y, const int m, const int i, const int c, const int h, const int w) {
  // Set the dataset x dimensions
  const auto xndims       = 4;
  const auto yndims       = 3;
  int input_dims[xndims]  = {i, c, h, w}; // you can vary this parameters to change the size
  int output_dims[yndims] = {m, h, w};    // you can vary this parameters to change the size

  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1] << " x " << input_dims[2] << " x "
            << input_dims[3] << "\n";

  std::cout << "input data values = "
            << "\n";

  // Set random values to the dataset x and y
  for (const auto i : range(0, input_dims[0])) {
    for (const auto c : range(0, input_dims[1])) {
      for (const auto h : range(0, input_dims[2])) {
        for (const auto w : range(0, input_dims[3])) {
          const auto xoffset = i * input_dims[1] * input_dims[2] * input_dims[3] + h * input_dims[2] * input_dims[3] +
                               w * input_dims[3] + c;
          x[xoffset] = rand() % 10; // random values: 0~9
          std::cout << x[xoffset] << " ";
        }
      }
    }
  }
  std::cout << "\n";

  for (const auto i : range(0, output_dims[0])) {
    for (const auto m : range(0, output_dims[3])) {   // for each output feature map
      for (const auto h : range(0, output_dims[1])) { // for each output element
        for (const auto w : range(0, output_dims[2])) {
          const auto yoffset = ((i * output_dims[1] + h) * output_dims[2] + w) * output_dims[3] + m;
          y[yoffset]         = rand() % 10; // random values: 0~9
        }
      }
    }
  }

  return 0;
}

// generate convolution filter
static void generateConvFilters(float *conv1, float *conv2, float *fc1, float *fc2, const int m, const int c,
                                const int h, const int w) {
  // Set the dataset x dimensions
  const auto convndims       = 4;
  int filter_dims[convndims] = {m, c, h, w};

  std::cout << "filter dimensions = " << filter_dims[0] << " x " << filter_dims[1] << " x " << filter_dims[2] << " x "
            << filter_dims[3] << "\n";

  // Set convolution filter values to 1
  for (const auto i : range(0, filter_dims[0])) {
    for (const auto c : range(0, filter_dims[1])) {
      for (const auto h : range(0, filter_dims[2])) {
        for (const auto w : range(0, filter_dims[3])) {
          const auto convoffset = i * filter_dims[1] * filter_dims[2] * filter_dims[3] +
                                  h * filter_dims[2] * filter_dims[3] + w * filter_dims[3] + c;
          conv1[convoffset] = 1;
        }
      }
    }
  }
}

// Rectified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Rectified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
static void average_pool(const float *X, const int xdims[4], const int pool_size, float *Y, const int ydims[4]) {
  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] + (pool_size * h + p) * xdims[2] * xdims[3] +
                                   (pool_size * w + q) * xdims[3] + m;
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
static void conv_forward_valid(const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];

  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {   // for each output feature map
      for (const auto w : range(0, ydims[2])) { // for each output element
        for (const auto h : range(0, ydims[1])) {
          const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
          Y[yoffset]         = 0;
          for (const auto p : range(0, filter_h)) {       // filter height
            for (const auto q : range(0, filter_w)) {     // filter width
              for (const auto c : range(0, in_channel)) { // sum over all input feature maps
                const auto xoffset =
                    i * xdims[1] * xdims[2] * xdims[3] + (h + p) * xdims[2] * xdims[3] + (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] + q * wdims[2] * wdims[3] + c * wdims[3] + m;
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
static void conv_backward_ygrad(const float *Y_orig, const float *Y, const int ydims[4], float *dE_dY,
                                const int dedydims[4]) {
  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {   // for each output feature map
      for (const auto w : range(0, ydims[2])) { // for each output element
        for (const auto h : range(0, ydims[1])) {
          const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
          dE_dY[yoffset]     = Y[yoffset] - Y_orig[yoffset];
        }
      }
    }
  }
  relu4(dE_dY, dedydims);
}

// backward propagation for dE/dW
static void conv_backward_wgrad(const float *X, const int xdims[4], const float *W, const int wdims[4],
                                const int ydims[4], const float *dE_dY, float *dE_dW) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];
  const auto out_h      = ydims[1] - filter_h + 1;
  const auto out_w      = ydims[2] - filter_w + 1;

  for (const auto m : range(0, ydims[3])) {         // for each output feature map
    for (const auto p : range(0, filter_h)) {       // filter height
      for (const auto q : range(0, filter_w)) {     // filter width
        for (const auto c : range(0, in_channel)) { // sum over all input feature maps
          const auto woffset = p * wdims[1] * wdims[2] * wdims[3] + q * wdims[2] * wdims[3] + c * wdims[3] + m;
          dE_dW[woffset]     = 0;
        }
      }
    }
  }

  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {   // for each output feature map
      for (const auto w : range(0, ydims[2])) { // for each output element
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, filter_h)) {       // filter height
            for (const auto q : range(0, filter_w)) {     // filter width
              for (const auto c : range(0, in_channel)) { // sum over all input feature maps
                const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                const auto xoffset =
                    i * xdims[1] * xdims[2] * xdims[3] + (h + p) * xdims[2] * xdims[3] + (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] + q * wdims[2] * wdims[3] + c * wdims[3] + m;
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
static void conv_backward_xgrad(const float *X, const int xdims[4], const float *W, const int wdims[4], const float *Y,
                                const int ydims[4], const float *dE_dY, float *dE_dX) {
  {
    const auto filter_h   = wdims[0];
    const auto filter_w   = wdims[1];
    const auto in_channel = wdims[2];
    const auto out_h      = ydims[1] - filter_h + 1;
    const auto out_w      = ydims[2] - filter_w + 1;

    for (const auto i : range(0, ydims[0])) {
      for (const auto w : range(0, ydims[2])) { // for each output element
        for (const auto h : range(0, ydims[1])) {
          for (const auto c : range(0, in_channel)) { // sum over all input feature maps
            const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] + h * xdims[2] * xdims[3] + w * xdims[3] + c;
            dE_dX[xoffset]     = 0;
          }
        }
      }
    }

    for (const auto i : range(0, ydims[0])) {
      for (const auto m : range(0, ydims[3])) {   // for each output feature map
        for (const auto w : range(0, ydims[2])) { // for each output element
          for (const auto h : range(0, ydims[1])) {
            for (const auto p : range(0, filter_h)) {       // filter height
              for (const auto q : range(0, filter_w)) {     // filter width
                for (const auto c : range(0, in_channel)) { // sum over all input feature maps
                  const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                  const auto xoffset =
                      i * xdims[1] * xdims[2] * xdims[3] + (h + p) * xdims[2] * xdims[3] + (w + q) * xdims[3] + c;
                  const auto woffset = p * wdims[1] * wdims[2] * wdims[3] + q * wdims[2] * wdims[3] + c * wdims[3] + m;
                  dE_dX[xoffset] += dE_dY[yoffset] * W[woffset];
                }
              }
            }
          }
        }
      }
    }
  }

  void fully_forward(const float *X, const int xdims[2], float *W, const int wdims[2], float *Y, const int ydims[2]) {
    for (const auto i : range(0, xdims[0])) {
      for (const auto j : range(0, wdims[1])) {
        float sum = 0;
        for (const auto k : range(0, xdims[1])) {
          sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
        }
        Y[i * wdims[1] + j] = sum;
      }
    }
  }

  // Leslie: update the function
  void fully_backward(const float *X, const int xdims[2], float *W, const int wdims[2], float *Y, const int ydims[2]) {
    for (const auto i : range(0, xdims[0])) {
      for (const auto j : range(0, wdims[1])) {
        float sum = 0;
        for (const auto k : range(0, xdims[1])) {
          sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
        }
        Y[i * wdims[1] + j] = sum;
      }
    }
  }

  // Choose the guess with largest score
  static void argmax(const float *X, const int xdims[2], int *Y) {
    for (const auto i : range(0, xdims[0])) {
      auto max_idx = 0;
      auto max     = X[i * xdims[1]];
      for (const auto j : range(0, xdims[1])) {
        const auto elem = X[(i * xdims[1]) + j];
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
    const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
    auto a            = zeros<float>(adims);
    conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

    /// relu layer
    relu4(a, adims);

    // average pooling
    const int pool_size = 2;
    const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
    auto b              = zeros<float>(bdims);
    average_pool(a, adims, pool_size, b, bdims);

    // conv layer
    const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
    auto c            = zeros<float>(cdims);
    conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

    // relu
    relu4(c, cdims);

    // average pooling
    const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
    auto d            = zeros<float>(ddims);
    average_pool(c, cdims, pool_size, d, ddims);

    // reshape
    const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

    // matrix multiplication
    const int edims[] = {ddims[0], fc1dims[1]};
    auto e            = zeros<float>(edims);
    fully_forward(d, ddims2, fc1, fc1dims, e, edims);

    // relu
    relu2(e, edims);

    // matrix multiplication
    const int fdims[] = {edims[0], fc2dims[1]};
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

  // Backward operation for the CNN, a combination of conv layer + average pooling + relu
  void backward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, float *y, const float *y_orig) {
    // conv layer
    const int dydims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
    auto dy            = zeros<float>(dydims);
    auto dw            = zeros<float>(dydims);
    auto dx            = zeros<float>(dydims);
    conv_backward_ygrad(y_orig, y, dy, dydims);
    conv_backward_wgrad(x, xdims, conv1, conv1dims, ydims, dy, dw);
    conv_backward_xgrad(x, xdims, conv1, conv1dims, ydims, dy, dx);
    /*
      /// relu layer
      relu4(dw, dwdims);
      relu4(dx, dxdims);

      // average pooling
      const int pool_size = 2;
      const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                           adims[3]};
      auto b = zeros<float>(bdims);
      average_pool(a, adims, pool_size, b, bdims);

      // matrix multiplication
      const int fdims[] = {edims[0], fc2dims[1]};
      auto f            = zeros<float>(fdims);
      fully_backward(e, edims, fc2, fc2dims, f, fdims);

      argmax(f, fdims, out);

      delete[] a;
      delete[] b;
      delete[] f;
    */
  }

  static void compare_solution(float *orig, float *comp) {
  }

  int main(int argc, char **argv) {
    /*
      if (argc != 3 && argc != 4) {
        std::cerr << "\n"
                  << "Sample usage: \n"
                  << argv[0]
                  << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
        return -1;
      }
      FLAGS_testdata = std::string(argv[1]);
      FLAGS_model    = std::string(argv[2]);
      } else if (argc == 4) {
        FLAGS_batch_size = atoi(argv[3]);
      }
    */

    enum Mode { CPU = 1, GPU_BASIC, GPU_MATRIX };
    Mode mode = (Mode) 1;

    // Initialize host variables
    // ----------------------------------------------
    xdims[0] = FLAGS_batch_size;
    rdims[0] = FLAGS_batch_size;

    // Generate data into x and y
    printf("Creating memory on host");

    float *x = allocate<float>(xdims);
    float *y = allocate<float>(rdims);
    float *x_dev;
    float *y_dev;
    generateData(x, y, xdims[1], xdims[0], conv1dims[1], xdims[2], xdims[3]);

    // Generate model
    float *conv1 = allocate<float>(conv1dims);
    float *conv2 = allocate<float>(conv2dims);
    float *fc1   = allocate<float>(fc1dims);
    float *fc2   = allocate<float>(fc2dims);
    float *conv1_dev;  // updated weights from GPU computation
    float *conv1_host; // mem-copied weights from GPU computation for solution check
    generateConvFilters(conv1, conv2, fc1, fc2, xdims[1], conv1dims[1], xdims[2], xdims[3]);

    int *out = zeros<int>(FLAGS_batch_size);

    // Allocate device variables
    // ----------------------------------------
    if (mode != CPU) {
      printf("Allocating GPU memory.");

      cudaMalloc((void **) &x_dev, xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float));
      cudaMalloc((void **) &conv1_dev, conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3] * sizeof(float));
      cudaMalloc((void **) &conv1_host, conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3] * sizeof(float));
      // cudaMalloc((void **)&y_dev, ydims[0] * ydims[1] * ydims[2] * ydims[3] * sizeof(float));
    }

    // Copy host variables to device
    // ----------------------------------------
    if (mode != CPU) {
      printf("Copying input memory to the GPU.");

      cudaMalloc(x_dev, x, xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float), cudaMemcpyHostToDevice);
      cudaMalloc(conv1_dev, conv1, conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3] * sizeof(float),
                 cudaMemcpyHostToDevice);

      cudaMemset(conv1_host, 0, conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3] * sizeof(float));
    }

    // Launch kernel
    // ----------------------------------------
    printf("Launching kernel ");

    if (mode == CPU) {
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

      // Calculate correctness
      int num_correct = 0;
      for (const auto i : range(0, FLAGS_batch_size)) {
        if (out[i] == ref[i]) {
          num_correct++;
        }
      }
      std::cout << "Done with " << FLAGS_batch_size << " queries in "
                << "elapsed = " << elapsed
                << " milliseconds. Correctness: " << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";
    } else if (mode == GPU_BASIC) {
      printf("Performing GPU Basic forward propagation");
      ConvForward(x, conv1, y, ydims[1], ydims[2], conv1dims[2], ydims[3]);
      cudaDeviceSynchronize();
    } else if (mode == GPU_MATRIX) {
      printf("Performing GPU forward propagation with matrix multiplication");
      // ConvForward(x, w, y, h, w, c, m);
      cudaDeviceSynchronize();
    } else {
    }

    // Copy device variables from host
    // ----------------------------------------
    printf("Copying output memory to the CPU");
    cudaMemcpy(conv1_host, conv1_dev, conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3] * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Verify correctness
    // ----------------------------------------
    printf("Verifying results.");
    compare_solution(conv1, conv1_dev);

    // Free memory
    // ----------------------------------------
    delete[] x;
    delete[] y;
    delete[] conv1;
    delete[] conv2;
    delete[] fc1;
    delete[] fc2;
    delete[] out;
    // delete[] ref;

    // free();

    if (mode != CPU) {
      printf("Freeing GPU Memory");
      cudaFree(x_dev);
      cudaFree(conv1_dev);
      cudaFree(y_dev);
    }

    return 0;
  }
