# Convolution Neural Networks lab

## Objective

The goal of this lab is to design a single iteration of the Convolutional Neural Network (CNN) algorithm using GPUs. You will implement forward and backward propagation and also optimize the execution speed using matrix multiplication. The sequential implementation provided follows the basic algorithm shown in Figure 16.4 and 16.5 of [textbook chapter 16](https://bw-course.ncsa.illinois.edu/pluginfile.php/1469/mod_resource/content/1/3rd-Edition-Chapter16-case-study-DNN-FINAL-corrected.pdf), chapter 16.2 as well as lecture notes 15-16.

## Background

Machine learning is a field of computer science which explores algorithms whose logic can be learned directly from data. Deep learning is a set of methods that allow a machine-learning system to automatically discover the complex features needed for detection directly from raw data. Deep learning procedures based on feed forward networks can achieve accurate pattern recognition results given enough data. These procedures are based on a particular type of feed forward network called the convolutional neural networks (CNN).

## CUDA Implementation

Textbook chapters 16.3 and 16.4 provide a basic CUDA implementation of forward propagation of a convolutional layer and a possible optimization using matrix multiplication. Your CUDA implementation for forward propagation and backward propagation will be compared with the CPU version for the correctness at the end of each step for correctness and evaluated based on its achieved performance. You should implement your own tiled/optimized matrix multiplication written in CUDA. Apply any optimization you think would bring benefit and feel free to modify any part of the code. You should not use `cuBLAS` or `cuDNN` for the implementation, but you are expected to compare your implementation with those libraries --- profiling the code as well as comparing the algorithms used (if algorithm information is publically available).

## Dataset Information

There is one dataset assigned in this lab. We will use random function to generate the input data images, hence, all the input test datasets between students and between each runs will be unique. Therefore, it is important to make sure that the output values match the results from the sequential code. For computation simplicity, convolution filter weights are all set to 1.

* Input Dataset `N x C x H x W = 100 x 1 x 28 x 28`: all random variables
* Filter `M x C x K x K =  32 x 1 x 5 x 5`: all 1's

## Instructions

In the provided sequential source code, you will find a function named `forward_operation`. This function implements sequential forward path of the convolution layer. In forward propagation, there are 6 steps: convolution layer 1, sub-sampling (or pooling), convolution layer 2, sub-sampling, fully connected layer 1, and fully connected layer 2 in order.

The function `backward_operation` implements backward propagation of the convolutional neural network. The functions `conv_backward_wgrad` and `conv_backward_xgrad` in the code compute error gradients in terms of the weights and inputs respectively. These values will be propagated along the path and used to update the filter weight values for next iteration in the training process. However, in this lab, we are only focusing on single iteration, single convolution layer for back-propagation. We compute the error gradient values on convolution layer 1 from forward propagation with randomly generated dE/dY. You don't have to modify these functions, but to call the function when executing and verifying your output of GPU implementation.

![image](/imgs/cnn_path.jpg "thumbnail")

You have to implement the host code to call GPU kernels, GPU kernel functions and CUDA memory management. Once you have finished with CUDA implementation, you will be using the function `compare_solution` to verify your solution with the results from the sequential code. You will check the output feature maps, Y (or out), after the forward propagation, and error gradient based on weights and input data, dE/dW, dE/dX, after the backward propagation.

Although your correctness and performance will be evaluated based on the default dataset we have provided, feel free to adjust the sizes and values to test your implementation and experiment with various approaches.

## Submissions

You are to submit the code with both CUDA implementation and the given CPU version. Then, you will be writing a 1-page report on how you have implemented your version of GPU, what optimizations you have tried and tested and what worked and/or what didn't work as expected. Create a tarball (ece508\_cnn\_<netid>.tar.gz) that includes all source codes and the report. The tarball should be submitted to the Compass.

## Remote Development Environment

The easiest way to develop the project is to use rai through the following prebuilt binaries. You can also use the Linux machines on [EWS](http://it.engineering.illinois.edu/ews) for RAI.

**NOTE:** Even if you use your own local development environment, your final code must run within the RAI system. 

See the [Client Documentation Page](https://github.com/rai-project/rai) for information on how to download, setup, and use the client on your own laptop or desktop.

## Profiling

Profiling can be performed using `nvprof`. Place the following build commands in your `rai-build.yml` file

```yaml
    - >-
      nvprof --cpu-profiling on --export-profile timeline.nvprof --
      ./mybinary -i input1,input2 -o output
    - >-
      nvprof --cpu-profiling on --export-profile analysis.nvprof --analysis-metrics --
      ./mybinary -i input1,input2 -o output
```

You could change the input and test datasets. This will output two files `timeline.nvprof` and `analysis.nvprof` which can be viewed using the `nvvp` tool (by performing a `file>import`). You will have to install the nvvp viewer on your machine to view these files.

_NOTE:_ `nvvp` will only show performance metrics for GPU invocations, so it may not show any analysis when you only have serial code.


## Utility Functions

We provide a some helper utility functions in the [`utils.hpp`][utilshpp] file.

### How to Time

In [`utils.hpp`][utilshpp] a function called `now()` which allows you to get the current time at a high resolution. To measure the overhead of a function `f(args...)`, the pattern to use is:

```{.cpp}
const auto tic = now();
f(args...);
const auto toc = now();
const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();;
std::cout << "Calling f(args...) took " << elapsed << "milliseconds\n";
```

### Range For Loops

Throughout the serial code, we use the [`range.hpp`][rangehpp] to make the code easier to understand. Essentially,

```{.cpp}
for (const auto ii : range(0, N)) {
    do_stuff(ii);
}
```

Is equivalent to

```{.cpp}
for (const auto ii = 0; ii < N; ii++) {
    do_stuff(ii);
}
```

The use of range introduces some overhead and you might get better speed if you remove it's usage.

### Checking Errors

To check for CUDA errors, specialize the `check_success` function in `utils.hpp` to also handle `cudaError_t`. For example:

```{.cpp}
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
```

`check_success` can then be used when calling CUDA functions:

```{.cpp}
check_success(cudaFree(deviceData));
```

## Reporting Issues

If emailing the TA with a problem, then please include the output of

```bash
rai version
```

as well as the output of

```bash
rai buildtime
```

In your bug report. You can also invoke the `rai` command with verbose and debug outputs using

```
rai --verbose --debug
```

Please use the [Github issue manager] to report any issues or suggestions about the lab.

[cmakedoc]: https://cmake.org/cmake/help/latest/

[hunterdoc]: https://docs.hunter.sh/en/latest/

[rangehpp]: https://github.com/harrism/cpp11-range

[hunter]: https://github.com/ruslo/hunter


## License

NCSA/UIUC © [Abdul Dakkak](http://impact.crhc.illinois.edu/Content_Page.aspx?student_pg=Default-dakkak)

[github issue manager]: https://github.com/rai-project/rai/issues

