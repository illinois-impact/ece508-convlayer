# Convolution Neural Networks lab

## Objective

The goal of this lab is to design single iteration step of the Convolutional Neural Network (CNN) algorithm using GPUs. You will implement forward and backward propagation to update the weights and also optimize using matrix multiplication. The sequential implementation provided follows the basic algorithm 16.4 and 16.5 decribed in [book chapter 16](https://bw-course.ncsa.illinois.edu/pluginfile.php/1469/mod_resource/content/1/3rd-Edition-Chapter16-case-study-DNN-FINAL-corrected.pdf) as well as lecture notes 15-16. The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## Background

Machine learning is a field of computer science which explores algorithms whose logic can be learned directly from data. Deep learning is a set of methods that allow a machine-learning system to automatically discover the complex features needed for detection directly from raw data. Deep learning procedures based on feed forward networks can achieve accurate pattern recognition results with given enough data. These procedures are based on a particular type of feed forward network called the convolutional neural networks (CNN).

## CUDA Implementation

Book chapters 16.3 and 16.4 provide a basic CUDA implementation of forward propagation of convolutional layer and possible optimization. Your CUDA implementation would be compared with the CPU version for the correctness and evaluated based on the performance. Apply any optimization you think would bring benefit and feel free to modify any part of the code. You should not use `cuBLAS` or `cuDNN` for the implementation, but you are expected to compare your implementation with those libraries --- profiling the code as well as comparing the algorithms used (if algorithm information is publically available).

## LESLIE: update this
## Dataset Information

There is 1 dataset assigned in this lab. We will use random function to generate the input data images, hence, all the test datasets between students and between each runs will be unique. Therefore, it is important to make sure that the output values match the results from the sequential code.

* Dataset H x W: 4096 x 3072
* Input Dataset I x H x W: 10 x 28 x 28
* Filter M x C x K x K: 64 x 1 x 3 x 3
* Output Dataset O x H x W: 64 x 28 x 28

## LESLIE: update this
## Instructions

In the provided source code, you will find functions named `forward_operation`. This function implements sequential forward path of the convolution layer. You don't have to modify this code, just call when verifying your output of GPU implementation. The functions `conv_backward_wgrad` and `conv_backward_dgrad` in the code implements backward propagation of the convolution layer which computes error gradient in terms of weights and inputs respectively to update the weight values for next iteration in the training process.

You have to implement the body of the `ConvLayerForward_Basic_Kernel`, `ConvLayerForward_MatMul_Kernel`, `ConvLayerBackward_WGrad_Kernel`, and `ConvLayerBackward_DGrad_Kernel` function. `ConvLayerForward_Basic_Kernel` function is the forward path of the convolution layer, CUDA version of `conv_forward_valid`. `ConvLayerForward_MatMul_Kernel` function do the same as `ConvLayerForward_Basic_Kernel` but using the matrix multiplication to improve the application performance. `ConvLayerBackward_WGrad_Kernel` and `ConvLayerBackward_DGrad_Kernel` are the backward propagation to compute the error gradients, which are the GPU equivalent of `conv_backward_wgrad` and `conv_backward_dgrad` respectively.

Although your correctness and performance will be evaluated based on the default dataset we have provided, feel free to adjust the sizes and values to test your implementation and experiment various approaches.

## Submissions

You are going to submit the code with both CUDA implementation and the given CPU version. Then, you will be writing 1-page report on how you have implemented your version of GPU, what optimizations you have tried and tested and what worked and/or what didn't work as expected. Refer to `What to deliver` section before for more details.


## Remote Development Environment

The easiest way to develop the project is to use rai through the following prebuilt binaries. You can also use the Linux machines on [EWS](http://it.engineering.illinois.edu/ews) for RAI.

**NOTE:** Even if you use your local development environment, your final code must run within the RAI system. Also, your final report performance measurements must be done within RAI.

### Download Binaries

The code is continuously built and published. The client can be downloaded from the following URLs (depending on your OS and Architecture):


| Operating System | Architecture | Stable Version Link                                                             |
| ---------------- | ------------ | ------------------------------------------------------------------------------- |
| Linux            | i386         | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-386.tar.gz)     |
| Linux            | amd64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-amd64.tar.gz)   |
| Linux            | armv5        | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-armv5.tar.gz)   |
| Linux            | armv6        | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-armv6.tar.gz)   |
| Linux            | armv7        | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-armv7.tar.gz)   |
| Linux            | arm64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-arm64.tar.gz)   |
| Linux            | ppc64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-ppc64.tar.gz)   |
| Linux            | ppc64le      | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-ppc64le.tar.gz) |
| OSX/Darwin       | i386         | [URL](http://files.rai-project.com/dist/rai/stable/latest/darwin-386.tar.gz)    |
| OSX/Darwin       | amd64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/darwin-amd64.tar.gz)  |
| Windows          | i386         | [URL](http://files.rai-project.com/dist/rai/stable/latest/windows-386.tar.gz)   |
| Windows          | amd64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/windows-amd64.tar.gz) |



### Client

#### Set up your Secret Key

Each team will be contacted by a TA and given a secret key to use this service. Do not share your key with other teams. The secret key is used to authenticate you with the server.

The `RAI_SECRET_KEY`, `RAI_TEAM_NAME`, and `RAI_ACCESS_KEY` should be specified in your `~/.rai.profile` (linux/OSX) or `%HOME%/.rai.profile` (Windows -- for me this is `C:\Users\abduld\.rai.profile`) in the following way.

```yaml
profile:
  firstname: Abdul
  lastname: Dakkak
  username: abduld
  email: dakkak@illinois.edu
  access_key: XXXXXXXXXXXXXXXXXXX
  secret_key: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

The above will need to match the email you received from `postmaster@webgpu.com` on March 31. If you did not receive the email, then contact the TA. Also, contact the TA with your team name as soon as possible if you are considering on using rai for your final project. Do not share your keys with other users or teams. The access and secret key is used to authenticate you with the server. Both the team name and the username are used to identify you to the system.

#### Run the Client

To run the client, use

```bash
rai -p <project folder>
```

From a user's point of view when the client runs, the local directory specified by `-p` gets uploaded to the server and extracted into the `/src` directory on the server. The server then executes the build commands from the `rai_build.yml` specification within the `/build` directory. Once the commands have been run, or there is an error, a zipped version of that `/build` directory is available from the server for download.

The server limits the task time to be an hour with a maximum of 8GB of memory being used within a session. The output `/build` directory is only available to be downloaded from the server for a short amount of time. Networking is also disabled on the execution server. Contact the teaching assistants if this is an issue.

#### Other Options

```
  -c, --color         Toggle color output.
  -d, --debug         Toggle debug mode.
  -p, --path string   Path to the directory you wish to submit. Defaults to the current working directory. (default "current working directory")
  -v, --verbose       Toggle verbose mode.
```

On Windows, it might be useful to disable the colored output. You can do that by using the `-c=false` option

#### Internal Details (Ignore if not Interested)

The client sends job submission requests to the rai server. The internal steps the client takes are as follows:

1.  The client creates an archive of your directory and posts it to Amazon S3
2.  The client creates a unique identifier (here called `ID`). These IDs are generated using [`NewObjectId`](https://godoc.org/labix.org/v2/mgo/bson#NewObjectId).
3.  The client creates a job request and publishes to the `tasks` topic on the queue. The job request has the ID field with the value `ID` and is mashaled using using the [`bson`](https://godoc.org/labix.org/v2/mgo/bson) library. The reason for using `bson` is that we will want to store the results in mongodb in the future.
4.  The client subscribes to the topic `log-ID` and prints the results on that topic.
5.  The client stops listening when the message on the topic has a tag `TagEnd`.

### Project Build Sepecification

The `rai_build.yml` must exist in your project directory. In some cases you may not be able to execute certain builtin bash commands, in this senario the current workaround is to create a bash file and insert the commands you need to run. You can then execute the bash script within `rai_build.yml`.

The `rai-build.yml` is written as a [Yaml](http://yaml.org/) ([Spec](http://www.yaml.org/spec/1.2/spec.html)) file and has the following structure.

```yaml
rai:
  version: 0.2 # this is required
  image: nvidia/cuda:8.0-devel-ubuntu16.04 # nvidia/cuda:8.0-devel-ubuntu16.04 is a docker image 
                                  # and can be viewed at https://hub.docker.com/r/nvidia/cuda/
                         				  # You can specify any image found on dockerhub
resources:
  gpus: 1 # tell the system that you're using a gpu
commands:
  build:
    - echo "Building project"
    # Use CMake to generate the build files. Remember that your directory gets uploaded to /src
    - cmake /src
    # Run the make file to compile the project.
    - make
    # here we break the long command into multiple lines. The Yaml
    # format supports this using a block-strip command. See
    # http://stackoverflow.com/a/21699210/3543720 for info
    - >-
      ./mybinary -i input1,input2 -o output
```

Syntax errors will be reported and the job will not be executed. You can check if your file is in a valid yaml format by using tools such as [Yaml Validator](http://codebeautify.org/yaml-validator).

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

## LESLIE: CHECK this. Change the build commands
### Project Submission

You will use the same client (with certain options) for the final submission. The submission system notify the teaching assistants. You will need the above credentials to make your final submission.

To submit your project, run

```bash
rai submit -d <project folder>
```

To perform the final project submission, you must have the `USAGE`, `README`, and `report.pdf` files in your project folder (as stated in the ["What to Deliver"](#what-to-deliver) section). The submission system ignores your `rai-build.yml` file and instead runs the following build file:

```yaml
rai:
  version: 0.1
resources:
  gpus: 1
commands:
  build:
    - echo "Submitting project"
    - cp -r /src /build/submission_code
    - cmake -DCONFIG_USE_HUNTER=OFF /src
    - make
    (REMOVE THIS LINE??)- /usr/bin/time ./ece508_cnn /src/data/testfull.hdf5 /src/data/model.hdf5 10000
```

**NOTE::** Only your last submission is recorded, so please make sure that your last submission is the one you'd want to be graded.

## Local Development Environment

**NOTE:** Even if you use your local development environment, your final code must run within the RAI system. Also, your final report performance measurements must be done within RAI.

The project requires a CUDA-supported operating system, C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page. Instructions on how to install the CUDA Toolkit are available in the [Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and [OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required to generate build scripts for your target IDE and compiler. On windows, we require Visual Studio 2015 (Service Pack 3) which you can download from the webstore. For other systems, a CUDA compatible compiler is required (e.g. for OSX the [clang compiler](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#system-requirements) is the only one supported).

## LESLIE: CHECK from here
### How to Build

There are two options to build this project, the first is using the [Hunter] package manager and the other is using [Docker](https://www.docker.com/). We sugguest using CMake along with Hunter, but it is known not to work on all operating systems. In this case, we suggest that you either using Docker or install the libraries needed (mainly `HDF5`).

#### Using Hunter Package Manager

By default, the compilation uses the [Hunter] --- a C package manager. This method requires that you have the CUDA toolkit installed on your machine.

Assuming that you have checked out the project into `$SRCDIR` do

```{.sh}
cd $SRCDIR
mkdir build
cd build
cmake $SRCDIR
```

This will download the required software needed for the project (see the [hunter docs][hunterdoc] for more information). You may see some warning while the system is compiling _HDF5_, which you can ignore. Once CMake has been run, a `Makefile` is generated so you can then perform `make` to buidl the project.

```{.sh}
make
```

If you do not plan on using `make`, examine the `cmake -G` option which allows you to generate XCode, Visual Studio, ... project configurations. You may also need to change the build type to enable/disable debugging and/or optimizations.

If you need to use another library, you need have to modify the [`CMakeLists.txt`](https://github.com/ece508/cnn/blob/master/CMakeLists.txt) and add the libraries to the `target_link_libraries` (and possibly the `include_directories`) section. Documentation on the CMake commands is found in the [documentation page][cmakedoc].

## How to Test

Test your implementation with small batch size first to verify the correctness. You can define the input data size in smaller values and adjust the filter size for verification.

## What to Deliver

A `.tar.gz` file which contains the report, code directory, the build scripts, and, possibly, the input dataset needs to be delivered to the Teaching Assistants.

-   Code:  A `USAGE` file needs to be placed included in the archive file which includes instructions on how to compile and run your code. If the report performs any profiling, the `USAGE` file must also specify how to run the performance measurements.
-   Report: A PDF version report must be included within the `.tar.gz` file. Single-page report should describe and evaluate the optimizations you tried. You should strive to be thorough, concise, and quantitative in your performance analysis.
    The report must be named `report.pdf`

Make sure you have a working CUDA implementation before applying any optimizations.

## Optimization Opportunities

The serial version of the code is amicable to many optimization opportunities, the following is an incomplete set of them:

-   Optimize the CUDA memory copies to decrease the overhead of memory transfers
-   Overlapping the memory transfer and the compute and/or independent computations using CUDA streams
-   Performing layout transformations to get coalesced accesses or to make better use of the cache
-   Using low precision to perform the computation (for example using `float16` or binary values)
-   Based on the size of the convolution, utilitize better algorithms to perform the computation (for example using the [Winograd Kernel][https://www.nervanasys.com/winograd-2/])

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
