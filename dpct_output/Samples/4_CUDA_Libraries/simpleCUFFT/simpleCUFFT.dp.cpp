/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dpct/fft_utils.hpp>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>
#include <dpct/lib_common_utils.hpp>

// Complex data type
typedef sycl::float2 Complex;
static inline Complex ComplexAdd(Complex, Complex);
static inline Complex ComplexScale(Complex, float);
static inline Complex ComplexMul(Complex, Complex);
static void ComplexPointwiseMulAndScale(Complex *, const Complex *,
                                                   int, float,
                                                   const sycl::nd_item<3> &item_ct1);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 50
#define FILTER_KERNEL_SIZE 11

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  printf("[simpleCUFFT] is starting...\n");

  findCudaDevice(argc, (const char **)argv);

  // Allocate host memory for the signal
  Complex *h_signal =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
    h_signal[i].x() = rand() / static_cast<float>(RAND_MAX);
    h_signal[i].y() = 0;
  }

  // Allocate host memory for the filter
  Complex *h_filter_kernel =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * FILTER_KERNEL_SIZE));

  // Initialize the memory for the filter
  for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
    h_filter_kernel[i].x() = rand() / static_cast<float>(RAND_MAX);
    h_filter_kernel[i].y() = 0;
  }

  // Pad signal and filter kernel
  Complex *h_padded_signal;
  Complex *h_padded_filter_kernel;
  int new_size =
      PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
              &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
  int mem_size = sizeof(Complex) * new_size;

  // Allocate device memory for signal
  Complex *d_signal;
  checkCudaErrors(DPCT_CHECK_ERROR(d_signal = (Complex *)sycl::malloc_device(
                                       mem_size, dpct::get_default_queue())));
  // Copy host memory to device
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_default_queue()
                           .memcpy(d_signal, h_padded_signal, mem_size)
                           .wait()));

  // Allocate device memory for filter kernel
  Complex *d_filter_kernel;
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_filter_kernel =
          (Complex *)sycl::malloc_device(mem_size, dpct::get_default_queue())));

  // Copy host memory to device
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_default_queue()
          .memcpy(d_filter_kernel, h_padded_filter_kernel, mem_size)
          .wait()));

  // CUFFT plan simple API
  dpct::fft::fft_engine_ptr plan;
  checkCudaErrors(DPCT_CHECK_ERROR(
      plan = dpct::fft::fft_engine::create(
          &dpct::get_default_queue(), new_size,
          dpct::fft::fft_type::complex_float_to_complex_float, 1)));

  // CUFFT plan advanced API
  dpct::fft::fft_engine_ptr plan_adv;
  size_t workSize;
  long long int new_size_long = new_size;

  checkCudaErrors(DPCT_CHECK_ERROR(plan_adv = dpct::fft::fft_engine::create()));
  /*
  DPCT1100:14: Currently the DFT external workspace feature in the Intel(R)
  oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the
  internal workspace if your code should run on non-GPU devices.
  */
  /*
  DPCT1099:15: Verify if the default value of the direction and placement used
  in the function "commit" is correct.
  */
  checkCudaErrors(DPCT_CHECK_ERROR(
      plan_adv->commit(&dpct::get_default_queue(), 1, &new_size_long, NULL, 1,
                       1, dpct::library_data_t::complex_float, NULL, 1, 1,
                       dpct::library_data_t::complex_float, 1, &workSize)));
  printf("Temporary buffer size %li bytes\n", workSize);

  // Transform signal and kernel
  printf("Transforming signal cufftExecC2C\n");
  checkCudaErrors(DPCT_CHECK_ERROR((plan->compute<sycl::float2, sycl::float2>(
      reinterpret_cast<sycl::float2 *>(d_signal),
      reinterpret_cast<sycl::float2 *>(d_signal),
      dpct::fft::fft_direction::forward))));
  checkCudaErrors(
      DPCT_CHECK_ERROR((plan_adv->compute<sycl::float2, sycl::float2>(
          reinterpret_cast<sycl::float2 *>(d_filter_kernel),
          reinterpret_cast<sycl::float2 *>(d_filter_kernel),
          dpct::fft::fft_direction::forward))));

  // Multiply the coefficients together and normalize the result
  printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 32) * sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
      [=](sycl::nd_item<3> item_ct1) {
        ComplexPointwiseMulAndScale(d_signal, d_filter_kernel, new_size,
                                    1.0f / new_size, item_ct1);
      });

  // Check if kernel execution generated and error
  getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

  // Transform signal back
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(DPCT_CHECK_ERROR((plan->compute<sycl::float2, sycl::float2>(
      reinterpret_cast<sycl::float2 *>(d_signal),
      reinterpret_cast<sycl::float2 *>(d_signal),
      dpct::fft::fft_direction::backward))));

  // Copy device memory to host
  Complex *h_convolved_signal = h_padded_signal;
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_default_queue()
                           .memcpy(h_convolved_signal, d_signal, mem_size)
                           .wait()));

  // Allocate host memory for the convolution result
  Complex *h_convolved_signal_ref =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));

  // Convolve on the host
  Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE,
           h_convolved_signal_ref);

  // check result
  bool bTestResult = sdkCompareL2fe(
      reinterpret_cast<float *>(h_convolved_signal_ref),
      reinterpret_cast<float *>(h_convolved_signal), 2 * SIGNAL_SIZE, 1e-5f);

  // Destroy CUFFT context
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::fft::fft_engine::destroy(plan)));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::fft::fft_engine::destroy(plan_adv)));

  // cleanup memory
  free(h_signal);
  free(h_filter_kernel);
  free(h_padded_signal);
  free(h_padded_filter_kernel);
  free(h_convolved_signal_ref);
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(d_signal, dpct::get_default_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(d_filter_kernel, dpct::get_default_queue())));

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
            const Complex *filter_kernel, Complex **padded_filter_kernel,
            int filter_kernel_size) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;
  int new_size = signal_size + maxRadius;

  // Pad signal
  Complex *new_data =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
  *padded_signal = new_data;

  // Pad filter
  new_data = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(Complex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(Complex));
  *padded_filter_kernel = new_data;

  return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host
void Convolve(const Complex *signal, int signal_size,
              const Complex *filter_kernel, int filter_kernel_size,
              Complex *filtered_signal) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;

  // Loop over output element indices
  for (int i = 0; i < signal_size; ++i) {
    filtered_signal[i].x() = filtered_signal[i].y() = 0;

    // Loop over convolution indices
    for (int j = -maxRadius + 1; j <= minRadius; ++j) {
      int k = i + j;

      if (k >= 0 && k < signal_size) {
        filtered_signal[i] =
            ComplexAdd(filtered_signal[i],
                       ComplexMul(signal[k], filter_kernel[minRadius - j]));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x() = a.x() + b.x();
  c.y() = a.y() + b.y();
  return c;
}

// Complex scale
static inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x() = s * a.x();
  c.y() = s * a.y();
  return c;
}

// Complex multiplication
static inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x() = a.x() * b.x() - a.y() * b.y();
  c.y() = a.x() * b.y() + a.y() * b.x();
  return c;
}

// Complex pointwise multiplication
static void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, float scale,
                                                   const sycl::nd_item<3> &item_ct1) {
  const int numThreads =
      item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
  const int threadID = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}
