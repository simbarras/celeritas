//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.cu
//---------------------------------------------------------------------------//
#include "DormandPrinceStepper.test.hh"

#include <typeinfo>

#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/field/detail/FieldUtils.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void test_kernel()
{
    // printf("Hello from block %d and thread %d\n", blockIdx.x, threadIdx.x);
    int i = threadIdx.x;
    int j = i;
    if (i < 4) {
        j = __shfl_down_sync(0x0000000f, i, 2);
        __syncthreads();
    }
    printf("Thread %d: before %d, after %d\n", threadIdx.x, i, j);
}

template<class Stepper_impl>
__device__ FieldStepperResult run_stepper(Stepper_impl stepper,
                                          int step,
                                          OdeState state,
                                          int id,
                                          int index,
                                          int num_states)
{
    return FieldStepperResult();
}

template<>
__device__ FieldStepperResult run_stepper(Stepper_uni stepper,
                                          int step,
                                          OdeState state,
                                          int id,
                                          int index,
                                          int num_states)
{
    if (index != 0) return FieldStepperResult();
    return stepper(step, state);
}

template<>
__device__ FieldStepperResult run_stepper(Stepper_multi stepper,
                                          int step,
                                          OdeState state,
                                          int id,
                                          int index,
                                          int num_states)
{
    // printf("thread %d, index %d\n", id, index);

    return stepper(step, state, id, index, num_states);
}

template<class Stepper_impl>
__global__ void dormand_test_arg_kernel(OdeState* states,
                                        FieldStepperResult* results,
                                        int* num_states,
                                        int* number_iterations,
                                        int* number_threads)
{
    constexpr double initial_step_size = 10000.0;
    constexpr double delta_chord = 1e-4;
    constexpr double half = 0.5;

    auto id = (blockIdx.x * blockDim.x + threadIdx.x) / *number_threads;

    if (id >= *num_states) return;

    auto index = (blockIdx.x * blockDim.x + threadIdx.x) % *number_threads;
    auto eval = make_dummy_equation(dormand_prince_dummy_field);
    Stepper_impl stepper{eval};
    FieldStepperResult res;
    auto state = states[id];
    auto step = initial_step_size;

    for (int i = 0; i < *number_iterations; ++i)
    {
        res = run_stepper(stepper,
                          step,
                          state,
                          id,
                          index,
                          *num_states);
        auto dchord
            = detail::distance_chord(state, res.mid_state, res.end_state);
        step *= max(std::sqrt(delta_chord / dchord), half);
    }
    results[id] = res;
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void test()
{
    test_kernel<<<1, 32>>>();
}

KernelResult simulate_multi_next_chord(int number_threads)
{
    KernelResult result;

    // Load initial states and results to device
    int *d_num_states, *d_number_iterations, *d_number_threads;

    FieldStepperResult *h_results, *d_results;
    h_results = new FieldStepperResult[number_of_states];
    for (int i = 0; i < number_of_states; ++i)
    {
        h_results[i] = FieldStepperResult();
    }

    OdeState *d_states;

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    cudaMalloc(&d_results, number_of_states * sizeof(FieldStepperResult));
    cudaMalloc(&d_states, number_of_states * sizeof(OdeState));
    cudaMalloc(&d_num_states, sizeof(int));
    cudaMalloc(&d_number_iterations, sizeof(int));
    cudaMalloc(&d_number_threads, sizeof(int));

    // Copy initial states to device
    cudaMemcpy(d_states,
               initial_states,
               number_of_states * sizeof(OdeState),
               cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_num_states, &number_of_states, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_number_iterations,
               &number_iterations,
               sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_number_threads, &number_threads, sizeof(int), cudaMemcpyHostToDevice);

    // Kernel configuration
    int thread_dimension = number_threads * number_of_states;
    int shared_memory = number_of_states * 7 * sizeof(OdeState) +
                    number_of_states * sizeof(OdeState) +
                    number_of_states * sizeof(FieldStepperResult);

    // Launch the kernel
    cudaEventRecord(start);
    if (number_threads > 1){
        dormand_test_arg_kernel<Stepper_multi>
            <<<1, thread_dimension, shared_memory>>>(d_states,
                                                       d_results,
                                                       d_num_states,
                                                       d_number_iterations,
                                                       d_number_threads);
    } else {
        dormand_test_arg_kernel<Stepper_uni>
            <<<1, thread_dimension>>>(d_states,
                                      d_results,
                                      d_num_states,
                                      d_number_iterations,
                                      d_number_threads);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // Compute the elapsed time
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&(result.milliseconds), start, stop);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy results back to host
    cudaMemcpy(h_results,
               d_results,
               number_of_states * sizeof(FieldStepperResult),
               cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_results);
    cudaFree(d_states);
    cudaFree(d_num_states);
    cudaFree(d_number_iterations);
    cudaFree(d_number_threads);

    // Return results
    result.results = h_results;
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
