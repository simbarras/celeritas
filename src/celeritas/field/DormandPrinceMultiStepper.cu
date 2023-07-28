//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceMultiStepper.cu
//---------------------------------------------------------------------------//
#include "DormandPrinceMultiStepper.hh"

#ifdef __CUDA_ARCH__

// #define debug_print(...) printf(__VA_ARGS__)
#ifndef debug_print
#   define debug_print(...)
#endif

namespace celeritas
{
template<class E>
CELER_FUNCTION auto
DormandPrinceMultiStepper<E>::operator()(real_type step,
                                         OdeState const& beg_state,
                                         int num_threads) const
    -> result_type
{
    constexpr int num_k = 7;
    constexpr int num_coef = 32;

    int num_states = (blockDim.x * gridDim.x) / num_threads;
    int id = (threadIdx.x + blockIdx.x * blockDim.x) / num_threads;
    int index = (threadIdx.x + blockIdx.x * blockDim.x) % num_threads;

    extern __shared__ void* shared_memory[];
    OdeState* shared_ks = (OdeState*)shared_memory;
    OdeState* shared_along_state = reinterpret_cast<OdeState*>(&shared_ks[num_k*num_states]);
    FieldStepperResult* shared_result = reinterpret_cast<FieldStepperResult*>(&shared_along_state[num_states]);


    int mask = (num_threads * num_threads - 1) << (id * num_threads);

    if (index == 0)
    {
        run_sequential(step, beg_state, id, mask, &shared_ks[num_k * id], &shared_along_state[id], &shared_result[id]);
    }
    else
    {
        R* shared_coef = reinterpret_cast<R*>(&shared_result[num_states]);
        run_aside(step, beg_state, id, index, mask, &shared_ks[num_k * id], &shared_along_state[id], &shared_result[id], &shared_coef[num_coef * id]);
    }

    // return *result;
    return shared_result[id];
}

template<class E>
CELER_FUNCTION void
DormandPrinceMultiStepper<E>::run_aside(real_type step,
                                        OdeState const& beg_state,
                                        int id,
                                        int index,
                                        int mask,
                                        OdeState* ks,
                                        OdeState* along_state,
                                        FieldStepperResult* result,
                                        R* computed_coef) const
{
    // Coefficients for Dormand-Prince Rks[4](4)7M
    constexpr R a11 = 0.2;

    constexpr R a21 = 0.075;
    constexpr R a22 = 0.225;

    constexpr R a31 = 44 / R(45);
    constexpr R a32 = -56 / R(15);
    constexpr R a33 = 32 / R(9);

    constexpr R a41 = 19372 / R(6561);
    constexpr R a42 = -25360 / R(2187);
    constexpr R a43 = 64448 / R(6561);
    constexpr R a44 = -212 / R(729);

    constexpr R a51 = 9017 / R(3168);
    constexpr R a52 = -355 / R(33);
    constexpr R a53 = 46732 / R(5247);
    constexpr R a54 = 49 / R(176);
    constexpr R a55 = -5103 / R(18656);

    constexpr R a61 = 35 / R(384);
    constexpr R a63 = 500 / R(1113);
    constexpr R a64 = 125 / R(192);
    constexpr R a65 = -2187 / R(6784);
    constexpr R a66 = 11 / R(84);

    constexpr R d71 = a61 - 5179 / R(57600);
    constexpr R d73 = a63 - 7571 / R(16695);
    constexpr R d74 = a64 - 393 / R(640);
    constexpr R d75 = a65 + 92097 / R(339200);
    constexpr R d76 = a66 - 187 / R(2100);
    constexpr R d77 = -1 / R(40);

    // Coefficients for the mid point calculation by Shampine
    constexpr R c71 = 6025192743 / R(30085553152);
    constexpr R c73 = 51252292925 / R(65400821598);
    constexpr R c74 = -2691868925 / R(45128329728);
    constexpr R c75 = 187940372067 / R(1594534317056);
    constexpr R c76 = -1776094331 / R(19743644256);
    constexpr R c77 = 11237099 / R(235043384);

    // Coefficients for the vector multiplication
    constexpr R coef[] = {
                         a11, a21, a22, // Start
                         a31, a32, a33, // Loop 0
                         a41, a42, a43, // loop 1
                         a44, a51, a52, // loop 2
                         a53, a54, a55, // loop 3
                         a61, a63, a64, // loop 4
                         a65, a66, d71, // After loop
                         d73, d74, d75, // Before end
                         d76, d77, c71,
                         c73, c74, c75,
                         c76, c77};
    constexpr int coef_offset = 6;
    int coef_counter = 0;
    int pre_coef_counter = index-1;

    #define COMPUTE_COEF(i) \
        computed_coef[*i] = step * coef[*i]; \
        *i += 3;

    // 0 to 2
    COMPUTE_COEF(&pre_coef_counter);

    // Vector multiplication for step one to five
    for (int i = 0; i < 5; i++){
        debug_print("Thread %d index %d waiting before step %d\n", id, index, i+1);
        __syncwarp(mask);
        for (int j = 0; j <= i; j++){
            UPDATE_STATE(
                index, (*along_state), computed_coef[coef_counter], ks[j]);
            coef_counter++;
        }
        __syncwarp(mask);
        // 3 to 5, 6 to 8, 9 to 11, 12 to 14, 15 to 17
        COMPUTE_COEF(&pre_coef_counter);
    }

    // 18 to 20
    COMPUTE_COEF(&pre_coef_counter);

    // Vector multiplication for step six: end state
    debug_print("Thread %d index %d waiting before step end_state\n", id, index);
    __syncwarp(mask);
    for (int j = 0; j < 6; j++){
        if (j==1) continue; // because a62 = 0
        UPDATE_STATE(index, result->end_state, computed_coef[coef_counter], ks[j]);
        coef_counter++;
    }
    __syncwarp(mask);

    // 21 to 23
    COMPUTE_COEF(&pre_coef_counter);
    if (pre_coef_counter < 32 - coef_offset){
        computed_coef[pre_coef_counter] = step * coef[pre_coef_counter];
        pre_coef_counter += 3;
    }
    while (pre_coef_counter < 32){
        computed_coef[pre_coef_counter] = step * coef[pre_coef_counter] / R(2);
        pre_coef_counter += 3;
    }

    // Vector mutltiplication for step eight and nine: error and mid state
    debug_print("Thread %d index %d waiting before step mid_state and err_state\n", id, index);
    __syncwarp(mask);
    for (int j = 0; j < 7; j++){
        if (j==1) continue; // because d72 and c72 = 0
        UPDATE_STATE(index, result->err_state, computed_coef[coef_counter], ks[j]);
        UPDATE_STATE(
            index, result->mid_state, computed_coef[coef_counter + coef_offset], ks[j]);
        coef_counter++;
    }
    __syncwarp(mask);
}

template<class E>
CELER_FUNCTION void
DormandPrinceMultiStepper<E>::run_sequential(real_type step,
                                             OdeState const& beg_state,
                                             int id,
                                             int mask,
                                             OdeState* ks,
                                             OdeState* along_state,
                                             FieldStepperResult* result) const
{

    // First step
    // debug_print("Step 1 ------------------\n");
    ks[0] = calc_rhs_(beg_state);
    *along_state = beg_state;
    debug_print("Main thread %d disptach to worker for step 1\n", id);
    DISPATCH_VECT_MULT(mask);

    // Second step
    // debug_print("Step 2 ------------------\n");
    ks[1] = calc_rhs_(*along_state);
    *along_state = beg_state;
    debug_print("Main thread %d disptach to worker for step 2\n", id);
    DISPATCH_VECT_MULT(mask);

    // Third step
    // debug_print("Step 3 ------------------\n");
    ks[2] = calc_rhs_(*along_state);
    *along_state = beg_state;
    debug_print("Main thread %d disptach to worker for step 3\n", id);
    DISPATCH_VECT_MULT(mask);

    // Fourth step
    // debug_print("Step 4 ------------------\n");
    ks[3] = calc_rhs_(*along_state);
    *along_state = beg_state;
    debug_print("Main thread %d disptach to worker for step 4\n", id);
    DISPATCH_VECT_MULT(mask);

    // Fifth step
    // debug_print("Step 5 ------------------\n");
    ks[4] = calc_rhs_(*along_state);
    *along_state = beg_state;
    debug_print("Main thread %d disptach to worker for step 5\n", id);
    DISPATCH_VECT_MULT(mask);

    // Sixth step
    // debug_print("Step 6 ------------------\n");
    ks[5] = calc_rhs_(*along_state);
    result->end_state = beg_state;
    debug_print("Main thread %d disptach to worker for step end state\n", id);
    DISPATCH_VECT_MULT(mask);

    // Seventh step: the final step
    ks[6] = calc_rhs_(result->end_state);

    // The error estimate and the mid point
    result->err_state = {{0, 0, 0}, {0, 0, 0}};
    result->mid_state = beg_state;
    debug_print("Main thread %d disptach to worker for step err state and mid state\n", id);
    DISPATCH_VECT_MULT(mask);

    // debug_print("Result mid state pos: %f %f %f\n",
    //             result->mid_state.pos[0],
    //             result->mid_state.pos[1],
    //             result->mid_state.pos[2]);
    // debug_print("Finish main task for thread %d index %d\n", id, 0);
}

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // __CUDA_ARCH__