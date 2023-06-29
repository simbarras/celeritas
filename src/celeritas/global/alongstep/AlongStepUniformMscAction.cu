//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepUniformMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "corecel/io/Logger.hh"

#include "detail/AlongStepKernels.hh"
#include "detail/PropagationApplier.hh"
#include "detail/UniformFieldPropagatorFactory.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepUniformMscAction::execute(CoreParams const& params,
                                        CoreStateDevice& state) const
{
    if (this->has_msc())
    {
        detail::launch_limit_msc_step(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        // Object that will be executed by the kernel
        auto execute_thread = make_along_step_track_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            this->action_id(),
            // Propagator
            detail::PropagationApplier{
                detail::UniformFieldPropagatorFactory{field_params_}});
        static ActionLauncher<decltype(execute_thread)> const launch_kernel(
            *this, "propagate");

        // HACK: Changed to launch 4 threads per track
        // launch_kernel(params, state, *this, execute_thread); // old
        size_type num_threads = state.size() * 4;
        StreamId stream_id = state.stream_id();
        CELER_LOG(debug) << "Launching " << num_threads << " threads";
        launch_kernel(num_threads, stream_id, execute_thread); // new
    }
    if (this->has_msc())
    {
        detail::launch_apply_msc(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    detail::launch_update_time(*this, params, state);
    detail::launch_apply_eloss(*this, params, state);
    detail::launch_update_track(*this, params, state);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
