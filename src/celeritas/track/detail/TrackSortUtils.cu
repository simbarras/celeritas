//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.cu
//---------------------------------------------------------------------------//
#include "TrackSortUtils.hh"

#include <random>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Stream.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//

template<class T>
using StateItems
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::device>;

template<class T>
using ThreadItems
    = Collection<T, Ownership::reference, MemSpace::device, ThreadId>;

using TrackSlots = ThreadItems<TrackSlotId::size_type>;

//---------------------------------------------------------------------------//

template<class F>
void partition_impl(TrackSlots const& track_slots, F&& func, StreamId stream_id)
{
    auto start = device_pointer_cast(track_slots.data());
    thrust::partition(
        thrust::device.on(celeritas::device().stream(stream_id).get()),
        start,
        start + track_slots.size(),
        std::forward<F>(func));
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//

template<class F>
void sort_impl(TrackSlots const& track_slots, F&& func, StreamId stream_id)
{
    auto start = device_pointer_cast(track_slots.data());
    thrust::sort(thrust::device.on(celeritas::device().stream(stream_id).get()),
                 start,
                 start + track_slots.size(),
                 std::forward<F>(func));
    CELER_DEVICE_CHECK_ERROR();
}

// PRE: get_action is sorted, i.e. i <= j ==> get_action(i) <=
// get_action(j)
template<class F>
__device__ void
tracks_per_action_impl(Span<ThreadId> offsets, size_type size, F&& get_action)
{
    ThreadId tid = celeritas::KernelParamCalculator::thread_id();

    if ((tid < size) && tid != ThreadId{0})
    {
        ActionId current_action = get_action(tid);
        ActionId previous_action = get_action(tid - 1);
        if (current_action && current_action != previous_action)
        {
            offsets[current_action.unchecked_get()] = tid;
        }
    }
    // needed if the first action range has only one element
    if (tid == ThreadId{0})
    {
        if (ActionId first = get_action(tid))
        {
            offsets[first.unchecked_get()] = tid;
        }
    }
}

__global__ void tracks_per_action_kernel(DeviceRef<CoreStateData> const states,
                                         Span<ThreadId> offsets,
                                         size_type size,
                                         TrackOrder order)
{
    switch (order)
    {
        case TrackOrder::sort_along_step_action:
            return tracks_per_action_impl(
                offsets,
                size,
                AlongStepActionAccessor{states.sim.along_step_action.data(),
                                        states.track_slots.data()});
        case TrackOrder::sort_step_limit_action:
            return tracks_per_action_impl(
                offsets,
                size,
                StepLimitActionAccessor{states.sim.step_limit.data(),
                                        states.track_slots.data()});
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Initialize default threads to track_slots mapping, track_slots[i] = i.
 *
 * TODO: move to global/detail
 */
template<>
void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type> track_slots)
{
    thrust::sequence(
        thrust::device_pointer_cast(track_slots.data()),
        thrust::device_pointer_cast(track_slots.data() + track_slots.size()),
        0);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Shuffle track slots.
 *
 * TODO: move to global/detail
 */
template<>
void shuffle_track_slots<MemSpace::device>(
    Span<TrackSlotId::size_type> track_slots)
{
    using result_type = thrust::default_random_engine::result_type;
    thrust::default_random_engine g{
        static_cast<result_type>(track_slots.size())};
    auto start = thrust::device_pointer_cast(track_slots.data());
    thrust::shuffle(thrust::device, start, start + track_slots.size(), g);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Sort or partition tracks.
 */
void sort_tracks(DeviceRef<CoreStateData> const& states, TrackOrder order)
{
    switch (order)
    {
        case TrackOrder::partition_status:
            return partition_impl(states.track_slots,
                                  alive_predicate{states.sim.status.data()},
                                  states.stream_id);
        case TrackOrder::sort_along_step_action:
            return sort_impl(
                states.track_slots,
                along_action_comparator{states.sim.along_step_action.data()},
                states.stream_id);
        case TrackOrder::sort_step_limit_action:
            return sort_impl(
                states.track_slots,
                step_limit_comparator{states.sim.step_limit.data()},
                states.stream_id);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Count tracks associated to each action that was used to sort them, specified
 * by order. Result is written in the output parameter offsets which sould be
 * of size num_actions + 1.
 */
void count_tracks_per_action(
    DeviceRef<CoreStateData> const& states,
    Span<ThreadId> offsets,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>& out,
    TrackOrder order)
{
    if (order == TrackOrder::sort_along_step_action
        || order == TrackOrder::sort_step_limit_action)
    {
        // dispatch in the kernel since CELER_LAUNCH_KERNEL doesn't work
        // with templated kernels
        auto start = device_pointer_cast(make_observer(offsets.data()));
        thrust::fill(start, start + offsets.size(), ThreadId{});
        CELER_DEVICE_CHECK_ERROR();
        CELER_LAUNCH_KERNEL(tracks_per_action,
                            celeritas::device().default_block_size(),
                            states.size(),
                            celeritas::device().stream(states.stream_id).get(),
                            states,
                            offsets,
                            states.size(),
                            order);
        Span<ThreadId> sout = out[AllItems<ThreadId, MemSpace::host>{}];
        Copier<ThreadId, MemSpace::host> copy_to_host{sout};
        copy_to_host(MemSpace::device, offsets);
        backfill_action_count(sout, states.size());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
