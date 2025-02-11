//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/detail/CuHipRngStateInit.cc
//---------------------------------------------------------------------------//
#include "CuHipRngStateInit.hh"

#include "corecel/cont/Range.hh"
#include "corecel/sys/ThreadId.hh"

#include "../CuHipRngData.hh"
#include "../CuHipRngEngine.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states from seeds randomly generated on host.
 */
void rng_state_init(HostRef<CuHipRngStateData> const& rng,
                    HostCRef<CuHipRngInitData> const& seeds)
{
    for (auto tid : range(TrackSlotId{seeds.size()}))
    {
        CuHipRngEngine engine(rng, tid);
        engine = seeds.seeds[tid];
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
