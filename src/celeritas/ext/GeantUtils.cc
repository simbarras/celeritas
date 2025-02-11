//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantUtils.cc
//---------------------------------------------------------------------------//
#include "GeantUtils.hh"

#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4Version.hh>

#if G4VERSION_NUMBER < 1070
#    include <G4MTRunManager.hh>
#endif
#if G4VERSION_NUMBER >= 1070
#    include <G4Backtrace.hh>
#endif

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Clear Geant4's signal handlers that get installed on startup/activation.
 *
 * This should be called before instantiating a run manager.
 */
void disable_geant_signal_handler()
{
#if G4VERSION_NUMBER >= 1070
    CELER_LOG(debug) << "Disabling Geant4 signal handlers";
    // Disable geant4 signal interception
    G4Backtrace::DefaultSignals() = {};
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of threads in a version-portable way.
 *
 * G4RunManager::GetNumberOfThreads isn't virtual before Geant4 v10.7.0 so we
 * need to explicitly dynamic cast to G4MTRunManager to get the number of
 * threads.
 */
int get_geant_num_threads(G4RunManager const& runman)
{
    // Default is 1 if not multithreaded
    int result{1};
#if G4VERSION_NUMBER >= 1070
    result = runman.GetNumberOfThreads();
#else
    if (auto const* runman_mt = dynamic_cast<G4MTRunManager const*>(&runman))
    {
        result = runman_mt->GetNumberOfThreads();
    }
#endif
    // TODO: even if run manager exists, it might not be initialized, so the
    // result might be nonpositive. See debug-vecgeom 'minimal' example
    // failure.
    CELER_ENSURE(result > 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of threads from the global run manager.
 */
int get_geant_num_threads()
{
    auto* run_man = G4RunManager::GetRunManager();
    CELER_VALIDATE(run_man,
                   << "cannot query global thread count before G4RunManager "
                      "is created");
    return get_geant_num_threads(*run_man);
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 thread ID.
 */
int get_geant_thread_id()
{
    // Thread ID is -1 when running serially
    if (G4Threading::IsMultithreadedApplication())
    {
        return G4Threading::G4GetThreadId();
    }
    return 0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
