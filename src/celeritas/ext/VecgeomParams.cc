//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomParams.cc
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include <cstddef>
#include <vector>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "celeritas_config.h"
#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedLimitSaver.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/ext/detail/VecgeomCompatibility.hh"
#if CELERITAS_USE_CUDA
#    include <VecGeom/management/CudaManager.h>
#    include <cuda_runtime_api.h>
#endif
#ifdef VECGEOM_USE_SURF
#    include <VecGeom/surfaces/BrepHelper.h>
#endif

#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedMem.hh"

#include "VecgeomData.hh"  // IWYU pragma: associated
#include "g4vg/Converter.hh"

#ifdef VECGEOM_USE_SURF
#    include "VecgeomParams.surface.hh"
#endif

static_assert(std::is_same_v<celeritas::real_type, vecgeom::Precision>,
              "Celeritas and VecGeom real types do not match");

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// MACROS
// VecGeom interfaces change based on whether CUDA and Surface capabilities
// are available. Use macros to hide the calls.
//---------------------------------------------------------------------------//

#ifdef VECGEOM_ENABLE_CUDA
#    define VG_CUDA_CALL(CODE) CODE
#else
#    define VG_CUDA_CALL(CODE) CELER_UNREACHABLE
#endif

#ifdef VECGEOM_USE_SURF
#    define VG_SURF_CALL(CODE) CODE
#else
#    define VG_SURF_CALL(CODE) \
        do                     \
        {                      \
        } while (0)
#endif

#if defined(VECGEOM_ENABLE_CUDA) && defined(VECGEOM_USE_SURF)
#    define VG_CUDASURF_CALL(CODE) CODE
#else
#    define VG_CUDASURF_CALL(CODE) CELER_UNREACHABLE
#endif

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the verbosity setting for vecgeom.
 */
int vecgeom_verbosity()
{
    static int const result = [] {
        std::string var = celeritas::getenv("VECGEOM_VERBOSE");
        if (var.empty())
        {
            return 0;
        }
        return std::stoi(var);
    }();
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether surface tracking is being used.
 */
bool VecgeomParams::use_surface_tracking()
{
#ifdef VECGEOM_USE_SURF
    return true;
#else
    return false;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
VecgeomParams::VecgeomParams(std::string const& filename)
{
    CELER_LOG(status) << "Loading VecGeom geometry from GDML at " << filename;
    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    ScopedMem record_mem("VecgeomParams.construct");
    {
        ScopedProfiling profile_this{"load-vecgeom"};
        ScopedMem record_mem("VecgeomParams.load_geant_geometry");
        ScopedTimeAndRedirect time_and_output_("vgdml::Frontend");
        vgdml::Frontend::Load(filename, /* validate_xml_schema = */ false);
    }

    this->build_tracking();
    this->build_data();
    this->build_metadata();

    CELER_ENSURE(this->num_volumes() > 0);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Translate a geometry from Geant4.
 */
VecgeomParams::VecgeomParams(G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    ScopedMem record_mem("VecgeomParams.construct");

    {
        // Convert the geometry to VecGeom
        ScopedProfiling profile_this{"load-vecgeom"};
        g4vg::Converter::Options opts;
        opts.compare_volumes
            = !celeritas::getenv("G4VG_COMPARE_VOLUMES").empty();
        g4vg::Converter convert{opts};
        auto result = convert(world);
        CELER_ASSERT(result.world != nullptr);
        g4log_volid_map_ = std::move(result.volumes);

        // Set as world volume
        auto& vg_manager = vecgeom::GeoManager::Instance();
        vg_manager.RegisterPlacedVolume(result.world);
        vg_manager.SetWorldAndClose(result.world);

        // NOTE: setting and closing changes the world
        CELER_ASSERT(vg_manager.GetWorld() != nullptr);
    }

    this->build_tracking();
    this->build_data();
    this->build_metadata();

    CELER_ENSURE(this->num_volumes() > 0);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up vecgeom on destruction.
 */
VecgeomParams::~VecgeomParams()
{
    if (device_ref_)
    {
        if (VecgeomParams::use_surface_tracking())
        {
            CELER_LOG(debug) << "Clearing VecGeom surface GPU data";
            VG_CUDASURF_CALL(teardown_surface_tracking_device());
        }
        else
        {
            CELER_LOG(debug) << "Clearing VecGeom GPU data";
            VG_CUDA_CALL(vecgeom::CudaManager::Instance().Clear());
        }
    }

    if (VecgeomParams::use_surface_tracking())
    {
        CELER_LOG(debug) << "Clearing SurfModel CPU data";
        VG_SURF_CALL(vgbrep::BrepHelper<real_type>::Instance().ClearData());
    }

    CELER_LOG(debug) << "Clearing VecGeom CPU data";
    vecgeom::GeoManager::Instance().Clear();
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a placed volume ID.
 */
Label const& VecgeomParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_.get(vol);
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label.
 */
auto VecgeomParams::find_volume(std::string const& name) const -> VolumeId
{
    auto result = vol_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "volume '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a label.
 *
 * If the label isn't in the geometry, a null ID will be returned.
 */
VolumeId VecgeomParams::find_volume(Label const& label) const
{
    return vol_labels_.find(label);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 logical volume.
 */
VolumeId VecgeomParams::find_volume(G4LogicalVolume const* volume) const
{
    VolumeId result{};
    if (volume)
    {
        auto iter = g4log_volid_map_.find(volume);
        if (iter != g4log_volid_map_.end())
            result = iter->second;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get zero or more volume IDs corresponding to a name.
 *
 * This is useful for volumes that are repeated in the geometry with different
 * uniquifying 'extensions' from Geant4.
 */
auto VecgeomParams::find_volumes(std::string const& name) const
    -> SpanConstVolumeId
{
    return vol_labels_.find_all(name);
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids, set up VecGeom tracking data and copy to GPU.
 */
void VecgeomParams::build_tracking()
{
    CELER_EXPECT(vecgeom::GeoManager::Instance().GetWorld());
    CELER_LOG(status) << "Initializing tracking information";
    ScopedProfiling profile_this{"initialize-vecgeom"};
    ScopedMem record_mem("VecgeomParams.build_tracking");

    if (VecgeomParams::use_surface_tracking())
    {
        this->build_surface_tracking();
    }
    else
    {
        this->build_volume_tracking();
    }

    // TODO: we stil lneed to make volume tracking information when using CUDA,
    // because we need a GPU world device pointer. We could probably just make
    // a single world physical/logical volume that have the correct IDs.
    if (CELERITAS_USE_CUDA && VecgeomParams::use_surface_tracking())
    {
        this->build_volume_tracking();
    }
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids, set up VecGeom surface data and copy to GPU.
 */
void VecgeomParams::build_surface_tracking()
{
    VG_SURF_CALL(auto& brep_helper = vgbrep::BrepHelper<real_type>::Instance());
    VG_SURF_CALL(brep_helper.SetVerbosity(vecgeom_verbosity()));

    {
        CELER_LOG(debug) << "Creating surfaces";
        ScopedTimeAndRedirect time_and_output_("BrepHelper::Convert");
        VG_SURF_CALL(CELER_VALIDATE(brep_helper.Convert(),
                                    << "failed to convert VecGeom to "
                                       "surfaces"));
        if (vecgeom_verbosity() > 1)
        {
            VG_SURF_CALL(brep_helper.PrintSurfData());
        }
    }

    if (celeritas::device())
    {
        CELER_LOG(debug) << "Transferring surface data to GPU";
        ScopedTimeAndRedirect time_and_output_(
            "BrepCudaManager::TransferSurfData");

        VG_CUDASURF_CALL(
            setup_surface_tracking_device(brep_helper.GetSurfData()));
        CELER_DEVICE_CHECK_ERROR();
    }
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids, set up VecGeom tracking data and copy to GPU.
 *
 * After instantiating the CUDA manager, which changes the stack limits, we
 * adjust the stack size based on a user variable due to VecGeom recursive
 * virtual function calls. This is necessary for deeply nested geometry such as
 * CMS, as well as certain cases with debug symbols and assertions.
 *
 * See https://github.com/celeritas-project/celeritas/issues/614
 */
void VecgeomParams::build_volume_tracking()
{
    CELER_EXPECT(vecgeom::GeoManager::Instance().GetWorld());

    {
        ScopedTimeAndRedirect time_and_output_("vecgeom::ABBoxManager");
        vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    }

    // Init the bounding volume hierarchy structure
    vecgeom::cxx::BVHManager::Init();

    if (celeritas::device())
    {
        {
            // NOTE: this *MUST* be the first time the CUDA manager is called,
            // otherwise we can't restore limits.
            ScopedLimitSaver save_cuda_limits;
            VG_CUDA_CALL(vecgeom::cxx::CudaManager::Instance());
        }

        // Set custom stack and heap size now that it's been initialized
        if (std::string var = celeritas::getenv("CUDA_STACK_SIZE");
            !var.empty())
        {
            int stack_size = std::stoi(var);
            CELER_VALIDATE(stack_size > 0,
                           << "invalid CUDA_STACK_SIZE=" << stack_size
                           << " (must be positive)");
            set_cuda_stack_size(stack_size);
        }
        else if constexpr (CELERITAS_DEBUG)
        {
            // Default to a large stack size due to debugging code.
            set_cuda_stack_size(16384);
        }

        if (std::string var = celeritas::getenv("CUDA_HEAP_SIZE"); !var.empty())
        {
            int heap_size = std::stoi(var);
            CELER_VALIDATE(heap_size > 0,
                           << "invalid CUDA_HEAP_SIZE=" << heap_size
                           << " (must be positive)");
            set_cuda_heap_size(heap_size);
        }

#if CELERITAS_USE_CUDA
        auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
        cuda_manager.set_verbose(vecgeom_verbosity());
        {
            CELER_LOG(debug) << "Converting to CUDA geometry";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::CudaManager.LoadGeometry");

            cuda_manager.LoadGeometry();
            CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
        }
        {
            CELER_LOG(debug) << "Transferring geometry to GPU";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::CudaManager.Synchronize");
            auto world_top_devptr = cuda_manager.Synchronize();
            CELER_DEVICE_CHECK_ERROR();
            CELER_VALIDATE(world_top_devptr != nullptr,
                           << "VecGeom failed to copy geometry to GPU");
        }
        {
            CELER_LOG(debug) << "Initializing BVH on GPU";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::BVHManager::DeviceInit");
            vecgeom::cxx::BVHManager::DeviceInit();
            CELER_DEVICE_CHECK_ERROR();
        }
#else
        CELER_NOT_CONFIGURED("CUDA");
#endif
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct host/device Celeritas data after setting up VecGeom tracking.
 */
void VecgeomParams::build_data()
{
    ScopedMem record_mem("VecgeomParams.build_data");
    // Save host data
    auto& vg_manager = vecgeom::GeoManager::Instance();
    host_ref_.world_volume = vg_manager.GetWorld();
    host_ref_.max_depth = vg_manager.getMaxDepth();

    if (celeritas::device())
    {
#if CELERITAS_USE_CUDA
        auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
        device_ref_.world_volume = cuda_manager.world_gpu();
#endif
        device_ref_.max_depth = host_ref_.max_depth;
        CELER_ENSURE(device_ref_.world_volume);
    }
    CELER_ENSURE(host_ref_);
    CELER_ENSURE(!celeritas::device() || device_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas host-only metadata.
 */
void VecgeomParams::build_metadata()
{
    ScopedMem record_mem("VecgeomParams.build_metadata");
    // Construct volume labels
    vol_labels_ = LabelIdMultiMap<VolumeId>([] {
        auto& vg_manager = vecgeom::GeoManager::Instance();
        CELER_EXPECT(vg_manager.GetRegisteredVolumesCount() > 0);

        std::vector<Label> result(vg_manager.GetRegisteredVolumesCount());

        for (auto vol_idx : range<VolumeId::size_type>(result.size()))
        {
            // Get label
            vecgeom::LogicalVolume const* vol
                = vg_manager.FindLogicalVolume(vol_idx);
            CELER_ASSERT(vol);

            auto label = [vol] {
                std::string const& label = vol->GetLabel();
                if (starts_with(label, "[TEMP]"))
                {
                    // Temporary logical volume not directly used in transport
                    return Label{};
                }
                return Label::from_geant(label);
            }();

            result[vol_idx] = std::move(label);
        }
        return result;
    }());

    // Check for duplicates
    {
        auto vol_dupes = vol_labels_.duplicates();
        if (!vol_dupes.empty())
        {
            auto streamed_label = [this](std::ostream& os, VolumeId v) {
                os << '"' << this->vol_labels_.get(v) << "\" ("
                   << v.unchecked_get() << ')';
            };

            CELER_LOG(warning) << "Geometry contains duplicate volume names: "
                               << join_stream(vol_dupes.begin(),
                                              vol_dupes.end(),
                                              ", ",
                                              streamed_label);
        }
    }

    // Save world bbox
    bbox_ = [] {
        using namespace vecgeom;

        // Get world logical volume
        VPlacedVolume const* pv = GeoManager::Instance().GetWorld();

        // Calculate bounding box
        Vector3D<real_type> lower, upper;
        ABBoxManager::Instance().ComputeABBox(pv, &lower, &upper);

        return BBox{detail::to_array(lower), detail::to_array(upper)};
    }();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
