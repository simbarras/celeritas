//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/WentzelModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;
class IsotopeView;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Wentzel Coulomb scattering model interaction.
 */
class WentzelModel final : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

    //! Wentzel model configuration options
    struct Options
    {
        //! Nuclear form factor model
        NuclearFormFactorType form_factor_model{
            NuclearFormFactorType::exponential};

        //! User defined screening factor
        real_type screening_factor{1};

        //! Whether to use integral method to sample interaction length
        bool use_integral_xs{true};

        //! Check if the options are valid
        explicit operator bool() const { return screening_factor > 0; }
    };

  public:
    // Construct from model ID and other necessary data
    WentzelModel(ActionId id,
                 ParticleParams const& particles,
                 MaterialParams const& materials,
                 Options const& options,
                 SPConstImported data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void execute(CoreParams const&, CoreStateDevice&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "coulomb-wentzel"; }

    //! Short description of the post-step action
    std::string description() const final
    {
        return "interact by Coulomb scattering (Wentzel)";
    }

    //!@{
    //! Access model data
    WentzelHostRef const& host_ref() const { return data_.host(); }
    WentzelDeviceRef const& device_ref() const { return data_.device(); }
    //!@}

  private:
    CollectionMirror<WentzelData> data_;
    ImportedModelAdapter imported_;

    // Construct per element data (loads Mott coefficients)
    void build_data(HostVal<WentzelData>& host_data,
                    MaterialParams const& materials);

    // Retrieve matrix of interpolated Mott coefficients
    static WentzelElementData::MottCoeffMatrix
    get_mott_coeff_matrix(AtomicNumber z);

    // Calculate the nuclear form prefactor
    static real_type calc_nuclear_form_prefactor(IsotopeView const& iso);
};

//---------------------------------------------------------------------------//
}  //  namespace celeritas
