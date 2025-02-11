//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/g4vg/Scaler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <G4TwoVector.hh>

#include "celeritas/Units.hh"

namespace celeritas
{
namespace g4vg
{
//---------------------------------------------------------------------------//
/*!
 * Convert a unit from Geant4 scale to another.
 *
 * Currently the scale is hardcoded as 1/mm but could easily be a class
 * attribute.
 */
class Scaler
{
  public:
    //! Convert a positional scalar
    double operator()(double val) const { return val * scale_; }

    //! Convert a two-vector
    std::pair<double, double> operator()(G4TwoVector const& vec) const
    {
        return {(*this)(vec.x()), (*this)(vec.y())};
    }

    //! Convert a three-vector
    vecgeom::Vector3D<double> operator()(G4ThreeVector const& vec) const
    {
        return {(*this)(vec.x()), (*this)(vec.y()), (*this)(vec.z())};
    }

  private:
    inline static constexpr double scale_ = units::millimeter;
};

//---------------------------------------------------------------------------//
}  // namespace g4vg
}  // namespace celeritas
