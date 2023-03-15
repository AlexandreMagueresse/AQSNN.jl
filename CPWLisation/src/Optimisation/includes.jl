module Optimisation

using Printf

using LinearAlgebra
using IterativeSolvers

using Knet
using AutoGrad
using Optim

using CPWLisation.Helpers
using CPWLisation.RegisteredFunctions

include("BoundaryCondition.jl")
export Asymptote
export PointFree
export PointValue
export PointSlope
export PointTangent

export getPoint

include("CPWL.jl")
export cpwl_ADAM
export cpwl_BFGS

include("Tangent.jl")
export tangent_ADAM
export tangent_BFGS

end # module Optimisation
